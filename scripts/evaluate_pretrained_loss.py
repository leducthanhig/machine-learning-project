"""
Evaluate a VITRA weights.pt checkpoint on a deterministic held-out sampler slice.

This script intentionally mirrors scripts/train.py for model construction and data
materialization, but runs forward-only loss evaluation from a chosen sampler step.
Use it to evaluate samples that were not consumed during a partial epoch training run.
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vitra.datasets.materialize import get_vla_dataset_and_collator
from vitra.datasets.data_mixture import HAND_MIXTURES
from vitra.models.vla_builder import build_vla, load_vla_checkpoint
from vitra.utils import set_global_seed, setup_seed
from vitra.utils.config_utils import load_config
from vitra.utils.overwatch import initialize_overwatch


os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

overwatch = initialize_overwatch(__name__)


class FixedBatchSampler:
    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        yield from self.batches

    def __len__(self):
        return len(self.batches)


def move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [move_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(v, device) for v in value)
    return value


def tensor_to_float(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().mean().cpu().item())
    return float(value)


def posix_to_str(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: posix_to_str(v) for k, v in value.items()}
    if isinstance(value, list):
        return [posix_to_str(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def resolve_weights_path(path: str) -> str:
    weights_path = Path(path)
    if weights_path.is_dir():
        weights_path = weights_path / "weights.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint weights not found: {weights_path}")
    return str(weights_path)


def cyclic_shift_batch(value: torch.Tensor) -> torch.Tensor:
    if value.shape[0] <= 1:
        return value
    return torch.roll(value, shifts=1, dims=0)


def get_local_rank() -> int:
    local_rank = getattr(overwatch, "local_rank", None)
    if callable(local_rank):
        return int(local_rank())
    return int(os.environ.get("LOCAL_RANK", 0))


def apply_inference_ablation(batch: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Apply input-only ablations for forward-loss evaluation."""
    state_mode = args.ablate_state
    if state_mode == "zero_state":
        batch["current_state"] = torch.zeros_like(batch["current_state"])
    elif state_mode == "no_state":
        batch["current_state"] = torch.zeros_like(batch["current_state"])
        batch["current_state_mask"] = torch.zeros_like(batch["current_state_mask"], dtype=torch.bool)
    elif state_mode == "shuffle_state":
        batch["current_state"] = cyclic_shift_batch(batch["current_state"])
        batch["current_state_mask"] = cyclic_shift_batch(batch["current_state_mask"])

    fov_mode = args.ablate_fov
    if fov_mode == "zero_fov":
        batch["fov"] = torch.zeros_like(batch["fov"])
    elif fov_mode == "shuffle_fov":
        batch["fov"] = cyclic_shift_batch(batch["fov"])

    return batch


def update_configs(configs: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    configs = copy.deepcopy(configs)

    if args.seed is not None:
        configs["seed"] = args.seed
    if args.data_mix is not None:
        configs["train_dataset"]["data_mix"] = args.data_mix
    if args.batch_size is not None:
        configs["batch_size"] = args.batch_size
    if args.total_batch_size is not None:
        configs["total_batch_size"] = args.total_batch_size
    if args.fwd_pred_next_n is not None:
        configs["fwd_pred_next_n"] = args.fwd_pred_next_n
    if args.repeated_diffusion_steps is not None:
        configs["repeated_diffusion_steps"] = args.repeated_diffusion_steps
    if args.use_bf16:
        configs["use_bf16"] = True

    configs["output_root"] = Path(configs["output_root"])
    configs["log_root"] = Path(configs["log_root"])
    configs["cache_root"] = Path(configs["cache_root"]) / configs["model"]
    if args.weights != "__dry_run_placeholder__":
        configs["model_load_path"] = resolve_weights_path(args.weights)
    else:
        configs["model_load_path"] = None  # dry_run: no checkpoint needed

    if not args.keep_train_augmentation:
        train_dataset = configs["train_dataset"]
        train_dataset["augmentation"] = False
        train_dataset["state_mask_prob"] = 0.0
        train_dataset["set_none_ratio"] = 0.0

    return configs


def collect_sampler_ids(batch_sampler, epoch: int, start_step: int, num_steps: int) -> set:
    batch_sampler.set_epoch(epoch, start_step)
    ids = set()
    for step_idx, batch in zip(range(num_steps), batch_sampler):
        ids.update(tuple(index) for index in batch)
    return ids


def count_available_samples(
    batch_sampler,
    epoch: int,
    cutoff_step: int,
    num_datasets: int,
    seen_ids: set = None,
    exclude_seen: bool = True,
) -> Dict[int, int]:
    """Count unique unseen samples available per dataset from cutoff_step to end-of-epoch.

    Returns a dict mapping dataset_id -> count of qualifying unique samples.
    This runs without loading the model and is fast enough to sweep across checkpoints.
    """
    batch_sampler.set_epoch(epoch, cutoff_step)
    seen_ids = seen_ids or set()
    per_dataset_seen: Dict[int, set] = {i: set() for i in range(num_datasets)}
    counts: Dict[int, int] = {i: 0 for i in range(num_datasets)}

    for mixed_batch in batch_sampler:
        for index in mixed_batch:
            index = tuple(index)
            dataset_id = index[0]
            if dataset_id not in counts:
                continue
            if exclude_seen and index in seen_ids:
                continue
            if index in per_dataset_seen[dataset_id]:
                continue
            per_dataset_seen[dataset_id].add(index)
            counts[dataset_id] += 1

    return counts


def get_dataset_names(vla_dataset) -> list:
    names = []
    for dataset in vla_dataset.datasets:
        names.append(getattr(dataset, "dataset_name", f"dataset_{len(names)}"))
    return names


def get_config_dataset_names(configs: Dict[str, Any]) -> list:
    data_mix = configs["train_dataset"]["data_mix"]
    if data_mix in HAND_MIXTURES:
        return [dataset_name for dataset_name, _ in HAND_MIXTURES[data_mix]]
    return [data_mix]


def collect_dataset_batches(
    batch_sampler,
    epoch: int,
    start_step: int,
    dataset_id: int,
    num_batches: int,
    batch_size: int,
    exclude_ids: set = None,
    unique: bool = True,
) -> list:
    """Filter the mixed sampler stream to fixed-size batches from one dataset id."""
    batch_sampler.set_epoch(epoch, start_step)
    batches = []
    current_batch = []
    exclude_ids = exclude_ids or set()
    used_ids = set()

    for mixed_batch in batch_sampler:
        for index in mixed_batch:
            index = tuple(index)
            if index[0] != dataset_id:
                continue
            if index in exclude_ids:
                continue
            if unique and index in used_ids:
                continue

            current_batch.append(index)
            used_ids.add(index)
            if len(current_batch) == batch_size:
                batches.append(current_batch)
                current_batch = []
                if len(batches) == num_batches:
                    return batches

    raise RuntimeError(
        f"Could only collect {len(batches)} batches for dataset_id={dataset_id}; "
        f"requested {num_batches}. Try a smaller --eval_batches, earlier --eval_sampler_step, "
        f"or disable --unique_per_dataset_eval."
    )


def make_dataset_and_sampler(configs: Dict[str, Any], processor, args: argparse.Namespace):
    batch_size = configs["batch_size"]
    train_dataset = configs["train_dataset"]

    vla_dataset, collator, batch_sampler = get_vla_dataset_and_collator(
        train_dataset["data_root_dir"],
        train_dataset["data_mix"],
        augmentation=train_dataset["augmentation"],
        shard_num=overwatch.world_size(),
        shard_index=overwatch.rank(),
        seed=configs["seed"],
        future_action_window_size=configs["fwd_pred_next_n"] - 1,
        processor=processor,
        batch_size=batch_size,
        normalization=train_dataset.get("normalization", True),
        flip_augmentation=train_dataset.get("flip_augmentation", 1.0),
        set_none_ratio=train_dataset.get("set_none_ratio", 0.0),
        action_type=train_dataset.get("action_type", "angle"),
        use_rel=train_dataset.get("use_rel", False),
        rel_mode=train_dataset.get("rel_mode", "step"),
        clip_len=train_dataset.get("clip_len", None),
        state_mask_prob=train_dataset.get("state_mask_prob", 0.0),
        target_image_height=train_dataset.get("target_image_height", 224),
    )

    setup_seed(configs["seed"], rank=overwatch.rank())
    return vla_dataset, collator, batch_sampler


def make_dataloader(vla_dataset, collator, batch_sampler, args: argparse.Namespace, configs: Dict[str, Any]):
    train_dataset = configs["train_dataset"]
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = train_dataset["num_workers"]
    prefetch_factor = args.prefetch_factor
    if prefetch_factor is None:
        prefetch_factor = train_dataset["prefetch_factor"]
    if num_workers == 0 or prefetch_factor == 0:
        prefetch_factor = None

    dataloader = DataLoader(
        vla_dataset,
        batch_sampler=batch_sampler,
        collate_fn=collator,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        worker_init_fn=set_global_seed(configs["seed"], get_worker_init_fn=True),
        persistent_workers=num_workers > 0,
        pin_memory=num_workers > 0,
    )
    return dataloader, batch_sampler


@torch.no_grad()
def evaluate(configs: Dict[str, Any], args: argparse.Namespace, dataset_name: str = None) -> Dict[str, Any]:
    device = torch.device(f"cuda:{get_local_rank()}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()

    model = build_vla(configs=configs)
    model = load_vla_checkpoint(model, configs["model_load_path"])
    model.model.use_bf16 = configs["use_bf16"]
    model.use_bf16 = configs["use_bf16"]
    model = model.to(device).eval()

    vla_dataset, collator, batch_sampler = make_dataset_and_sampler(configs, model.processor, args)
    dataset_names = get_dataset_names(vla_dataset)

    overlap_count = None
    seen_ids = None
    if args.seen_sampler_steps is not None:
        seen_ids = collect_sampler_ids(batch_sampler, args.seen_epoch, 0, args.seen_sampler_steps)

    if dataset_name is None:
        if seen_ids is not None:
            eval_ids = collect_sampler_ids(batch_sampler, args.eval_epoch, args.eval_sampler_step, args.eval_batches)
            overlap = seen_ids & eval_ids
            overlap_count = len(overlap)
            if overlap_count > 0 and not args.allow_seen_overlap:
                raise RuntimeError(
                    f"Evaluation sampler slice overlaps the seen sampler slice by {overlap_count} sample ids. "
                    f"This can happen with weighted oversampling. Increase --eval_sampler_step or pass "
                    f"--allow_seen_overlap for a weighted reference-set comparison."
                )
        batch_sampler.set_epoch(args.eval_epoch, args.eval_sampler_step)
        eval_batch_sampler = batch_sampler
        eval_dataset_id = None
    else:
        if dataset_name not in dataset_names:
            raise ValueError(f"Unknown dataset {dataset_name!r}. Available datasets: {dataset_names}")
        eval_dataset_id = dataset_names.index(dataset_name)
        fixed_batches = collect_dataset_batches(
            batch_sampler,
            args.eval_epoch,
            args.eval_sampler_step,
            eval_dataset_id,
            args.eval_batches,
            configs["batch_size"],
            exclude_ids=seen_ids if args.exclude_seen_ids else None,
            unique=args.unique_per_dataset_eval,
        )
        eval_batch_sampler = FixedBatchSampler(fixed_batches)

        if args.seen_sampler_steps is not None:
            eval_ids = {index for batch in fixed_batches for index in batch}
            overlap_count = len(seen_ids & eval_ids)
            if overlap_count > 0:
                raise RuntimeError(
                    f"Per-dataset evaluation for {dataset_name} overlaps the seen sampler slice by "
                    f"{overlap_count} sample ids. Increase --eval_sampler_step."
                )

    dataloader, _ = make_dataloader(vla_dataset, collator, eval_batch_sampler, args, configs)

    output_jsonl = Path(args.output_jsonl)
    if dataset_name is not None:
        output_jsonl = output_jsonl.with_name(f"{output_jsonl.stem}.{dataset_name}{output_jsonl.suffix}")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    component_sums: Dict[str, float] = {}
    component_values: Dict[str, list] = {}
    num_batches = 0
    num_examples = 0

    progress = tqdm(
        zip(range(args.eval_batches), dataloader),
        total=args.eval_batches,
        disable=not overwatch.is_rank_zero(),
        desc="evaluating",
    )
    with open(output_jsonl, "w") as f:
        for local_step, batch in progress:
            batch = move_to_device(batch, device)
            batch = apply_inference_ablation(batch, args)
            prediction = model.forward(
                batch["pixel_values"],
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                action_labels=batch["actions"],
                action_masks=batch["action_masks"],
                current_state_mask=batch["current_state_mask"],
                current_state=batch["current_state"],
                data_source=["action"],
                fov=batch["fov"],
            )

            record = {
                "eval_batch": local_step,
                "sampler_epoch": args.eval_epoch,
                "sampler_step": args.eval_sampler_step + local_step if dataset_name is None else None,
                "source_start_sampler_step": args.eval_sampler_step,
                "batch_size": int(batch["input_ids"].shape[0]),
                "dataset": dataset_name or "mixed",
                "ablate_state": args.ablate_state,
                "ablate_fov": args.ablate_fov,
                "repeated_diffusion_steps": configs.get("repeated_diffusion_steps"),
            }
            for key, value in prediction.items():
                record[key] = tensor_to_float(value)
                component_sums[key] = component_sums.get(key, 0.0) + record[key]
                component_values.setdefault(key, []).append(record[key])

            num_batches += 1
            num_examples += record["batch_size"]
            progress.set_postfix(loss=record.get("loss"))
            f.write(json.dumps(record) + "\n")

    summary = {
        "checkpoint": configs["model_load_path"],
        "dataset": dataset_name or "mixed",
        "dataset_id": eval_dataset_id,
        "dataset_names": dataset_names,
        "config": posix_to_str(configs),
        "eval_epoch": args.eval_epoch,
        "eval_sampler_step": args.eval_sampler_step,
        "eval_batches": num_batches,
        "eval_examples": num_examples,
        "seen_epoch": args.seen_epoch,
        "seen_sampler_steps": args.seen_sampler_steps,
        "seen_eval_overlap": overlap_count,
        "exclude_seen_ids": args.exclude_seen_ids if dataset_name is not None else False,
        "unique_per_dataset_eval": args.unique_per_dataset_eval if dataset_name is not None else False,
        "sampler_num_iters": batch_sampler.num_iters,
        "sampler_dataset_lengths": batch_sampler._dataset_lengths,
        "sampler_weights": batch_sampler.weights,
        "ablation": {
            "ablate_state": args.ablate_state,
            "ablate_fov": args.ablate_fov,
            "repeated_diffusion_steps": configs.get("repeated_diffusion_steps"),
        },
        "metrics": {},
    }
    for key, values in component_values.items():
        array = np.asarray(values, dtype=np.float64)
        summary["metrics"][key] = {
            "mean": float(array.mean()),
            "std": float(array.std()),
            "min": float(array.min()),
            "max": float(array.max()),
            "p10": float(np.quantile(array, 0.10)),
            "p50": float(np.quantile(array, 0.50)),
            "p90": float(np.quantile(array, 0.90)),
        }

    summary_path = output_jsonl.with_suffix(".summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VITRA weights.pt loss on a deterministic sampler slice.")
    parser.add_argument("--config", required=True, type=str, help="Training config JSON/YAML used for the run.")
    parser.add_argument(
        "--weights",
        default=None,
        type=str,
        help="Path to weights.pt or a checkpoint directory containing weights.pt. Not required for --dry_run.",
    )
    parser.add_argument("--output_jsonl", default=".tmp/eval_loss.jsonl", type=str, help="Path for per-batch JSONL metrics.")
    parser.add_argument("--eval_dataset", default=None, type=str, help="Evaluate one dataset from the configured mixture, e.g. epic or ssv2.")
    parser.add_argument("--eval_each_dataset", action="store_true", help="Evaluate each dataset in the configured mixture separately.")
    parser.add_argument("--eval_epoch", default=0, type=int, help="Sampler epoch to evaluate.")
    parser.add_argument("--eval_sampler_step", default=20000, type=int, help="Sampler micro-batch step to start evaluation from.")
    parser.add_argument("--eval_batches", default=200, type=int, help="Number of sampler batches to evaluate.")
    parser.add_argument("--seen_epoch", default=0, type=int, help="Sampler epoch for the seen slice overlap check.")
    parser.add_argument("--seen_sampler_steps", default=None, type=int, help="If set, assert eval ids do not overlap sampler steps [0, N).")
    parser.add_argument("--seen_optimizer_steps", default=None, type=int, help="Convert optimizer steps to seen sampler steps using gradient accumulation.")
    parser.add_argument("--grad_accumulation_steps", default=None, type=int, help="Override gradient accumulation when using --seen_optimizer_steps.")
    parser.add_argument(
        "--allow_seen_overlap",
        action="store_true",
        help="Allow raw sample-id overlap in mixed weighted evaluation. Useful because weighted oversampling can repeat ids.",
    )
    parser.add_argument(
        "--exclude_seen_ids",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For per-dataset evaluation, skip raw sample ids consumed in the seen sampler slice.",
    )
    parser.add_argument(
        "--unique_per_dataset_eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For per-dataset evaluation, avoid duplicate raw sample ids in the eval batches.",
    )
    parser.add_argument("--data_mix", default=None, type=str, help="Override train_dataset.data_mix.")
    parser.add_argument("--seed", default=None, type=int, help="Override config seed.")
    parser.add_argument("--batch_size", default=None, type=int, help="Override per-device batch size.")
    parser.add_argument("--total_batch_size", default=None, type=int, help="Override global batch size for metadata consistency.")
    parser.add_argument("--fwd_pred_next_n", default=None, type=int, help="Override future action horizon.")
    parser.add_argument("--repeated_diffusion_steps", default=None, type=int, help="Override repeated diffusion steps for forward-loss evaluation.")
    parser.add_argument("--num_workers", default=None, type=int, help="Override DataLoader workers.")
    parser.add_argument("--prefetch_factor", default=None, type=int, help="Override DataLoader prefetch factor.")
    parser.add_argument("--use_bf16", action="store_true", help="Force model use_bf16=True.")
    parser.add_argument(
        "--ablate_state",
        default="none",
        choices=["none", "zero_state", "no_state", "shuffle_state"],
        help=(
            "Inference-only state ablation. zero_state zeros state values but keeps masks; "
            "no_state zeros values and masks; shuffle_state cyclically shifts states within each batch."
        ),
    )
    parser.add_argument(
        "--ablate_fov",
        default="none",
        choices=["none", "zero_fov", "shuffle_fov"],
        help="Inference-only FOV ablation. shuffle_fov cyclically shifts FOV values within each batch.",
    )
    parser.add_argument(
        "--keep_train_augmentation",
        action="store_true",
        help="Keep train_dataset augmentation/state masking settings. Default disables stochastic eval transforms.",
    )
    # ── Dry-run / counting mode ────────────────────────────────────────────────
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help=(
            "Count unique unseen samples per dataset for each cutoff in --sweep_cutoff_steps "
            "without loading the model. For each cutoff C, samples from [0, C) are treated as "
            "seen (inferred automatically); samples from [C, epoch-end) that are unique and "
            "not in the seen set are counted. Total corpus size is derived from the dataset "
            "lengths. --seen_sampler_steps is ignored in this mode."
        ),
    )
    parser.add_argument(
        "--sweep_cutoff_steps",
        default=None,
        type=str,
        help=(
            "Comma-separated list of sampler cutoff steps to sweep in --dry_run mode "
            "(overrides --eval_sampler_step for each entry). "
            "Example: --sweep_cutoff_steps 64000,80000,96000,112000"
        ),
    )
    return parser.parse_args()


def run_dry_run(configs: Dict[str, Any], args: argparse.Namespace) -> None:
    """Count unique unseen samples available per dataset from one or more cutoff steps.

    Builds the sampler once (no model load). For each cutoff step C:
      - seen_ids are collected from sampler steps [0, C) automatically.
      - count_available_samples walks [C, epoch-end) and counts unique unseen samples.
      - total_corpus_size is derived from sum(batch_sampler._dataset_lengths).

    Prints a table of (cutoff, total_available, fraction_of_corpus, per_dataset)
    so you can identify which 16k-step checkpoint boundary leaves ~20% for eval.
    """
    import time

    overwatch.info("[dry_run] Building dataset and sampler (no model load) …")
    t0 = time.monotonic()

    # Build only the sampler; we need a processor to call get_vla_dataset_and_collator.
    # Construct the model CPU-only just to extract the processor, then discard it.
    from vitra.models.vla_builder import build_vla  # noqa: F811

    tmp_model = build_vla(configs=copy.deepcopy(configs))
    processor = tmp_model.processor
    del tmp_model

    vla_dataset, _collator, batch_sampler = make_dataset_and_sampler(configs, processor, args)
    dataset_names = get_dataset_names(vla_dataset)
    num_datasets = len(dataset_names)

    # Total unique corpus size is the sum of per-dataset lengths from the sampler.
    total_corpus = sum(batch_sampler._dataset_lengths)
    overwatch.info(
        f"[dry_run] Sampler ready in {time.monotonic() - t0:.1f}s — "
        f"{num_datasets} datasets: {dataset_names}, "
        f"total_corpus={total_corpus:,}"
    )

    # Determine which cutoff steps to sweep.
    if args.sweep_cutoff_steps is not None:
        cutoff_steps = [int(s.strip()) for s in args.sweep_cutoff_steps.split(",")]
    else:
        cutoff_steps = [args.eval_sampler_step]

    results = []
    for cutoff in cutoff_steps:
        # Seen samples = everything the model was trained on up to this checkpoint.
        # Inferred directly from the cutoff step; --seen_sampler_steps is ignored here.
        t1 = time.monotonic()
        overwatch.info(f"[dry_run] cutoff={cutoff}: collecting seen_ids from [0, {cutoff}) …")
        seen_ids = collect_sampler_ids(batch_sampler, args.seen_epoch, 0, cutoff)
        overwatch.info(
            f"[dry_run] cutoff={cutoff}: {len(seen_ids):,} seen sample-ids "
            f"in {time.monotonic() - t1:.1f}s"
        )

        t2 = time.monotonic()
        counts = count_available_samples(
            batch_sampler,
            epoch=args.eval_epoch,
            cutoff_step=cutoff,
            num_datasets=num_datasets,
            seen_ids=seen_ids if args.exclude_seen_ids else set(),
            exclude_seen=args.exclude_seen_ids,
        )
        elapsed = time.monotonic() - t2
        total_available = sum(counts.values())
        fraction = round(total_available / total_corpus, 4) if total_corpus > 0 else None
        entry = {
            "cutoff_step": cutoff,
            "seen_ids_count": len(seen_ids),
            "total_available": total_available,
            "fraction_of_corpus": fraction,
            "elapsed_s": round(elapsed, 2),
            "per_dataset": {dataset_names[i]: counts[i] for i in range(num_datasets)},
        }
        results.append(entry)
        overwatch.info(
            f"[dry_run] cutoff={cutoff}: available={total_available:,} "
            f"({fraction:.1%} of corpus) in {elapsed:.1f}s"
        )
        for name, cnt in entry["per_dataset"].items():
            overwatch.info(f"  {name}: {cnt:,}")

    # Write JSON summary.
    output_path = Path(args.output_jsonl).with_suffix(".dry_run.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dry_run_summary = {
        "mode": "dry_run",
        "exclude_seen_ids": args.exclude_seen_ids,
        "dataset_names": dataset_names,
        "total_corpus": total_corpus,
        "sampler_num_iters": batch_sampler.num_iters,
        "sampler_dataset_lengths": batch_sampler._dataset_lengths,
        "sampler_weights": batch_sampler.weights,
        "cutoff_sweep": results,
    }
    with open(output_path, "w") as f:
        json.dump(dry_run_summary, f, indent=2)
    overwatch.info(f"[dry_run] Summary written to {output_path}")
    if overwatch.is_rank_zero():
        print(json.dumps(dry_run_summary, indent=2))


if __name__ == "__main__":
    if not dist.is_initialized():
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29501")
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    args = parse_args()

    # In dry_run mode --weights is not required.
    if not args.dry_run and args.weights is None:
        raise ValueError("--weights is required unless --dry_run is set.")

    # Patch resolve_weights_path to be a no-op when weights not provided.
    if args.weights is None:
        args.weights = "__dry_run_placeholder__"

    configs = update_configs(load_config(args.config), args)

    if args.seen_optimizer_steps is not None and args.seen_sampler_steps is None:
        grad_accumulation_steps = args.grad_accumulation_steps
        if grad_accumulation_steps is None:
            grad_accumulation_steps = configs["total_batch_size"] // configs["batch_size"] // overwatch.world_size()
        args.seen_sampler_steps = args.seen_optimizer_steps * grad_accumulation_steps

    if args.eval_each_dataset and args.eval_dataset is not None:
        raise ValueError("Use either --eval_each_dataset or --eval_dataset, not both.")

    if args.dry_run:
        run_dry_run(configs, args)
    elif args.eval_each_dataset:
        dataset_names = get_config_dataset_names(configs)
        summaries = [evaluate(configs, args, dataset_name=name) for name in dataset_names]
        summary = {
            "datasets": [item["dataset"] for item in summaries],
            "summary_paths": [
                str(Path(args.output_jsonl).with_name(f"{Path(args.output_jsonl).stem}.{item['dataset']}{Path(args.output_jsonl).suffix}").with_suffix(".summary.json"))
                for item in summaries
            ],
            "metrics": {item["dataset"]: item["metrics"] for item in summaries},
        }
        if overwatch.is_rank_zero():
            print(json.dumps({"summary_paths": summary["summary_paths"], "metrics": summary["metrics"]}, indent=2))
    else:
        summary = evaluate(configs, args, dataset_name=args.eval_dataset)
        if overwatch.is_rank_zero():
            if args.eval_dataset is not None:
                output_jsonl = Path(args.output_jsonl)
                summary_path = str(output_jsonl.with_name(f"{output_jsonl.stem}.{args.eval_dataset}{output_jsonl.suffix}").with_suffix(".summary.json"))
            else:
                summary_path = str(Path(args.output_jsonl).with_suffix(".summary.json"))
            print(json.dumps({"summary_path": summary_path, "metrics": summary["metrics"]}, indent=2))

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
