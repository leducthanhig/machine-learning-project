# VITRA Pretraining and Benchmarking Project

This repository contains the source code, training configurations, notebooks, and evaluation protocols for pretraining and benchmarking the VITRA Vision-Language-Action (VLA) model. This project builds upon the official Microsoft VITRA repository: [microsoft/VITRA](https://github.com/microsoft/VITRA/).

Our work focuses on replicating and analyzing the partial human-pretraining phase of VITRA under constrained compute environments, evaluating model convergence, testing generalization on clean splits, and analyzing key model input ablations.

---

## 1. Introduction

VITRA is a VLA model designed to learn control policies from human video datasets. The model consists of two core components:
1. **Vision Language Model (VLM)**: Utilizing a SigLIP vision encoder and Gemma-2 language model to parse visual frames, instructions, and camera parameters (FoV) into a unified cognitive feature representation (`cognition token`).
2. **Action Expert (DiT)**: A Diffusion Transformer conditioned on the VLM's cognition feature to predict precise 3D trajectory actions (6D hand poses and joint rotations) via iterative denoising.

This project focuses on the **Human Pretraining** phase of VITRA, scaling pretraining across human action datasets (Something-Something-V2 and EPIC-Kitchens) to learn strong generalized features before any downstream robot fine-tuning.

---

## 2. Implementation (Config & Pretraining)

### Environment & Resources
- **Runtime Environment**: Kaggle Notebook (Google G4 instance).
- **GPU Hardware**: 1x NVIDIA RTX Pro 6000 (96 GB VRAM).
- **GPU Training Time**: ~100 GPU hours in total.

### Dataset Mixture
- **SSV2**: 5x oversampled Something-Something-V2 dataset (5x 1,125,013 frames).
- **EPIC**: 1x EPIC-Kitchens dataset (1x 4,836,063 frames).
- **Total Dataset Size**: 10,461,128 frames.
- **Iterations / Steps**: 
  - Local batch size: `64`
  - Gradient Accumulation: `8 steps` (effective batch size: `512`)
  - Total iterations: `163,455` (1 epoch over the mixture dataset).
  - Total optimizer steps: `~20,000 steps`.

### Preprocessing Optimization
To avoid performance bottlenecks during training and speed up video decoding, we resized all source videos beforehand (resizing the short edge to `224` pixels while preserving the aspect ratio). This preprocessing step took **~40 minutes** in total.

### Pretraining Strategy & Command
Due to Kaggle's 12-hour session runtime limit, the training was split into **10 separate sequential runs** (2,000 steps per run). After each run, the `model_load_path` (pointing to the last checkpoint) and `max_steps` configurations were updated in `vitra/configs/human_pretrain.json`.

- **Training Config File**: [human_pretrain.json](vitra/configs/human_pretrain.json)
- **Training Command**:
  ```bash
  python scripts/train.py --config vitra/configs/human_pretrain.json
  ```

For step-by-step guidance on setting up environments and executing pretraining, refer to the pretraining notebook:  
- **Pretraining Notebook**: [notebooks/pretrain.ipynb](notebooks/pretrain.ipynb)

---

## 3. Evaluation (2 Protocols)

To evaluate model convergence and generalization, we established two evaluation protocols. The notebook implementation of the training and evaluation runs is contained in `notebooks/pretrain.ipynb`, while visual prediction inference is located in `notebooks/inference.ipynb`.
- **Training & Evaluation Notebook**: [notebooks/pretrain.ipynb](notebooks/pretrain.ipynb)
- **Visual Inference Notebook**: [notebooks/inference.ipynb](notebooks/inference.ipynb)

### Protocol 1: Test-split Evaluation (Main Benchmark)
Used to measure the model's zero-shot generalization capabilities on unseen data.
- **Dataset**: EPIC-Kitchens clean held-out test split.
- **Test Split Statistics**:
  - **Cutoff Step**: `128,000` iterations.
  - **Seen Samples Filtered**: `4,911,432` frames (previously seen training frames are filtered out to avoid data leakage).
  - **Clean Test Sample Count**: `1,049,637` frames (entirely from the `epic` dataset, representing `17.61%` of the EPIC corpus; `ssv2` is `0` because oversampling makes a clean split impossible).
- **Execution Time**: ~3 hours per checkpoint (~16,400 batches of size 64).
- **Evaluation Command**:
  ```bash
  python scripts/evaluate_pretrained_loss.py \
      --config vitra/configs/human_pretrain.json \
      --weights /path/to/weights.pt \
      --eval_dataset epic \
      --eval_sampler_step 128000 \
      --eval_batches 16400 \
      --seen_sampler_steps 128000 \
      --output_jsonl /path/to/output/file
  ```

### Protocol 2: Reference Evaluation (Fast Validation)
A lightweight benchmark used to quickly monitor convergence behavior on both datasets simultaneously during checkpoints verification.
- **Dataset**: 200 batches (~12.8k frames) taken directly after sampler step 161,000 on both SSV2 and EPIC datasets.
- **Execution Time**: ~2.5 minutes per checkpoint.
- **Evaluation Command**:
  ```bash
  python scripts/evaluate_pretrained_loss.py \
      --config vitra/configs/human_pretrain.json \
      --weights /path/to/weights.pt \
      --eval_each_dataset \
      --eval_sampler_step 161000 \
      --eval_batches 200 \
      --output_jsonl /path/to/output/file
  ```

---

## 4. Ablation Study

We performed an inference-only ablation study on our fully-trained checkpoint (`step=16000`) to evaluate the relative contributions of key model input components (e.g., current hand states, camera field of view (FOV), and repeated diffusion steps) without the need to retrain the model.

Key configurations analyzed include:
- `baseline`: Standard evaluation without modifications.
- `no_state`: Setting `current_state` to 0 and masking it out to verify if hand states aid trajectory predictions.
- `zero_fov`: Zeroing out camera FOV details to test spatial/camera alignment contribution.
- `rds1` / `rds4`: Overriding `repeated_diffusion_steps` to 1 or 4 to test hyperparameter inference trade-offs.

For full ablation scripts, configuration setup, and output formats, refer directly to the ablation directory:
- **Ablation Documentation**: [ablation/README.md](ablation/README.md)
- **Ablation Kaggle Notebook**: [ablation/vitra_ablation_kaggle.ipynb](ablation/vitra_ablation_kaggle.ipynb)

---

## 5. Supplementary

### Kaggle Dataset Preparation (EPIC-Kitchens)
To facilitate seamless training on Kaggle notebooks and avoid disk overflow or long setup delays, we created a utility pipeline to partition and transfer EPIC-Kitchens video data directly from the Hugging Face Hub (`a1raman/epic_kitchens_100`) to Kaggle datasets.

- **Dataset Preparation Notebook**: [notebooks/prepare-epic-dataset.ipynb](notebooks/prepare-epic-dataset.ipynb)
- **Key Implementation Details**:
  - **Partition Filtering**: Downloads and filters only targeted participant directories (e.g., `P12`, `P02`, `P03`, `P10`) using `huggingface_hub.hf_hub_download` to reduce resource footprints.
  - **Custom GCS Uploads**: Re-creates structured directories and uploads video assets file-by-file using low-level, internal `kagglehub` SDK components (`_upload_file` and `UploadDirectoryInfo`) to build dataset versions.
  - **Disk Space Management**: Automatically deletes temporary downloaded `.MP4` files immediately after upload to avoid exceeding Kaggle's local container storage limits.
