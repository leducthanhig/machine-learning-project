# VITRA Inference-Only Ablation

Thư mục này chứa notebook Kaggle để chạy ablation chi phí thấp cho checkpoint VITRA. Mục tiêu là có kết quả ablation để báo cáo mà không cần train lại model.

## Ablation Đang Làm Gì?

Notebook giữ nguyên checkpoint và EPIC held-out sampler split, sau đó chỉ thay đổi input hoặc hyperparameter ở lúc inference/evaluation.

Checkpoint mục tiêu:

```text
final-epoch=0-step=16000.ckpt/weights.pt
```

Thiết lập EPIC clean split:

```text
eval_dataset = epic
CUTOFF = 128000
eval_sampler_step = 128000
seen_sampler_steps = 128000
```

Các setting ablation:

| Setting | Ý nghĩa |
| --- | --- |
| `baseline` | Eval bình thường, không ablation |
| `zero_state` | Đưa giá trị state về 0 nhưng vẫn giữ state mask |
| `no_state` | Đưa giá trị state về 0 và mask state khỏi model |
| `shuffle_state` | Đảo/shift state giữa các sample trong cùng batch |
| `zero_fov` | Đưa input FOV về 0 |
| `shuffle_fov` | Đảo/shift FOV giữa các sample trong cùng batch |
| `rds1` | Override `repeated_diffusion_steps=1` |
| `rds4` | Override `repeated_diffusion_steps=4` |

Nếu thời gian hạn chế, các setting quan trọng nhất để đưa vào báo cáo là:

```text
baseline
no_state
zero_fov
rds1
```

Như vậy vẫn có đủ ablation dạng component/hyperparameter:

- Ablation thành phần state: `no_state`
- Ablation thành phần FOV: `zero_fov`
- Ablation hyperparameter inference: `repeated_diffusion_steps=1`

## Cách Chạy Trên Kaggle

Notebook cần chạy:

```text
ablation/vitra_ablation_kaggle.ipynb
```

Cấu hình Kaggle khuyến nghị:

```text
Accelerator: RTX Pro 6000
Internet: Off
```

Attach các input sau:

```text
/kaggle/input/notebooks/ldthanh/prepare-vitra-resources
/kaggle/input/models/ldthanh/vitra-vla-3b
/kaggle/input/models/ldthanh/paligemma2-3b-mix-224
/kaggle/input/datasets/ldthanh/vitra-1m
/kaggle/input/datasets/ldthanh/epic-kitchens-100-p*
/kaggle/input/datasets/nahidsiddique/something-something-v2
```

Chạy notebook từ đầu. Các dòng log nên thấy:

```text
Patched human_dataset.py with missing-video filtering: True
Patched data_mixture.py with magic_mix_epic_ssv2: True
Patched scripts/evaluate_pretrained_loss.py with inference-only ablation flags
Resized EPIC root: /tmp/epic-kitchens-100-224
[INFO] Linked resized EPIC videos from /tmp/epic-kitchens-100-224
Using checkpoint: ...final-epoch=0-step=16000.ckpt/weights.pt
Using PaliGemma backbone: ...paligemma2-3b-mix-224...
```

## Nếu Thời Gian GPU Hạn Chế

Không bật full evaluation ngay.

Nên dùng:

```python
RUN_FULL = False
SMOKE_BATCHES = 16
```

Nếu vẫn quá chậm, giảm `SMOKE_SETTINGS` còn:

```python
SMOKE_SETTINGS = [
    dict(name="baseline", ablate_state="none", ablate_fov="none"),
    dict(name="no_state", ablate_state="no_state", ablate_fov="none"),
    dict(name="zero_fov", ablate_state="none", ablate_fov="zero_fov"),
    dict(name="rds1", ablate_state="none", ablate_fov="none", repeated_diffusion_steps=1),
]
```

Nếu còn nhiều thời gian GPU hơn, có thể tăng:

```python
SMOKE_BATCHES = 32
```

Chỉ cân nhắc `RUN_FULL=True` sau khi ablation nhỏ đã chạy xong thành công.

## Output Cần Lấy

Notebook sẽ ghi các file:

```text
ablation_results/smoke/*.jsonl
ablation_results/smoke/*.summary.json
ablation_results/smoke_summary.csv
ablation_results/ablation_summary.csv
../vitra_ablation_results.zip
```

File nên dùng để đưa vào báo cáo:

```text
ablation_results/ablation_summary.csv
```

Cột quan trọng nhất là `loss`. Hai cột `delta_vs_baseline` và `relative_vs_baseline_pct` cho biết loss thay đổi bao nhiêu so với baseline.