# LAS-VAD review and real-feature run — 2026-09-21

## Findings and fixes

1. **Incorrect earlier environment assessment.** `datasets/UCFClipFeatures` is a symlink to `/home/islab/thanh/datasets/UCF-Crime/UCFClipFeatures`, not an empty dataset. All 16,100 training paths and 290 test paths in the existing CSVs resolve. CUDA is unavailable inside the restricted sandbox but works outside it on the host's RTX 2080 Ti GPUs.
2. **Feature preprocessing differed subtly from VadCLIP.** The initial loader converted float16 arrays to float32 before pooling. It now directly reuses `utils.tools.process_feat` and converts its output to float32, matching the baseline's pooling/padding exactly. Tests cover lengths shorter than, equal to, and longer than the target length. Real arrays were also compared exactly.
3. **UCF batch construction did not follow the baseline.** The initial implementation shuffled a single mixed dataset. The default now shuffles normal and anomaly pools independently, combines half of each per batch, and stops at the smaller pool with incomplete batches dropped. A total batch size of 64 means 32 normal plus 32 anomaly; this preserves the paper's total batch size while following VadCLIP's sampling approach. The current dataset contains 8,000 normal and 8,100 anomaly crop features, so each epoch trains on 16,000 examples in 250 batches; the omitted 100 anomaly examples vary with the shuffle.
4. **Relative paths depended on the working directory.** CSV and feature paths now support repository-root resolution, plus explicit feature-root overrides. All file paths are checked before loading CLIP. Array dimensions are checked when features are loaded.
5. **Training was difficult to monitor.** The trainer now records step/epoch losses, timing, sampling settings and checkpoint locations in JSONL. Nonfinite losses or gradient norms stop an update. Both UCF balanced and XD shuffled checkpoint/resume workflows are tested.

This review retains the documented model assumptions in [LAS_VAD.md](LAS_VAD.md), including text fusion, contrastive-loss interpretation and substitute attribute descriptions. It does not claim to resolve ambiguities in the paper or to reproduce the authors' full training setup.

## Actual run

The following commands completed on GPU `cuda:0` (RTX 2080 Ti):

```bash
.venv/bin/python src/las_vad.py train \
  --dataset ucf \
  --checkpoint model/las_ucf_review/checkpoint.pt \
  --epochs 10 --batch-size 64 --device cuda:0 --workers 4 --log-every 10

.venv/bin/python src/las_vad.py evaluate \
  --checkpoint model/las_ucf_review/checkpoint.pt \
  --frame-gt list/gt_ucf.npy \
  --output model/las_ucf_review/evaluation --device cuda:0
```

The model uses width 512, visual length 256, and the remaining default `LASConfig` values; the exact config, model state, optimizer, RNG states and epoch are saved in the checkpoint. Training finished one full balanced epoch in 77.79 seconds (excluding initial CLIP loading). No nonfinite loss/gradient errors occurred.

Epoch-average training losses:

| Term | Value |
|---|---:|
| Total | 3.291645 |
| Binary MIL | 0.485492 |
| Fine-grained MIL | 1.770074 |
| ACC auxiliary | 0.163512 |
| Consistency | 0.117379 |
| Contrastive | 0.837353 |

Evaluation covered all 290 videos (69,368 snippets, repeated into 1,109,888 frames), matching the provided ground truth length exactly. For the paper's fused anomaly score:

| Frame metric | Result |
|---|---:|
| ROC AUC | 80.1093% |
| Average precision | 18.1547% |

These are **one-epoch verification results**, not a completed 10-epoch experiment or reproduced paper results. No detection mAP was computed in this run.

Artifacts (ignored by git under `model/`):

- `model/las_ucf_review/checkpoint.pt`
- `model/las_ucf_review/checkpoint.jsonl`
- `model/las_ucf_review/evaluation/metrics.json`
- `model/las_ucf_review/evaluation/manifest.json`
- Per-video `.npz` predictions and `proposals.json` in the evaluation directory.

To continue this checkpoint to a total of ten epochs:

```bash
.venv/bin/python src/las_vad.py train \
  --dataset ucf \
  --resume model/las_ucf_review/checkpoint.pt \
  --checkpoint model/las_ucf_review/checkpoint.pt \
  --epochs 10 --batch-size 64 --device cuda:0 --workers 4
```

The evaluation files above describe epoch 1; rerun evaluation after further training to obtain updated metrics.

## Periodic-validation follow-up

The revised trainer was exercised by resuming the epoch-1 checkpoint for one additional full epoch:

```bash
.venv/bin/python src/las_vad.py train \
  --dataset ucf --resume model/las_ucf_review/checkpoint.pt \
  --checkpoint model/las_ucf_validation/latest.pt \
  --epochs 2 --batch-size 64 --device cuda:0 --workers 4
```

Using defaults, epoch 2 validated 13 times: every 1,280 processed examples, plus its end at 16,000 examples. Each validation evaluated all 290 videos. Latest and best checkpoints were written separately; a check against the JSONL confirmed the saved best AUC equals the maximum of all validation records. Loading that checkpoint and evaluating again reproduced all recorded metrics exactly.

- Latest: `model/las_ucf_validation/latest.pt`
- Best: `model/las_ucf_validation/latest_best.pt`
- Log: `model/las_ucf_validation/latest.jsonl`
- Reloaded best metrics: `model/las_ucf_validation/best_evaluation/metrics.json`
- Best fused AUC: **82.792991%**; AP: **20.643292%**; Ano-AUC: **57.765671%**.
- Average detection mAP: **2.527716%**, using LAS-VAD proposals and this implementation's all-points AP.

The best in this particular run occurred at epoch 2's last batch. Separate controlled tests force an earlier validation to be best and verify that later lower scores do not overwrite it, and that resuming from that mid-epoch checkpoint reproduces uninterrupted training exactly. All **19 tests** passed, covering both dataset workflows, validation mode/prototype/RNG preservation, frame-to-snippet annotation conversion, metrics, and local versus global feature propagation.

This is still a short implementation-validation experiment, not a complete 10-epoch reproduction. The local/global encoder's computations remain unchanged; the new explicit `encode_local` and `encode_global` methods expose the existing components and retain old checkpoint compatibility. The architectural differences from VadCLIP's LGT adapter are documented in `LAS_VAD.md`.
