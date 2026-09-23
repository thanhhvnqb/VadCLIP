# Audit of the 84% run — 2026-09-22

## Evidence from the existing run

`model/las_ucf_best.pt` reached frame AUC **84.2328%** at epoch 10, batch 60. Its abnormal-video-only AUC is **61.1586%**, AP **26.2550%**, and average detection mAP **4.8793%**. The binary branch AUC is 84.1156% and the fine-grained branch AUC is 84.0151%; changing the fusion alone is therefore not supported as the main fix by these results.

The run used the original `simple` adapter: width 512, window 32, eight heads, one Transformer layer and one non-residual graph projection. This is not the original VadCLIP LGT adapter. Validation frame counts and best-checkpoint selection were already checked against the existing annotations/logs.

A deterministic diagnostic sampled 32 training CSV rows uniformly across its order, using the exact training preprocessing and saved best weights. The diagnostic does not use frame-level ground truth:

| Measurement | Legacy best checkpoint |
|---|---:|
| Mean normalized temporal variation, raw CLIP features | 0.074138 |
| Mean normalized temporal variation, after local Transformer | 0.565832 |
| Mean normalized temporal variation, after global graph | 0.003343 |
| Mean within-video pairwise cosine, after graph (including diagonal) | 0.997457 |
| Videos with exactly one ACC connected component | 32 / 32 |

Normalized temporal variation means `mean((X - temporal_mean(X))²) / mean(X²)`. The large drop through the global graph shows that almost all temporal distinctions are lost in these sampled representations. ACC consequently supplies nearly video-wide targets on this sample rather than separating normal and anomalous intervals. This is strong evidence of a localization bottleneck, not proof of the sole cause of the accuracy gap.

The legacy graph normalizes raw cosine similarities directly with softmax. Since cosine lies between -1 and 1, a row's largest/smallest weight ratio cannot exceed `exp(2)`; without a residual path, the graph strongly averages frame features. In this checkpoint only prototypes 0 and 9 have norms far from their unit-norm initialization. The remaining twelve retain norms approximately one, suggesting that their confidence-gated EMA updates are largely inactive; this observation is not an exact update counter.

The legacy positional embedding has standard deviation **0.998774**, versus the **0.01** initialization used in VadCLIP. This initialization difference was another unintended departure from the baseline adapter.

## Changes

- Added `src/las_lgt.py`: the original VadCLIP Transformer/QuickGELU and two residual graph branches, with correct masking and device-independent distance adjacency.
- New CLI runs use LGT by default. UCF presets are window 8, one head and two layers; XD presets are window 64, one head and one layer. Explicit flags override these presets.
- The simple adapter, its weights and initialization remain available with `--adapter simple` for reproducibility. Existing checkpoints without the adapter config field still load as simple.
- Kept all other heads, losses and attribute descriptions unchanged for the initial comparison. Switching architecture requires fresh training; `--resume` retains the checkpoint architecture.
- Added a numerical comparison test against the original `CLIPVAD.encode_video` for unpadded inputs, plus padding/short-video/backpropagation checks.
- Added `src/las_diagnose.py` so temporal-collapse measurements can be reproduced on any checkpoint.

```bash
.venv/bin/python src/las_diagnose.py \
  --checkpoint model/las_ucf_best.pt \
  --output model/las_audit/simple_diagnostics.json

.venv/bin/python src/las_vad.py train \
  --dataset ucf --adapter lgt \
  --checkpoint model/las_ucf_lgt_audit/latest.pt \
  --epochs 2 --batch-size 64 --device cuda:0 --workers 4
```

This comparison changes the whole adapter configuration, including initialization/window/head/layer settings. It does not isolate the causal effect of a single residual connection. The paper's unspecified text fusion and contrastive weighting/mining details remain reproduction limitations; a higher validation score alone does not settle those questions. This historical experiment used substitute attributes. A later review located the published Table 10 descriptions in the official CVPR supplement and imported them for new runs.

## Completed LGT experiment

The new run completed two epochs from scratch, with 26 validations on the existing 290-video test split. The best checkpoint occurred at epoch 2, batch 80. Reloading it reproduced **every recorded metric exactly**, and its AUC equals the maximum of the validation log.

| Metric | Simple, best from 10 epochs | LGT, best from 2 epochs |
|---|---:|---:|
| Frame AUC | 84.2328% | 86.0950% |
| Abnormal-video-only AUC | 61.1586% | 64.7151% |
| Frame AP | 26.2550% | 26.4524% |
| Average detection mAP | 4.8793% | 4.0815% |

The AUC gain is **1.8622 percentage points**. This is a short, single-seed comparison with different training durations, not a statistically established superiority claim. Detection mAP has not improved. The saved checkpoint is selected by AUC, not by mAP/AP.

On the same 32 training examples, the LGT checkpoint's normalized temporal variation after graph is **0.047698** (versus 0.003343 previously), and mean pairwise cosine is **0.965699** (versus 0.997457). The roughly **14.27x** increase in temporal variation supports the diagnosis that the simple graph was oversmoothing features.

**Remaining limitations:** ACC still returns one component for every sampled video with the current threshold/rectification rule. In the LGT best checkpoint, only the normal prototype has a norm far from its initialization. Thus improved AUC does not mean ACC and IAM have been fully validated or that the paper's reported performance has been reproduced. Their grouping/calibration and the contrastive/text-fusion assumptions still need controlled ablation; they were not silently retuned against the test results in this experiment. The [subsequent equation audit](LAS_VAD_PAPER_AUDIT.md) identified and corrected the probability-versus-cosine ACC input error; this historical run predates that fix.

Artifacts:

- `model/las_ucf_lgt_audit/latest.pt`: latest state at the end of epoch 2.
- `model/las_ucf_lgt_audit/latest_best.pt`: best AUC checkpoint.
- `model/las_ucf_lgt_audit/latest.jsonl`: all training/validation metrics.
- `model/las_ucf_lgt_audit/best_evaluation/metrics.json`: independently reloaded metrics.
- `model/las_audit/simple_diagnostics.json`, `lgt_diagnostics.json`, `comparison.json`: diagnostic evidence.

All 21 tests passed, including numerical parity with the original LGT adapter and training/resume workflows. The previous user-trained simple checkpoints were not overwritten.

To continue this LGT run to ten total epochs:

```bash
.venv/bin/python src/las_vad.py train \
  --dataset ucf \
  --resume model/las_ucf_lgt_audit/latest.pt \
  --checkpoint model/las_ucf_lgt_audit/latest.pt \
  --epochs 10 --batch-size 64 --device cuda:0 --workers 4
```
