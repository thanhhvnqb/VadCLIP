# LAS-VAD training diagnosis — 2026-09-22

Final verification completed 2026-09-23: all six experiments finished ten epochs;
all reloaded best-checkpoint metrics match their logs. **90.86% AUC has not been
reproduced.** The best new LAS-VAD ablation reaches 86.150464%, only 0.022484
percentage points above the existing 86.127980% model. This is not evidence of a
meaningful improvement. Machine-readable results and completion checks are in
`model/las_final_review/experiment_summary.json` and `verification.json`.

## Scope and sources

This review checks the current implementation against the
[arXiv paper](https://arxiv.org/pdf/2603.00550), the
[accepted CVPR paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_Weakly_Supervised_Video_Anomaly_Detection_with_Anomaly-Connected_Components_and_Intention_CVPR_2026_paper.pdf),
and its [official supplement](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Wang_Weakly_Supervised_Video_CVPR_2026_supplemental.pdf).
Table 2 and supplementary Table 8 report **90.86% UCF-Crime AUC with CLIP**.
The 91.05% result uses I3D. The main-text 90.96/AP wording conflicts with
the tables; this review uses the tabulated CLIP AUC.

The user's running training process and checkpoints were not modified. A copy of
its best checkpoint was saved as `model/las_final_review/baseline_best.pt` before
independent evaluation. Existing implementation changes predating this review
(LGT, signed-cosine ACC, validation and resume fixes) are not new fixes here.

## Confirmed correction: published attributes were available

Previous documentation incorrectly called the authors' descriptions unavailable.
They are published in supplementary Table 10. The previous default
`configs/las_attributes.json` contains hand-written substitutes.

New runs now default to `configs/las_attributes_paper.json`, imported directly
from that table with `src/las_import_attributes.py`. Separate UCF and XD mappings
preserve their description differences; the `roadAccidents` key is mapped to
the existing class name without changing class indices. The importer verifies
all 14/7 classes and records the supplementary PDF's SHA-256. Old flat attribute
files remain supported, and old checkpoints retain their stored text embeddings.

Training records the text source and its hash, the full model configuration and
effective optimizer settings. Changing `--attributes` while resuming does not
change the saved text embeddings: a fresh run is required for this comparison.

## Data and evaluation checks

An independent reconstruction from `Temporal_Anomaly_Annotation.txt` found:

- 16,100 training crop rows from 1,610 unique videos;
- 290 test videos, with no train/test video-ID overlap;
- exactly 1,109,888 evaluated frames;
- zero differences between reconstructed frame labels and `gt_ucf.npy`.

Artifact: `model/las_final_review/data_audit.json`. Feature dimension and existing
preprocessing checks cannot establish the provenance of the original visual
extractor, crop/resolution settings or its weights; that still requires feature
extraction metadata.

Reloading the user's saved best model reproduced AUC **86.127980%**, AP
**31.200112%**, abnormal-video-only AUC **65.497466%**, and current-evaluator
detection mAP **5.155138%**. Its binary/alignment AUCs are 85.930820%/85.849686%.
The peak is epoch 3, update 600. It is not an incorrect best-checkpoint selection
or a simple inference-fusion discrepancy.

Individual head auditing also gives closely matched frame AUCs: visual
85.306726%, text 85.281253%, IAM 85.321686%. No individual head already approaches
90% on this checkpoint. `evaluate --audit-heads` now saves these scores and
additional metrics without changing fusion. Evidence:
`model/las_final_review/baseline_heads/metrics.json`.

The existing **original VadCLIP** checkpoint
`model/original-v1/model_ucf.pth` was also independently evaluated with its
original evaluator: **88.14% AUC and 70.72% Ano-AUC**, at console precision.
Thus the supplied features/annotations support higher performance than the
current LAS-VAD reconstruction. This is a different architecture and inference
branch, not a LAS-VAD result or a new model trained by this review. Its original
detection evaluator also differs from the LAS-VAD evaluator. Evidence:
`model/las_final_review/vadclip_reference.json`.

## Why the early peak needs more than a learning-rate change

The current run's video-level losses become very small while test localization
degrades. On 32 deterministic, uniformly spaced training rows at its best model:

- all 32 ACC graphs have exactly one connected component;
- mean rectified inter-frame edge density is 96.3045%;
- even visual-only graphs have one component in all 32 videos;
- mean within-video feature cosine is 0.957214;
- the normal prototype has 374 EMA updates, and all 13 anomaly prototypes have 0.

The ACC labels therefore become video-wide soft targets on this sample. IAM's
random prototype initialization plus confidence-gated updates can leave anomaly
prototypes inactive. Neither observation establishes that DFS or EMA is coded
incorrectly: both follow the documented equations under the implementation's
initialization choices. Arbitrarily forcing extra components or bootstrapping
prototypes would add an algorithm absent from the published specification.

`src/las_diagnose.py --gradient-samples N` now compares base/contrastive gradient
norms and their cosine on shared visual parameters. For eight sampled training
rows at the baseline best checkpoint, nonzero contrastive visual gradient norms
were smaller than the base gradient norms; two of the five nonzero pairs had
negative cosine. Thus a contrastive scalar loss around 2.3 is **not** sufficient
evidence of gradient domination. These measurements are a small diagnostic
sample, not an estimate over the whole training distribution.

Artifacts: `model/las_final_review/current_best_diagnostics.json` and
`model/las_final_review/baseline_evaluation/metrics.json`.

## Controlled experiments

All new experiments use seed 234, LGT, signed-cosine ACC, total batch 64, AdamW
at 2e-5, and 10 full epochs (2,500 updates; 160,000 processed crop examples).
Validation occurs every 4,000 processed examples plus epoch end: 40 evaluations
per run on all 290 test videos. The user's run evaluated more frequently
(1,280 examples), so its observed peak has a denser selection grid.

| Experiment | Attributes | Best AUC | Best epoch / update | Final AUC |
|---|---|---:|---:|---:|
| User's existing extended objective | Substitute | 86.127980% | 3 / 600 | See user's run log |
| Printed Eq. 9 objective | Substitute | 85.971341% | 1 / 250 | 83.533801% |
| Eq. 9, mean text initialization | Substitute | 86.104942% | 2 / 313 | 83.416130% |
| Eq. 9, ACC auxiliary weight zero | Substitute | 86.150464% | 3 / 625 | 84.541951% |
| Printed Eq. 9 objective | Official Table 10 | 85.988842% | 1 / 250 | 83.891716% |
| Extended objective | Official Table 10 | 85.883132% | 2 / 313 | 83.282493% |
| Extended objective, alpha 0.1 | Official Table 10 | 86.076005% | 2 / 313 | 83.269034% |

The three completed substitute-attribute experiments each had their best weights
reloaded and all saved evaluation metrics reproduced exactly. Their directories
are `model/las_eq9_review`, `model/las_eq9_mean_review`, and
`model/las_eq9_noaux_review`, with `latest.pt`, `latest_best.pt`, `latest.jsonl`,
and `best_evaluation/metrics.json`.

The mean initialization is `[I/2,I/2]` for the trainable text projection, with
zero bias. It is exposed as `--text-init mean`, retaining `random` as default.
The ACC ablation is `--aux-weight 0`, retaining the paper's weight 1 by default.
Neither alternative has established a substantial AUC improvement, and the
ACC-disabled run is not the full LAS-VAD method. No default was changed based
on its 0.0225 percentage-point gain.

These comparisons are exploratory, single-seed experiments using the benchmark
test split for checkpoint selection, as the existing workflow does. They do not
establish statistical superiority or independent held-out tuning performance.
The completed official-attribute runs are recorded separately under
`model/las_paper_attributes_eq9` and `model/las_paper_attributes_extended`.
An additional completed IAM experiment uses `alpha=0.1` with official attributes
and the extended objective (`model/las_paper_alpha01`). Alpha is not specified in
the paper; this experiment tests the inactive-prototype hypothesis, not a known
author configuration. All remaining settings match the official extended run.

Replacing substitute descriptions with the published table fixes a confirmed
reproduction mismatch but did not improve AUC in these trials. At their best
checkpoints, both official-attribute runs still have zero anomaly-prototype EMA
updates and 32/32 sampled videos with one ACC component. Their reloaded metrics
match the training logs exactly. The `alpha=0.1` best checkpoint activates 10/13
anomaly prototypes, but still has one ACC component in all 32 sampled videos.

## Remaining reproduction gaps

The accepted paper and supplement still do not settle several choices:

- concatenating two D-dimensional text embeddings yields 2D channels, but the
  cosine comparison uses D-dimensional visual features; projection details are
  not specified;
- the figure contains a cross-modal enhancement block whose exact parameterization
  is not defined by the equations; this implementation does not reconstruct
  unseen author code for that block;
- the text branch is described with cosine values but fused with probabilities
  and passed to log losses; this implementation calibrates it with softmax;
- Eq. 9 omits the contrastive term introduced in Eq. 11; both interpretations
  are explicit options, and neither tested interpretation resolved the AUC gap;
- prototype initialization, confidence threshold alpha, partner-label generation,
  mining scope and M are not sufficiently specified;
- the simplified GCN/overlapping-window description and the actual VadCLIP LGT
  differ. The implementation uses the explicitly documented LGT choice;
- temporal sampling, feature extraction provenance, and the detection evaluator
  must be matched before claiming exact benchmark reproduction.

The implemented EMA, graph construction, MIL and inference rule have regression
tests, but passing them does not certify 90.86% AUC. A reproduction claim still
requires author implementation details and a successful controlled experiment.

## Commands

```bash
# Published attributes with the printed objective.
.venv/bin/python src/las_vad.py train \
  --dataset ucf --adapter lgt --objective eq9 \
  --attributes configs/las_attributes_paper.json \
  --checkpoint model/las_paper_attributes_eq9/latest.pt \
  --epochs 10 --batch-size 64 --device cuda:1 --workers 4 \
  --validate-every 4000 --log-every 4000

# To compare the text's contrastive-learning interpretation, use a NEW path
# and --objective extended. Do not overwrite an existing completed experiment.

# Reproduce substitute-attribute ablations by adding:
#   --attributes configs/las_attributes.json
# Eq. 9 control: --objective eq9
# Mean projection: --objective eq9 --text-init mean
# No ACC objective: --objective eq9 --aux-weight 0
# The official-attribute IAM experiment uses:
#   --objective extended --alpha 0.1
# Use a different --checkpoint destination for every fresh experiment.

.venv/bin/python -m unittest discover -s tests -v
```

The complete suite passed 37 tests, including real table schema checks,
dataset-specific attribute parsing, legacy configurations/checkpoints, objective
ablation isolation, mean-initialization round trip, gradient diagnostics without
state mutation, independent-head auditing without fusion changes, frame metrics
and exact mid-epoch resume.

The startup console prints the effective objective, alpha, auxiliary weight,
text initialization and text source. Resume restores these from checkpoint;
changing architecture/loss/text flags on a resume command does not create a
new ablation.
