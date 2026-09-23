# LAS-VAD paper/code audit — 2026-09-22

Reference: [arXiv:2603.00550v1](https://arxiv.org/html/2603.00550v1). This is an independent implementation. Passing equation tests does not establish benchmark reproduction or resolve missing author implementation details.

**Later correction:** The official CVPR supplementary material publishes the
attribute descriptions in Table 10. They have now been imported as
`configs/las_attributes_paper.json` and made the default for new runs. Historical
experiments below used substitute attributes; their results are unchanged.

## Confirmed ACC error and correction

The old implementation passed temperature-scaled softmax probabilities to Eq. 13. The paper defines its semantic input as cross-modal cosine similarity. These are different quantities: softmax removes negative semantic consistency and changes the magnitude of edge rectification.

`LASVAD.forward` now exposes both `language_similarity` (signed cosine) and `language` (classification probabilities). New models use the former for ACC. `acc_affinity` implements Eqs. 12–13; its max-min accumulator starts at negative infinity, so negative consistency is preserved. Thresholding remains strictly `> 0.9`, eta remains 0.5. Connected components use exact DFS, independently within each valid video. Component feature/semantic means and nearest-prototype assignments exclude padding and remain detached targets.

Tests compare the affinity against an independent scalar calculation and check negative semantics, temperature independence, transitive connectivity, isolated vertices, and component labels. The diagnostic tool and training now share the same affinity/DFS functions.

## Why a correct DFS can still return one component

One connected component does not require all frame pairs to pass the threshold. A path through intermediate frames is sufficient. The paper does not prescribe a minimum number of groups.

For the existing `model/las_ucf_lgt_audit/latest_best.pt`, the same 32 uniformly sampled training rows include 16 abnormal examples. Recomputing all three graphs on identical saved features gives:

| Graph | Videos with one component | Mean inter-frame edge density |
|---|---:|---:|
| Visual cosine only, Eq. 12 | 32/32 | 90.4047% |
| Corrected signed-cosine rectification | 32/32 | 97.3790% |
| Legacy softmax rectification | 32/32 | 99.6152% |

Evidence: `model/las_acc_audit/pretrained_graph_comparison.json`. Densities exclude self edges. The corrected input reduces connectivity but does **not** fix the observed single-component outcome on this checkpoint. Forcing multiple groups, changing tau to fit these examples, adding a temporal edge restriction, or using k-means would change the stated algorithm. None is silently applied.

## Other audited parts

| Area | Verified behavior / remaining qualification |
|---|---|
| Local/global encoder | LGT numerically matches the original VadCLIP adapter on unpadded inputs; padding and CPU/device handling have separate tests. It is not literally Eq. 1: the paper's simplified equation and overlapping-window description differ from VadCLIP's actual residual dual-graph/block-window implementation. Both `lgt` and the earlier `simple` implementation remain explicit choices. |
| IAM motion and EMA | Hand-calculated tests check position/velocity/acceleration, gated differences, padded boundaries, confidence-selected frame means and beta=0.1 EMA. Exact per-class update counters now expose inactive prototypes. No arbitrary bootstrap is added. |
| Eq. 11 | Added `--eq11-exact`: raw dot-product scoring, negative-only denominator and division by all valid time steps per video. Cosine partner mining and zero contribution when a partner is unavailable remain assumptions. Default normalized InfoNCE is an explicit alternative. The old `--literal-contrast` is not an exact implementation of the printed equation. |
| Eq. 9 | Added `--objective eq9` to exclude contrastive loss from the total, as printed. Default `extended` includes it to implement the accompanying contrastive-learning description. A test verifies that the eq9 total is independent of contrastive weight. |
| Data/evaluation | Existing tests cover original float16 feature pooling, balanced sampling, full snippet coverage, frame annotation alignment, validation mode/RNG/prototype preservation, highest-score checkpoint selection and deterministic mid-epoch resume. |

## Unresolved reproduction details

- The text concatenation produces 2D channels but visual cosine requires D channels; the projection is not specified. The learned projection here is an assumption. The historical runs below used manually authored substitute attributes; new runs use the published Table 10 descriptions.
- The paper combines cosine scores with softmax scores, calls their average logits, then takes a log in its loss without specifying calibration. The probability calibration used here is not an exact transcription of that ambiguous sequence.
- The binary head is described as two sigmoid outputs while its loss is called BCE. This implementation uses one anomaly sigmoid and its complement with ordinary MIL BCE.
- The paper calls Eq. 11 InfoNCE but prints a negative-only denominator; Eq. 9 omits this loss. Raw-dot Eq. 11 is unbounded below. The two explicit CLI choices expose this inconsistency rather than claiming to know the authors' intended objective.
- Prototype initialization, alpha, M, mining scope, temporal boundary convention, and the D/3 rounding at D=512 are incompletely specified. Current choices are recorded in `LAS_VAD.md` and checkpoint config. Inactive prototype counters indicate a practical training limitation, not permission to invent an initialization procedure.
- Detection outer extent/NMS and AP convention are implementation choices; the mAP result is not certified against the authors' evaluator.

## Reproduce the checks

```bash
.venv/bin/python -m unittest discover -s tests -v

.venv/bin/python src/las_diagnose.py \
  --checkpoint model/las_ucf_lgt_audit/latest_best.pt \
  --samples 32 --output model/las_acc_audit/pretrained_graph_comparison.json

# Isolate the ACC input correction; keep the existing extended objective.
.venv/bin/python src/las_vad.py train \
  --dataset ucf --adapter lgt --acc-similarity cosine \
  --checkpoint model/las_acc_cosine_audit/latest.pt \
  --epochs 1 --batch-size 64 --device cuda:0 --workers 4
```

Use a new checkpoint destination for a new run. Resume intentionally restores checkpoint configuration: checkpoints predating this fix retain `acc_similarity=probability`. Their missing historical prototype counters are marked incomplete. New logs/checkpoints record the actual ACC mode, component statistics and prototype updates. Losses are epoch-running means; ACC statistics describe the logged batch.

## Completed real-feature verification

The corrected-cosine run above completed one epoch: 250 updates, 16,000 processed examples, and 13 validations over all 290 test videos. All 30 tests passed (full suite of 29, then the expanded nine-test equation suite including the new video-isolation test). Best checkpoint selection and every saved metric were checked against predictions produced after reloading the best checkpoint.

| Result | Value |
|---|---:|
| Best optimizer step | 220 |
| AUC | 85.7293% |
| AP | 25.6208% |
| Ano-AUC | 64.2024% |
| Detection mAP, current evaluator | 3.0088% |
| Best checkpoint: sampled single-component videos | 32/32 |
| Best checkpoint: prototype EMA updates | 0 for every class |
| End-of-epoch checkpoint: prototype EMA updates | 24 for normal, 0 for anomaly classes |

These results verify execution, **not resolution of representation/prototype collapse**. The experiment keeps the previous extended objective and initialization to isolate the ACC correction. It is not a ten-epoch reproduction or evidence that the correction improves AUC. The best checkpoint can precede the first prototype update.

Artifacts:

- `model/las_acc_cosine_audit/latest_best.pt` and `latest.pt`
- `model/las_acc_cosine_audit/latest.jsonl`
- `model/las_acc_cosine_audit/diagnostics.json`
- `model/las_acc_cosine_audit/best_evaluation/metrics.json`
- `model/las_acc_cosine_audit/verification.json`
