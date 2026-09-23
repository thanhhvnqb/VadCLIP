# LAS-VAD implementation

This is an independent implementation of [Weakly Supervised Video Anomaly Detection with Anomaly-Connected Components and Intention Reasoning](https://arxiv.org/pdf/2603.00550), arXiv:2603.00550v1. It is not the authors' code, and benchmark results have not been reproduced. The original VadCLIP scripts and checkpoints are unchanged.

## Run

Use a Python environment with PyTorch, torchvision and the dependencies in `requirements.txt`. The existing `.venv` is sufficient in this workspace. Run commands from the repository root:

```bash
# UCF-Crime
.venv/bin/python src/las_vad.py train \
  --dataset ucf --train-list list/ucf_CLIP_rgb.csv \
  --checkpoint model/las_ucf.pt

# XD-Violence
.venv/bin/python src/las_vad.py train \
  --dataset xd --train-list list/xd_CLIP_rgb.csv \
  --checkpoint model/las_xd.pt

# Continue to a total of 10 epochs; architecture/settings come from the checkpoint.
.venv/bin/python src/las_vad.py train \
  --dataset ucf --train-list list/ucf_CLIP_rgb.csv \
  --resume model/las_ucf.pt --checkpoint model/las_ucf.pt --epochs 10

# Predictions and optional frame-level metrics
.venv/bin/python src/las_vad.py evaluate \
  --checkpoint model/las_ucf_best.pt --test-list list/ucf_CLIP_rgbtest.csv \
  --output results/las_ucf --frame-gt list/gt_ucf.npy
```

Feature files must actually exist; the checked-in CSVs do not supply the arrays. Each `.npy` must be finite `[snippets, input_dim]` features (512 for CLIP ViT-B/16). CSV columns are `path,label`, optionally `video_id`. Paths resolve from `--feature-root`, otherwise the current directory, repository root, then CSV directory. Missing files fail before CLIP loads; dimensions are checked when arrays are read. UCF uses labels such as `Normal` and `Abuse`; XD uses `A`, `B1`, `B2`, `B4`, `B5`, `B6`, `G`, including compound anomaly labels such as `B1-B2`. Normal is always class index zero. Class order is defined in `src/las_data.py`.

Training calls the original VadCLIP `utils.tools.process_feat`: long sequences are uniformly averaged into `--max-length` bins and shorter sequences are zero padded. Raw float16 features retain VadCLIP's pooling behavior before conversion to float32 for the model. For UCF, `--sampling auto` pairs equally sized shuffled normal/anomaly pools and drops incomplete batches from the smaller pool, matching VadCLIP's two-loader approach. Here `--batch-size 64` means **32 normal + 32 anomaly**, not 64 of each. XD defaults to ordinary shuffled batches; `--sampling shuffle` or `--sampling balanced` overrides either default.

Evaluation preserves all snippets by chunking, including short videos and exact multiples of the chunk size, without using the original `process_split` function's empty trailing chunk for exact multiples. Crops with a common `video_id` are averaged; otherwise IDs follow the original dataset classes (UCF: before the first `__`; XD: remove the final `__crop` suffix). Unique videos retain their first-appearance CSV order.

`--train-list` and `--test-list` can be omitted to use the matching repository CSV. Step/epoch losses are printed and appended to the checkpoint's `.jsonl` companion file; customize with `--log-file` and `--log-every`. Checkpoints also save sampling settings and epoch-average losses. Model and optimizer resume settings come from the checkpoint; CLI data/batch/sampling settings apply to the new run. Losses and gradient norms are checked for finiteness before an optimizer update.

Use `--device cpu` when necessary. The default batch size is 64, learning rate is 2e-5, optimizer is AdamW, and training runs for 10 epochs. All architecture and loss options appear in `train --help`. For I3D/C3D features set `--input-dim` to the actual dimension. Feature extraction from raw videos is not included.

## Text features and attribute descriptions

The frozen CLIP ViT-B/16 encoder embeds category prompts and attribute descriptions once before training. It uses this repository's modified two-argument `encode_text` API. Checkpoints contain those embeddings, so inference and resume do not load or download CLIP. `--clip-model` accepts a model name or a local CLIP checkpoint.

New runs default to `configs/las_attributes_paper.json`, imported from Table 10 of
the [official CVPR supplementary material](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Wang_Weakly_Supervised_Video_CVPR_2026_supplemental.pdf).
The file stores separate `ucf` and `xd` mappings and the PDF SHA-256. The
`roadAccidents` key is mapped to the repository's `road accident` class name.
Earlier audits incorrectly described these published attributes as unavailable.

`configs/las_attributes.json` retains the previous manually authored substitutes
for reproducing old runs. Select it explicitly with `--attributes`. Both the old
flat mapping and the new dataset-specific mapping are supported. Resume uses the
embeddings already saved in the checkpoint; it does not replace attributes. New
runs record the text-source path/hash in their logs/checkpoints. No external LLM
service is called.

To regenerate the official configuration from a downloaded supplementary PDF:

```bash
.venv/bin/python src/las_import_attributes.py \
  --pdf /path/to/Wang_Weakly_Supervised_Video_CVPR_2026_supplemental.pdf \
  --output configs/las_attributes_paper.json
```

This importer requires `pdftotext`, validates both class sets, and rejects missing
or duplicate entries rather than silently using incomplete attributes.

An offline `--text-features path.pt` file can contain:

```python
{
    'category': category_embeddings,  # float tensor [classes, text_dim]
    'attribute': attribute_embeddings,  # same shape
    'class_names': class_names('ucf'),  # from las_data; exact ordering is checked
}
```

## Method coverage and explicit implementation choices

`src/las_model.py` implements selectable local/global adapters, binary and visual category heads, category/attribute text fusion, intention features and prototype confidence, connected-component pseudo labels, temporal MIL, auxiliary/consistency losses, and hard cross-intention contrastive learning. `src/las_evaluation.py` implements score fusion and temporal detection.

Several details are ambiguous or unspecified in the paper. The following choices are part of this implementation, not claims about the authors' hidden implementation:

| Detail | Implementation choice |
|---|---|
| Temporal encoder | New CLI runs default to `--adapter lgt`: original VadCLIP local Transformer plus residual similarity/distance GCN branches, with padding/device fixes. UCF defaults: window 8, one head, two layers; XD: window 64, one head, one layer. `--adapter simple` retains the earlier overlapping-window/single-graph implementation and its defaults (window 32, eight heads, one layer). |
| Text concatenation dimensions | Concatenate frozen category and attribute embeddings, then learn a linear projection into the visual width. No learned CLIP prompt tokens or additional unspecified cross-modal enhancement block. |
| Cross-modal scores | Softmax of cosine similarity divided by temperature 0.07, so all three fused branches represent probabilities. |
| Binary branch | One sigmoid anomaly probability; normal probability is its complement. |
| Intention dimension | Each of position, velocity and acceleration has `floor(width/3)` channels. At width 512, their concatenation has 510 channels. No silent requirement that 512 be divisible by three. |
| Motion boundaries | Absolute backward differences with a zero initial difference; convolution kernel three with symmetric zero padding. Invalid time steps are masked before differences and attention. |
| Prototype initialization | Random unit vectors stored as checkpoint buffers; selected-frame EMA updates only during training. Confidence is the raw cosine, including negative values. Threshold alpha defaults to 0.7; beta is 0.1. |
| Contrastive categories | Detached argmax of the intention logits, mined independently within each video. The anchor itself is excluded; anchors without both a positive and a negative are skipped. M defaults to 10. |
| Eq. 11 denominator | Default is conventional normalized InfoNCE. `--eq11-exact` uses raw dot products, the printed negative-only denominator and a full valid-time denominator per video. Its loss can be negative and is unbounded below. Partner mining still uses cosine within a video (an unspecified choice). The old `--literal-contrast` only changes the denominator of normalized InfoNCE; it is **not** exact Eq. 11. |
| Contrastive weight | Eq. 9 omits the separately introduced contrastive term. `--objective eq9` follows that printed total. Default `--objective extended` adds `contrast_weight * Lcst`, weight 1, an explicit extension. |
| ACC edges | Eqs. 12–13 use signed cross-modal **cosine**, independently of classification temperature, followed by strict `> tau` and exact DFS. New runs use `--acc-similarity cosine`; `probability` retains the old, incorrect-for-Eq.-13 input for checkpoint reproducibility. Defaults: tau 0.9, eta 0.5. |
| ACC targets | Detached, soft component-average fused probabilities; assign each frame the probabilities of its nearest feature prototype. No ground-truth frame labels are used. A connected graph legitimately produces one component; no minimum cluster count is imposed. |
| Loss reductions | Per-video temporal means followed by a batch mean; MIL uses `max(floor(valid_length/16), 1)`. Multi-label video targets are normalized for the fine-grained loss. Consistency weight is 0.3. |
| Detection | Video threshold 0.1, snippet threshold 0.2, half-open intervals, outer context on each side equal to 25% of interval length (at least one snippet), inner mean minus outer mean, per-class NMS IoU 0.6. Outer extent and NMS value are assumptions. |

The CPU DFS step transfers each video's boolean adjacency to the host. It is exact, but can become a training bottleneck at large batch sizes or sequence lengths. Attention and adjacency costs are bounded by `max_length`; inference processes long videos in independent chunks.

### Controlled reproduction diagnostics

`--text-init mean` is an optional initialization experiment: initialize the trainable
text concatenation projection as `[I/2, I/2]`, with zero bias. It preserves the
average category/attribute CLIP embedding initially and requires visual width to
equal text embedding width. `random` remains the default and preserves old
checkpoint behavior. Neither initialization is specified by the paper.

`--aux-weight 0` disables the ACC auxiliary contribution to the objective for an
ablation, while still measuring the graph and auxiliary loss. The default `1`
retains Eq. 6/7. A run with a different weight must not be reported as the full
paper model. These options are checkpointed and restored on resume.

To inspect whether a large contrastive loss actually has a large or conflicting
gradient, use training features only:

```bash
.venv/bin/python src/las_diagnose.py \
  --checkpoint model/las_ucf_best.pt --samples 32 --gradient-samples 8 \
  --output model/las_diagnostics.json
```

The optional gradient report compares the base and contrastive objectives on the
shared visual encoder, with independent norms and gradient cosine. It does not
update parameters, prototype buffers, or parameter `.grad` fields. A large scalar
loss is not evidence that its gradients dominate. Check the per-class prototype
update counters and ACC connectivity as well as AUC.

For frame-level metrics of each individual visual/text/IAM head, add
`--audit-heads` to `evaluate`. This saves their anomaly scores (`1 - normal
probability`) and reports additional AUC/AP keys in `metrics.json`, while
preserving the ordinary fused score and checkpoint-selection metric.

See [equation-level audit and remaining reproduction gaps](LAS_VAD_PAPER_AUDIT.md). Old checkpoints lacking `acc_similarity` load with `probability`, including on resume. To apply the corrected ACC, start fresh training with a new checkpoint destination. The console prints the actual ACC input. Merely reevaluating an old checkpoint does not retrain its representation.

## Evaluation outputs

Evaluation writes `manifest.json`, one numbered `.npz` per unique video (`fused` class probabilities, `anomaly` scores, and `binary`/`alignment` branch scores), `proposals.json`, and `metrics.json`. Proposal `start` and `end` are half-open **snippet indices**, not seconds or original frame indices. Scores and JSON metrics use the range 0–1; console metrics are percentages.

`--frame-gt` accepts a flattened binary NumPy array. Predictions are repeated `--frames-per-snippet` times (default 16) in unique-video CSV order. The total length must match exactly; the evaluator refuses to silently truncate ground truth. If source videos need per-video trimming, align the saved snippet predictions with the source frame annotations explicitly before measuring metrics.

For detection mAP, pass `--segment-gt annotations.json` with the exact same video IDs, including empty lists for normal videos:

```json
{
  "Normal_Videos_001_x264": [],
  "Abuse001_x264": [{"class_id": 1, "start": 12, "end": 28}]
}
```

JSON ground truth coordinates must also be snippets. Alternatively, use the existing paired `--gt-segment-path list/gt_segment_ucf.npy --gt-label-path list/gt_label_ucf.npy` files: these are explicitly interpreted as frame coordinates and converted by the evaluator. Detection evaluation uses one-to-one matching, all-points interpolated AP, and IoUs 0.1–0.5; classes absent from ground truth are excluded. This evaluator avoids the original utilities' hard-coded video counts, but its AP convention may differ from the benchmark authors' evaluation scripts.

## Verification

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Tests cover MIL pooling, transitive connected components, cross-modal rectification, detached pseudo labels, hard positive/negative mining, EMA updates, padding invariance, finite gradients, short/chunked inference, label validation, proposals, and detection mAP. A CPU integration check also exercised cached CLIP text encoding, training, saving, exact deterministic resume, crop aggregation, frame metrics, and detection metrics with synthetic features. Synthetic tests establish functionality; they do not establish the paper's reported accuracy.

Feature-loader tests additionally compare float16 pooling and padding exactly against `UCFDataset`, check balanced batches, and check repository-relative path resolution. The 2026-09-21 review confirmed this workspace's `datasets/UCFClipFeatures` is a valid symlink to `/home/islab/thanh/datasets/UCF-Crime/UCFClipFeatures`: all 16,100 train and 290 test feature files exist. See [real-feature training review](LAS_VAD_REVIEW.md) for the run and metrics.

## Periodic validation, best checkpoint and logging

Training now validates periodically **and at the end of each epoch**. The default intervals follow the sample counters in the original VadCLIP scripts:

| Dataset | Loss log interval | Validation interval | Best checkpoint metric |
|---|---:|---:|---|
| UCF-Crime | 1,280 processed examples | 1,280 processed examples | Fused frame AUC |
| XD-Violence | 4,800 processed examples | 4,800 processed examples | Fused frame AP |

With total batch size 64, UCF validates every 20 optimizer updates and XD every 75. `--log-every` and `--validate-every` are **example counts, not batch counts**. A boundary crossed by a batch triggers the event, even if the interval is not divisible by batch size. Counters restart each epoch; the true number of processed examples is printed as `step`. This fixes the original scripts' zero-based counter offset. Unlike the original XD trainer (which validates only at epoch end), LAS-VAD also validates at its periodic interval. Set `--validate-every 0` to disable validation for an offline training-only run.

```bash
.venv/bin/python src/las_vad.py train \
  --dataset ucf --checkpoint model/las_ucf/latest.pt \
  --best-checkpoint model/las_ucf/best.pt \
  --batch-size 64 --epochs 10 --device cuda:0 --workers 4
```

Without `--test-list`, validation uses the existing benchmark test CSV and the matching frame/segment/label NPY files, following VadCLIP's split convention. For a separate validation split, supply `--test-list`, `--frame-gt`, and either `--segment-gt` JSON or the pair `--gt-segment-path` / `--gt-label-path`. The paired VadCLIP NPY files use frame coordinates and are converted to snippet coordinates by dividing by `--frames-per-snippet`; normal (`A`/`Normal`) annotations are excluded from anomaly detection mAP. Arrays must follow the unique-video CSV order.

Console losses use VadCLIP's `epoch: ... | step: ... | loss1: ... | loss2: ...` format, with LAS-VAD's ACC, consistency and contrastive losses named explicitly. Each validation prints:

- Fused frame AUC/AP and abnormal-video-only Ano-AUC/Ano-AP.
- mAP at IoU 0.1 through 0.5 and average mAP when detection annotations are provided.
- C-branch AUC/AP (binary head) and A-branch AUC/AP (`1 - p_f(normal)`, the combined fine-grained branch).
- Abnormal-video and frame counts. A one-class abnormal subset reports N/A rather than failing validation.

Loss logs also report the current batch's mean ACC component count and fraction of videos with one component. JSONL adds inter-frame edge density, largest-component fraction, and per-class prototype EMA update counters. ACC statistics describe the current batch; displayed losses are running epoch means. Legacy checkpoints mark prototype update history incomplete rather than treating missing historical counters as zero updates.

LAS-VAD's checkpoint score is the average of the binary and fine-grained anomaly scores, as described by its inference rule. This differs from VadCLIP's original selection branches (C-branch for UCF, alignment branch for XD). Detection metrics use LAS-VAD proposals and the documented all-points AP convention, not the original hard-coded VadCLIP proposal/evaluation routines.

`--checkpoint` stores the **latest** state after every validation and epoch completion. `--best-checkpoint` stores only strictly improving validation states; its default is `<checkpoint_stem>_best<suffix>`. JSONL retains all losses, validation metrics, optimizer step numbers and selection decisions in machine-readable form. Best and latest paths must differ. The model remains at its latest training state and is not rolled back to the best weights between epochs.

Both checkpoints contain optimizer/RNG state, the epoch's sampler-start RNG state, consumed batches, running losses, and best score. Resume from either latest or best is supported, including a best checkpoint from the middle of an epoch. Mid-epoch resume requires the same training CSV, batch size, sampling strategy and worker count. Validation restores the previous train/eval mode, never updates intention prototypes, and does not consume the training RNG. Old epoch-complete `las-vad-v1` checkpoints remain loadable.

## Local/global encoder versus VadCLIP LGT

The legacy `--adapter simple` encoder contains both components, exposed as explicit methods in `src/las_model.py`:

```text
CLIP image features
  -> encode_local: positional embeddings + self.temporal (local Transformer)
  -> encode_global: cosine adjacency + self.graph (graph projection)
  -> binary / fine-grained / intention branches
```

| Component | Legacy `--adapter simple` | Original `CLIPVAD.encode_video` LGT |
|---|---|---|
| Local attention | Overlapping windows, stride half the window, average overlapping outputs | Fixed non-overlapping block attention mask |
| Transformer | PyTorch pre-norm TransformerEncoder, GELU, configurable dropout | Custom residual attention blocks, QuickGELU |
| Similarity graph | Softmax of cosine adjacency over valid frames; one projection and GELU | Cosine threshold 0.7; two residual graph convolution layers |
| Distance graph | Not present | A separate distance-adjacency branch with two residual graph convolution layers |
| Graph output | `GELU(softmax(A) @ X @ W)` | Concatenate the two graph branches, then linear projection |
| Padding | Masked in local attention, graph edges and output | The original temporal call passes no padding mask; distance graph includes padded positions |

The legacy adapter is **not the identical LGT adapter** from VadCLIP. Its `temporal.*`, `positions.*` and `graph.*` state-dict names remain unchanged for reproducibility. Checkpoints without an `adapter` config field are interpreted as `simple`.

New training now defaults to `--adapter lgt`, implemented in `src/las_lgt.py`. This reuses the original Transformer, QuickGELU and residual GraphConvolution classes, both graph branches, and positional initialization with standard deviation 0.01. The distance graph uses the same formula as VadCLIP, but is constructed on the feature device rather than hard-coded CUDA. Padding is masked at each layer. A numerical test compares its output with `CLIPVAD.encode_video` using identical weights on unpadded inputs; separate tests cover padding, short clips and backward gradients. `--dropout` applies only to `simple`; LGT retains the original adapter's activation/dropout choices.

`--resume` always takes architecture from the checkpoint, even if CLI defaults now select LGT. To switch from simple to LGT, start a **new training run**, not a resume of the simple checkpoint. Other LAS-VAD heads/losses remain unchanged; an improvement from switching adapters is not proof that the paper has been exactly reproduced.
