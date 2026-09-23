"""LAS-VAD thresholding, outer-inner confidence, NMS and temporal detection AP."""
import numpy as np
import torch


def temporal_iou(a, b):
    intersection = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return intersection / max(a[1] - a[0] + b[1] - b[0] - intersection, 1e-12)


def proposals(scores, video_threshold=0.1, snippet_threshold=0.2, nms_threshold=0.6, outer_ratio=0.25):
    """Return class/start/end/score dictionaries in half-open snippet coordinates."""
    scores = np.asarray(scores)
    if scores.ndim != 2 or not len(scores):
        raise ValueError('scores must be a nonempty [time, classes] array')
    k = max(len(scores) // 16, 1)
    video_scores = np.sort(scores, axis=0)[-k:].mean(0)
    result = []
    for c in range(1, scores.shape[1]):
        if video_scores[c] <= video_threshold:
            continue
        values = scores[:, c]
        edges = np.diff(np.r_[0, (values > snippet_threshold).astype(int), 0])
        candidates = []
        for start, end in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
            margin = max(1, int(np.ceil((end - start) * outer_ratio)))
            outer = np.r_[values[max(0, start-margin):start], values[end:min(len(values), end+margin)]]
            confidence = values[start:end].mean() - (outer.mean() if len(outer) else 0)
            candidates.append(dict(class_id=c, start=int(start), end=int(end), score=float(confidence)))
        kept = []
        for candidate in sorted(candidates, key=lambda p: p['score'], reverse=True):
            if all(temporal_iou((candidate['start'], candidate['end']), (p['start'], p['end'])) <= nms_threshold for p in kept):
                kept.append(candidate)
        result.extend(kept)
    return result


@torch.no_grad()
def predict_video(model, features, chunk_length=None):
    scores = predict_video_branches(model, features, chunk_length)
    return scores['fused'], scores['anomaly']


@torch.no_grad()
def predict_video_branches(model, features, chunk_length=None, include_heads=False):
    """Split once without an empty trailing chunk; preserve every original snippet."""
    device = next(model.parameters()).device
    size = chunk_length or model.config.max_length
    if not 1 <= size <= model.config.max_length or len(features) == 0:
        raise ValueError('Invalid chunk size or empty video')
    parts = {name: [] for name in ('fused', 'anomaly', 'binary')}
    if include_heads:
        parts.update({name: [] for name in ('multiclass', 'language', 'intention')})
    for start in range(0, len(features), size):
        part = features[start:start+size].to(device)
        output = model(part.unsqueeze(0), [len(part)])
        for name in parts:
            values = output[name][0]
            if name in ('multiclass', 'language', 'intention'):
                values = 1 - values[:, 0]
            parts[name].append(values.cpu())
    result = {name: torch.cat(values).numpy() for name, values in parts.items()}
    result['alignment'] = 1 - result['fused'][:, 0]
    return result


def detection_map(predictions, ground_truth, class_count, thresholds=(0.1, 0.2, 0.3, 0.4, 0.5)):
    """Ground truth: {video_id: [{class_id, start, end}]}, in snippet units.

    Classes absent from ground truth are excluded from the macro average.
    Score-ranked one-to-one matching and all-points interpolated AP are used.
    """
    results = {}
    for threshold in thresholds:
        aps = []
        for c in range(1, class_count):
            targets = {v: [s for s in segments if s['class_id'] == c] for v, segments in ground_truth.items()}
            count = sum(map(len, targets.values()))
            if count == 0:
                continue
            ranked = sorted([(v, s) for v, segments in predictions.items() for s in segments if s['class_id'] == c],
                            key=lambda item: item[1]['score'], reverse=True)
            matched, hits = set(), []
            for video, segment in ranked:
                candidates = [(temporal_iou((segment['start'], segment['end']), (s['start'], s['end'])), i)
                              for i, s in enumerate(targets.get(video, [])) if (video, i) not in matched]
                overlap, index = max(candidates, default=(0, -1))
                hit = overlap >= threshold
                hits.append(hit)
                if hit:
                    matched.add((video, index))
            if not hits:
                aps.append(0.)
                continue
            tp = np.cumsum(hits)
            recall = np.r_[0, tp / count, 1]
            precision = np.r_[0, tp / np.arange(1, len(hits)+1), 0]
            precision = np.maximum.accumulate(precision[::-1])[::-1]
            changes = np.where(recall[1:] != recall[:-1])[0]
            aps.append(float(np.sum((recall[changes+1]-recall[changes]) * precision[changes+1])))
        results[f'mAP@{threshold:.1f}'] = float(np.mean(aps)) if aps else None
    valid = [value for value in results.values() if value is not None]
    results['mAP'] = float(np.mean(valid)) if valid else None
    return results
