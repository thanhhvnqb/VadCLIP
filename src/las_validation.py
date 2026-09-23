"""Shared LAS-VAD validation, branch diagnostics and VadCLIP-style metric logs."""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from las_data import REPOSITORY_ROOT, UCF_LABELS, XD_LABELS
from las_evaluation import detection_map, predict_video_branches, proposals


def resolve_path(path):
    path = Path(path).expanduser()
    if not path.is_absolute() and not path.exists():
        path = REPOSITORY_ROOT / path
    return path


class ValidationData:
    """Load annotations once; align with unique-video order in the feature CSV."""
    def __init__(self, dataset, frame_gt=None, segment_gt=None, gt_segment_path=None,
                 gt_label_path=None, frames_per_snippet=16):
        if frames_per_snippet < 1:
            raise ValueError('frames_per_snippet must be positive')
        self.dataset = dataset
        self.frames_per_snippet = frames_per_snippet
        self.frame_gt = None
        if frame_gt:
            self.frame_gt = np.load(resolve_path(frame_gt), allow_pickle=False).reshape(-1)
            if not np.isin(self.frame_gt, [0, 1]).all() or len(np.unique(self.frame_gt)) != 2:
                raise ValueError('Frame GT must contain both binary classes')
        if segment_gt and (gt_segment_path or gt_label_path):
            raise ValueError('Use JSON segment GT OR the paired VadCLIP NPY files')
        if bool(gt_segment_path) != bool(gt_label_path):
            raise ValueError('gt-segment-path and gt-label-path must be provided together')
        self.segment_gt = None
        if segment_gt:
            with resolve_path(segment_gt).open() as stream:
                self.segment_gt = json.load(stream)
        self.legacy_segments = self.legacy_labels = None
        if gt_segment_path:
            # The repository stores variable-length annotations as object arrays.
            self.legacy_segments = np.load(resolve_path(gt_segment_path), allow_pickle=True)
            self.legacy_labels = np.load(resolve_path(gt_label_path), allow_pickle=True)

    def detection_targets(self, videos):
        if self.segment_gt is not None:
            if set(self.segment_gt) != set(videos):
                raise ValueError('Segment GT must contain exactly the evaluated video IDs')
            return self.segment_gt
        if self.legacy_segments is None:
            return None
        if len(self.legacy_segments) != len(videos) or len(self.legacy_labels) != len(videos):
            raise ValueError('VadCLIP annotation rows must match unique-video CSV order')
        keys = UCF_LABELS if self.dataset.dataset == 'ucf' else XD_LABELS
        targets = {}
        for video, segments, labels in zip(videos, self.legacy_segments, self.legacy_labels):
            if len(segments) != len(labels):
                raise ValueError(f'Mismatched segment/label count for {video}')
            targets[video] = []
            for interval, label in zip(segments, labels):
                if label in ('A', 'Normal'):
                    continue
                if label not in keys:
                    raise ValueError(f'Unknown detection class {label!r}')
                start, end = map(float, interval)
                if not np.isfinite([start, end]).all() or start < 0 or end <= start:
                    raise ValueError(f'Invalid detection interval for {video}: {interval}')
                # Existing gt_segment*.npy annotations use FRAME coordinates.
                targets[video].append(dict(class_id=keys.index(label), start=start/self.frames_per_snippet,
                                           end=end/self.frames_per_snippet))
        return targets


def collect_predictions(model, dataset, include_heads=False):
    """Evaluate without changing dropout mode or intention prototypes during training."""
    was_training = model.training
    model.eval()
    grouped = {}
    try:
        for index in range(len(dataset)):
            features, label, length, video = dataset[index]
            scores = predict_video_branches(model, features, include_heads=include_heads)
            abnormal = not bool(label[0])
            if video in grouped:
                entry = grouped[video]
                if len(entry['anomaly']) != length or entry['abnormal'] != abnormal:
                    raise ValueError(f'Inconsistent crop length/label for {video}')
                for name, values in scores.items():
                    entry[name] += values
                entry['count'] += 1
            else:
                grouped[video] = dict(**scores, abnormal=abnormal, count=1)
        for entry in grouped.values():
            for name in ('fused', 'anomaly', 'binary', 'alignment', 'multiclass', 'language', 'intention'):
                if name not in entry:
                    continue
                entry[name] /= entry['count']
    finally:
        model.train(was_training)
    return grouped


def compute_metrics(grouped, annotations, video_threshold=.1, snippet_threshold=.2, nms_threshold=.6):
    detections = {video: proposals(entry['fused'], video_threshold, snippet_threshold, nms_threshold)
                  for video, entry in grouped.items()}
    metrics = {}
    if annotations.frame_gt is not None:
        gt = annotations.frame_gt
        branches = [('anomaly', ''), ('binary', 'C-branch '), ('alignment', 'A-branch ')]
        for name in ('multiclass', 'language', 'intention'):
            if all(name in entry for entry in grouped.values()):
                branches.append((name, name+' head '))
        scores = {name: np.concatenate([np.repeat(entry[name], annotations.frames_per_snippet)
                                       for entry in grouped.values()])
                  for name, _ in branches}
        if len(gt) != len(scores['anomaly']):
            raise ValueError(f"Frame GT has {len(gt)} entries but predictions have {len(scores['anomaly'])}; "
                             'verify CSV order, crop IDs and frames-per-snippet')
        if any(not np.isfinite(values).all() for values in scores.values()):
            raise ValueError('Nonfinite validation scores')
        for name, prefix in branches:
            metrics[prefix+'AUC'] = float(roc_auc_score(gt, scores[name]))
            metrics[prefix+'AP'] = float(average_precision_score(gt, scores[name]))
        abnormal = np.concatenate([np.full(len(entry['anomaly'])*annotations.frames_per_snippet,
                                          entry['abnormal'], dtype=bool) for entry in grouped.values()])
        metrics.update(abnormal_videos=sum(entry['abnormal'] for entry in grouped.values()),
                       abnormal_video_frames=int(abnormal.sum()),
                       abnormal_video_positive_frames=int(gt[abnormal].sum()))
        usable = abnormal.any() and len(np.unique(gt[abnormal])) == 2
        metrics['Ano-AUC'] = float(roc_auc_score(gt[abnormal], scores['anomaly'][abnormal])) if usable else None
        metrics['Ano-AP'] = float(average_precision_score(gt[abnormal], scores['anomaly'][abnormal])) if usable else None
    targets = annotations.detection_targets(list(grouped))
    if targets is not None:
        metrics.update(detection_map(detections, targets, len(annotations.dataset.labels[0])))
    return metrics, detections


def validate(model, annotations, include_heads=False, **thresholds):
    predictions = collect_predictions(model, annotations.dataset, include_heads=include_heads)
    metrics, detections = compute_metrics(predictions, annotations, **thresholds)
    return metrics, predictions, detections


def print_metrics(metrics, dataset):
    print('=' * 60)
    print(f"{'UCF-Crime' if dataset == 'ucf' else 'XD-Violence'} LAS-VAD Evaluation")
    print('=' * 60)
    print('[Coarse-grained anomaly detection - fused paper score]')
    for key in ('AUC', 'AP', 'Ano-AUC', 'Ano-AP'):
        if key in metrics:
            value = metrics[key]
            print(f'{key:30s}: {value * 100:.2f}%' if value is not None else f'{key:30s}: N/A (one-class or empty subset)')
    if 'mAP' in metrics:
        print('\n[Fine-grained anomaly detection]')
        for key in ('mAP@0.1', 'mAP@0.2', 'mAP@0.3', 'mAP@0.4', 'mAP@0.5', 'mAP'):
            value = metrics[key]
            print(f'{key:30s}: {value * 100:.2f}%' if value is not None else f'{key:30s}: N/A')
    print('\n[Auxiliary branch diagnostics]')
    for key in ('C-branch AUC', 'C-branch AP', 'A-branch AUC', 'A-branch AP'):
        if key in metrics:
            print(f'{key:30s}: {metrics[key] * 100:.2f}%')
    if 'abnormal_videos' in metrics:
        print(f"Abnormal-only: videos={metrics['abnormal_videos']}, frames={metrics['abnormal_video_frames']}, "
              f"abnormal_frames={metrics['abnormal_video_positive_frames']}, "
              f"normal_frames={metrics['abnormal_video_frames'] - metrics['abnormal_video_positive_frames']}")
    print('=' * 60, flush=True)
