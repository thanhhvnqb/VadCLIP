"""Feature CSV loading and frozen text encoding for LAS-VAD."""
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from utils.tools import process_feat

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

UCF_LABELS = ['Normal', 'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion',
              'Fighting', 'RoadAccidents', 'Robbery', 'Shooting', 'Shoplifting', 'Stealing', 'Vandalism']
XD_LABELS = ['A', 'B1', 'B2', 'B4', 'B5', 'B6', 'G']
UCF_NAMES = ['normal', 'abuse', 'arrest', 'arson', 'assault', 'burglary', 'explosion',
             'fighting', 'road accident', 'robbery', 'shooting', 'shoplifting', 'stealing', 'vandalism']
XD_NAMES = ['normal', 'fighting', 'shooting', 'riot', 'abuse', 'car accident', 'explosion']


def class_names(dataset):
    if dataset not in ('ucf', 'xd'):
        raise ValueError(f'Unknown dataset: {dataset}')
    return UCF_NAMES if dataset == 'ucf' else XD_NAMES


def encode_labels(label, dataset):
    keys = UCF_LABELS if dataset == 'ucf' else XD_LABELS
    parts = [label] if dataset == 'ucf' else label.split('-')
    if any(part not in keys for part in parts):
        raise ValueError(f'Unknown {dataset} label: {label!r}')
    target = torch.zeros(len(keys))
    for part in parts:
        target[keys.index(part)] = 1
    if target[0] and target[1:].any():
        raise ValueError(f'Mixed normal and abnormal labels: {label}')
    return target


class FeatureDataset(Dataset):
    def __init__(self, csv_path, dataset, max_length=256, training=True, feature_root=None, input_dim=512):
        class_names(dataset)
        if max_length < 1:
            raise ValueError('max_length must be positive')
        self.csv_path = Path(csv_path).expanduser()
        if not self.csv_path.is_absolute() and not self.csv_path.is_file():
            self.csv_path = REPOSITORY_ROOT / self.csv_path
        self.csv_path = self.csv_path.resolve()
        with self.csv_path.open(newline='') as stream:
            reader = csv.DictReader(stream)
            if not {'path', 'label'} <= set(reader.fieldnames or []):
                raise ValueError('CSV must contain path and label columns')
            self.rows = list(reader)
        if not self.rows:
            raise ValueError('Feature CSV is empty')
        self.dataset, self.max_length, self.training = dataset, max_length, training
        self.feature_root = Path(feature_root).expanduser().resolve() if feature_root else None
        self.input_dim = input_dim
        self.paths, self.labels = [], []
        for row in self.rows:
            self.labels.append(encode_labels(row['label'], dataset))
            path = Path(row['path']).expanduser()
            if not path.is_absolute():
                if self.feature_root is not None:
                    path = self.feature_root / path
                else:
                    candidates = [Path.cwd() / path, REPOSITORY_ROOT / path, self.csv_path.parent / path]
                    path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
            self.paths.append(path)
        missing = [str(path) for path in self.paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f'{len(missing)} missing feature files; first: {missing[0]}')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.paths[index]
        array = np.load(path, allow_pickle=False)
        if array.ndim != 2 or len(array) == 0 or not np.isfinite(array).all():
            raise ValueError(f'{path}: expected finite, nonempty [time, dimension] features')
        if self.input_dim is not None and array.shape[1] != self.input_dim:
            raise ValueError(f'{path}: expected feature dimension {self.input_dim}, got {array.shape[1]}')
        if self.training:
            # Use VadCLIP's exact temporal pooling/padding, including float16
            # accumulation behavior, then convert to float32 for the network.
            array, length = process_feat(array, self.max_length)
        else:
            length = len(array)
        default_id = path.stem.split('__')[0] if self.dataset == 'ucf' else path.stem.rsplit('__', 1)[0]
        video_id = row.get('video_id') or default_id
        return torch.from_numpy(array.astype(np.float32, copy=False)), self.labels[index].clone(), length, video_id


class BalancedBatchSampler(Sampler):
    """VadCLIP UCF sampling: shuffle each pool and pair normal/anomaly halves.

    Stop when the smaller pool is exhausted and drop incomplete halves, as in
    zip(normal_loader, anomaly_loader) with drop_last=True. batch_size is TOTAL.
    """
    def __init__(self, dataset, batch_size):
        if batch_size < 2 or batch_size % 2:
            raise ValueError('Balanced sampling requires an even total batch size >= 2')
        self.half = batch_size // 2
        self.normal = [i for i, label in enumerate(dataset.labels) if label[0] == 1]
        self.anomaly = [i for i, label in enumerate(dataset.labels) if label[0] == 0]
        if len(self) == 0:
            raise ValueError('Each pool must have at least batch_size/2 examples')

    def __len__(self):
        return min(len(self.normal), len(self.anomaly)) // self.half

    def __iter__(self):
        normal = torch.tensor(self.normal)[torch.randperm(len(self.normal))].tolist()
        anomaly = torch.tensor(self.anomaly)[torch.randperm(len(self.anomaly))].tolist()
        for batch in range(len(self)):
            start = batch * self.half
            yield normal[start:start+self.half] + anomaly[start:start+self.half]


def attribute_descriptions(dataset, attributes_path):
    names = class_names(dataset)
    with open(attributes_path) as stream:
        descriptions = json.load(stream)
    # Official Table 10 has separate descriptions for each dataset. Continue
    # accepting the old flat mapping for existing user-authored configurations.
    if dataset in descriptions:
        descriptions = descriptions[dataset]
    missing = set(names) - descriptions.keys()
    if missing:
        raise ValueError(f'Missing attribute descriptions: {sorted(missing)}')
    if any(not isinstance(descriptions[name], str) or not descriptions[name].strip() for name in names):
        raise ValueError('Attribute descriptions must be nonempty strings')
    return [descriptions[name] for name in names]


@torch.no_grad()
def text_embeddings(dataset, attributes_path, device='cpu', clip_model='ViT-B/16'):
    # This repository's CLIP fork accepts token embeddings AND token IDs.
    from clip import clip
    names = class_names(dataset)
    descriptions = attribute_descriptions(dataset, attributes_path)
    model, _ = clip.load(clip_model, device=device)
    model.eval()
    tokens = clip.tokenize([f'a video of {name}' for name in names] +
                           descriptions, truncate=True).to(device)
    embeddings = model.encode_text(model.encode_token(tokens), tokens).float().cpu()
    return embeddings[:len(names)], embeddings[len(names):]
