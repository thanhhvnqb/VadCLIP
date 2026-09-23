"""Compatibility checks against the existing VadCLIP feature loader."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from las_data import BalancedBatchSampler, FeatureDataset
from utils.dataset import UCFDataset


class FeatureLoaderTests(unittest.TestCase):
    def make_csv(self, root):
        rows = []
        rng = np.random.default_rng(3)
        for index, length in enumerate([2, 4, 17, 2, 4, 17]):
            path = root / f'video{index}__0.npy'
            np.save(path, rng.normal(size=(length, 512)).astype(np.float16))
            rows.append(dict(path=str(path), label='Normal' if index < 3 else 'Abuse'))
        csv_path = root / 'features.csv'
        with csv_path.open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=['path', 'label'])
            writer.writeheader()
            writer.writerows(rows)
        return csv_path

    def test_matches_vadclip_float16_pooling_and_padding(self):
        with tempfile.TemporaryDirectory() as folder:
            csv_path = self.make_csv(Path(folder))
            actual = FeatureDataset(csv_path, 'ucf', max_length=4)
            for normal, offset in [(True, 0), (False, 3)]:
                baseline = UCFDataset(4, str(csv_path), False, {'Normal': 'normal', 'Abuse': 'abuse'}, normal)
                for index in range(len(baseline)):
                    expected, _, expected_length = baseline[index]
                    features, _, length, _ = actual[offset + index]
                    torch.testing.assert_close(features, expected.float(), rtol=0, atol=0)
                    self.assertEqual(length, expected_length)

    def test_balanced_sampler_drops_incomplete_halves(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset = FeatureDataset(self.make_csv(Path(folder)), 'ucf', max_length=4)
            sampler = BalancedBatchSampler(dataset, 4)
            self.assertEqual(len(sampler), 1)
            batch = next(iter(sampler))
            self.assertEqual(len(set(batch)), 4)
            self.assertEqual(sum(int(dataset.labels[i][0]) for i in batch), 2)
            with self.assertRaises(ValueError):
                BalancedBatchSampler(dataset, 3)

    def test_repository_relative_paths_from_another_cwd(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            feature = root / 'datasets/example.npy'
            feature.parent.mkdir()
            np.save(feature, np.zeros((2, 512), dtype=np.float16))
            csv_path = root / 'features.csv'
            csv_path.write_text('path,label\ndatasets/example.npy,Normal\n')
            with patch('las_data.REPOSITORY_ROOT', root):
                data = FeatureDataset(csv_path, 'ucf')
                self.assertEqual(data.paths[0], feature)
                self.assertEqual(data[0][0].shape, (256, 512))
            with self.assertRaises(FileNotFoundError):
                FeatureDataset(csv_path, 'ucf', feature_root=root/'missing')

    def test_evaluation_keeps_all_snippets_and_validates_dimension(self):
        with tempfile.TemporaryDirectory() as folder:
            csv_path = self.make_csv(Path(folder))
            data = FeatureDataset(csv_path, 'ucf', max_length=4, training=False)
            self.assertEqual(data[2][0].shape, (17, 512))
            self.assertEqual(data[2][3], 'video2')
            data = FeatureDataset(csv_path, 'ucf', input_dim=256)
            with self.assertRaisesRegex(ValueError, 'expected feature dimension'):
                data[0]


if __name__ == '__main__':
    unittest.main()
