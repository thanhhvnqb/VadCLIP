import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from las_data import FeatureDataset, class_names
from las_model import LASConfig, LASVAD
from las_validation import ValidationData, collect_predictions, compute_metrics
from las_vad import parser, train


class ValidationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(5)

    def fixture(self, root):
        rows = []
        rng = np.random.default_rng(5)
        for index, label in enumerate(['A', 'B1', 'A', 'B1']):
            path = root / f'video{index}.npy'
            np.save(path, rng.standard_normal((4, 12)).astype('float32'))
            rows.append(dict(path=str(path), label=label))
        with (root/'features.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=['path', 'label'])
            writer.writeheader()
            writer.writerows(rows)
        np.save(root/'gt.npy', np.array([0, 0, 0, 0, 0, 1, 1, 0]*2))
        torch.save(dict(category=torch.eye(7), attribute=torch.eye(7), class_names=class_names('xd')), root/'text.pt')
        return FeatureDataset(root/'features.csv', 'xd', training=False, input_dim=12)

    def test_validation_restores_training_mode_and_buffers(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = self.fixture(root)
            model = LASVAD(torch.eye(7), torch.eye(7), LASConfig(input_dim=12, width=12, heads=3, max_length=8, window=4))
            model.train()
            buffers = {key: value.clone() for key, value in model.named_buffers()}
            rng = torch.get_rng_state().clone()
            predictions = collect_predictions(model, data)
            self.assertTrue(model.training)
            torch.testing.assert_close(torch.get_rng_state(), rng)
            for key, value in model.named_buffers():
                torch.testing.assert_close(value, buffers[key])
            metrics, _ = compute_metrics(predictions, ValidationData(data, root/'gt.npy', frames_per_snippet=1))
            for key in ['AUC', 'AP', 'Ano-AUC', 'Ano-AP', 'C-branch AUC', 'A-branch AP']:
                self.assertTrue(0 <= metrics[key] <= 1)
            self.assertEqual(metrics['abnormal_videos'], 2)
            output = model(torch.randn(1, 4, 12), [4])
            model.loss(output, torch.tensor([[0., 1., 0., 0., 0., 0., 0.]]), [4])['total'].backward()
            self.assertTrue(torch.isfinite(model.graph.weight.grad).all())
            # Restoring eval mode is equally important for standalone inference.
            model.eval()
            collect_predictions(model, data)
            self.assertFalse(model.training)

    def test_frame_annotation_conversion_and_perfect_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = self.fixture(root)
            segments = np.empty(4, dtype=object)
            labels = np.empty(4, dtype=object)
            for index in range(4):
                segments[index] = [[0, 64]] if index % 2 == 0 else [[16, 48]]
                labels[index] = ['A'] if index % 2 == 0 else ['B1']
            np.save(root/'segments.npy', segments)
            np.save(root/'labels.npy', labels)
            gt = ValidationData(data, gt_segment_path=root/'segments.npy', gt_label_path=root/'labels.npy')
            converted = gt.detection_targets([f'video{i}' for i in range(4)])
            self.assertEqual(converted['video0'], [])
            self.assertEqual(converted['video1'], [dict(class_id=1, start=1., end=3.)])
            # Construct perfect class and binary probabilities, without model noise.
            predictions = {}
            for index in range(4):
                values = np.array([0., 1., 1., 0.]) if index % 2 else np.zeros(4)
                scores = np.zeros((4, 7))
                scores[:, 0], scores[:, 1] = 1-values, values
                predictions[f'video{index}'] = dict(fused=scores, anomaly=values, binary=values,
                                                    alignment=values, abnormal=bool(index % 2), count=1)
            gt.frame_gt = np.repeat(np.load(root/'gt.npy'), 16)
            metrics, _ = compute_metrics(predictions, gt)
            for key in ['AUC', 'AP', 'Ano-AUC', 'Ano-AP', 'mAP']:
                self.assertEqual(metrics[key], 1.)

    def test_head_audit_preserves_fusion_and_existing_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = self.fixture(root)
            model = LASVAD(torch.eye(7), torch.eye(7),
                           LASConfig(input_dim=12, width=12, heads=3, max_length=3, window=2)).eval()
            ordinary = collect_predictions(model, data)
            audited = collect_predictions(model, data, include_heads=True)
            for video, entry in audited.items():
                np.testing.assert_array_equal(entry['anomaly'], ordinary[video]['anomaly'])
                average = (entry['multiclass']+entry['language']+entry['intention'])/3
                np.testing.assert_allclose(average, entry['alignment'], atol=1e-7)
            annotations = ValidationData(data, root/'gt.npy', frames_per_snippet=1)
            old, _ = compute_metrics(ordinary, annotations)
            new, _ = compute_metrics(audited, annotations)
            self.assertEqual(old, {key: new[key] for key in old})
            for head in ('multiclass', 'language', 'intention'):
                self.assertTrue(0 <= new[head+' head AUC'] <= 1)

    def test_best_is_not_overwritten_and_mid_epoch_resume_is_exact(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            common = ['train', '--dataset', 'xd', '--train-list', str(root/'features.csv'),
                      '--text-features', str(root/'text.pt'), '--input-dim', '12', '--width', '12',
                      '--heads', '3', '--window', '4', '--max-length', '8', '--batch-size', '2', '--device', 'cpu',
                      '--threads', '1', '--test-list', str(root/'features.csv'), '--frame-gt', str(root/'gt.npy'),
                      '--frames-per-snippet', '1', '--validate-every', '2', '--log-every', '3', '--epochs', '2']
            scores = [0.9, 0.8, 0.7, 0.6]
            with contextlib.redirect_stdout(io.StringIO()), patch('las_vad.validate', side_effect=[({'AP': score}, {}, {}) for score in scores]):
                train(parser().parse_args(common + ['--checkpoint', str(root/'latest.pt')]))
            latest = torch.load(root/'latest.pt', weights_only=True)
            best = torch.load(root/'latest_best.pt', weights_only=True)
            self.assertEqual(best['metrics']['AP'], .9)
            self.assertEqual(latest['metrics']['AP'], .6)
            self.assertEqual(best['epoch'], 1)
            self.assertFalse(best['epoch_complete'])
            self.assertEqual(best['step_in_epoch'], 1)
            records = [json.loads(line) for line in (root/'latest.jsonl').read_text().splitlines()]
            # Batch size 2 crosses log threshold 3 at example 4, without relying
            # on exact divisibility or producing a redundant first-step log.
            self.assertEqual([entry['examples'] for entry in records if entry['event'] == 'step'], [4, 4])
            with contextlib.redirect_stdout(io.StringIO()), patch('las_vad.validate', side_effect=[({'AP': score}, {}, {}) for score in scores[1:]]):
                train(parser().parse_args(common + ['--checkpoint', str(root/'resumed.pt'), '--resume', str(root/'latest_best.pt')]))
            resumed = torch.load(root/'resumed.pt', weights_only=True)
            self.assertEqual(resumed['global_step'], latest['global_step'])
            self.assertEqual(resumed['best_score'], .9)
            for name, tensor in latest['model'].items():
                torch.testing.assert_close(resumed['model'][name], tensor, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
