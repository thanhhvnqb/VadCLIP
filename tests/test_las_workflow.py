"""Offline CLI integration check, requiring neither datasets nor a CLIP download."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from las_data import class_names


class WorkflowTests(unittest.TestCase):
    def test_train_resume_and_evaluate_xd(self):
        self.run_workflow('xd')

    def test_train_resume_and_evaluate_ucf_balanced(self):
        self.run_workflow('ucf')

    def run_workflow(self, dataset):
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(0)
            rows = []
            labels = [('normal', 'A'), ('anomaly', 'B1-B2')] if dataset == 'xd' else [('normal', 'Normal'), ('anomaly', 'Abuse')]
            for video, label in labels:
                for crop in range(2):
                    path = root / f'{video}__{crop}.npy'
                    np.save(path, rng.standard_normal((9, 12)).astype('float32'))
                    rows.append(dict(path=str(path), label=label))
            with (root / 'features.csv').open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=['path', 'label'])
                writer.writeheader()
                writer.writerows(rows)
            count = len(class_names(dataset))
            torch.save(dict(category=torch.eye(count), attribute=torch.eye(count), class_names=class_names(dataset)), root / 'text.pt')
            np.save(root / 'gt.npy', np.r_[np.zeros(144), np.ones(144)])
            (root / 'segments.json').write_text(json.dumps({'normal': [], 'anomaly': [dict(class_id=1, start=0, end=9)]}))
            environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')

            def run(arguments):
                result = subprocess.run([sys.executable, str(repository / 'src/las_vad.py')] + arguments,
                                        cwd=repository, env=environment, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            training = ['train', '--dataset', dataset, '--train-list', str(root/'features.csv'),
                        '--text-features', str(root/'text.pt'), '--input-dim', '12', '--width', '12',
                        '--heads', '3', '--window', '4', '--max-length', '8', '--batch-size', '2', '--device', 'cpu',
                        '--test-list', str(root/'features.csv'), '--frame-gt', str(root/'gt.npy'),
                        '--segment-gt', str(root/'segments.json'), '--validate-every', '2', '--log-every', '2']
            run(training + ['--epochs', '1', '--checkpoint', str(root/'model.pt')])
            run(training + ['--epochs', '2', '--checkpoint', str(root/'model.pt'), '--resume', str(root/'model.pt')])
            run(training + ['--epochs', '2', '--checkpoint', str(root/'continuous.pt')])
            resumed = torch.load(root/'model.pt', weights_only=True)
            continuous = torch.load(root/'continuous.pt', weights_only=True)
            self.assertEqual(resumed['epoch'], 2)
            self.assertEqual(resumed['run_config']['sampling'], 'balanced' if dataset == 'ucf' else 'shuffle')
            best = torch.load(root/'model_best.pt', weights_only=True)
            records = [json.loads(line) for line in (root/'model.jsonl').read_text().splitlines()]
            validations = [record for record in records if record['event'] == 'validation']
            metric = 'AUC' if dataset == 'ucf' else 'AP'
            self.assertEqual(len(validations), 4)
            self.assertEqual(best['metrics'][metric], max(record['metrics'][metric] for record in validations))
            for key in resumed['model']:
                torch.testing.assert_close(resumed['model'][key], continuous['model'][key], rtol=0, atol=0)
            run(['evaluate', '--checkpoint', str(root/'model.pt'), '--test-list', str(root/'features.csv'),
                 '--frame-gt', str(root/'gt.npy'), '--segment-gt', str(root/'segments.json'),
                 '--output', str(root/'results'), '--device', 'cpu'])
            manifest = json.loads((root/'results/manifest.json').read_text())
            self.assertEqual([entry['video_id'] for entry in manifest], ['normal', 'anomaly'])
            self.assertTrue(all(entry['crops'] == 2 and entry['snippets'] == 9 for entry in manifest))
            metrics = json.loads((root/'results/metrics.json').read_text())
            for key in ('AP', 'AUC', 'mAP'):
                self.assertTrue(0 <= metrics[key] <= 1)


if __name__ == '__main__':
    unittest.main()
