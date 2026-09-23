import sys
from pathlib import Path
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from las_model import LASConfig, LASVAD, connected_pseudo_labels, cross_intention_loss, mil_pool
from las_evaluation import detection_map, predict_video, proposals
from las_data import encode_labels


class LASVADTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        torch.set_num_threads(1)
        self.model = LASVAD(torch.randn(3, 8), torch.randn(3, 8),
                            LASConfig(input_dim=12, width=12, heads=3, max_length=32, window=4, dropout=0., alpha=0.))

    def test_mil_floor_and_lengths(self):
        scores = torch.arange(33.).view(1, 33, 1)
        self.assertEqual(mil_pool(scores, [32]).item(), 30.5)
        self.assertEqual(mil_pool(scores, [1]).item(), 0.)

    def test_connected_components_transitive_and_detached(self):
        # Adjacent directions connect but endpoints do not: DFS must merge all three.
        angles = torch.tensor([0., .4, .8, 3.14])
        features = torch.stack((angles.cos(), angles.sin()), -1)[None].requires_grad_()
        scores = torch.tensor([[[1., 0.], [0., 1.], [0., 1.], [1., 0.]]])
        pseudo = connected_pseudo_labels(features, scores, scores, [4], tau=.9, eta=0.)
        torch.testing.assert_close(pseudo[0, :3], torch.tensor([[1/3, 2/3]]).expand(3, -1))
        torch.testing.assert_close(pseudo[0, 3], scores[0, 3])
        self.assertFalse(pseudo.requires_grad)

    def test_rectification_changes_connectivity(self):
        features = torch.tensor([[[1., 0.], [.8, .6]]])
        language = torch.ones(1, 2, 2) / 2
        fused = torch.eye(2)[None]
        separate = connected_pseudo_labels(features, language, fused, [2], eta=0.)
        merged = connected_pseudo_labels(features, language, fused, [2], eta=.5)
        torch.testing.assert_close(separate, fused)
        torch.testing.assert_close(merged, torch.ones_like(fused) / 2)

    def test_padding_invariance_and_backward(self):
        self.model.eval()
        x = torch.randn(2, 8, 12)
        original = self.model(x, [8, 3])
        x[1, 3:] = 1e6
        altered = self.model(x, [8, 3])
        torch.testing.assert_close(original['fused'], altered['fused'])
        # Appending padding must also preserve outputs for the valid sequence.
        short = self.model(x[1:2, :3], [3])
        torch.testing.assert_close(short['fused'][0], original['fused'][1, :3], atol=1e-6, rtol=1e-5)
        self.model.train()
        output = self.model(x, [8, 3])
        before = self.model.intention.prototypes.clone()
        loss = self.model.loss(output, torch.tensor([[1., 0., 0.], [0., 1., 0.]]), [8, 3])
        loss['total'].backward()
        self.assertTrue(torch.isfinite(loss['total']))
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertFalse(torch.equal(before, self.model.intention.prototypes))

    def test_ema_and_evaluation_no_mutation(self):
        module = self.model.intention
        features = torch.randn(1, 2, module.prototypes.shape[1])
        scores = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]])
        mask = torch.tensor([[True, False]])
        before = module.prototypes.clone()
        module.update_prototypes(features, scores, mask)
        torch.testing.assert_close(module.prototypes[0], .9 * before[0] + .1 * features[0, 0])
        torch.testing.assert_close(module.prototypes[1:], before[1:])
        self.model.eval()
        before = module.prototypes.clone()
        output = self.model(torch.randn(1, 2, 12), [2])
        self.model.loss(output, torch.tensor([[1., 0., 0.]]), [2])
        torch.testing.assert_close(before, module.prototypes)

    def test_contrastive_hard_mining(self):
        features = torch.tensor([[[1., 0.], [0., 1.], [-1., 0.], [.8, .6]]], requires_grad=True)
        labels = torch.tensor([[0, 0, 0, 1]])
        mask = torch.ones(1, 4, dtype=torch.bool)
        actual = cross_intention_loss(features, labels, mask, negatives=1)
        expected = torch.stack([torch.nn.functional.softplus(torch.tensor(1.8)),
                                torch.nn.functional.softplus(torch.tensor(.6)),
                                torch.nn.functional.softplus(torch.tensor(.2))]).mean()
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        empty = cross_intention_loss(features, torch.zeros_like(labels), mask)
        self.assertEqual(empty.item(), 0.)

    def test_short_and_exact_chunk_inference(self):
        self.model.eval()
        for length in (1, 32, 33, 64):
            fused, anomaly = predict_video(self.model, torch.randn(length, 12))
            self.assertEqual(fused.shape, (length, 3))
            self.assertEqual(anomaly.shape, (length,))
            self.assertTrue(np.isfinite(fused).all())
            np.testing.assert_allclose(fused.sum(-1), 1., atol=1e-6)

    def test_local_transformer_is_local_and_graph_is_global(self):
        self.model.eval()
        x = torch.randn(1, 8, 12)
        mask = torch.ones(1, 8, dtype=torch.bool)
        changed = x.clone()
        changed[:, 7] += 10
        local = self.model.encode_local(x, mask)
        altered = self.model.encode_local(changed, mask)
        # Frame zero has no local window containing frame seven.
        torch.testing.assert_close(local[:, 0], altered[:, 0], rtol=0, atol=0)
        global_original = self.model.encode_global(local, mask)
        global_changed = self.model.encode_global(altered, mask)
        self.assertGreater((global_original[:, 0] - global_changed[:, 0]).abs().max().item(), 1e-5)

    def test_proposals_and_map(self):
        scores = np.array([[.95, .05], [.1, .9], [.1, .9], [.95, .05]])
        found = proposals(scores)
        self.assertEqual((found[0]['start'], found[0]['end']), (1, 3))
        self.assertAlmostEqual(found[0]['score'], .85)
        truth = {'video': [dict(class_id=1, start=1, end=3)]}
        self.assertEqual(detection_map({'video': found}, truth, 2)['mAP'], 1.)
        self.assertEqual(detection_map({'video': []}, truth, 2)['mAP'], 0.)
        self.assertEqual(proposals(np.tile([.99, .01], (4, 1))), [])

    def test_invalid_lengths_and_multilabel(self):
        for lengths in ([0], [4], [1.5]):
            with self.assertRaises(ValueError):
                self.model(torch.randn(1, 3, 12), lengths)
        target = encode_labels('B1-B2', 'xd')
        self.assertEqual(target.sum(), 2)
        with self.assertRaises(ValueError):
            encode_labels('A-B1', 'xd')


if __name__ == '__main__':
    unittest.main()
