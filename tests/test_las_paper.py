"""Independent numerical checks of explicitly specified LAS-VAD equations."""
import copy
from dataclasses import asdict
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from las_model import (LASConfig, LASVAD, IntentionAwareness, acc_affinity,
                       connected_pseudo_labels, equation11_loss, graph_components)
from las_vad import load_checkpoint
from las_diagnose import gradient_diagnostics


class PaperEquationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        torch.set_num_threads(1)

    def make_model(self):
        return LASVAD(torch.randn(3, 6), torch.randn(3, 6),
                      LASConfig(input_dim=6, width=6, heads=2, max_length=8, window=4, dropout=0.))

    def test_equations_12_13_against_scalar_reference(self):
        x = torch.randn(5, 7, dtype=torch.float64)
        q = torch.rand(5, 3, dtype=torch.float64)*2-1
        visual, actual = acc_affinity(x, q, eta=.5)
        expected = torch.zeros_like(actual)
        for i in range(len(x)):
            for j in range(len(x)):
                cosine = torch.dot(x[i], x[j])/(x[i].norm()*x[j].norm())
                consistency = max(min(q[i, c], q[j, c]) for c in range(q.shape[1]))
                expected[i, j] = cosine*(1+.5*consistency)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(visual, visual.T)
        torch.testing.assert_close(actual, actual.T)

    def test_mean_text_initialization_and_checkpoint_roundtrip(self):
        category, attribute = torch.randn(3, 6), torch.randn(3, 6)
        cfg = LASConfig(input_dim=6, width=6, heads=2, max_length=8, text_init='mean')
        model = LASVAD(category, attribute, cfg).eval()
        text = model.text_fusion(torch.cat((category, attribute), -1))
        torch.testing.assert_close(text, (category+attribute)/2)
        text.square().sum().backward()
        self.assertGreater(model.text_fusion.weight.grad.norm().item(), 0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'model.pt'
            torch.save(dict(format='las-vad-v1', config=asdict(cfg), model=model.state_dict()), path)
            loaded, _ = load_checkpoint(path, 'cpu')
            loaded.eval()
            x = torch.randn(1, 3, 6)
            torch.testing.assert_close(loaded(x, [3])['fused'], model(x, [3])['fused'])
        with self.assertRaisesRegex(ValueError, 'width == text'):
            LASVAD(category, attribute, LASConfig(width=12, heads=2, text_init='mean'))

    def test_gradient_diagnostic_does_not_update_training_state(self):
        model = self.make_model().train()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        data = [(torch.randn(8, 6), torch.tensor([0., 1., 0.]), 5, 'video')]
        rng = torch.get_rng_state().clone()
        report = gradient_diagnostics(model, data, samples=1)
        self.assertTrue(model.training)
        self.assertGreater(report[0]['visual_base_grad_norm'], 0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_negative_semantics_remove_edges_and_softmax_is_not_equivalent(self):
        x = torch.tensor([[1., 0.], [1., 0.]])
        q = torch.tensor([[-.8, -.5], [-.8, -.5]])
        _, raw = acc_affinity(x, q)
        _, probability = acc_affinity(x, q.softmax(-1))
        torch.testing.assert_close(raw, torch.full((2, 2), .75))
        self.assertEqual(len(graph_components(raw > .9)), 2)
        self.assertEqual(len(graph_components(probability > .9)), 1)

    def test_acc_cosine_is_not_affected_by_temperature(self):
        model = self.make_model().eval()
        changed = copy.deepcopy(model)
        changed.config.temperature = .9
        x = torch.randn(1, 7, 6)
        a, b = model(x, [7]), changed(x, [7])
        torch.testing.assert_close(a['acc_language'], a['language_similarity'])
        torch.testing.assert_close(a['acc_language'], b['acc_language'])
        self.assertFalse(torch.allclose(a['language'], b['language']))
        _, edges_a = acc_affinity(a['features'][0], a['acc_language'][0])
        _, edges_b = acc_affinity(b['features'][0], b['acc_language'][0])
        torch.testing.assert_close(edges_a, edges_b)

    def test_connected_graph_is_not_artificially_split(self):
        # The endpoints are not neighbors, but the middle node bridges them.
        graph = torch.tensor([[False, True, False], [True, False, True], [False, True, False]])
        self.assertEqual(sorted(graph_components(graph)[0]), [0, 1, 2])
        features = torch.ones(1, 3, 2)
        scores = torch.tensor([[[.1, .9], [.5, .5], [.9, .1]]])
        labels, stats = connected_pseudo_labels(features, torch.ones(1, 3, 2), scores, [3], return_stats=True)
        self.assertEqual(stats['components_mean'], 1.)
        torch.testing.assert_close(labels, torch.full_like(labels, .5))
        self.assertFalse(labels.requires_grad)

    def test_acc_never_merges_videos_or_padded_frames(self):
        # Identical features across videos must not share semantic prototypes.
        features = torch.ones(2, 3, 2)
        language = torch.ones(2, 3, 2)
        scores = torch.tensor([[[1., 0.], [1., 0.], [0., 1.]],
                               [[0., 1.], [0., 1.], [0., 1.]]])
        pseudo, stats = connected_pseudo_labels(features, language, scores, [2, 3], return_stats=True)
        torch.testing.assert_close(pseudo[0, :2], torch.tensor([[1., 0.], [1., 0.]]))
        torch.testing.assert_close(pseudo[0, 2], torch.zeros(2))
        torch.testing.assert_close(pseudo[1], scores[1])
        self.assertEqual(stats['components_mean'], 1.)
        self.assertEqual(stats['edge_density'], 1.)

    def test_equation10_and_ema_ignore_padding(self):
        module = IntentionAwareness(6, 2, alpha=.7, beta=.1)
        with torch.no_grad():
            module.position.weight.zero_()
            module.position.weight[:, :2] = torch.eye(2)
            module.position.bias.zero_()
            for gate in (module.velocity_gate, module.acceleration_gate):
                gate.weight.zero_()
                gate.bias.zero_()
        x = torch.tensor([[[1., 2., 0., 0., 0., 0.], [4., 6., 0., 0., 0., 0.],
                           [8., 12., 0., 0., 0., 0.], [100., 100., 0., 0., 0., 0.]]])
        mask = torch.tensor([[True, True, True, False]])
        features, _, _ = module(x, mask)
        expected = torch.tensor([[[1., 2., 0., 0., 0., 0.], [4., 6., 1.5, 2., .75, 1.],
                                  [8., 12., 2., 3., .25, .5], [0., 0., 0., 0., 0., 0.]]])
        torch.testing.assert_close(features, expected)
        before = module.prototypes.clone()
        scores = torch.tensor([[[.1, .9], [.1, .9], [.5, .5], [.01, .99]]])
        module.update_prototypes(features, scores, mask)
        torch.testing.assert_close(module.prototypes[1], .9*before[1]+.1*features[0, :2].mean(0))
        torch.testing.assert_close(module.prototypes[0], before[0])
        self.assertEqual(module.prototype_updates.tolist(), [0, 1])

    def test_equation11_uses_raw_dot_and_full_temporal_denominator(self):
        x = torch.tensor([[[2., 0.], [0., 3.], [-2., 0.]]], requires_grad=True)
        categories = torch.tensor([[0, 0, 1]])
        mask = torch.ones(1, 3, dtype=torch.bool)
        # Anchor 0: log(exp(-4))-0 = -4. Anchor 1: 0. Anchor 2:
        # no positive, documented zero contribution. Divide by all 3 frames.
        actual = equation11_loss(x, categories, mask, negatives=1)
        torch.testing.assert_close(actual, torch.tensor(-4/3))
        actual.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertEqual(equation11_loss(x, torch.zeros_like(categories), mask).item(), 0.)
        torch.testing.assert_close(equation11_loss(2*x, categories, mask, negatives=1), 4*actual)

    def test_equation9_excludes_contrast_even_if_weight_is_nonzero(self):
        model = self.make_model().eval()
        model.config.objective = 'eq9'
        model.config.eq11_exact = True
        model.config.contrast_weight = 123.
        output = model(torch.randn(1, 3, 6), [3])
        output['categories'] = torch.tensor([[0, 0, 1]])
        losses = model.loss(output, torch.tensor([[0., 1., 0.]]), [3])
        expected = losses['agnostic']+losses['fine']+losses['auxiliary']+.3*losses['regularization']
        torch.testing.assert_close(losses['total'], expected)

    def test_acc_ablation_only_removes_auxiliary_objective(self):
        model = self.make_model().eval()
        output = model(torch.randn(1, 3, 6), [3])
        labels = torch.tensor([[0., 1., 0.]])
        original = model.loss(output, labels, [3])
        model.config.aux_weight = 0.
        ablated = model.loss(output, labels, [3])
        torch.testing.assert_close(original['total']-ablated['total'], original['auxiliary'])
        for key in ('agnostic', 'fine', 'contrast', 'regularization', 'auxiliary'):
            torch.testing.assert_close(original[key], ablated[key])

    def test_legacy_checkpoint_preserves_probability_acc_and_unknown_counters(self):
        model = self.make_model().eval()
        model.config.acc_similarity = 'probability'
        state = model.state_dict()
        del state['intention.prototype_updates']
        del state['intention.prototype_update_history_complete']
        config = asdict(model.config)
        for name in ('acc_similarity', 'objective', 'eq11_exact'):
            del config[name]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'old.pt'
            torch.save(dict(format='las-vad-v1', config=config, model=state), path)
            loaded, _ = load_checkpoint(path, 'cpu')
            loaded.eval()
            self.assertEqual(loaded.config.acc_similarity, 'probability')
            self.assertFalse(loaded.intention.prototype_update_history_complete)
            x = torch.randn(1, 3, 6)
            torch.testing.assert_close(loaded(x, [3])['fused'], model(x, [3])['fused'])


if __name__ == '__main__':
    unittest.main()
