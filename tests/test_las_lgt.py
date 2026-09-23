import copy
import math
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from las_lgt import LGTAdapter
from las_model import LASConfig, LASVAD, valid_mask
from model import CLIPVAD


class CPUDistance(nn.Module):
    def forward(self, batch, length):
        t = torch.arange(length, dtype=torch.float32)
        return torch.exp(-(t[:, None]-t[None, :]).abs()/math.e).expand(batch, -1, -1)


class LGTTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)

    def test_matches_original_vadclip_adapter_without_padding(self):
        adapter = LGTAdapter(12, 12, 4, 3, 2).eval()
        # Instantiate only the original visual modules, without loading CLIP.
        reference = CLIPVAD.__new__(CLIPVAD)
        nn.Module.__init__(reference)
        reference.device = 'cpu'
        reference.visual_length = 12
        reference.temporal = copy.deepcopy(adapter.temporal)
        for block in reference.temporal.resblocks:
            block.attn_mask = reference.build_attention_mask(4)
        for name in ('gc1', 'gc2', 'gc3', 'gc4', 'linear', 'gelu'):
            setattr(reference, name, copy.deepcopy(getattr(adapter, name)))
        reference.frame_position_embeddings = copy.deepcopy(adapter.positions)
        reference.disAdj = CPUDistance()
        reference.eval()
        x = torch.randn(2, 12, 12)
        mask = valid_mask(x, [12, 12])
        expected = reference.encode_video(x, None, [12, 12])
        actual = adapter.encode_global(adapter.encode_local(x, mask), mask)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)

    def test_short_padded_batch_and_gradient(self):
        model = LASVAD(torch.randn(3, 8), torch.randn(3, 8),
                       LASConfig(adapter='lgt', input_dim=12, width=12, max_length=12, window=4, heads=3, layers=2))
        x = torch.randn(2, 12, 12)
        lengths = [12, 3]
        model.eval()
        full = model(x, lengths)
        short = model(x[1:2, :3], [3])
        torch.testing.assert_close(full['fused'][1, :3], short['fused'][0], atol=2e-6, rtol=1e-5)
        x[1, 3:] = 1e9
        torch.testing.assert_close(model(x, lengths)['fused'], full['fused'])
        model.train()
        output = model(x, lengths)
        model.loss(output, torch.tensor([[1., 0., 0.], [0., 1., 0.]]), lengths)['total'].backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)


if __name__ == '__main__':
    unittest.main()
