"""VadCLIP local/global temporal adapter, with padding and device handling fixed.

Reuses the original Transformer/QuickGELU and residual GraphConvolution classes.
No CLIP weights are downloaded or loaded by this adapter.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from model import Transformer, QuickGELU
from utils.layers import GraphConvolution


class LGTAdapter(nn.Module):
    def __init__(self, width, max_length, window, heads, layers):
        super().__init__()
        if width % 2:
            raise ValueError('VadCLIP LGT requires an even width for its two graph branches')
        self.max_length, self.window = max_length, window
        self.positions = nn.Embedding(max_length, width)
        nn.init.normal_(self.positions.weight, std=.01)
        self.temporal = Transformer(width, layers, heads)
        self.gc1 = GraphConvolution(width, width // 2, residual=True)
        self.gc2 = GraphConvolution(width // 2, width // 2, residual=True)
        self.gc3 = GraphConvolution(width, width // 2, residual=True)
        self.gc4 = GraphConvolution(width // 2, width // 2, residual=True)
        self.linear = nn.Linear(width, width)
        self.gelu = QuickGELU()

    def encode_local(self, x, valid):
        _, length, _ = x.shape
        if length > self.max_length:
            raise ValueError('Sequence exceeds max_length')
        x = (x.float() + self.positions(torch.arange(length, device=x.device))) * valid[..., None]
        pieces = []
        # Blocks are independent, exactly as with the original block-diagonal
        # attention mask. Skip fully padded blocks to avoid all-masked softmax.
        for start in range(0, length, self.window):
            end = min(start + self.window, length)
            block_mask = valid[:, start:end]
            active = block_mask.any(-1)
            block = torch.zeros_like(x[:, start:end])
            if active.any():
                encoded, _ = self.temporal((x[active, start:end].transpose(0, 1), ~block_mask[active]))
                block[active] = encoded.transpose(0, 1) * block_mask[active, :, None]
            pieces.append(block)
        return torch.cat(pieces, dim=1)

    def encode_global(self, x, valid):
        unit = F.normalize(x, dim=-1)
        similarity = F.threshold(unit @ unit.transpose(1, 2), .7, 0)
        similarity = similarity.masked_fill(~valid[:, None, :], -torch.inf).softmax(-1)
        similarity = similarity * valid[..., None]
        positions = torch.arange(x.shape[1], device=x.device, dtype=x.dtype)
        # Original DistanceAdj = exp(-|i-j| / exp(1)); it is NOT row normalized.
        distance = torch.exp(-(positions[:, None] - positions[None, :]).abs() / math.e)
        distance = distance[None] * valid[:, :, None] * valid[:, None, :]
        mask = valid[..., None]
        similar = self.gelu(self.gc1(x, similarity)) * mask
        similar = self.gelu(self.gc2(similar, similarity)) * mask
        temporal = self.gelu(self.gc3(x, distance)) * mask
        temporal = self.gelu(self.gc4(temporal, distance)) * mask
        return self.linear(torch.cat((similar, temporal), dim=-1)) * mask
