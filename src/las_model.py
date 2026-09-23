"""LAS-VAD (arXiv:2603.00550), with explicit choices documented in docs/LAS_VAD.md."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class LASConfig:
    # Missing field in old checkpoints must retain the original architecture.
    # The CLI selects lgt for new training runs.
    adapter: str = 'simple'
    acc_similarity: str = 'cosine'
    objective: str = 'extended'
    eq11_exact: bool = False
    text_init: str = 'random'
    input_dim: int = 512
    width: int = 512
    max_length: int = 256
    window: int = 32
    heads: int = 8
    layers: int = 1
    dropout: float = 0.1
    alpha: float = 0.7
    beta: float = 0.1
    tau: float = 0.9
    eta: float = 0.5
    reg_weight: float = 0.3
    aux_weight: float = 1.0
    contrast_weight: float = 1.0
    negatives: int = 10
    temperature: float = 0.07
    literal_contrast: bool = False

    def __post_init__(self):
        if self.adapter not in ('simple', 'lgt'):
            raise ValueError('adapter must be simple or lgt')
        if self.acc_similarity not in ('cosine', 'probability'):
            raise ValueError('acc_similarity must be cosine or probability (legacy)')
        if self.objective not in ('eq9', 'extended'):
            raise ValueError('objective must be eq9 or extended')
        if self.text_init not in ('random', 'mean'):
            raise ValueError('text_init must be random or mean')
        if self.eq11_exact and self.literal_contrast:
            raise ValueError('Choose eq11_exact OR legacy literal_contrast, not both')
        if min(self.input_dim, self.width, self.max_length, self.window, self.heads,
               self.layers, self.negatives) < 1 or self.width < 3:
            raise ValueError('Dimensions, lengths, layers and negatives must be positive; width >= 3')
        if self.width % self.heads:
            raise ValueError('width must be divisible by heads')
        if not 0 <= self.alpha <= 1 or not 0 <= self.beta <= 1:
            raise ValueError('alpha and beta must be in [0, 1]')
        if self.temperature <= 0 or not 0 <= self.dropout < 1:
            raise ValueError('temperature must be positive and dropout in [0, 1)')
        if not self.aux_weight >= 0:
            raise ValueError('aux_weight must be nonnegative')


def valid_mask(x, lengths):
    lengths = torch.as_tensor(lengths, device=x.device)
    if lengths.shape != (x.shape[0],) or torch.any(lengths < 1) or torch.any(lengths > x.shape[1]):
        raise ValueError('lengths must contain one nonempty valid length per video')
    if lengths.is_floating_point() and torch.any(lengths != lengths.floor()):
        raise ValueError('lengths must be integers')
    return torch.arange(x.shape[1], device=x.device)[None] < lengths[:, None]


def mil_pool(scores, lengths):
    """Per-class top-K probabilities, K=max(floor(T/16), 1), Eq. 2/4."""
    valid_mask(scores, lengths)
    return torch.stack([s[:int(n)].topk(max(int(n) // 16, 1), dim=0).values.mean(0)
                        for s, n in zip(scores, lengths)])


def acc_affinity(features, language_similarity, eta=0.5):
    """Eqs. 12-13 on one valid video. q_l is SIGNED cosine, not softmax.

    A negative max-min consistency must remain negative so rectification can
    weaken a positive visual edge. Do not clamp or initialize the max at zero.
    """
    if features.ndim != 2 or language_similarity.ndim != 2 or len(features) != len(language_similarity):
        raise ValueError('ACC expects matching [time, dimension] and [time, classes] arrays')
    if language_similarity.shape[1] == 0:
        raise ValueError('ACC requires at least one semantic class')
    unit = F.normalize(features, dim=-1)
    visual = unit @ unit.T
    consistency = torch.full_like(visual, -torch.inf)
    for c in range(language_similarity.shape[1]):
        q = language_similarity[:, c]
        consistency = torch.maximum(consistency, torch.minimum(q[:, None], q[None, :]))
    return visual, visual * (1 + eta * consistency)


def graph_components(adjacency):
    """Exact connected components, including transitive links and isolated nodes."""
    neighbors = [row.nonzero().flatten().tolist() for row in adjacency.detach().cpu()]
    seen, components = set(), []
    for root in range(len(neighbors)):
        if root in seen:
            continue
        seen.add(root)
        stack, component = [root], []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in neighbors[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


@torch.no_grad()
def connected_pseudo_labels(features, language_scores, fused_scores, lengths, tau=0.9, eta=0.5,
                            return_stats=False):
    """Eq. 12/13, undirected DFS, and nearest component prototype soft labels."""
    valid_mask(features, lengths)
    result = torch.zeros_like(fused_scores)
    counts, densities, largest = [], [], []
    for b, length in enumerate(lengths):
        n = int(length)
        x = features[b, :n]
        unit = F.normalize(x, dim=-1)
        _, affinity = acc_affinity(x, language_scores[b, :n], eta)
        adjacency = (affinity > tau).cpu()
        components = graph_components(adjacency)
        counts.append(len(components))
        # Exclude self edges so density means inter-frame connectivity.
        densities.append(float((adjacency.sum()-adjacency.diagonal().sum()) / max(n*(n-1), 1)))
        largest.append(max(map(len, components))/n)
        centers = torch.stack([x[ids].mean(0) for ids in components])
        semantics = torch.stack([fused_scores[b, ids].mean(0) for ids in components])
        nearest = (unit @ F.normalize(centers, dim=-1).T).argmax(-1)
        result[b, :n] = semantics[nearest]
    if return_stats:
        size = len(counts)
        return result, dict(components_mean=sum(counts)/size,
                            single_component_fraction=sum(c == 1 for c in counts)/size,
                            edge_density=sum(densities)/size, largest_component_fraction=sum(largest)/size)
    return result


def equation11_loss(features, categories, mask, negatives=10):
    """Printed Eq. 11: raw dot products, negative-only denominator, 1/T.

    Cosine similarity mines partners; missing partners contribute zero. These
    two cases and the within-video mining scope are unspecified in the paper.
    Unlike conventional InfoNCE, the printed expression is not bounded below.
    """
    if negatives < 1:
        raise ValueError('negatives must be positive')
    video_losses = []
    for features_i, labels, valid in zip(features, categories, mask):
        x, labels = features_i[valid], labels[valid]
        if len(x) == 0:
            raise ValueError('Contrastive loss requires nonempty videos')
        unit = F.normalize(x, dim=-1)
        total = x.sum() * 0
        for start in range(0, len(x), 128):
            end = min(start+128, len(x))
            similarity = unit[start:end] @ unit.T
            same = labels[start:end, None] == labels[None]
            other = ~same
            same[torch.arange(end-start, device=x.device), torch.arange(start, end, device=x.device)] = False
            usable = same.any(-1) & other.any(-1)
            if not usable.any():
                continue
            positive_ids = similarity.masked_fill(~same, torch.inf).argmin(-1)
            negative_similarity, negative_ids = similarity.masked_fill(~other, -torch.inf).topk(min(negatives, len(x)), -1)
            anchor = x[start:end][usable]
            positive = (anchor*x[positive_ids[usable]]).sum(-1)
            negative = (anchor[:, None]*x[negative_ids[usable]]).sum(-1)
            negative = negative.masked_fill(~torch.isfinite(negative_similarity[usable]), -torch.inf)
            total = total + (torch.logsumexp(negative, -1)-positive).sum()
        video_losses.append(total/len(x))
    return torch.stack(video_losses).mean()


def cross_intention_loss(features, categories, mask, negatives=10, literal=False, eq11_exact=False):
    """Hardest same-class positive and top-M different-class negatives per video.

    Chunk anchors to bound memory. Skip anchors lacking either kind of partner.
    literal=True is the legacy normalized negative-only variant, NOT exact Eq. 11.
    eq11_exact=True uses the printed expression and temporal reduction.
    """
    if eq11_exact:
        return equation11_loss(features, categories, mask, negatives)
    total, count = features.sum() * 0, 0
    for x, labels, valid in zip(features, categories, mask):
        x, labels = F.normalize(x[valid], dim=-1), labels[valid]
        n = len(x)
        for start in range(0, n, 128):
            end = min(start + 128, n)
            similarity = x[start:end] @ x.T
            same = labels[start:end, None] == labels[None]
            other = ~same
            same[torch.arange(end-start, device=x.device), torch.arange(start, end, device=x.device)] = False
            usable = same.any(-1) & other.any(-1)
            if not usable.any():
                continue
            positive = similarity.masked_fill(~same, torch.inf).min(-1).values[usable]
            negative = similarity.masked_fill(~other, -torch.inf).topk(min(negatives, n), dim=-1).values[usable]
            denominator = negative if literal else torch.cat((positive[:, None], negative), -1)
            total = total + (torch.logsumexp(denominator, -1) - positive).sum()
            count += int(usable.sum())
    return total / max(count, 1)


class IntentionAwareness(nn.Module):
    def __init__(self, width, classes, alpha, beta):
        super().__init__()
        part = width // 3
        self.position = nn.Linear(width, part)
        self.velocity_gate = nn.Conv1d(part, part, 3, padding=1)
        self.acceleration_gate = nn.Conv1d(part, part, 3, padding=1)
        self.classifier = nn.Sequential(nn.Linear(3 * part, width), nn.ReLU(), nn.Linear(width, classes))
        self.register_buffer('prototypes', F.normalize(torch.randn(classes, 3 * part), dim=-1))
        self.register_buffer('prototype_updates', torch.zeros(classes, dtype=torch.long))
        self.register_buffer('prototype_update_history_complete', torch.tensor(True))
        self.alpha, self.beta = alpha, beta

    def forward(self, x, mask):
        p = self.position(x) * mask[..., None]
        difference = torch.zeros_like(p)
        difference[:, 1:] = (p[:, 1:] - p[:, :-1]).abs()
        difference = difference * mask[..., None]
        v = torch.sigmoid(self.velocity_gate(difference.transpose(1, 2)).transpose(1, 2)) * difference
        acceleration = torch.zeros_like(v)
        acceleration[:, 1:] = (v[:, 1:] - v[:, :-1]).abs()
        acceleration = acceleration * mask[..., None]
        a = torch.sigmoid(self.acceleration_gate(acceleration.transpose(1, 2)).transpose(1, 2)) * acceleration
        intention = torch.cat((p, v, a), -1)
        logits = self.classifier(intention)
        categories = logits.argmax(-1)
        # clone avoids a version-counter conflict when EMA is updated before backward.
        confidence = F.cosine_similarity(intention, self.prototypes.detach().clone()[categories], dim=-1)
        scores = (logits * confidence[..., None]).softmax(-1)
        return intention, scores, categories

    @torch.no_grad()
    def update_prototypes(self, features, scores, mask):
        if not self.training:
            return
        for c in range(len(self.prototypes)):
            selected = mask & (scores[..., c] > self.alpha)
            if selected.any():
                self.prototypes[c].lerp_(features[selected].mean(0), self.beta)
                self.prototype_updates[c] += 1


class LASVAD(nn.Module):
    """Takes pre-extracted visual features and frozen category/attribute CLIP embeddings."""
    def __init__(self, category_features, attribute_features, config=None):
        super().__init__()
        self.config = config or LASConfig()
        cfg = self.config
        category_features = torch.as_tensor(category_features).detach().float()
        attribute_features = torch.as_tensor(attribute_features).detach().float()
        if category_features.ndim != 2 or category_features.shape != attribute_features.shape or len(category_features) < 2:
            raise ValueError('Category and attribute features must have matching [classes>=2, text_dim] shapes')
        self.register_buffer('category_features', category_features)
        self.register_buffer('attribute_features', attribute_features)
        classes, text_dim = category_features.shape
        self.input_projection = nn.Linear(cfg.input_dim, cfg.width) if cfg.input_dim != cfg.width else nn.Identity()
        if cfg.adapter == 'lgt':
            from las_lgt import LGTAdapter
            self.lgt = LGTAdapter(cfg.width, cfg.max_length, cfg.window, cfg.heads, cfg.layers)
        else:
            # Preserve legacy parameter names/initialization for reproducibility.
            self.positions = nn.Embedding(cfg.max_length, cfg.width)
            layer = nn.TransformerEncoderLayer(cfg.width, cfg.heads, cfg.width * 4, cfg.dropout,
                                               activation='gelu', batch_first=True, norm_first=True)
            self.temporal = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
            self.graph = nn.Linear(cfg.width, cfg.width, bias=False)
        self.binary = nn.Linear(cfg.width, 1)
        self.multiclass = nn.Linear(cfg.width, classes)
        self.text_fusion = nn.Linear(text_dim * 2, cfg.width)
        self.intention = IntentionAwareness(cfg.width, classes, cfg.alpha, cfg.beta)
        if cfg.text_init == 'mean':
            if cfg.width != text_dim:
                raise ValueError('mean text initialization requires width == text embedding dimension')
            # An explicit initialization ablation, not an author-specified setting.
            # Preserve the frozen CLIP coordinate system at initialization; the
            # concatenation projection remains fully trainable afterwards.
            with torch.no_grad():
                self.text_fusion.weight.copy_(torch.cat((torch.eye(text_dim), torch.eye(text_dim)), 1) / 2)
                self.text_fusion.bias.zero_()

    def encode_video(self, visual, mask):
        """Local Transformer -> configured global GCN (simple or VadCLIP LGT).

        Keep `temporal` and `graph` parameter names for checkpoint compatibility.
        See docs/LAS_VAD.md for the comparison with the original LGT adapter.
        """
        return self.encode_global(self.encode_local(visual, mask), mask)

    def encode_local(self, visual, mask):
        """Overlapping-window local Transformer with learned temporal positions."""
        if self.config.adapter == 'lgt':
            return self.lgt.encode_local(self.input_projection(visual.float()), mask)
        cfg = self.config
        b, t, _ = visual.shape
        if t > cfg.max_length:
            raise ValueError('Sequence exceeds max_length; split videos into chunks for inference')
        x = self.input_projection(visual.float())
        x = (x + self.positions(torch.arange(t, device=x.device))) * mask[..., None]
        # Overlapping local windows, averaged where they overlap. Process only
        # nonempty windows so padding cannot create all-masked attention rows.
        local, counts = torch.zeros_like(x), torch.zeros_like(x[..., :1])
        for start in range(0, t, max(cfg.window // 2, 1)):
            end = min(start + cfg.window, t)
            active = mask[:, start:end].any(-1)
            if active.any():
                value = self.temporal(x[active, start:end], src_key_padding_mask=~mask[active, start:end])
                local[active, start:end] = local[active, start:end] + value * mask[active, start:end, None]
                counts[active, start:end] += mask[active, start:end, None]
        return local / counts.clamp_min(1)

    def encode_global(self, local, mask):
        """Eq. 1: GELU(softmax(cosine(X, X)) @ X @ W), over valid frames."""
        if self.config.adapter == 'lgt':
            return self.lgt.encode_global(local, mask)
        unit = F.normalize(local, dim=-1)
        adjacency = (unit @ unit.transpose(1, 2)).masked_fill(~mask[:, None, :], -torch.inf).softmax(-1)
        return F.gelu(self.graph(adjacency @ local)) * mask[..., None]

    def forward(self, visual, lengths):
        if visual.ndim != 3 or visual.shape[-1] != self.config.input_dim:
            raise ValueError('visual must have shape [batch, time, input_dim]')
        mask = valid_mask(visual, lengths)
        # Remove padding before projection, including arbitrary values in padded slots.
        features = self.encode_video(visual.masked_fill(~mask[..., None], 0), mask)
        binary = self.binary(features).sigmoid().squeeze(-1)
        multiclass = self.multiclass(features).softmax(-1)
        text = self.text_fusion(torch.cat((self.category_features, self.attribute_features), -1))
        # Keep the exact cross-modal cosine for ACC (Eq. 13) separate from the
        # calibrated class probabilities used by the classification branch.
        language_similarity = F.normalize(features, dim=-1) @ F.normalize(text, dim=-1).T
        language = (language_similarity / self.config.temperature).softmax(-1)
        acc_language = language_similarity if self.config.acc_similarity == 'cosine' else language
        intention_features, intention, categories = self.intention(features, mask)
        fused = (multiclass + language + intention) / 3
        return dict(features=features, binary=binary, multiclass=multiclass, language=language,
                    language_similarity=language_similarity, acc_language=acc_language,
                    intention=intention, intention_features=intention_features, categories=categories,
                    fused=fused, anomaly=(binary + 1 - fused[..., 0]) / 2, mask=mask)

    def loss(self, output, labels, lengths, update_prototypes=True):
        cfg = self.config
        labels = labels.to(output['fused'])
        if labels.shape != (output['fused'].shape[0], output['fused'].shape[-1]):
            raise ValueError('labels must be [batch, classes], normal at index zero')
        if torch.any(labels < 0) or torch.any(labels.sum(-1) <= 0):
            raise ValueError('Each video must have a nonnegative, nonempty label vector')
        if torch.any((labels[:, 0] > 0) & (labels[:, 1:].sum(-1) > 0)):
            raise ValueError('A video cannot be both normal and abnormal')
        targets = labels / labels.sum(-1, keepdim=True)
        agnostic = F.binary_cross_entropy(mil_pool(output['binary'][..., None], lengths).squeeze(-1),
                                         (labels[:, 0] == 0).to(labels))
        fine = -(targets * mil_pool(output['fused'], lengths).clamp_min(1e-8).log()).sum(-1).mean()
        pseudo, output['acc_stats'] = connected_pseudo_labels(
            output['features'], output['acc_language'], output['fused'], lengths, cfg.tau, cfg.eta, return_stats=True)
        mask = output['mask']
        # Average each video's valid time steps, then average the batch.
        def temporal_mean(values):
            return ((values * mask).sum(-1) / mask.sum(-1)).mean()
        auxiliary = temporal_mean((pseudo - output['multiclass']).abs().sum(-1))
        regularization = temporal_mean((1 - output['fused'][..., 0] - output['binary']).abs())
        contrast = cross_intention_loss(output['intention_features'], output['categories'].detach(), mask,
                                       cfg.negatives, cfg.literal_contrast, cfg.eq11_exact)
        total = agnostic + fine + cfg.aux_weight * auxiliary + cfg.reg_weight * regularization
        if cfg.objective == 'extended':
            total = total + cfg.contrast_weight * contrast
        if update_prototypes and self.training:
            self.intention.update_prototypes(output['intention_features'].detach(), output['intention'].detach(), mask)
        return dict(total=total, agnostic=agnostic, fine=fine, auxiliary=auxiliary,
                    regularization=regularization, contrast=contrast)
