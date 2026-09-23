"""Measure temporal collapse and ACC grouping on training features (no GT frames)."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


from las_data import FeatureDataset
from las_model import acc_affinity, graph_components, valid_mask
from las_vad import load_checkpoint


def gradient_diagnostics(model, dataset, samples=8):
    """Compare objective gradients on deterministic training rows, without updates.

    Loss magnitude alone cannot establish which objective drives optimization.
    Report the cosine of base/contrast gradients on shared visual parameters,
    and gradient norms of the two objectives independently.
    """
    if samples < 1:
        raise ValueError('samples must be positive')
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    names, parameters = zip(*[(n, p) for n, p in model.named_parameters() if p.requires_grad])
    visual = [i for i, n in enumerate(names) if n.startswith(
        ('lgt.', 'temporal.', 'graph.', 'positions.', 'input_projection.'))]
    rows = []
    try:
        for index in np.linspace(0, len(dataset)-1, min(samples, len(dataset)), dtype=int):
            x, label, length, video = dataset[index]
            output = model(x[None].to(device), [length])
            losses = model.loss(output, label[None].to(device), [length], update_prototypes=False)
            base = losses['agnostic'] + losses['fine'] + model.config.aux_weight*losses['auxiliary'] + model.config.reg_weight*losses['regularization']
            ga = torch.autograd.grad(base, parameters, retain_graph=True, allow_unused=True)
            gb = torch.autograd.grad(losses['contrast'], parameters, allow_unused=True)
            def norm(grads, indices):
                return float(sum((grads[i].square().sum() for i in indices if grads[i] is not None),
                                 torch.zeros((), device=device)).sqrt())
            na, nb = norm(ga, visual), norm(gb, visual)
            dot = sum((ga[i].mul(gb[i]).sum() for i in visual if ga[i] is not None and gb[i] is not None),
                      torch.zeros((), device=device))
            rows.append(dict(video=video, base_loss=float(base.detach()), contrast_loss=float(losses['contrast'].detach()),
                             visual_base_grad_norm=na, visual_contrast_grad_norm=nb,
                             visual_gradient_cosine=float(dot)/(na*nb) if na*nb else None,
                             base_head_grad_norms={prefix: norm(ga, [i for i, n in enumerate(names) if n.startswith(prefix)])
                                                   for prefix in ('binary.', 'multiclass.', 'text_fusion.', 'intention.classifier.')},
                             intention_classes=output['categories'][0, :length].unique().tolist()))
    finally:
        model.train(was_training)
    return rows


@torch.no_grad()
def diagnose(model, dataset, samples=32):
    if samples < 1:
        raise ValueError('samples must be positive')
    model.eval()
    rows = []
    device = next(model.parameters()).device
    for index in np.linspace(0, len(dataset)-1, min(samples, len(dataset)), dtype=int):
        x, labels, length, video = dataset[index]
        x = x[None].to(device)
        mask = valid_mask(x, [length])
        local = model.encode_local(x, mask)
        output = model(x, [length])
        global_features = output['features'][0, :length]
        cosine, rectified = acc_affinity(global_features, output['language_similarity'][0, :length], model.config.eta)
        _, legacy = acc_affinity(global_features, output['language'][0, :length], model.config.eta)
        graphs = {'visual': cosine > model.config.tau, 'cosine': rectified > model.config.tau,
                  'probability': legacy > model.config.tau}
        graph_stats = {}
        for name, adjacency in graphs.items():
            components = graph_components(adjacency)
            pairs = max(length*(length-1), 1)
            graph_stats[name] = dict(components=len(components), largest_fraction=max(map(len, components))/length,
                                     edge_density=float((adjacency.sum()-adjacency.diagonal().sum())/pairs))
        selected = graph_stats[model.config.acc_similarity]
        def temporal_variation(z):
            return float((z-z.mean(0)).square().mean()/z.square().mean().clamp_min(1e-12))
        rows.append(dict(video=video, abnormal=not bool(labels[0]), graphs=graph_stats, raw_variation=temporal_variation(x[0, :length]),
                         local_variation=temporal_variation(local[0, :length]),
                         global_variation=temporal_variation(global_features), cosine=float(cosine.mean()),
                         components=selected['components'], edge_density=selected['edge_density'],
                         intention_classes=output['categories'][0, :length].unique().cpu().tolist(),
                         intention_confidence=float(output['intention'][0, :length].max(-1).values.mean())))
    summary = {key: float(np.mean([row[key] for row in rows])) for key in
               ('raw_variation', 'local_variation', 'global_variation', 'cosine', 'components', 'edge_density', 'intention_confidence')}
    summary['single_component_videos'] = sum(row['components'] == 1 for row in rows)
    summary['sampled_videos'] = len(rows)
    summary['graph_comparison'] = {name: dict(
        components_mean=float(np.mean([r['graphs'][name]['components'] for r in rows])),
        single_component_videos=sum(r['graphs'][name]['components'] == 1 for r in rows),
        edge_density=float(np.mean([r['graphs'][name]['edge_density'] for r in rows])))
        for name in ('visual', 'cosine', 'probability')}
    summary['abnormal_sampled_videos'] = sum(r['abnormal'] for r in rows)
    return dict(adapter=model.config.adapter, acc_similarity=model.config.acc_similarity, summary=summary,
                prototype_updates=model.intention.prototype_updates.cpu().tolist(),
                prototype_update_history_complete=bool(model.intention.prototype_update_history_complete),
                prototype_norms=model.intention.prototypes.norm(dim=-1).cpu().tolist(), videos=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--train-list')
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--gradient-samples', type=int, default=0,
                        help='Also compare objective gradients on this many training rows (no updates)')
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, checkpoint = load_checkpoint(args.checkpoint, args.device)
    dataset = FeatureDataset(args.train_list or f"list/{checkpoint['dataset']}_CLIP_rgb.csv", checkpoint['dataset'],
                             max_length=model.config.max_length, input_dim=model.config.input_dim)
    report = diagnose(model, dataset, args.samples)
    if args.gradient_samples:
        report['gradient_diagnostics'] = gradient_diagnostics(model, dataset, args.gradient_samples)
    report['checkpoint'] = args.checkpoint
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report['summary']))
