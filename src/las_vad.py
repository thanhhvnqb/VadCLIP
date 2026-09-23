"""Train and evaluate LAS-VAD on the repository's UCF/XD feature CSVs."""
import argparse
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from las_data import BalancedBatchSampler, FeatureDataset, class_names, text_embeddings
from las_validation import ValidationData, print_metrics, validate
from las_model import LASConfig, LASVAD


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('format') != 'las-vad-v1':
        raise ValueError('Expected a LAS-VAD checkpoint, not a VadCLIP checkpoint')
    state = checkpoint['model']
    config = dict(checkpoint['config'])
    # Old checkpoints used softmax probabilities in ACC. Preserve their behavior
    # on resume; corrected cosine semantics apply to new models/checkpoints.
    config.setdefault('acc_similarity', 'probability')
    model = LASVAD(state['category_features'], state['attribute_features'], LASConfig(**config))
    if 'intention.prototype_updates' not in state:
        state['intention.prototype_updates'] = torch.zeros(len(state['category_features']), dtype=torch.long)
        state['intention.prototype_update_history_complete'] = torch.tensor(False)
    model.load_state_dict(state)
    return model.to(device), checkpoint


def capture_rng():
    state = np.random.get_state()
    return dict(torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
                numpy_rng=(state[0], state[1].tolist(), *state[2:]),
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    torch.set_rng_state(state['torch_rng'])
    random.setstate(state['python_rng'])
    numpy_rng = state['numpy_rng']
    np.random.set_state((numpy_rng[0], np.asarray(numpy_rng[1], dtype=np.uint32), *numpy_rng[2:]))
    if torch.cuda.is_available() and state.get('cuda_rng'):
        torch.cuda.set_rng_state_all(state['cuda_rng'])


def save_checkpoint(checkpoint, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(destination)


def make_validation(args, dataset_name, config, training=False):
    test_list = args.test_list or f'list/{dataset_name}_CLIP_rgbtest.csv'
    dataset = FeatureDataset(test_list, dataset_name, training=False,
                             feature_root=args.feature_root, input_dim=config.input_dim)
    frame_gt = args.frame_gt
    segments, labels = args.gt_segment_path, args.gt_label_path
    if training and not frame_gt:
        frame_gt = 'list/gt_ucf.npy' if dataset_name == 'ucf' else 'list/gt.npy'
    # Use the repository's paired detection annotations by default when using
    # its default test split. Custom splits must supply their own annotations.
    if training and not args.test_list and not args.segment_gt and not segments and not labels:
        suffix = '_ucf' if dataset_name == 'ucf' else ''
        segments, labels = f'list/gt_segment{suffix}.npy', f'list/gt_label{suffix}.npy'
    return ValidationData(dataset, frame_gt, args.segment_gt, segments, labels, args.frames_per_snippet)


def train(args):
    seed_all(args.seed)
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    cadence = 1280 if args.dataset == 'ucf' else 4800
    log_every = args.log_every if args.log_every is not None else cadence
    validate_every = args.validate_every if args.validate_every is not None else cadence
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or log_every < 1 or validate_every < 0:
        raise ValueError('epochs, batch-size, lr and log-every must be positive; validate-every >= 0')
    train_list = args.train_list or f'list/{args.dataset}_CLIP_rgb.csv'
    sampling = args.sampling
    if sampling == 'auto':
        sampling = 'balanced' if args.dataset == 'ucf' else 'shuffle'
    destination = Path(args.checkpoint).resolve()
    best_destination = Path(args.best_checkpoint).resolve() if args.best_checkpoint else destination.with_stem(destination.stem + '_best')
    if destination == best_destination:
        raise ValueError('Latest and best checkpoint paths must differ')
    log_path = Path(args.log_file) if args.log_file else destination.with_suffix('.jsonl')
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def report(record):
        line = json.dumps(record, allow_nan=False)
        if record['event'] == 'step':
            print(f"epoch: {record['epoch']} | step: {record['examples']} | "
                  f"loss1: {record['agnostic']:.6f} | loss2: {record['fine']:.6f} | "
                  f"loss_acc: {record['auxiliary']:.6f} | loss_reg: {record['regularization']:.6f} | "
                  f"loss_cst: {record['contrast']:.6f} | loss: {record['total']:.6f} | "
                  f"ACC components: {record['acc']['components_mean']:.2f} | "
                  f"ACC single: {record['acc']['single_component_fraction']:.1%}", flush=True)
        elif record['event'] == 'validation':
            print(f"epoch: {record['epoch']} | step: {record['examples']} | "
                  f"best {record['best_metric']}: {record['best_score'] * 100:.2f}% | "
                  f"improved: {record['improved']}", flush=True)
        elif record['event'] == 'epoch':
            print(f"epoch: {record['epoch']} complete | loss: {record['total']:.6f} | "
                  f"seconds: {record['seconds']:.2f} | latest: {record['checkpoint']}", flush=True)
        else:
            print(f"Training {args.dataset}: {record['examples']} features | {record['steps_per_epoch']} batches/epoch | "
                  f"batch size: {record['batch_size']} | log every: {log_every} examples | "
                  f"validate every: {validate_every} examples | best metric: {metric_name} | "
                  f"adapter: {record['adapter']} (window={cfg.window}, layers={cfg.layers}, heads={cfg.heads}) | "
                  f"ACC input: {cfg.acc_similarity} | objective: {cfg.objective} | "
                  f"alpha: {cfg.alpha} | aux weight: {cfg.aux_weight} | text init: {cfg.text_init}", flush=True)
            print(f"Text source: {record['text_source']} | "
                  f"configuration: {'restored from checkpoint' if args.resume else 'fresh run'}", flush=True)
        with log_path.open('a') as stream:
            stream.write(line + '\n')

    if args.resume:
        model, previous = load_checkpoint(args.resume, args.device)
        if previous['dataset'] != args.dataset:
            raise ValueError('Resume dataset does not match checkpoint')
        cfg = model.config
    else:
        values = {field.name: getattr(args, field.name) for field in fields(LASConfig)}
        presets = dict(window=8 if args.dataset == 'ucf' else 64, heads=1,
                       layers=2 if args.dataset == 'ucf' else 1) if args.adapter == 'lgt' else dict(window=32, heads=8, layers=1)
        for key, default in presets.items():
            if values[key] is None:
                values[key] = default
        cfg = LASConfig(**values)
        previous = None
    dataset = FeatureDataset(train_list, args.dataset, cfg.max_length,
                             feature_root=args.feature_root, input_dim=cfg.input_dim)
    loader_options = dict(num_workers=args.workers, pin_memory=str(args.device).startswith('cuda'))
    if sampling == 'balanced':
        loader = DataLoader(dataset, batch_sampler=BalancedBatchSampler(dataset, args.batch_size), **loader_options)
    else:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, **loader_options)
    dataset[0]
    annotations = make_validation(args, args.dataset, cfg, training=True) if validate_every else None
    if not args.resume:
        if args.text_features:
            embeddings = torch.load(args.text_features, map_location='cpu', weights_only=True)
            if embeddings.get('class_names') != class_names(args.dataset):
                raise ValueError('Text feature class_names must match the dataset class order')
            category, attribute = embeddings['category'], embeddings['attribute']
        else:
            category, attribute = text_embeddings(args.dataset, args.attributes, args.device, args.clip_model)
        model = LASVAD(category, attribute, cfg).to(args.device)
    if previous:
        text_source = previous.get('run_config', {}).get('text_source', {'kind': 'legacy_checkpoint'})
    else:
        source_path = Path(args.text_features or args.attributes).resolve()
        text_source = dict(kind='embeddings' if args.text_features else 'attributes', path=str(source_path),
                           sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(), clip_model=args.clip_model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    first_epoch, resume_step, global_step = 0, 0, 0
    best_score = None
    metric_name = 'AUC' if args.dataset == 'ucf' else 'AP'
    if previous:
        optimizer.load_state_dict(previous['optimizer'])
        first_epoch = previous['epoch']
        if not previous.get('epoch_complete', True):
            first_epoch -= 1
            resume_step = previous['step_in_epoch']
        global_step = previous.get('global_step', first_epoch * len(loader) + resume_step)
        restore_rng(previous)
        if previous.get('best_metric') == metric_name:
            best_score = previous.get('best_score')
        # Preserve the previous best when continuing into a new run directory.
        if best_score is not None:
            source = Path(previous.get('best_checkpoint', best_destination))
            if not source.is_file():
                raise FileNotFoundError(f'Previous best checkpoint is missing: {source}')
            if source.resolve() != best_destination:
                import shutil
                best_destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, best_destination)
    run_config = dict(train_list=str(dataset.csv_path), batch_size=args.batch_size, sampling=sampling,
                      seed=args.seed, workers=args.workers, device=args.device, adapter=cfg.adapter,
                      acc_similarity=cfg.acc_similarity,
                      objective=cfg.objective, eq11_exact=cfg.eq11_exact,
                      model_config=asdict(cfg),
                      text_source=text_source,
                      optimizer_config={k: v for k, v in optimizer.param_groups[0].items() if k != 'params'},
                      log_every=log_every, validate_every=validate_every, interval_unit='examples',
                      test_list=str(annotations.dataset.csv_path) if annotations else None)
    if resume_step:
        for key in ('train_list', 'batch_size', 'sampling', 'workers'):
            if previous['run_config'][key] != run_config[key]:
                raise ValueError(f'Mid-epoch resume requires the same {key}')
    normal_count = sum(int(label[0]) for label in dataset.labels)
    report(dict(event='start', examples=len(dataset), normal=normal_count,
                anomaly=len(dataset)-normal_count, steps_per_epoch=len(loader),
                first_epoch=first_epoch+1, epochs=args.epochs, best_metric=metric_name,
                best_checkpoint=str(best_destination), **run_config))
    thresholds = dict(video_threshold=args.video_threshold, snippet_threshold=args.snippet_threshold,
                      nms_threshold=args.nms_threshold)
    for epoch in range(first_epoch, args.epochs):
        model.train()
        sums, seen = {}, 0
        epoch_start_rng = capture_rng()
        if resume_step:
            epoch_start_rng = previous['epoch_start_rng']
            restore_rng(epoch_start_rng)
        iterator = iter(loader)
        if resume_step:
            # Rebuild the shuffled sampler position before restoring the saved
            # network RNG (dropout). Feature preprocessing is deterministic.
            for _ in range(resume_step):
                next(iterator)
            restore_rng(previous)
            sums, seen = dict(previous['epoch_sums']), previous['examples_in_epoch']
        next_log = (seen // log_every + 1) * log_every
        next_validation = (seen // validate_every + 1) * validate_every if validate_every else None
        started = time.monotonic()
        for step, (features, labels, lengths, _) in enumerate(iterator, resume_step + 1):
            optimizer.zero_grad(set_to_none=True)
            output = model(features.to(args.device), lengths)
            losses = model.loss(output, labels.to(args.device), lengths, update_prototypes=False)
            if not torch.isfinite(losses['total']):
                raise FloatingPointError('Nonfinite LAS-VAD loss')
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
            optimizer.step()
            model.intention.update_prototypes(output['intention_features'].detach(), output['intention'].detach(), output['mask'])
            for key, value in losses.items():
                sums[key] = sums.get(key, 0.) + value.detach().item() * len(features)
            seen += len(features)
            global_step += 1
            epoch_complete = step == len(loader)
            mean_losses = {key: value / seen for key, value in sums.items()}
            if seen >= next_log or epoch_complete:
                report(dict(event='step', epoch=epoch+1, step=step, global_step=global_step,
                            steps=len(loader), examples=seen, seconds=round(time.monotonic()-started, 2),
                            acc=output['acc_stats'], prototype_updates=model.intention.prototype_updates.tolist(),
                            prototype_update_history_complete=bool(model.intention.prototype_update_history_complete),
                            **mean_losses))
                next_log = (seen // log_every + 1) * log_every
            metrics, improved = None, False
            if annotations and (seen >= next_validation or epoch_complete):
                metrics, _, _ = validate(model, annotations, **thresholds)
                print_metrics(metrics, args.dataset)
                score = metrics[metric_name]
                if not np.isfinite(score):
                    raise FloatingPointError('Nonfinite checkpoint selection metric')
                improved = best_score is None or score > best_score
                if improved:
                    best_score = score
                report(dict(event='validation', epoch=epoch+1, step=step, global_step=global_step,
                            examples=seen, metrics=metrics, best_metric=metric_name, best_score=best_score,
                            improved=improved, best_checkpoint=str(best_destination)))
                next_validation = (seen // validate_every + 1) * validate_every
            if metrics is not None or epoch_complete:
                checkpoint = dict(format='las-vad-v1', config=asdict(cfg), dataset=args.dataset,
                                  class_names=class_names(args.dataset), model=model.state_dict(), optimizer=optimizer.state_dict(),
                                  epoch=epoch+1, epoch_complete=epoch_complete, step_in_epoch=step, global_step=global_step,
                                  examples_in_epoch=seen, epoch_sums=sums, epoch_start_rng=epoch_start_rng,
                                  run_config=run_config, losses=mean_losses, metrics=metrics,
                                  acc_stats=output['acc_stats'],
                                  best_metric=metric_name, best_score=best_score, best_checkpoint=str(best_destination),
                                  **capture_rng())
                if improved:
                    save_checkpoint(checkpoint, best_destination)
                    print(f'Best checkpoint saved: {best_destination} | {metric_name}: {best_score * 100:.2f}%', flush=True)
                save_checkpoint(checkpoint, destination)
        resume_step = 0
        report(dict(event='epoch', epoch=epoch+1, checkpoint=str(destination), best_score=best_score,
                    seconds=round(time.monotonic()-started, 2), **mean_losses))


def evaluate(args):
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    model, checkpoint = load_checkpoint(args.checkpoint, args.device)
    annotations = make_validation(args, checkpoint['dataset'], model.config)
    metrics, grouped, detections = validate(model, annotations, include_heads=args.audit_heads, video_threshold=args.video_threshold,
                                            snippet_threshold=args.snippet_threshold, nms_threshold=args.nms_threshold)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, (video, entry) in enumerate(grouped.items()):
        filename = f'{index:05d}.npz'
        np.savez_compressed(output_dir / filename, **{name: entry[name] for name in
                            ('fused', 'anomaly', 'binary', 'alignment', 'multiclass', 'language', 'intention') if name in entry})
        manifest.append(dict(video_id=video, file=filename, snippets=len(entry['anomaly']), crops=entry['count']))
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    (output_dir / 'proposals.json').write_text(json.dumps(detections, indent=2))
    (output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False))
    print_metrics(metrics, checkpoint['dataset'])
    print(json.dumps(dict(videos=len(grouped), output=str(output_dir), **metrics)))


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest='command', required=True)
    training = sub.add_parser('train')
    training.add_argument('--dataset', choices=['ucf', 'xd'], required=True)
    training.add_argument('--train-list', help='Defaults to list/{dataset}_CLIP_rgb.csv')
    training.add_argument('--attributes', default=str(Path(__file__).resolve().parents[1] / 'configs/las_attributes_paper.json'),
                          help='Dataset-specific official Table 10 descriptions; old flat JSON also supported')
    training.add_argument('--clip-model', default='ViT-B/16', help='CLIP model name or local checkpoint path')
    training.add_argument('--text-features', help='Offline torch file: category, attribute, class_names')
    training.add_argument('--resume')
    training.add_argument('--epochs', type=int, default=10)
    training.add_argument('--batch-size', type=int, default=64)
    training.add_argument('--lr', type=float, default=2e-5)
    training.add_argument('--seed', type=int, default=234)
    training.add_argument('--workers', type=int, default=0)
    training.add_argument('--sampling', choices=['auto', 'balanced', 'shuffle'], default='auto',
                          help='auto: balanced normal/anomaly batches for UCF, shuffle for XD; batch-size is total')
    training.add_argument('--log-every', type=int, help='Processed examples between loss logs; UCF 1280, XD 4800')
    training.add_argument('--validate-every', type=int, help='Processed examples between validations; UCF 1280, XD 4800; 0 disables')
    training.add_argument('--best-checkpoint', help='Highest-score checkpoint; defaults to <checkpoint_stem>_best.pt')
    training.add_argument('--log-file', help='JSONL log path; defaults to checkpoint path with .jsonl suffix')
    defaults = LASConfig()
    for field in fields(LASConfig):
        value = getattr(defaults, field.name)
        if field.name == 'adapter':
            training.add_argument('--adapter', choices=['lgt', 'simple'], default='lgt',
                                  help='New runs use VadCLIP LGT; simple reproduces the original LAS-VAD implementation')
        elif field.name == 'acc_similarity':
            training.add_argument('--acc-similarity', choices=['cosine', 'probability'], default='cosine',
                                  help='Eq.13 uses signed cosine; probability reproduces the previous bug')
        elif field.name == 'objective':
            training.add_argument('--objective', choices=['extended', 'eq9'], default='extended',
                                  help='eq9: printed objective without Lcst; extended: add weighted contrastive loss')
        elif field.name == 'text_init':
            training.add_argument('--text-init', choices=['random', 'mean'], default='random',
                                  help='Explicit ablation: mean initializes concatenation projection as [I/2,I/2]')
        elif field.name in ('heads', 'layers', 'window'):
            training.add_argument('--' + field.name, type=int, default=None, help='Defaults follow dataset/adapter preset')
        elif isinstance(value, bool):
            training.add_argument('--' + field.name.replace('_', '-'), action='store_true', default=value)
        else:
            training.add_argument('--' + field.name.replace('_', '-'), type=type(value), default=value)
    evaluation = sub.add_parser('evaluate')
    evaluation.add_argument('--audit-heads', action='store_true',
                            help='Also save and score individual visual/text/IAM heads; inference fusion is unchanged')
    evaluation.add_argument('--output', default='results/las_vad')
    for command in (training, evaluation):
        command.add_argument('--test-list', help='Defaults to the dataset test CSV')
        command.add_argument('--frame-gt', '--gt-path', dest='frame_gt', help='Flat binary frame GT; training defaults to dataset GT')
        command.add_argument('--segment-gt', help='JSON detection annotations in snippet coordinates')
        command.add_argument('--gt-segment-path', help='VadCLIP gt_segment*.npy, in frame coordinates')
        command.add_argument('--gt-label-path', help='VadCLIP gt_label*.npy, paired with gt-segment-path')
        command.add_argument('--frames-per-snippet', type=int, default=16)
        command.add_argument('--video-threshold', type=float, default=0.1)
        command.add_argument('--snippet-threshold', type=float, default=0.2)
        command.add_argument('--nms-threshold', type=float, default=0.6)
        command.add_argument('--checkpoint', required=True)
        command.add_argument('--feature-root', help='Root prepended to relative CSV feature paths')
        command.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
        command.add_argument('--threads', type=int, default=4, help='PyTorch CPU threads')
    return root


if __name__ == '__main__':
    args = parser().parse_args()
    if args.command == 'evaluate' and args.frames_per_snippet < 1:
        raise ValueError('frames-per-snippet must be positive')
    (train if args.command == 'train' else evaluate)(args)
