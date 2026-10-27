"""Prepare a frozen multi-model queue and manual Windows launch commands; never train.

The catalog is ordered for architectural breadth. A budget is a hard ceiling,
not a promise that every listed model converges or that every row fits.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PureWindowsPath

from infra.windows.experiment_queue import config_contract

DEFAULT_CODE = 'D:/mahjong/code-v5-ready-20261026'
TRAIN_PYTHON = 'C:/Users/TRAINING_USER/miniforge3/envs/mahjong-train/python.exe'
YOLO_PYTHON = 'C:/Users/TRAINING_USER/miniforge3/envs/mahjong-yolo/python.exe'


def digest(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def make_suite(root: Path, *, code_root=DEFAULT_CODE, queue_id='v5-breadth-24h-20261026', hours=24,
               allowed_gpu_paths=(), train_python=TRAIN_PYTHON, yolo_python=YOLO_PYTHON, asset_manifest=None, environment_snapshots=()):
    if not 5 <= hours <= 24:
        raise ValueError('Choose a 5–24 hour hard budget')
    code = code_root.rstrip('/\\')
    data = 'D:/mahjong/data/prepared-v4-main'
    extended = 'D:/mahjong/data/prepared-v4-extended'
    prepared = data + '-yolo42'
    generic = data + '-yolo1'
    crops = data + '-crops42'
    qdir = 'D:/mahjong/queues/' + queue_id
    manifest = code + '/docs/data/split-v4.jsonl'
    checks = []
    jobs = []

    def add(name, module, args, *, py=train_python, estimate, timeout, reports=(), dependencies=(), preflight=True, steps_after=()):
        identifier = queue_id + '-' + name
        # queue IDs remain human readable without exceeding the engine's limit.
        if len(identifier) > 96:
            raise ValueError('Queue ID is too long for experiment names')
        run = 'D:/mahjong/runs/' + identifier
        argv = [py, '-m', module] + [str(v).replace('{run}', run) for v in args]
        job = {'id': identifier, 'run_dir': run, 'cwd': code, 'estimated_minutes': estimate,
               'timeout_minutes': timeout, 'steps': [{'name': 'train' if preflight else 'evaluate', 'argv': argv}],
               'report_paths': [str(v).replace('{run}', run) for v in reports]}
        if dependencies:
            job['depends_on'] = list(dependencies)
        for step_name, step_argv, step_timeout in steps_after:
            job['steps'].append({'name': step_name, 'argv': [str(v).replace('{run}', run) for v in step_argv], 'timeout_minutes': step_timeout})
        jobs.append(job)
        if preflight:
            checks.append({'name': name, 'argv': argv + ['--preflight-only'], 'timeout_seconds': 180})
        return identifier, run

    # First reproduce a strong familiar baseline, then separate detection from
    # glyph classification. Generic and crop training do not depend on one another.
    add('yolov8s-960', 'ml.train.yolo_train',
        ['--dataset', data, '--prepared', prepared, '--weights', 'D:/mahjong/models/yolov8s.pt', '--output', '{run}',
         '--resolution', 960, '--epochs', 50, '--batch', 8, '--augmentation', 'photometric'],
        py=yolo_python, estimate=95, timeout=140, reports=['{run}/validation/report.json'],
        steps_after=[('validate', [yolo_python, '-m', 'ml.eval.evaluate_yolo_candidates', '--dataset', data, '--run', '{run}/run.json',
                      '--checkpoint', '{run}/train/weights/best.pt', '--out', '{run}/validation'], 20)])
    det_id, det_run = add('tile-detector-v8-960', 'ml.train.yolo_train',
        ['--dataset', data, '--prepared', generic, '--generic', '--weights', 'D:/mahjong/models/yolov8s.pt', '--output', '{run}',
         '--resolution', 960, '--epochs', 50, '--batch', 8, '--augmentation', 'photometric'],
        py=yolo_python, estimate=80, timeout=130)
    cls_id, cls_run = add('efficientnetv2-s-320', 'ml.train.cascade_train',
        ['--dataset', data, '--crops', crops, '--weights', 'D:/mahjong/models/efficientnet_v2_s-dd5fe13b.pth', '--output', '{run}',
         '--resolution', 320, '--epochs', 35, '--batch', 16, '--gradient-accumulation', 2],
        py=yolo_python, estimate=55, timeout=90)
    add('cascade-v8-end-to-end', 'ml.eval.evaluate_yolo_candidates',
        ['--dataset', data, '--checkpoint', det_run + '/train/weights/best.pt', '--run', det_run + '/run.json',
         '--classifier', cls_run + '/best.pt', '--classifier-run', cls_run + '/run.json', '--out', '{run}/validation'],
        py=yolo_python, estimate=10, timeout=25, reports=['{run}/validation/report.json'],
        dependencies=[det_id, cls_id], preflight=False)
    add('yolo26m-960', 'ml.train.yolo_train',
        ['--dataset', data, '--prepared', prepared, '--weights', 'D:/mahjong/models/yolo26m.pt', '--output', '{run}',
         '--resolution', 960, '--epochs', 50, '--batch', 4, '--augmentation', 'photometric'],
        py=yolo_python, estimate=130, timeout=190, reports=['{run}/validation/report.json'],
        steps_after=[('validate', [yolo_python, '-m', 'ml.eval.evaluate_yolo_candidates', '--dataset', data, '--run', '{run}/run.json',
                      '--checkpoint', '{run}/train/weights/best.pt', '--out', '{run}/validation'], 20)])

    modern_det_id, modern_det_run = add('tile-detector-v26-960', 'ml.train.yolo_train',
        ['--dataset', data, '--prepared', generic, '--generic', '--weights', 'D:/mahjong/models/yolo26s.pt', '--output', '{run}',
         '--resolution', 960, '--epochs', 50, '--batch', 8, '--augmentation', 'photometric'],
        py=yolo_python, estimate=80, timeout=130)
    add('cascade-v26-end-to-end', 'ml.eval.evaluate_yolo_candidates',
        ['--dataset', data, '--checkpoint', modern_det_run + '/train/weights/best.pt', '--run', modern_det_run + '/run.json',
         '--classifier', cls_run + '/best.pt', '--classifier-run', cls_run + '/run.json', '--out', '{run}/validation'],
        py=yolo_python, estimate=10, timeout=25, reports=['{run}/validation/report.json'],
        dependencies=[modern_det_id, cls_id], preflight=False)

    def rf(name, architecture, resolution, batch, accumulation, epochs, augmentation, estimate, timeout, source=data, selection=manifest):
        return add(name, 'ml.train.rfdetr_train',
            ['--dataset', source, '--data-manifest', selection, '--output', '{run}', '--architecture', architecture,
             '--resolution', resolution, '--epochs', epochs, '--batch-size', batch, '--accumulation', accumulation,
             '--workers', 2, '--seed', 42, '--augmentation', augmentation, '--learning-rate', '0.0001',
             '--encoder-learning-rate', '0.00001', '--scheduler', 'cosine', '--warmup-epochs', 2,
             '--eval-interval', 2, '--checkpoint-interval', 5, '--patience', 15],
            estimate=estimate, timeout=timeout, reports=['{run}/validation/report.json'],
            steps_after=[('validate', [train_python, '-m', 'ml.eval.evaluate_candidates', '--dataset', source,
                          '--run-dir', '{run}', '--out', '{run}/validation', '--max-checkpoints', '2'], 30)])

    rf('rf-small-640-control', 'small', 640, 8, 2, 35, 'none', 110, 165)
    rf('rf-small-640-photo', 'small', 640, 8, 2, 35, 'photometric', 115, 170)
    rf('rf-small-960-photo', 'small', 960, 4, 4, 35, 'photometric', 170, 235)
    # RT and RF high-resolution runs precede larger variants to expose input
    # geometry and resolution gains without changing every factor at once.
    for name, resolution, mode, estimate, timeout in [('rt-r18-640-warp', 640, 'warp', 110, 170),
                                                     ('rt-r18-960-warp', 960, 'warp', 165, 225),
                                                     ('rt-r18-960-letterbox', 960, 'letterbox', 165, 225)]:
        add(name, 'ml.train.rtdetr_train',
            ['--dataset', data, '--data-manifest', manifest, '--source-directory', 'D:/mahjong/pretrained/rtdetr-v2-r18vd',
             '--output', '{run}', '--resolution', resolution, '--epochs', 35, '--batch-size', 2, '--accumulation', 8,
             '--workers', 0, '--seed', 42, '--augmentation', 'photometric', '--input-mode', mode,
             '--validation-interval', 2, '--min-epochs', 15, '--patience', 8],
            estimate=estimate, timeout=timeout, reports=['{run}/best/validation-report.json'])
    rf('rf-large-768-photo', 'large', 768, 2, 8, 30, 'photometric', 190, 255)
    rf('rf-medium-960-photo', 'medium', 960, 2, 8, 30, 'photometric', 230, 300)
    rf('rf-small-960-extended', 'small', 960, 4, 4, 30, 'photometric', 250, 315,
       source=extended, selection=code + '/docs/data/split-v4-extended.jsonl')
    # A shorter final candidate can use spare time when an earlier large run no
    # longer fits. This is still a real independent experiment, not a busy loop.
    add('yolo26s-640-short', 'ml.train.yolo_train',
        ['--dataset', data, '--prepared', prepared, '--weights', 'D:/mahjong/models/yolo26s.pt', '--output', '{run}',
         '--resolution', 640, '--epochs', 25, '--batch', 8, '--augmentation', 'photometric'],
        py=yolo_python, estimate=45, timeout=90, reports=['{run}/validation/report.json'],
        steps_after=[('validate', [yolo_python, '-m', 'ml.eval.evaluate_yolo_candidates', '--dataset', data, '--run', '{run}/run.json',
                      '--checkpoint', '{run}/train/weights/best.pt', '--out', '{run}/validation'], 15)])
    config = {'schemaVersion': 1, 'queue_id': queue_id, 'queue_dir': qdir, 'deadline_hours': hours,
              'reserve_minutes': 15, 'poll_seconds': 2, 'telemetry_seconds': 60, 'wait_timeout_minutes': 15,
              'gpu_policy': 'exclusive', 'allowed_gpu_process_paths': list(allowed_gpu_paths),
              'wait_for_status': [],
              'env': {'RF_HOME': 'D:/mahjong/models', 'HF_HUB_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
                      'WANDB_MODE': 'disabled', 'PYTHONUTF8': '1', 'OMP_NUM_THREADS': '8'},
              'experiments': jobs}
    config_contract(config)
    source_files = sorted(set(root.glob('ml/**/*.py')) | set(root.glob('infra/windows/*.py')))
    sources = [{'path': code + '/' + p.relative_to(root).as_posix(), 'sha256': digest(p)} for p in source_files if p.is_file()]
    dataset_files = ['class-map.json', 'final-dataset-v4.json', 'split-v4.jsonl', 'final-dataset-v4-extended.json', 'split-v4-extended.jsonl']
    sources += [{'path': code + '/docs/data/' + n, 'sha256': digest(root / 'docs/data' / n)} for n in dataset_files]
    preparation = [
        {'name': 'yolo42', 'argv': [yolo_python, '-m', 'ml.data.prepare_yolo', '--dataset', data, '--out', prepared]},
        {'name': 'yolo1', 'argv': [yolo_python, '-m', 'ml.data.prepare_yolo', '--dataset', data, '--out', generic, '--generic']},
        {'name': 'crops42', 'argv': [yolo_python, '-m', 'ml.data.prepare_crops', '--dataset', data, '--out', crops]},
    ]
    # Preparing derivatives is a separate explicit CPU action. Launch verifies
    # them and never silently turns a missing dataset into a training attempt.
    checks = [{'name': 'derivative-' + p['name'], 'argv': p['argv'] + ['--preflight-only'], 'timeout_seconds': 180,
               'requireDerivativeReady': True} for p in preparation] + checks
    checks = [{'name': 'pixel-integrity-' + name, 'argv': [train_python, '-m', 'ml.train.rfdetr_train', '--dataset', dataset, '--output', qdir + '/unused-preflight-' + name, '--preflight-only', '--verify-images'], 'timeout_seconds': 600} for name, dataset in [('main', data), ('extended', extended)]] + checks
    suite = {'schemaVersion': 1, 'kind': 'prepared_experiment_suite', 'trainingStarted': False,
             'createdAt': datetime.now(timezone.utc).isoformat(), 'codeRoot': code, 'queueConfig': config,
             'sourceFiles': sources, 'preparationCommands': preparation, 'cpuPreflightCommands': checks,
             'pythonPaths': [train_python, yolo_python],
             'requiredFiles': ['D:/mahjong/models/' + n for n in ['rf-detr-small.pth', 'rf-detr-medium.pth', 'rf-detr-large-2026.pth',
                 'yolov8s.pt', 'yolo26s.pt', 'yolo26m.pt', 'efficientnet_v2_s-dd5fe13b.pth']] +
                 ['D:/mahjong/pretrained/rtdetr-v2-r18vd/' + n for n in ['config.json', 'preprocessor_config.json', 'model.safetensors']],
             'note': 'No training is launched by generation or check-only. Only an explicit launch_prepared_queue --start submits WMI. Estimates are provisional small-probe extrapolations; budget may skip later rows. Test remains sealed.'}
    suite['environmentSnapshots'] = []
    for filename in environment_snapshots:
        file = Path(filename)
        snapshot = json.loads(file.read_text(encoding='utf-8-sig'))
        executable = snapshot['python_executable'].replace('\\', '/')
        suite['environmentSnapshots'].append({'python': executable, 'version': snapshot['python'], 'packages': snapshot['packages'], 'snapshotSha256': digest(file)})
        try:
            relative = file.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
        else:
            suite['sourceFiles'].append({'path': code + '/' + relative, 'sha256': digest(file)})
    suite['assetsVerifiedByPreparation'] = asset_manifest is not None
    if asset_manifest is not None:
        asset_path = Path(asset_manifest)
        assets = json.loads(asset_path.read_text(encoding='utf-8'))['assets']
        lookup = {a['path'].replace('\\', '/').casefold(): a for a in assets}
        required = []
        for name in suite['requiredFiles']:
            item = lookup.get(name.replace('\\', '/').casefold())
            if not item or len(item.get('sha256', '')) != 64:
                raise ValueError('Asset manifest needs an exact SHA256 for ' + name)
            required.append({'path': name, 'sha256': item['sha256']})
        suite['requiredFiles'] = required
        try:
            relative = asset_path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
        else:
            suite['sourceFiles'].append({'path': code + '/' + relative, 'sha256': digest(asset_path)})
    return suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--hours', type=float, default=24)
    parser.add_argument('--queue-id', default='v5-breadth-24h-20261026')
    parser.add_argument('--code-root', default=DEFAULT_CODE)
    parser.add_argument('--gpu-background-paths', type=Path)
    parser.add_argument('--asset-manifest', type=Path, help='Pinned Windows asset paths and SHA256 values; required to make the suite launch-ready')
    parser.add_argument('--environment-snapshot', type=Path, action='append', default=[], help='Fresh capture_environment environment.json for each Python environment')
    args = parser.parse_args()
    allowed = json.loads(args.gpu_background_paths.read_text(encoding='utf-8-sig')) if args.gpu_background_paths else []
    if isinstance(allowed, dict):
        allowed = allowed.get('allowed_gpu_process_paths', allowed.get('paths', []))
    suite = make_suite(args.root, code_root=args.code_root, queue_id=args.queue_id, hours=args.hours, allowed_gpu_paths=allowed, asset_manifest=args.asset_manifest, environment_snapshots=args.environment_snapshot)
    if args.out.exists():
        raise ValueError('Use a fresh immutable preparation directory')
    args.out.mkdir(parents=True)
    write_json(args.out / 'suite.json', suite)
    write_json(args.out / 'queue.json', suite['queueConfig'])
    code = args.code_root.rstrip('/\\')
    suite_on_win = code + '/prepared-suites/' + args.out.name + '/suite.json'
    launcher = code + '/infra/windows/launch_prepared_queue.py'
    command = '"' + TRAIN_PYTHON + '" "' + launcher + '" --suite "' + suite_on_win + '"'
    (args.out / 'check.cmd').write_text('@echo off\r\n' + command + '\r\n', encoding='utf-8')
    (args.out / 'submit.cmd').write_text('@echo off\r\n' + command + ' --start\r\n', encoding='utf-8')
    (args.out / 'watch.cmd').write_text('@echo off\r\n"' + TRAIN_PYTHON + '" "' + code + '/infra/windows/watch_queue.py" --queue "' + suite['queueConfig']['queue_dir'] + '" --follow\r\n', encoding='utf-8')
    print(json.dumps({'state': 'prepared_only', 'trainingStarted': False, 'experiments': len(suite['queueConfig']['experiments']),
                      'hardBudgetHours': args.hours, 'catalogEstimatedHours': round(sum(j['estimated_minutes'] for j in suite['queueConfig']['experiments'])/60, 2),
                      'out': str(args.out)}))


if __name__ == '__main__':
    main()
