"""Freeze a new offline single-stage 42-face training/paired CPU validation queue."""
import argparse
import json
from pathlib import Path

from infra.windows.experiment_queue import config_contract
from infra.windows.offline_bundle import digest
from infra.windows.prepare_refinement_suite import make_suite as base_suite, PYTHON
from infra.windows.prepare_portable_suite_v9 import bundle, write_suite_files

CODE = 'D:/mahjong/code-v11-direct-20261125'
QUEUE = 'v11-direct-latency-12h-20261125'
ROOT = Path(__file__).resolve().parents[2]
DATA = 'D:/mahjong/data/prepared-v4-main'
RECORD = 'docs/models/compact-v11-baseline.json'


def make_suite(root=ROOT, code=CODE, queue=QUEUE, allowed=()):
    root = Path(root)
    suite = base_suite(root, code, queue, allowed, parent_assets=[])
    nano = json.loads((root/'docs/models/nano-assets-v9.json').read_text())['assets'][0]
    suite['requiredFiles'] = [{'path': nano['path'], 'sha256': nano['sha256']}]
    baseline = json.loads((root / RECORD).read_text(encoding='utf-8'))
    for role in ('detector', 'classifier'):
        suite['requiredFiles'].extend(baseline[role][key] for key in ('checkpoint', 'receipt', 'run'))
    extras = [RECORD, 'docs/models/nano-assets-v9.json', 'docs/models/V11_DIRECT_LATENCY.md',
              'infra/portable/candidate_benchmark.py', 'infra/portable/quality_gate.py']
    suite['sourceFiles'] += [{'path': code+'/'+name, 'sha256': digest(root/name)} for name in extras]
    prep = [PYTHON, '-m', 'ml.data.prepare_yolo', '--dataset', DATA, '--out', DATA+'-yolo42']
    jobs, checks = [], []
    for resolution, train_minutes in [(640, 135), (800, 170), (960, 190)]:
        identifier = f'{queue}-faces42-{resolution}'
        run = 'D:/mahjong/runs/'+identifier
        train = [PYTHON, '-m', 'ml.train.yolo_train_v9', '--dataset', DATA, '--prepared', DATA+'-yolo42',
                 '--face-manifest', code+'/docs/data/final-dataset-v4.json', '--weights', nano['path'], '--output', run,
                 '--resolution', str(resolution), '--epochs', '60', '--batch', '8', '--workers', '2', '--seed', '2026',
                 '--lr', '.0003', '--augmentation', 'photometric', '--geometry', 'mild', '--max-train-minutes', str(train_minutes)]
        selection = [PYTHON, '-m', 'ml.eval.recover_terminal', '--dataset', DATA, '--run', run+'/run.json',
                     '--checkpoint', run+'/train/weights/best.pt', '--out', run+'/validation', '--max-candidates', '2',
                     '--threads', '2', '--locked-operating-point']
        speed = [PYTHON, '-m', 'ml.eval.direct_latency', '--dataset', DATA, '--run', run+'/run.json',
                 '--selection', run+'/validation/report.json', '--control-record', code+'/'+RECORD, '--out', run+'/paired-cpu.json']
        checks.append({'name': 'faces42-'+str(resolution), 'argv': train+['--preflight-only'],
                       'timeout_seconds': 900, 'requireDerivativeReady': True})
        jobs.append({'id': identifier, 'run_dir': run, 'cwd': code, 'estimated_minutes': train_minutes+55,
                     'timeout_minutes': train_minutes+60, 'steps': [
                         {'name': 'train', 'argv': train, 'timeout_minutes': train_minutes+5},
                         {'name': 'select-locked-validation', 'argv': selection, 'timeout_minutes': 30, 'env': {'CUDA_VISIBLE_DEVICES': '-1'}},
                         {'name': 'paired-cpu', 'argv': speed, 'timeout_minutes': 25, 'env': {'CUDA_VISIBLE_DEVICES': '-1'}}],
                     'report_paths': [run+'/validation/report.json']})
    config = suite['queueConfig']
    config.update(experiments=jobs, deadline_hours=12, reserve_minutes=20, requires_new_window=True)
    config_contract(config)
    hard = sum(j['timeout_minutes'] for j in jobs)
    if hard+config['reserve_minutes'] > 12*60:
        raise ValueError('Serial hard caps and the reserve exceed the window')
    suite.update(requiresNewWindow=True, newWindowAuthorized=False, trainingStarted=False, submitted=False,
                 preparationStatus='offline_direct_v11_unsubmitted', assetEvidence='Frozen V9 checkpoint hashes and signed metadata',
                 preparationCommands=[{'name': 'faces42', 'argv': prep}], cpuPreflightCommands=checks,
                 budget={'hardHours': 12, 'sumJobHardCapsMinutes': hard, 'reserveMinutes': 20},
                 note='Single-stage 42-face nano runs; open a fresh time-bounded window at launch. No test or automatic deployment.')
    return suite


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--bundle-out', type=Path)
    p.add_argument('--code-root', default=CODE)
    p.add_argument('--queue-id', default=QUEUE)
    p.add_argument('--gpu-background-paths', type=Path)
    a = p.parse_args()
    allowed = json.loads(a.gpu_background_paths.read_text(encoding='utf-8-sig')) if a.gpu_background_paths else []
    if isinstance(allowed, dict):
        allowed = allowed['allowed_gpu_process_paths']
    suite = write_suite_files(make_suite(ROOT, a.code_root, a.queue_id, allowed), a.out)
    result = {'trainingStarted': False, 'submitted': False, 'budget': suite['budget']}
    if a.bundle_out:
        result['bundle'] = bundle(ROOT, a.out, a.bundle_out)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
