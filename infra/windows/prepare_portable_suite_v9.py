"""Freeze a new 12h efficiency queue offline. Never SSH, train, submit or deploy."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import stat
import tempfile
import zipfile
from infra.windows.prepare_efficiency_suite_v7 import make_suite as old_suite, bundle as old_bundle
from infra.windows.prepare_refinement_suite import PYTHON, BASE_PYTHON, PREFIX
from infra.windows.experiment_queue import config_contract
from infra.windows.offline_bundle import MANIFEST, digest, verify, safe_name

CODE = 'D:/mahjong/code-v9-portable-20261106-r2'
QUEUE = 'v9-portable-12h-20261106-r2'
# name, parent recipe, resolution, epochs, alpha, training cap, evaluation cap
RECIPES = (
    ('student-ce224', 'student-ce', 224, 40, 0., 50, 20),
    ('v26s-640', 'v26-background-geometry-960', 640, 30, None, 55, 25),
    ('v8n-800', 'nano', 800, 45, None, 80, 25),
    ('student-kd224-a02', 'student-kd', 224, 40, .2, 90, 20),
    ('v8n-960', 'nano', 960, 45, None, 95, 25),
    ('v26s-800', 'v26-background-geometry-960', 800, 30, None, 70, 25),
    ('student-ce192', 'student-ce', 192, 40, 0., 45, 20),
)


def replace_flag(argv, flag, value):
    argv[argv.index(flag)+1] = str(value)


def make_suite(root, code_root=CODE, queue_id=QUEUE, allowed_gpu_paths=()):
    root = Path(root)
    suite = old_suite(root, code_root, queue_id, allowed_gpu_paths)
    config = suite['queueConfig']
    previous = {j['id'].removeprefix(queue_id+'-'): j for j in config['experiments']}
    data = 'D:/mahjong/data/prepared-v4-main'
    assets = json.loads((root/'docs/models/nano-assets-v9.json').read_text())['assets']
    if len(assets) != 1 or Path(assets[0]['path']).name != 'yolov8n.pt' or len(assets[0]['sha256']) != 64:
        raise ValueError('One frozen official nano asset is required')
    nano = assets[0]
    jobs, checks = [], [c for c in suite['cpuPreflightCommands'] if c['name'] in ('reviewed-train-backgrounds', 'lossless-crop-cache')]
    for name, parent, resolution, epochs, alpha, soft, evaluation_minutes in RECIPES:
        identifier = queue_id+'-'+name
        run = 'D:/mahjong/runs/'+identifier
        if parent == 'nano':
            train = [PYTHON, '-m', 'ml.train.yolo_train_v9', '--dataset', data,
                     '--prepared', 'D:/mahjong/data/v6-background-reviewed', '--weights', nano['path'],
                     '--background-review', code_root+'/docs/data/background-v6-review.json',
                     '--output', run, '--resolution', str(resolution), '--epochs', str(epochs), '--batch', '8',
                     '--workers', '2', '--seed', '2026', '--lr', '.0003', '--augmentation', 'photometric',
                     '--geometry', 'mild', '--max-train-minutes', str(soft)]
            evaluation = [PYTHON, '-m', 'ml.eval.recover_terminal', '--dataset', data, '--run', run+'/run.json',
                          '--checkpoint', run+'/train/weights/best.pt', '--classifier-run', PREFIX+'efficientnetv2-s-320/run.json',
                          '--classifier', PREFIX+'efficientnetv2-s-320/last.pt',
                          '--classifier-terminal-receipt', PREFIX+'efficientnetv2-s-320/status.json',
                          '--out', run+'/validation', '--max-candidates', '2', '--threads', '2']
        else:
            old_id = queue_id+'-'+parent
            job = json.loads(json.dumps(previous[parent]).replace(old_id, identifier))
            train, evaluation = job['steps'][0]['argv'], job['steps'][1]['argv']
            for flag, value in (('--resolution', resolution), ('--epochs', epochs), ('--max-train-minutes', soft)):
                replace_flag(train, flag, value)
            if alpha is not None:
                replace_flag(train, '--distillation-alpha', alpha)
        evaluation += ['--locked-operating-point']
        jobs.append({'id': identifier, 'run_dir': run, 'cwd': code_root,
                     'estimated_minutes': soft+evaluation_minutes, 'timeout_minutes': soft+evaluation_minutes+5,
                     'steps': [{'name': 'train', 'argv': train, 'timeout_minutes': soft+5},
                               {'name': 'validate-locked-complete-hands', 'argv': evaluation,
                                'timeout_minutes': evaluation_minutes, 'env': {'CUDA_VISIBLE_DEVICES': '-1'}}],
                     'report_paths': [run+'/validation/report.json']})
        checks.append({'name': name, 'argv': train+['--preflight-only'], 'timeout_seconds': 900, 'requireDerivativeReady': True})
    config.update(experiments=jobs, deadline_hours=12, reserve_minutes=30)
    config_contract(config)
    # V9 never uses the old v8s parent. Keep only the actual frozen input assets.
    suite['requiredFiles'] = [a for a in suite['requiredFiles'] if 'tile-detector-v8-960' not in a['path']]
    suite['requiredFiles'].append({'path': nano['path'], 'sha256': nano['sha256']})
    extra = list(root.glob('infra/portable/*.py')) + [root/'docs/models/nano-assets-v9.json']
    for p in (root/'docs/models/V9_PORTABLE_12H.md', root/'docs/models/v9-runtime-matrix.json'):
        if p.is_file(): extra.append(p)
    suite['sourceFiles'] += [{'path': code_root+'/'+p.relative_to(root).as_posix(), 'sha256': digest(p)} for p in sorted(extra)]
    suite.update(cpuPreflightCommands=checks, preparationStatus='offline_v9_prepared_windows_unverified',
                 trainingStarted=False, submitted=False, newWindowAuthorized=False, windowsConnected=False,
                 note='Fresh independent V9 runs: V26 signed warm starts; nano COCO initialization; student ImageNet initialization. No training authorized.',
                 assetEvidence='Frozen V5 parent, official ImageNet and official nano hashes; fresh Windows validation required',
                 smallModelStage='Independent localization/student experiments; pair finalists after fixed-point validation',
                 calibrationScope='No quantization in training queue; ONNX precision experiments follow FP32 quality gate',
                 budget={'hardHours': 12, 'reserveMinutes': 30,
                         'trainSoftCapsMinutes': sum(r[5] for r in RECIPES),
                         'evaluationCapsMinutes': sum(r[6] for r in RECIPES),
                         'estimatedMinutes': sum(j['estimated_minutes'] for j in jobs),
                         'sumJobHardCapsMinutes': sum(j['timeout_minutes'] for j in jobs),
                         'estimatesAreProvisional': True,
                         'policy': 'Includes validation. Fail-next serial execution; hard wall and monotonic deadline; no automatic promotion'},
                 priorControlNotRepeated='Frozen V4 detector960/Eff320 and V7 CE/KD224 remain controls; no repeat of regressed FX INT8',
                 qualityGate={'fusedStrictGreaterThan': .35, 'acceptThreshold': .7, 'detectorThreshold': .05,
                              'validationImages': 150, 'primaryImages': 82, 'normalHandImages': 45,
                              'protectBaselineAutomaticExactIds': True, 'protectBaselineModelExactIds': True,
                              'newWrongAcceptsAllowed': 0, 'testUsedForSelection': False},
                 runtimeTarget={'warmPipelineP95Milliseconds': 2000, 'normalHandTiles': [13, 14],
                                'downloadLoadColdStartAndNetworkReportedSeparately': True,
                                'phoneHardwareMeasuredSeparately': True},
                 limitations=['Windows is off: environment/assets/GPU/free disk not freshly verified.',
                              '12h is a hard ceiling; epoch caps do not assert completed epochs or convergence.',
                              'New nano weights require separate transfer; never download inside the offline queue.',
                              'No new independent training data or phone benchmark is claimed.',
                              'Smaller resolution reduces compute, not parameter count; nano/student change model size.',
                              'GPU background allowlist defaults empty; review current full executable paths before start.'],
                 nextStep='Only after new user authorization: transfer fresh bundle/assets, prepare/check on Windows, review GPU, submit.cmd.')
    if suite['budget']['sumJobHardCapsMinutes']+config['reserve_minutes'] > 720:
        raise ValueError('Even nominal job hard caps plus reserve must fit twelve hours')
    return suite


def write_plan(root, out, **options):
    out = Path(out)
    if out.exists(): raise FileExistsError('Choose a new immutable prepared plan')
    suite = make_suite(root, **options)
    out.mkdir(parents=True)
    for name, value in (('suite.json', suite), ('queue.json', suite['queueConfig'])):
        (out/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    code = suite['codeRoot']
    remote = code+'/prepared-suites/'+out.name+'/suite.json'
    launch = f'"{PYTHON}" "{code}/infra/windows/launch_prepared_queue.py" --suite "{remote}"'
    scripts = {
        'prepare.cmd': f'set CUDA_VISIBLE_DEVICES=-1\r\n"{BASE_PYTHON}" "{code}/infra/windows/prepare_v6_host.py" --suite "{remote}"',
        'check.cmd': 'set CUDA_VISIBLE_DEVICES=-1\r\n'+launch,
        'submit.cmd': launch+' --start',
        'watch.cmd': f'"{PYTHON}" "{code}/infra/windows/watch_queue.py" --queue "{suite["queueConfig"]["queue_dir"]}" --follow',
    }
    for name, command in scripts.items():
        (out/name).write_bytes(('@echo off\r\nsetlocal\r\n'+command+'\r\nexit /b %errorlevel%\r\n').encode())
    return suite


def bundle(root, plan, out):
    root, plan, out = map(Path, (root, plan, out))
    if out.exists(): raise FileExistsError('Choose a new immutable bundle')
    with tempfile.TemporaryDirectory() as temporary:
        previous = Path(temporary)/'base.zip'
        old_bundle(root, plan, previous)
        with zipfile.ZipFile(previous) as source:
            contents = {n: source.read(n) for n in source.namelist()}
    manifest = json.loads(contents.pop(MANIFEST))
    # Include exactly every extra source bound by the suite, including portable tools.
    suite = json.loads((plan/'suite.json').read_text())
    prefix = suite['codeRoot']+'/'
    for item in suite['sourceFiles']:
        name = safe_name(item['path'].removeprefix(prefix))
        data = (root/name).read_bytes()
        if hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError('Source changed after plan freeze: '+name)
        contents[name] = data
    manifest.update(preparationStatus=suite['preparationStatus'], training_started=False)
    manifest['files'] = {n: {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)} for n, data in sorted(contents.items())}
    contents[MANIFEST] = (json.dumps(manifest, indent=2)+'\n').encode()
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, 'x', compression=zipfile.ZIP_DEFLATED) as target:
        for name, data in sorted(contents.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            target.writestr(info, data)
    checksum = digest(out)
    verify(out, checksum)
    verifier = out.with_suffix('.verify.py')
    verifier.write_bytes((root/'infra/windows/offline_bundle.py').read_bytes())
    receipt = {'bundle': str(out), 'sha256': checksum, 'bytes': out.stat().st_size,
               'payloadFiles': len(manifest['files']), 'sourceFiles': len(suite['sourceFiles']),
               'verifier': str(verifier), 'verifierSha256': digest(verifier), 'localBundleVerificationPassed': True,
               'windowsConnected': False, 'windowsChecked': False, 'trainingStarted': False, 'submitted': False,
               'modelWeightsIncluded': False, 'budget': suite['budget']}
    out.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    out.with_suffix(out.suffix+'.sha256').write_text(checksum+'\n')
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--bundle-out', type=Path)
    p.add_argument('--code-root', default=CODE)
    p.add_argument('--queue-id', default=QUEUE)
    p.add_argument('--gpu-background-paths', type=Path)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    allowed = json.loads(a.gpu_background_paths.read_text()) if a.gpu_background_paths else []
    if isinstance(allowed, dict): allowed = allowed['allowed_gpu_process_paths']
    suite = write_plan(root, a.out, code_root=a.code_root, queue_id=a.queue_id, allowed_gpu_paths=allowed)
    result = {'trainingStarted': False, 'windowsConnected': False, 'jobs': len(RECIPES), 'budget': suite['budget']}
    if a.bundle_out: result['bundle'] = bundle(root, a.out, a.bundle_out)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
