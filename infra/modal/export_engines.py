"""Download exactly three verified engine files through one CPU-only Modal App.

python infra/modal/export_engines.py --self-test
python infra/modal/export_engines.py --run
Existing output directories are never overwritten. A failed .partial directory
can be resumed explicitly with --resume-partial, after its identity is checked.
No inference, training, GPU allocation or publication is performed.
"""
import argparse
from datetime import datetime, timezone
import getpass
import hashlib
from importlib import import_module
import json
import os
from pathlib import Path
import re
import socket
import sys
import time

IMAGE_ID = 'im-ugEjBxqJMY0x3jlYNf05pU'
MODEL_VERSION = 'mahjong-cascade-v26-effv2s-v4-20261027'
ENGINE_SHA256 = {
    'classifier.engine': 'febaf0a0f0e6399e118ce8cf0107f3a9b34a5b42ff2f260081c745453864a604',
    'detector.engine': '111c46fe23deafc01a3c4a549173f23b901558598079b6b7a0f48aa90009141c',
}
MODEL_SHA256 = {
    'classifier.pt': 'bc1ce8fa421ae3a81636f2de31da91b376a7c6c13160fdc79afff4bfa5820cfc',
    'detector.pt': '92f4091baa4dea3af611ee8b1e9211b49a58d72fdd8732cddff0e2344cc12afb',
}
FILES = ('classifier.engine', 'detector.engine', 'runtime.json')
DEFAULT_OUT = Path('artifacts/models/releases/mahjong-runtime-trt-a10-fp16-20261029')


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def validate_files(files, expected_engines=None):
    expected_engines = ENGINE_SHA256 if expected_engines is None else expected_engines
    if set(files) != set(FILES) or any(type(files[name]) is not bytes or not files[name] for name in FILES):
        raise ValueError('Only nonempty bytes for the three allowlisted files are accepted')
    receipt = json.loads(files['runtime.json'])
    if (receipt.get('schemaVersion') != 2 or receipt.get('modelVersion') != MODEL_VERSION
            or receipt.get('precision') != 'fp16' or receipt.get('calibrationDataUsed') is not False
            or receipt.get('modelFilesSha256') != MODEL_SHA256
            or receipt.get('hardwareCompatibility') != 'AMPERE_PLUS'
            or receipt.get('engineHardwareCompatibility') != {name: 'AMPERE_PLUS' for name in expected_engines}
            or receipt.get('enginesSha256') != expected_engines):
        raise ValueError('Engine receipt differs from the frozen compatible build')
    for name, expected in expected_engines.items():
        if sha256(files[name]) != expected:
            raise ValueError('Engine SHA256 mismatch: ' + name)
    return {name: {'bytes': len(files[name]), 'sha256': sha256(files[name])} for name in FILES}


def read_engine_files(expected_engines):
    """Remote module function: stdlib only; fixed paths; no GPU requested or used."""
    if expected_engines != ENGINE_SHA256:
        raise ValueError('Export request differs from the allowlisted frozen engine hashes')
    root = Path('/engines')
    files = {}
    for name in FILES:
        path = root / name
        if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= 512 * 1024**2:
            raise ValueError('Unexpected engine file or size: ' + name)
        files[name] = path.read_bytes()
    metadata = validate_files(files)
    return {'files': files, 'metadata': metadata}


def privacy_scan(files, private_needles=()):
    """Pattern audit of all raw bytes; never prints or returns suspected values."""
    patterns = {
        'home_path': rb'/(?:Users|home)/[^\x00\x09\x0a\x0d <>"\']{2,160}',
        'windows_user_path': rb'[A-Za-z]:[\\/]Users[\\/][^\x00\x09\x0a\x0d <>"\']{2,160}',
        'private_ipv4': rb'(?<![0-9.])(?:10\.(?:[0-9]{1,3}\.){2}[0-9]{1,3}|192\.168\.[0-9]{1,3}\.[0-9]{1,3}|172\.(?:1[6-9]|2[0-9]|3[01])\.[0-9]{1,3}\.[0-9]{1,3})(?![0-9.])',
        'credential_marker': rb'(?i)(?:Bearer [A-Za-z0-9._-]{16,}|gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)',
    }
    reports = {}
    for name, data in files.items():
        issues = []
        for kind, pattern in patterns.items():
            for match in re.finditer(pattern, data):
                # This pinned TensorRT version is not a private IPv4 address.
                if kind == 'private_ipv4' and match.group() == b'10.13.3.9':
                    continue
                issues.append({'kind': kind, 'offset': match.start(), 'matchedBytes': len(match.group()),
                               'matchedSha256': sha256(match.group())})
        lowered = data.lower()
        for needle in private_needles:
            value = needle.encode().lower() if isinstance(needle, str) else needle.lower()
            if len(value) < 4:
                continue
            offset = lowered.find(value)
            if offset >= 0:
                issues.append({'kind': 'known_private_identifier', 'offset': offset,
                               'matchedBytes': len(value), 'matchedSha256': sha256(value)})
        reports[name] = {'bytesScanned': len(data), 'sha256': sha256(data), 'issues': issues}
    return {'schemaVersion': 1, 'filesModified': False, 'suspectedValuesPrinted': False,
            'rawBytesScanned': True, 'passed': not any(r['issues'] for r in reports.values()),
            'files': reports,
            'limitation': 'Pattern/known-identifier audit only; not proof that arbitrary binary metadata contains no private information.'}


def register_export(image_id):
    if image_id != IMAGE_ID:
        raise ValueError('This exporter is pinned to the independently verified image')
    import modal
    image = (modal.Image.from_id(image_id)
        .add_local_file(Path(__file__), '/app/infra/modal/export_engines.py', copy=True)
        .env({'CUDA_VISIBLE_DEVICES': '-1'}))
    app = modal.App('mahjong-export-verified-engines')
    remote = app.function(image=image, gpu=None, cpu=1, memory=1024,
        min_containers=0, max_containers=1, scaledown_window=2, timeout=180, retries=0)(
            import_module('infra.modal.export_engines').read_engine_files)
    return app, remote


def transient_network_failure(error):
    message = str(error).lower()
    return any(word in message for word in ('ssl', 'dns', 'connection', 'r2.', 'timed out',
                                             'timeout', 'name resolution', 'network', 'temporary failure'))


def export(out, *, resume_partial=False):
    out = Path(out).resolve()
    partial = out.with_name(out.name + '.partial')
    if out.exists():
        raise FileExistsError('Existing export is immutable; choose a new directory')
    identity = {'sourceImageId': IMAGE_ID, 'modelVersion': MODEL_VERSION, 'expectedEnginesSha256': ENGINE_SHA256,
                'gpuRequested': False, 'trainingOrInferenceExecuted': False}
    state_path = partial / '.export-state.json'
    if partial.exists():
        if not resume_partial or json.loads(state_path.read_text()).get('identity') != identity:
            raise FileExistsError('Existing partial export requires explicit --resume-partial and matching identity')
    else:
        partial.mkdir(parents=True)
    state_path.write_text(json.dumps({'identity': identity, 'state': 'downloading'}, indent=2) + '\n')
    import modal
    try:
        app, remote = register_export(IMAGE_ID)
        with modal.enable_output(), app.run():
            for attempt in range(2):
                try:
                    result = remote.remote(ENGINE_SHA256)
                    break
                except Exception as error:
                    if attempt or not transient_network_failure(error):
                        raise
                    print(json.dumps({'event': 'export_network_retry', 'retry': 1,
                                      'errorType': type(error).__name__, 'gpuRequested': False}), flush=True)
                    time.sleep(2)
        files = result['files']
        metadata = validate_files(files)
        if metadata != result['metadata']:
            raise ValueError('Downloaded bytes differ from the remote export metadata')
        for name in FILES:
            path = partial / name
            if path.exists():
                if path.read_bytes() != files[name]:
                    raise ValueError('Existing partial file differs; preserve it for inspection: ' + name)
            else:
                with path.open('xb') as stream:
                    stream.write(files[name])
        needles = (getpass.getuser(), str(Path.home()), socket.gethostname(), 'avalon-poker')
        privacy = privacy_scan(files, needles)
        scan_bytes = (json.dumps(privacy, indent=2) + '\n').encode()
        receipt = {'schemaVersion': 1, 'files': metadata}
        receipt_bytes = (json.dumps(receipt, indent=2) + '\n').encode()
        extras = {'privacy-scan.json': scan_bytes, 'export-receipt.json': receipt_bytes}
        for name, data in extras.items():
            path = partial / name
            if path.exists():
                if path.read_bytes() != data:
                    raise ValueError('Existing export metadata differs: ' + name)
            else:
                path.write_bytes(data)
        sums = ''.join(f'{sha256(data)}  {name}\n' for name, data in sorted({**files, **extras}.items()))
        sum_path = partial / 'SHA256SUMS'
        if sum_path.exists() and sum_path.read_text() != sums:
            raise ValueError('Existing checksum list differs')
        sum_path.write_text(sums)
        state_path.unlink()
        if out.exists():
            raise FileExistsError('Output appeared while exporting; preserve partial directory')
        partial.rename(out)
        summary = {'exported': True, 'directory': str(out), 'files': metadata,
                   'privacyScanPassed': privacy['passed'], 'gpuRequested': False,
                   'trainingOrInferenceExecuted': False, 'published': False}
        print(json.dumps(summary, indent=2), flush=True)
        return summary
    except Exception as error:
        if partial.exists():
            state_path.write_text(json.dumps({'identity': identity, 'state': 'partial',
                'errorType': type(error).__name__, 'updatedAt': datetime.now(timezone.utc).isoformat()}, indent=2) + '\n')
        raise


def self_test():
    import tempfile
    import types
    from unittest.mock import patch
    sample_engines = {'classifier.engine': b'classifier-fixture', 'detector.engine': b'detector-fixture'}
    expected = {name: sha256(data) for name, data in sample_engines.items()}
    receipt = {'schemaVersion': 2, 'modelVersion': MODEL_VERSION, 'precision': 'fp16',
               'calibrationDataUsed': False, 'modelFilesSha256': MODEL_SHA256,
               'hardwareCompatibility': 'AMPERE_PLUS',
               'engineHardwareCompatibility': {name: 'AMPERE_PLUS' for name in expected}, 'enginesSha256': expected}
    files = dict(sample_engines, **{'runtime.json': json.dumps(receipt).encode()})
    assert validate_files(files, expected)['classifier.engine']['bytes'] == len(sample_engines['classifier.engine'])
    for bad in (dict(files, **{'../escape': b'x'}), dict(files, **{'classifier.engine': b'corrupt'})):
        try:
            validate_files(bad, expected)
        except ValueError:
            pass
        else:
            raise AssertionError('Unsafe/corrupt payload accepted')
    assert privacy_scan(files)['passed']
    assert not privacy_scan({'fixture': b'private /Users/example/private and ghp_' + b'A' * 30})['passed']
    assert privacy_scan({'runtime': b'10.13.3.9'})['passed']
    options = {}
    image = types.SimpleNamespace(add_local_file=lambda *a, **k: image, env=lambda *a, **k: image)
    class App:
        def __init__(self, name): pass
        def function(self, **kwargs):
            options.update(kwargs)
            return lambda function: function
    stub = types.SimpleNamespace(App=App, Image=types.SimpleNamespace(from_id=lambda _: image))
    with patch.dict(sys.modules, {'modal': stub, 'infra.modal.app': None}):
        _, function = register_export(IMAGE_ID)
    assert options['gpu'] is None and options['max_containers'] == 1 and not options.get('serialized', False)
    assert function.__module__ == 'infra.modal.export_engines'
    with tempfile.TemporaryDirectory() as temp:
        try:
            export(Path(temp))
        except FileExistsError:
            pass
        else:
            raise AssertionError('Existing output was not rejected')
    print(json.dumps({'selfTestsPassed': True, 'gpuRequested': False, 'networkUsed': False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--resume-partial', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if not args.run:
        print(json.dumps({'sourceImageId': IMAGE_ID, 'allowlistedFiles': FILES,
                          'out': str(args.out), 'expectedEnginesSha256': ENGINE_SHA256,
                          'gpuRequested': False, 'downloadStarted': False}, indent=2))
        return
    return export(args.out, resume_partial=args.resume_partial)


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    main()
