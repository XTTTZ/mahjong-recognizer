"""Check a prepared queue; submit via WMI only when --start is explicitly present.

Running without --start is a CPU-only integrity/preflight operation. It never
starts training, creates queue receipts, or launches background work.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

# Support both `python -m ...` and the absolute .py invocation in submit.cmd.
if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from infra.windows.experiment_queue import config_contract, pid_alive, read_json, signature


def digest(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def last_json(text):
    # Some preflights print one compact object, others indent a full receipt.
    # Decode only a complete trailing object, never a nested fragment or list.
    decoder = json.JSONDecoder()
    offsets = [0] + [index + 1 for index, value in enumerate(text) if value == "\n"]
    for offset in reversed(offsets):
        candidate = text[offset:].lstrip()
        if not candidate.startswith("{"):
            continue
        try:
            value, end = decoder.raw_decode(candidate)
        except ValueError:
            continue
        if isinstance(value, dict) and not candidate[end:].strip():
            return value
    raise ValueError('CPU preflight did not produce a complete trailing JSON receipt')


def validate_suite(suite, *, run_commands=True):
    if suite.get('schemaVersion') != 1 or suite.get('kind') != 'prepared_experiment_suite' or suite.get('trainingStarted') is not False:
        raise ValueError('Expected a prepared, unstarted suite')
    config = config_contract(suite['queueConfig'])
    if suite.get('assetsVerifiedByPreparation') is not True:
        raise ValueError('Regenerate with --asset-manifest before checking/submitting: pretrained hashes are not frozen')
    for item in suite['sourceFiles']:
        path = Path(item['path'])
        if not path.is_file() or digest(path) != item['sha256']:
            raise ValueError('Frozen source/data metadata differs or is missing: ' + str(path))
    for name in suite['pythonPaths']:
        if not Path(name).is_file():
            raise ValueError('Prepared Python environment is missing: ' + name)
    snapshots = suite.get('environmentSnapshots', [])
    normalize = lambda name: name.replace('\\', '/').casefold()
    if {normalize(s['python']) for s in snapshots} != {normalize(p) for p in suite['pythonPaths']}:
        raise ValueError('Fresh pinned package snapshots are required for both Python environments')
    for snapshot in snapshots:
        program = "import importlib.metadata as m,json,platform;print(json.dumps({'python':platform.python_version(),'packages':{d.metadata['Name'].lower().replace('_','-'):d.version for d in m.distributions() if d.metadata.get('Name')}}))"
        result = subprocess.run([snapshot['python'], '-c', program], check=True, capture_output=True, text=True, encoding='utf-8', timeout=30)
        actual = last_json(result.stdout)
        expected = {n.lower().replace('_', '-'): v for n, v in snapshot['packages'].items()}
        if actual['python'] != snapshot['version'] or actual['packages'] != expected:
            raise ValueError('Pinned Python/package set drifted: ' + snapshot['python'])
        result = subprocess.run([snapshot['python'], '-m', 'pip', 'check'], check=False, capture_output=True, text=True, encoding='utf-8', timeout=60)
        if result.returncode:
            raise ValueError('Dependency check failed: ' + snapshot['python'] + '\n' + result.stdout[-3000:] + result.stderr[-3000:])
    for item in suite['requiredFiles']:
        name = item['path'] if isinstance(item, dict) else item
        path = Path(name)
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError('Required offline pretrained asset is missing: ' + name)
        if isinstance(item, dict) and item.get('sha256') and digest(path) != item['sha256']:
            raise ValueError('Offline pretrained asset checksum differs: ' + name)
    root = Path(config['queue_dir'])
    if (root / 'state.json').is_file():
        previous = read_json(root / 'state.json')
        if previous.get('config_sha256') != signature(config):
            raise ValueError('Queue directory belongs to another frozen configuration')
        if previous.get('state') in {'starting', 'running', 'waiting'} and pid_alive(previous.get('coordinator_pid')):
            raise ValueError('The prepared queue already has a live coordinator')
    # Preflight is useful before resume too, but already-used run directories are
    # accepted only when the queue owns a matching immutable receipt.
    for job in config['experiments']:
        if Path(job['run_dir']).exists():
            receipt = root / 'jobs' / (job['id'] + '.json')
            if not receipt.is_file() or read_json(receipt).get('experiment_sha256') != signature(job):
                raise ValueError('Run directory already exists outside this queue: ' + job['run_dir'])
    receipts = []
    if run_commands:
        parser_receipts = []
        for job in config['experiments']:
            for step in job['steps']:
                argv = step['argv']
                result = subprocess.run([argv[0], suite['codeRoot'].rstrip('/\\') + '/infra/windows/validate_cli.py', '--argv-json', json.dumps(argv)],
                                        cwd=suite['codeRoot'], capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=60, check=False)
                if result.returncode:
                    raise ValueError('Prepared argv failed parsing: ' + job['id'] + '/' + step['name'] + '\n' + result.stderr[-4000:])
                parsed = last_json(result.stdout)
                if parsed.get('state') != 'argv_parsed_only' or parsed.get('trainingStarted') is not False:
                    raise ValueError('Prepared parser check failed to stop at the argument boundary')
                parser_receipts.append({'experiment': job['id'], 'step': step['name'], 'argvSha256': signature(argv), 'module': parsed['module']})
        receipts.append({'name': 'all-entrypoint-argument-boundaries', 'parsers': parser_receipts})
        environment = os.environ.copy()
        environment.update({str(k): str(v) for k, v in config.get('env', {}).items()})
        for command in suite['cpuPreflightCommands']:
            argv = command['argv']
            # This launcher only executes preflights. No arbitrary train argv can
            # be smuggled in via a malformed prepared configuration.
            if '--preflight-only' not in argv or not any(v.startswith(('ml.train.', 'ml.data.')) for v in argv):
                raise ValueError('CPU check must be an explicit ML/data preflight')
            result = subprocess.run(argv, cwd=suite['codeRoot'], env=environment, text=True, encoding='utf-8',
                                    errors='replace', stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    timeout=command.get('timeout_seconds', 180), check=False)
            if result.returncode:
                raise RuntimeError('CPU preflight failed: ' + command['name'] + '\n' + result.stderr[-6000:])
            value = last_json(result.stdout)
            if command.get('requireDerivativeReady') and value.get('derivativeReady') is not True:
                raise ValueError('Prepared derivative is not ready: ' + command['name'])
            receipts.append({'name': command['name'], 'argvSha256': signature(argv), 'receipt': value})
            print(json.dumps({'event': 'cpu_preflight_passed', 'name': command['name']}), flush=True)
    return {'state': 'prepared_checks_passed', 'trainingStarted': False,
            'configSha256': signature(config), 'checkedSourceFiles': len(suite['sourceFiles']),
            'checkedAssets': len(suite['requiredFiles']), 'preflights': receipts}


def submit(suite_path, suite, receipt):
    if os.name != 'nt':
        raise RuntimeError('WMI submission must be run on the prepared Windows host')
    config = suite['queueConfig']
    identifier = config['queue_id'] + '-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    launch = Path('D:/mahjong/queue-launches') / identifier
    launch.mkdir(parents=True, exist_ok=False)
    config_path = launch / 'queue.json'
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
    (launch / 'preflight.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding='utf-8')
    code = suite['codeRoot'].rstrip('/\\')
    python = suite['pythonPaths'][0]
    spec = {'run_dir': str(launch / 'runner'), 'cwd': code, 'env': config.get('env', {}),
            'argv': [python, code + '/infra/windows/experiment_queue.py', '--config', str(config_path)]}
    spec_path = launch / 'runner-spec.json'
    spec_path.write_text(json.dumps(spec, indent=2), encoding='utf-8')
    argv = [python, code + '/infra/windows/job_runner.py', '--spec', str(spec_path)]
    # list2cmdline implements Windows argv quoting. Encode the resulting command
    # as UTF-16 PowerShell data, never concatenate credential or shell fragments.
    command = subprocess.list2cmdline(argv)
    escaped = command.replace("'", "''")
    script = "$ErrorActionPreference='Stop';$r=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='" + escaped + "'}; $r | Select-Object ReturnValue,ProcessId | ConvertTo-Json -Compress"
    encoded = base64.b64encode(script.encode('utf-16le')).decode('ascii')
    result = subprocess.run(['powershell.exe', '-NoProfile', '-EncodedCommand', encoded], capture_output=True,
                            text=True, encoding='utf-8', errors='replace', timeout=30, check=True)
    created = last_json(result.stdout)
    if created.get('ReturnValue') != 0:
        raise RuntimeError('WMI did not create a queue process: ' + json.dumps(created))
    deadline = time.monotonic() + 15
    state = None
    while time.monotonic() < deadline:
        file = Path(config['queue_dir']) / 'state.json'
        if file.is_file():
            try:
                state = read_json(file)
                if state.get('config_sha256') == signature(config) and state.get('coordinator_pid') and state.get('state') != 'blocked':
                    break
            except ValueError:
                pass
        runner_status = launch / 'runner/status.json'
        if runner_status.is_file() and read_json(runner_status).get('state') == 'failed':
            raise RuntimeError('Queue launcher failed; inspect ' + str(launch / 'runner/stderr.log'))
        time.sleep(.25)
    result = {'state': 'submitted', 'trainingStarted': True, 'wmiProcessId': created.get('ProcessId'),
              'queueDirectory': config['queue_dir'], 'queueState': state.get('state') if state else 'starting_unconfirmed',
              'launchDirectory': str(launch), 'note': 'Independent of SSH/chat. Inspect queue state for actual jobs; a submission is not proof of training success.'}
    (launch / 'submission.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--start', action='store_true', help='Explicitly authorize WMI background training submission')
    args = parser.parse_args()
    suite = read_json(args.suite)
    receipt = validate_suite(suite)
    print(json.dumps(submit(args.suite, suite, receipt) if args.start else receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
