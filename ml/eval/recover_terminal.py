"""CPU-only validation of signed checkpoints from terminal runs; never rewrite run history."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path

from ml.eval.evaluate_yolo_candidates import ranking
from ml.eval.yolo_evaluate import evaluate
from ml.train.provenance import atomic_json, check_evaluation_data, dataset_contract, sha256
from ml.train.yolo_train import verify_receipt


def checked_run(run_path, contract, terminal_receipt=None):
    run_path = Path(run_path)
    run = json.loads(run_path.read_text(encoding='utf-8'))
    check_evaluation_data(run, contract)
    status = {'trainingCompletion': 'completed'}
    if not run.get('finishedAt'):
        if terminal_receipt is None:
            raise ValueError('Unfinished runs need an explicit terminal receipt, never a forged finishedAt')
        path = Path(terminal_receipt)
        receipt = json.loads(path.read_text(encoding='utf-8'))
        if receipt.get('state') not in {'timed_out', 'interrupted', 'failed'} or receipt.get('child_pid') is not None or not receipt.get('finished_at'):
            raise ValueError('Recovery requires an inactive terminal run')
        if Path(receipt.get('run_dir', '')).resolve() != run_path.parent.resolve():
            raise ValueError('Terminal receipt belongs to another run')
        status = {'trainingCompletion': receipt['state'] + '_checkpoint_recovery',
                  'terminalReceiptSha256': sha256(path), 'terminalReceipt': str(path),
                  'terminalFinishedAt': receipt['finished_at']}
    return run, status


def checked_candidates(run_path, checkpoint, contract, terminal_receipt=None):
    run, status = checked_run(run_path, contract, terminal_receipt)
    results, seen = [], set()
    for path in (Path(checkpoint), Path(checkpoint).with_name('last.pt')):
        if not path.is_file():
            continue
        verify_receipt(path, run)
        digest = sha256(path)
        if digest not in seen:
            seen.add(digest)
            results.append((path, digest))
    if not results:
        raise ValueError('No intact signed checkpoints to recover')
    return run, status, results


def save_best(out, rows, total, complete=False):
    valid = [row for row in rows if row['state'] == 'succeeded']
    if not valid:
        return
    best = max(valid, key=lambda row: tuple(row['ranking']))
    report = json.loads(Path(best['report']).read_text())
    report.update(selectedCheckpoint=best['checkpoint'], selectedClassifier=best['classifier'],
                  candidateReports=rows, checkpointCandidatesTotal=total, checkpointCandidatesEvaluated=len(rows),
                  checkpointSelectionComplete=complete and len(rows)==total, recoveryEvaluationFinished=complete)
    atomic_json(Path(out) / 'report.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--terminal-receipt', type=Path)
    parser.add_argument('--classifier-run', type=Path)
    parser.add_argument('--classifier', type=Path)
    parser.add_argument('--classifier-terminal-receipt', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--max-candidates', type=int, default=4)
    args = parser.parse_args()
    if bool(args.classifier) != bool(args.classifier_run) or not 1 <= args.threads <= 4 or not 1 <= args.max_candidates <= 4:
        raise ValueError('Supply a complete classifier pair and1..4 CPU threads')
    if args.out.exists():
        raise FileExistsError('Use a new immutable recovery output directory')
    # Only the recovery process is masked/prioritized. The existing GPU queue is untouched.
    os.environ.update(CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
                      HF_HUB_OFFLINE='1', YOLO_OFFLINE='true', YOLO_AUTOINSTALL='false',
                      YOLO_CONFIG_DIR=str(args.out / 'yolo-config'))
    if os.name == 'nt':
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    contract = dataset_contract(args.dataset, verify_images=True)
    run, run_status, detectors = checked_candidates(args.run, args.checkpoint, contract, args.terminal_receipt)
    classifier_run, classifier_status, classifiers = None, None, [(None, None)]
    if args.classifier:
        classifier_run, classifier_status, classifiers = checked_candidates(args.classifier_run, args.classifier, contract, args.classifier_terminal_receipt)
        if not run.get('generic') or classifier_run['architecture'] != 'efficientnet-v2-s':
            raise ValueError('Cascade requires a generic detector and the reviewed classifier')
    elif run.get('generic'):
        raise ValueError('A generic detector alone is not tile recognition')
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError('CPU recovery must never see a CUDA device')
    from ml.inference.yolo_model import RawYOLO
    from ml.inference.cascade_model import RawCascade
    args.out.mkdir(parents=True)
    rows = []
    pairs = list(itertools.product(detectors, classifiers))
    for index, ((checkpoint, digest), (classifier, classifier_digest)) in enumerate(pairs[:args.max_candidates]):
        predictor = None
        row = {'checkpoint': str(checkpoint), 'checkpointSha256': digest,
               'classifier': str(classifier) if classifier else None, 'classifierSha256': classifier_digest}
        try:
            metadata = {'checkpointSha256': digest, 'runSha256': sha256(args.run),
                'resumeSignatureSha256': hashlib.sha256(json.dumps(run['resumeSignature'], sort_keys=True).encode()).hexdigest(),
                'trainAnnotationSha256': contract['splits']['train']['annotationSha256'],
                'validationAnnotationSha256': contract['splits']['valid']['annotationSha256'],
                'dataContract': contract, 'architecture': run['architecture'], 'resolution': run['resolution'],
                'evaluationDevice': 'cpu', 'evaluationThreads': args.threads, 'recoveryCodeSha256': sha256(__file__),
                'sourceTrainingStatus': run_status, 'recoveryNotTrainingCompletion': True,
                'gpuDeploymentRevalidationRequired': True, 'predictionFloor': .05, 'nmsIou': .6}
            detector = RawYOLO(checkpoint, run['classes'], run['resolution'], device='cpu', threshold=.05, generic=bool(classifier))
            predictor = detector
            if classifier:
                config = classifier_run['config']
                predictor = RawCascade(detector, classifier, run['classes'], config['resolution'], config['context'], device='cpu')
                metadata.update(architecture='yolo-efficientnet-v2-s-cascade',
                    classifierCheckpointSha256=classifier_digest, classifierRunSha256=sha256(args.classifier_run),
                    classifierResumeSignatureSha256=hashlib.sha256(json.dumps(classifier_run['resumeSignature'], sort_keys=True).encode()).hexdigest(),
                    classifierResolution=config['resolution'], cropContext=config['context'], detectorThreshold=.05,
                    scoreFusion='detector_probability_times_classifier_probability', localizationScope='predicted_detector_boxes',
                    sourceClassifierTrainingStatus=classifier_status)
            destination = args.out / f'candidate-{index:02d}'
            report = evaluate(predictor, args.dataset, destination, metadata)
            if sha256(checkpoint) != digest or (classifier and sha256(classifier) != classifier_digest):
                raise ValueError('Checkpoint changed during recovery evaluation')
            row.update(state='succeeded', report=str(destination / 'report.json'), ranking=ranking(report), inferenceFailedImages=report['inferenceFailedImages'])
        except Exception as error:
            row.update(state='failed', error=f'{type(error).__name__}: {error}')
        finally:
            predictor = None
            if 'detector' in locals():
                del detector
            gc.collect()
        rows.append(row)
        atomic_json(args.out / 'selection.json', {'candidates': rows, 'trainingStarted': False})
        save_best(args.out, rows, len(pairs), complete=False)
        print(json.dumps({'event': 'recovery_candidate_finished', **row}), flush=True)
    winners = [row for row in rows if row['state'] == 'succeeded']
    if not winners:
        raise RuntimeError('No recovery candidate produced valid full-image validation')
    winner = max(winners, key=lambda row: tuple(row['ranking']))
    report = json.loads(Path(winner['report']).read_text())
    report.update(selectedCheckpoint=winner['checkpoint'], selectedClassifier=winner['classifier'],
                  candidateReports=rows, checkpointCandidatesTotal=len(pairs), checkpointCandidatesEvaluated=len(rows),
                  checkpointSelectionComplete=len(rows)==len(pairs), recoveryEvaluationFinished=True,
                  recoveryFinishedAt=datetime.now(timezone.utc).isoformat())
    atomic_json(args.out / 'report.json', report)
    print(json.dumps({'event': 'cpu_recovery_completed', 'out': str(args.out), 'trainingStarted': False}), flush=True)


if __name__ == '__main__': main()
