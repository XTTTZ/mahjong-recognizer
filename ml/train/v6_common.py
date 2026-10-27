"""Pure provenance and bounded-epoch policies for new v6 fine-tuning runs."""
import hashlib
import json
import math
from pathlib import Path
from ml.train.provenance import check_evaluation_data, sha256
from ml.train.yolo_train import verify_receipt


def parent_contract(run_file, checkpoint, data, terminal_receipt=None, *, generic=None):
    run_file, checkpoint = Path(run_file), Path(checkpoint)
    run = json.loads(run_file.read_text(encoding='utf-8'))
    check_evaluation_data(run, data)
    verify_receipt(checkpoint, run)
    if generic is not None and bool(run.get('generic')) != generic:
        raise ValueError('Parent task/head differs from the new fine-tuning branch')
    status = {'trainingCompletion': 'completed'}
    if not run.get('finishedAt'):
        if terminal_receipt is None:
            raise ValueError('Unfinished parent requires an explicit terminal receipt')
        path = Path(terminal_receipt)
        receipt = json.loads(path.read_text(encoding='utf-8'))
        if receipt.get('state') not in {'timed_out','interrupted','failed'} or receipt.get('child_pid') is not None or not receipt.get('finished_at'):
            raise ValueError('Parent checkpoint must come from an inactive terminal run')
        if Path(receipt.get('run_dir','')).resolve() != run_file.parent.resolve():
            raise ValueError('Terminal receipt belongs to another parent run')
        status = {'trainingCompletion': 'terminal_checkpoint_warm_start', 'terminalState': receipt['state'],
                  'terminalReceiptSha256': sha256(path)}
    return run, {'runSha256': sha256(run_file), 'checkpointSha256': sha256(checkpoint),
                 'signatureSha256': hashlib.sha256(json.dumps(run['resumeSignature'],sort_keys=True).encode()).hexdigest(),
                 'sourceStatus': status, 'semantics': 'new independent fine-tune; optimizer/schedule reset; never an exact resume of v5'}


def completed_epoch_budget(completed, planned, elapsed_seconds, maximum_minutes, last_epoch_seconds):
    """Call only after validation and an atomic complete checkpoint were saved."""
    if completed < 1 or planned < 1 or not all(math.isfinite(v) for v in (elapsed_seconds,maximum_minutes,last_epoch_seconds)) or maximum_minutes <= 0:
        raise ValueError("Budget is defined only for a complete saved epoch and finite positive time")
    if completed >= planned:
        return 'epoch_limit_reached'
    if elapsed_seconds >= maximum_minutes*60 or elapsed_seconds+last_epoch_seconds*1.15 >= maximum_minutes*60:
        return 'budget_finished'
    return None
