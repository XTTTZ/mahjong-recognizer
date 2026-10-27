"""Validate immutable terminal RF checkpoints on GPU; never resume or rewrite training."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from ml.eval.recover_terminal import checked_run, save_best
from ml.eval.evaluate_yolo_candidates import ranking
from ml.train.provenance import atomic_json, dataset_contract, sha256


def checkpoint_candidates(directory, maximum=2):
    names=('checkpoint_best_total.pth','checkpoint_best_ema.pth','last.ckpt')
    paths=[];seen=set()
    for name in names:
        path=Path(directory)/name
        if not path.is_file():continue
        digest=sha256(path)
        if digest not in seen:
            paths.append((path,digest));seen.add(digest)
        if len(paths)>=maximum:break
    if not paths:raise ValueError('No intact candidate files available')
    return paths


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--max-candidates',type=int,default=2)
    args=parser.parse_args()
    if not 1<=args.max_candidates<=3:raise ValueError('Use1..3 candidates')
    if args.out.exists():raise FileExistsError('Use a fresh recovery output directory')
    run_path=args.run_dir/'run.json';status_path=args.run_dir/'status.json'
    contract=dataset_contract(args.dataset,verify_images=True)
    run,proof=checked_run(run_path,contract,status_path)
    if not run['architecture'].startswith('rfdetr-'):raise ValueError('Expected RF-DETR')
    candidates=checkpoint_candidates(args.run_dir,args.max_candidates)
    run_digest=sha256(run_path);status_digest=sha256(status_path)
    args.out.mkdir(parents=True)
    env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',NO_ALBUMENTATIONS_UPDATE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    rows=[]
    for index,(checkpoint,digest) in enumerate(candidates):
        destination=args.out/f'candidate-{index:02d}'
        command=[sys.executable,'-m','ml.eval.rfdetr_evaluate','--run',str(run_path),'--dataset',str(args.dataset),
                 '--checkpoint',str(checkpoint),'--out',str(destination)]
        row={'checkpoint':str(checkpoint),'checkpointSha256':digest,'classifier':None}
        with (args.out/f'candidate-{index:02d}.stdout.log').open('wb') as stdout,(args.out/f'candidate-{index:02d}.stderr.log').open('wb') as stderr:
            code=subprocess.run(command,env=env,stdout=stdout,stderr=stderr,check=False).returncode
        report_path=destination/'report.json'
        if code==0 and report_path.is_file():
            report=json.loads(report_path.read_text())
            if report.get('partial') or report.get('split')!='validation_only' or report['checkpointSha256']!=digest or sha256(checkpoint)!=digest:
                raise ValueError('Candidate changed or evaluation incomplete')
            if sha256(run_path)!=run_digest or sha256(status_path)!=status_digest:raise ValueError('Training lineage changed')
            report.update(sourceTrainingStatus=proof,recoveryNotTrainingCompletion=True,evaluationDevice='cuda',
                          recoveryCodeSha256=sha256(__file__),productionValidated=False)
            atomic_json(report_path,report)
            row.update(state='succeeded',report=str(report_path),ranking=ranking(report),inferenceFailedImages=report['inferenceFailedImages'])
        else:row.update(state='failed',exitCode=code,error='RF checkpoint load/validation failed; retained candidate log')
        rows.append(row);atomic_json(args.out/'selection.json',{'candidates':rows,'trainingStarted':False})
        save_best(args.out,rows,len(candidates),complete=False)
        print(json.dumps({'event':'rf_recovery_candidate',**row}),flush=True)
    if not any(row['state']=='succeeded' for row in rows):raise RuntimeError('All RF recovery candidates failed validation')
    save_best(args.out,rows,len(candidates),complete=True)
    report=json.loads((args.out/'report.json').read_text());report['recoveryFinishedAt']=datetime.now(timezone.utc).isoformat()
    atomic_json(args.out/'report.json',report)
    print(json.dumps({'event':'rf_recovery_completed','report':str(args.out/'report.json')}),flush=True)


if __name__=='__main__':main()
