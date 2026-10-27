"""Bounded generic-detector fine-tunes from a signed v5 parent; no test access."""
import argparse
import csv
from datetime import datetime,timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time
from unittest.mock import patch
from ml.data.prepare_yolo import verify_prepared
from ml.train.provenance import atomic_json,dataset_contract,sha256
from ml.train.yolo_train import (ULTRALYTICS_VERSION,admit_output,enforce_fixed_batch,
    training_options,shared_signature,checkpoint_receipt)
from ml.train.v6_common import parent_contract,completed_epoch_budget


def options(args):
    config=training_options(args)
    config.update(warmup_epochs=1.,patience=0)
    if args.geometry=='mild':config.update(degrees=8.,translate=.04,scale=.1,perspective=.0004)
    return config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--parent-run',type=Path,required=True);p.add_argument('--parent-checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--resolution',type=int,default=960)
    p.add_argument('--epochs',type=int,default=12);p.add_argument('--batch',type=int,default=8)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--lr',type=float,default=.00005);p.add_argument('--augmentation',choices=('none','photometric'),default='photometric')
    p.add_argument('--geometry',choices=('none','mild'),default='none')
    p.add_argument('--max-train-minutes',type=float,default=45);p.add_argument('--preflight-only',action='store_true')
    args=p.parse_args()
    if args.resolution<320 or args.resolution%32 or min(args.epochs,args.batch)<1 or args.workers<0 or not 0<args.lr<.1 or args.max_train_minutes<=0 or not math.isfinite(args.max_train_minutes):
        raise ValueError('Invalid v6 detector recipe')
    contract=dataset_contract(args.dataset,verify_images=True)
    parent,evidence=parent_contract(args.parent_run,args.parent_checkpoint,contract,generic=True)
    if parent['architecture']not in {'yolov8s-generic','yolo26s-generic'}:raise ValueError('Use the reviewed generic parent')
    ready=(args.prepared/'prepared.json').is_file()
    derivative=verify_prepared(args.prepared,contract,True)if ready else{}
    train_options=options(args);config={k:v for k,v in train_options.items()if k not in {'data','project','name'}}
    config.update(geometry=args.geometry,maximumTrainMinutes=args.max_train_minutes,oomPolicy='fail_without_batch_changes')
    signature=shared_signature(contract,config,args.parent_checkpoint)
    signature.update(parent=evidence,derivativeSha256=sha256(args.prepared/'prepared.json')if ready else None)
    signature['code'].update({'train/yolo_train_v6.py':sha256(__file__),
        'train/v6_common.py':sha256(Path(__file__).with_name('v6_common.py')),
        'data/prepare_background_v6.py':sha256(Path(__file__).resolve().parents[1]/'data/prepare_background_v6.py')})
    if args.preflight_only:
        print(json.dumps({'trainingStarted':False,'derivativeReady':ready,'config':config,'parent':evidence,
                          'backgrounds':len(derivative.get('backgrounds',[])),'dataContract':contract}));return
    if not ready:raise FileNotFoundError('Prepare/review the signed derivative first')
    if (args.output/'run.json').exists():raise FileExistsError('Use an independent fresh run')
    if importlib.metadata.version('ultralytics')!=ULTRALYTICS_VERSION:raise RuntimeError('Use the pinned independent v6 environment')
    import torch
    if not torch.cuda.is_available()or not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA BF16 is required')
    admit_output(args.output)
    os.environ.update(YOLO_OFFLINE='true',YOLO_AUTOINSTALL='false',YOLO_CONFIG_DIR=str(args.output/'yolo-config'))
    from ultralytics import YOLO,settings
    settings.update({'sync':False,'wandb':False,'mlflow':False,'clearml':False,'comet':False,'dvc':False,'raytune':False,'tensorboard':False})
    run={'architecture':parent['architecture'],'generic':True,'resolution':args.resolution,'classes':contract['classes'],
         'ontology':'faces42','resumeSignature':signature,'config':config,'parentWarmStart':evidence,
         'startedAt':datetime.now(timezone.utc).isoformat(),'plannedEpochs':args.epochs,
         'annotationSha256':contract['splits']['train']['annotationSha256'],
         'validationAnnotationSha256':contract['splits']['valid']['annotationSha256'],'testOpened':False}
    atomic_json(args.output/'run.json',run)
    model=YOLO(str(args.parent_checkpoint),task='detect')
    started=time.monotonic();completed=0;last_saved=-1;reason=None
    def saved(trainer):
        nonlocal last_saved
        for checkpoint in (trainer.last,trainer.best):
            if checkpoint.is_file():checkpoint_receipt(checkpoint,signature,{'0':'tile'})
        last_saved=trainer.epoch
    def fit_end(trainer):
        nonlocal completed,reason
        # final_eval invokes this again without saving a new training epoch.
        if trainer.epoch!=last_saved:return
        if not trainer.csv.is_file():raise RuntimeError('Validated epoch has no metrics receipt')
        rows=list(csv.DictReader(trainer.csv.open(encoding='utf-8')))
        with (args.output/'metrics.csv').open('w',newline='',encoding='utf-8')as stream:
            writer=csv.DictWriter(stream,fieldnames=['epoch','train/loss','mAP50','mAP50_95']);writer.writeheader()
            for row in rows:
                row={k.strip():v for k,v in row.items()};losses=[float(v)for k,v in row.items()if k.startswith('train/')and k.endswith('loss')]
                if not losses or not all(__import__('math').isfinite(v)for v in losses):raise RuntimeError('Non-finite detector loss')
                writer.writerow({'epoch':int(row['epoch'])-1,'train/loss':sum(losses),'mAP50':row.get('metrics/mAP50(B)',''),'mAP50_95':row.get('metrics/mAP50-95(B)','')})
        completed=trainer.epoch+1
        reason=completed_epoch_budget(completed,args.epochs,time.monotonic()-started,args.max_train_minutes,trainer.epoch_time or 0)
        atomic_json(args.output/'epoch-completion.json',{'completedEpochs':completed,'checkpointSavedAndValidated':True,'stopReason':reason})
        if reason:trainer.stop=True
    model.add_callback('on_train_epoch_start',lambda trainer:enforce_fixed_batch(trainer,args.batch))
    model.add_callback('on_model_save',saved);model.add_callback('on_fit_epoch_end',fit_end)
    with patch('ultralytics.data.utils.check_font',return_value=None):model.train(**train_options)
    for checkpoint in (args.output/'train/weights/last.pt',args.output/'train/weights/best.pt'):
        if checkpoint.is_file():checkpoint_receipt(checkpoint,signature,{'0':'tile'})
    if not completed:raise RuntimeError('No complete validated detector epoch was saved')
    run.update(finishedAt=datetime.now(timezone.utc).isoformat(),completedEpochs=completed,plannedEpochs=args.epochs,
        completionReason=reason or 'epoch_limit_reached',convergenceClaim=False,elapsedTrainSeconds=time.monotonic()-started,
        actualBatch=model.trainer.batch_size,actualNominalAccumulate=model.trainer.accumulate,
        note='Independent warm-start; soft budget stops only after validation and checkpoint. No convergence claim.')
    atomic_json(args.output/'run.json',run)

if __name__=='__main__':main()
