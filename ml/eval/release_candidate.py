"""GPU revalidation of a validation-selected pair. Never opens test or trains."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from ml.eval.recover_terminal import checked_run
from ml.eval.yolo_evaluate import evaluate
from ml.train.provenance import dataset_contract, sha256, atomic_json
from ml.train.yolo_train import verify_receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection-report',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--classifier-run',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    selection=json.loads(args.selection_report.read_text())
    if selection.get('partial') or selection.get('split')!='validation_only' or not selection.get('checkpointSelectionComplete'):
        raise ValueError('GPU release checks require a completed full validation candidate selection')
    contract=dataset_contract(args.dataset,verify_images=True)
    run,status=checked_run(args.run,contract,args.run.with_name('status.json'))
    checkpoint=Path(selection['selectedCheckpoint']);verify_receipt(checkpoint,run)
    if sha256(checkpoint)!=selection['checkpointSha256'] or sha256(args.run)!=selection['runSha256']:
        raise ValueError('Selected detector lineage changed')
    os.environ.update(YOLO_OFFLINE='true',YOLO_AUTOINSTALL='false',YOLO_CONFIG_DIR=str(args.out.parent/(args.out.name+'-settings')))
    import torch
    if not torch.cuda.is_available():raise RuntimeError('GPU validation needs CUDA')
    torch.set_num_threads(2)
    from ml.inference.yolo_model import RawYOLO
    from ml.inference.cascade_model import RawCascade
    classifier=Path(selection['selectedClassifier']) if selection.get('selectedClassifier') else None
    detector=RawYOLO(checkpoint,run['classes'],run['resolution'],device='cuda',threshold=.05,generic=bool(classifier))
    predictor=detector
    metadata=dict(selection)
    for key in ['candidateReports','checkpointCandidatesTotal','checkpointCandidatesEvaluated','checkpointSelectionComplete','thresholds','cocoAP']:
        metadata.pop(key,None)
    metadata.update(evaluationDevice='cuda',gpuName=torch.cuda.get_device_name(0),
        cpuReferenceReportSha256=sha256(args.selection_report),sourceTrainingStatus=status,
        gpuDeploymentRevalidationRequired=False,releaseValidationCodeSha256=sha256(__file__))
    if classifier:
        if not args.classifier_run:raise ValueError('Cascade needs classifier run')
        classifier_run,classifier_status=checked_run(args.classifier_run,contract,args.classifier_run.with_name('status.json'))
        verify_receipt(classifier,classifier_run)
        if sha256(classifier)!=selection['classifierCheckpointSha256'] or sha256(args.classifier_run)!=selection['classifierRunSha256']:
            raise ValueError('Selected classifier lineage changed')
        config=classifier_run['config']
        predictor=RawCascade(detector,classifier,run['classes'],config['resolution'],config['context'],device='cuda')
        metadata.update(sourceClassifierTrainingStatus=classifier_status)
    report=evaluate(predictor,args.dataset,args.out,metadata)
    report.update(selectedCheckpoint=str(checkpoint),selectedClassifier=str(classifier) if classifier else None,
                  checkpointSelectionComplete=True,checkpointCandidatesEvaluated=1,checkpointCandidatesTotal=1)
    group=report['selectionGroup'];threshold=str(report['bestThreshold'])
    cpu_metrics=selection['thresholds'][threshold][group]
    gpu_metrics=report['thresholds'][threshold][group]
    report['cpuGpuMetricComparison']={key:{'cpu':cpu_metrics[key],'gpu':gpu_metrics[key]} for key in ['images','successes','model_exact','accepted_coverage','accepted_exact','failed_requests']}
    if sha256(checkpoint)!=selection['checkpointSha256'] or (classifier and sha256(classifier)!=selection['classifierCheckpointSha256']):
        raise ValueError('Source checkpoint changed during GPU validation')
    atomic_json(args.out/'report.json',report)
    print(json.dumps({'event':'gpu_release_validation_complete','report':str(args.out/'report.json'),'comparison':report['cpuGpuMetricComparison']}),flush=True)


if __name__=='__main__':main()
