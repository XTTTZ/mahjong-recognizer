"""Full-image CPU validation of signed small classifier best/last pairs; no test."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from ml.eval.recover_terminal import checked_run, checked_candidates
from ml.eval.evaluate_yolo_candidates import ranking
from ml.eval.yolo_evaluate import evaluate
from ml.train.provenance import atomic_json, dataset_contract, sha256
from ml.train.student_common import ARCHITECTURE


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','run','checkpoint','detector-run','detector-checkpoint','out'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--terminal-receipt',type=Path); p.add_argument('--threads',type=int,default=2)
    a=p.parse_args()
    if not 1<=a.threads<=4: raise ValueError('Use1..4 CPU threads')
    if a.out.exists(): raise FileExistsError('Use an immutable new validation directory')
    os.environ.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS=str(a.threads),MKL_NUM_THREADS=str(a.threads),YOLO_OFFLINE='true',YOLO_AUTOINSTALL='false',YOLO_CONFIG_DIR=str(a.out/'yolo-config'))
    contract=dataset_contract(a.dataset,verify_images=True)
    run,status,candidates=checked_candidates(a.run,a.checkpoint,contract,a.terminal_receipt)
    detector_run,detector_status,detectors=checked_candidates(a.detector_run,a.detector_checkpoint,contract)
    # The requested frozen detector is fixed; last/best substitution is forbidden.
    detectors=[pair for pair in detectors if pair[0]==a.detector_checkpoint]
    if run['architecture']!=ARCHITECTURE or not detector_run.get('generic') or len(detectors)!=1: raise ValueError('Expected a signed student and the fixed generic detector')
    import torch
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
    if torch.cuda.is_available(): raise RuntimeError('CPU evaluation cannot see CUDA')
    from ml.inference.yolo_model import RawYOLO
    from ml.inference.student_model import RawStudentCascade
    rows=[];a.out.mkdir(parents=True)
    for index,(checkpoint,digest) in enumerate(candidates):
        row={'checkpoint':str(a.detector_checkpoint),'checkpointSha256':detectors[0][1],
             'classifier':str(checkpoint),'classifierSha256':digest}
        predictor=None
        try:
            detector=RawYOLO(a.detector_checkpoint,run['classes'],detector_run['resolution'],device='cpu',threshold=.05,generic=True)
            predictor=RawStudentCascade(detector,checkpoint,run['classes'],run['config'])
            metadata={'architecture':'yolo-mobilenet-v3-large-cascade','classifierArchitecture':ARCHITECTURE,
                'checkpointSha256':detectors[0][1],'classifierCheckpointSha256':digest,'runSha256':sha256(a.detector_run),
                'classifierRunSha256':sha256(a.run),'classifierResumeSignatureSha256':hashlib.sha256(json.dumps(run['resumeSignature'],sort_keys=True).encode()).hexdigest(),
                'resumeSignatureSha256':hashlib.sha256(json.dumps(detector_run['resumeSignature'],sort_keys=True).encode()).hexdigest(),
                'dataContract':contract,'trainAnnotationSha256':contract['splits']['train']['annotationSha256'],
                'validationAnnotationSha256':contract['splits']['valid']['annotationSha256'], 'resolution':detector_run['resolution'],
                'classifierResolution':224,'cropContext':run['config']['context'],'evaluationDevice':'cpu','evaluationThreads':a.threads,
                'scoreFusion':'detector_probability_times_classifier_probability','localizationScope':'predicted_detector_boxes',
                'predictionFloor':.05,'detectorThreshold':.05,'nmsIou':.6,'sourceTrainingStatus':detector_status,'sourceClassifierTrainingStatus':status,
                'gpuDeploymentRevalidationRequired':True,'evaluationCodeSha256':sha256(__file__)}
            destination=a.out/f'candidate-{index:02d}'
            report=evaluate(predictor,a.dataset,destination,metadata)
            if sha256(checkpoint)!=digest or sha256(a.detector_checkpoint)!=detectors[0][1]: raise ValueError('Weight changed during evaluation')
            if report['inferenceFailedImages']:raise RuntimeError('Candidate inference failures retained; refusing successful complete selection')
            row.update(state='succeeded',ranking=ranking(report),report=str(destination/'report.json'),inferenceFailedImages=report['inferenceFailedImages'])
        except Exception as error: row.update(state='failed',error=f'{type(error).__name__}: {error}')
        finally:
            predictor=None
            if 'detector' in locals(): del detector
            gc.collect()
        rows.append(row);atomic_json(a.out/'selection.json',{'candidates':rows,'trainingStarted':False})
        print(json.dumps(row),flush=True)
    valid=[r for r in rows if r['state']=='succeeded']
    if not valid: raise RuntimeError('No student candidate produced full-image validation')
    winner=max(valid,key=lambda r:tuple(r['ranking']));report=json.loads(Path(winner['report']).read_text())
    report.update(selectedCheckpoint=winner['checkpoint'],selectedClassifier=winner['classifier'],candidateReports=rows,
        checkpointCandidatesTotal=len(candidates),checkpointCandidatesEvaluated=len(rows),checkpointSelectionComplete=len(valid)==len(candidates),
        recoveryEvaluationFinished=True,recoveryFinishedAt=datetime.now(timezone.utc).isoformat())
    atomic_json(a.out/'report.json',report)
    if len(valid)!=len(candidates): raise RuntimeError('Partial student selection retained; not a complete successful job')


if __name__=='__main__': main()
