"""Research CPU FX-INT8 export, train-only calibration and locked-point validation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from ml.data.prepare_crops import verify_prepared
from ml.data.prepare_crop_cache_v6 import verify_cache
from ml.train.provenance import atomic_json, dataset_contract, sha256, check_evaluation_data
from ml.train.yolo_train import admit_output, verify_receipt
from ml.train.student_common import ARCHITECTURE, build_student, calibration_indices


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','crops','cache','run','selection','detector-run','detector-checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--calibration-limit',type=int,default=672);p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--preflight-only',action='store_true')
    a=p.parse_args()
    if not 42<=a.calibration_limit<=2048 or not 1<=a.threads<=4:raise ValueError('Bound calibration and CPU threads')
    # Future student outputs need not exist for argument/config preflight.
    if a.preflight_only:
        print(json.dumps({'trainingStarted':False,'quantizationExecuted':False,'futureStudentCheckpointRequired':True,
            'method':'FX static INT8 CPU fbgemm','calibrationSplit':'train_only','testOpened':False}));return
    os.environ.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS=str(a.threads),MKL_NUM_THREADS=str(a.threads),YOLO_OFFLINE='true',YOLO_AUTOINSTALL='false',YOLO_CONFIG_DIR=str(a.output/'yolo-config'))
    contract=dataset_contract(a.dataset,verify_images=True);verify_prepared(a.crops,contract);verify_cache(a.cache,contract,a.crops)
    run=json.loads(a.run.read_text());detector_run=json.loads(a.detector_run.read_text());selection=json.loads(a.selection.read_text())
    check_evaluation_data(run,contract);check_evaluation_data(detector_run,contract)
    if run['architecture']!=ARCHITECTURE or not run.get('finishedAt') or not detector_run.get('finishedAt') or not detector_run.get('generic'):
        raise ValueError('Quantization requires completed immutable student/detector runs')
    if not selection.get('checkpointSelectionComplete') or selection.get('partial') or selection.get('inferenceFailedImages') or selection.get('split')!='validation_only':
        raise ValueError('Complete FP32 student validation required before quantization')
    checkpoint=Path(selection['selectedClassifier']);verify_receipt(checkpoint,run);verify_receipt(a.detector_checkpoint,detector_run)
    if selection.get('classifierCheckpointSha256')!=sha256(checkpoint) or selection.get('checkpointSha256')!=sha256(a.detector_checkpoint) or selection.get('classifierRunSha256')!=sha256(a.run) or selection.get('dataContract')!=contract:
        raise ValueError('Selection does not bind the requested cascade')
    import torch
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import prepare_fx,convert_fx
    if torch.cuda.is_available():raise RuntimeError('CPU INT8 must never see CUDA')
    if torch.__version__.split('+')[0]!='2.8.0':raise RuntimeError('Use pinned Torch2.8 for FX artifact reproducibility')
    if 'fbgemm' not in torch.backends.quantized.supported_engines:raise RuntimeError('Pinned x86 fbgemm backend unavailable')
    torch.backends.quantized.engine='fbgemm';torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
    admit_output(a.output)
    from ml.inference.cascade_model import image_tensor
    from ml.train.v6_augment import cached_crop_face
    from PIL import Image
    records=json.loads((a.cache/'train.json').read_text())['records'];chosen=calibration_indices(records,a.calibration_limit,a.seed)
    allowed_groups={r['groupId'] for r in json.loads((a.crops/'train.json').read_text())['records']}
    valid_groups={r['groupId'] for r in json.loads((a.crops/'valid.json').read_text())['records']}
    if allowed_groups&valid_groups or any(records[i]['groupId']not in allowed_groups for i in chosen):raise ValueError('Calibration source-group leakage')
    calibration={'split':'train','seed':a.seed,'records':[{'annotationId':records[i]['annotationId'],'groupId':records[i]['groupId'],'label':records[i]['label']} for i in chosen],
        'sourceCacheSha256':sha256(a.cache/'prepared.json'),'validationUsedForCalibration':False,'testOpened':False}
    atomic_json(a.output/'calibration.json',calibration)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    signature_sha=hashlib.sha256(json.dumps(run['resumeSignature'],sort_keys=True).encode()).hexdigest()
    if saved['signatureSha256']!=signature_sha or saved['config']!=run['config'] or saved['classes']!=run['classes']:raise ValueError('Student payload differs from signed source')
    model=build_student();model.load_state_dict(saved['model'],strict=True);model.eval()
    example=(torch.zeros(1,3,224,224),)
    prepared=prepare_fx(model,get_default_qconfig_mapping('fbgemm'),example)
    started=time.monotonic()
    with torch.inference_mode():
        for start in range(0,len(chosen),16):
            faces=[]
            for i in chosen[start:start+16]:
                row=records[i];path=(a.cache/row['cacheFile']).resolve()
                if not path.is_relative_to(a.cache.resolve()):raise ValueError('Calibration path escape')
                with Image.open(path)as image:faces.append(image_tensor(cached_crop_face(image,row,224,run['config']['context'])))
            prepared(torch.stack(faces))
    quantized=convert_fx(prepared).eval()
    convs=sum('quantized' in type(m).__module__ and 'Conv' in type(m).__name__ for m in quantized.modules())
    if not convs:raise RuntimeError('No convolutions quantized; refusing a mislabeled INT8 artifact')
    scripted=torch.jit.script(quantized);scripted.save(str(a.output/'classifier-int8.ts'))
    from ml.inference.yolo_model import RawYOLO
    from ml.inference.student_model import RawStudentCascade
    from ml.eval.yolo_evaluate import evaluate
    detector=RawYOLO(a.detector_checkpoint,run['classes'],detector_run['resolution'],device='cpu',threshold=.05,generic=True)
    predictor=RawStudentCascade(detector,checkpoint,run['classes'],run['config'],scripted=a.output/'classifier-int8.ts')
    meta={'architecture':'yolo-mobilenet-v3-large-int8-cascade','classifierArchitecture':ARCHITECTURE,'quantization':'FX_static_INT8_fbgemm',
        'checkpointSha256':sha256(a.detector_checkpoint),'classifierCheckpointSha256':sha256(a.output/'classifier-int8.ts'),
        'sourceFP32ClassifierSha256':sha256(checkpoint),'sourceSelectionSha256':sha256(a.selection),'classifierRunSha256':sha256(a.run),
        'dataContract':contract,'trainAnnotationSha256':contract['splits']['train']['annotationSha256'],
        'validationAnnotationSha256':contract['splits']['valid']['annotationSha256'],'evaluationDevice':'cpu','evaluationThreads':a.threads,
        'resolution':detector_run['resolution'],'classifierResolution':224,'cropContext':run['config']['context'],
        'localizationScope':'predicted_detector_boxes','scoreFusion':'detector_probability_times_classifier_probability',
        'predictionFloor':.05,'detectorThreshold':.05,'nmsIou':.6,'calibrationSha256':sha256(a.output/'calibration.json'),
        'calibrationExamples':len(chosen),'quantizedConvolutions':convs,'calibrationSeconds':time.monotonic()-started,
        'sourceThresholdLocked':selection['bestThreshold'],'cpuOnlyResearchArtifact':True,'testOpened':False}
    root=Path(__file__).resolve().parents[1]
    meta['quantizationCodeSha256']={name:sha256(root/name) for name in ('train/student_quantize_v7.py','train/student_common.py','inference/student_model.py','inference/cascade_model.py','inference/postprocess.py','eval/operating_point.py','eval/metrics.py','eval/yolo_evaluate.py')}
    report=evaluate(predictor,a.dataset,a.output/'validation',meta,thresholds=(selection['bestThreshold'],),accept_threshold=selection['acceptThreshold'],select_threshold=False)
    if report['inferenceFailedImages']:raise RuntimeError('INT8 inference failed on validation images; rejecting artifact')
    primary=selection['selectionGroup'];before=selection['thresholds'][str(selection['bestThreshold'])][primary];after=report['thresholds'][str(report['bestThreshold'])][primary]
    report.update(fp32PrimaryMetrics=before,int8PrimaryMetrics=after,
        primaryAutoExactDelta=after['successes']-before['successes'],sourceTrainingCompleted=True,trainingStarted=False,
        quantizationFinishedAt=datetime.now(timezone.utc).isoformat(),artifactBytes=(a.output/'classifier-int8.ts').stat().st_size,
        note='CPU-only post-training experiment. No validation calibration, no threshold retuning, no deployment or latency claim. Full GPU pipeline remains unchanged.')
    atomic_json(a.output/'report.json',report)
    atomic_json(a.output/'quantization.json',{'method':'FX_static_INT8_fbgemm','artifactSha256':sha256(a.output/'classifier-int8.ts'),
        'sourceCheckpointSha256':sha256(checkpoint),'runSha256':sha256(a.run),'calibrationSha256':sha256(a.output/'calibration.json'),
        'torchVersion':torch.__version__,'engine':torch.backends.quantized.engine,'quantizedConvolutions':convs,
        'codeSha256':meta['quantizationCodeSha256'],'testOpened':False})


if __name__=='__main__':main()
