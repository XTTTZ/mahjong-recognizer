"""One-time sealed test of a frozen package and operating point; never selects models."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

from ml.inference.model import inspect_model, load_predictor
from ml.train.provenance import atomic_json, sha256, dataset_contract


def freeze_lock(model_dir, metadata, out):
    model_dir,metadata,out=Path(model_dir),Path(metadata),Path(out)
    manifest=inspect_model(model_dir)
    report_path=model_dir/'validation-report.json'
    report=json.loads(report_path.read_text())
    if report.get('evaluationDevice')!='cuda' or report.get('partial') or report.get('split')!='validation_only':
        raise ValueError('Freeze the winner only after full GPU validation')
    if sha256(report_path)!=manifest['validationReportSha256'] or manifest['threshold']!=report['bestThreshold'] or manifest['acceptThreshold']!=report['acceptThreshold']:
        raise ValueError('Package operating point/report differs from validation')
    source=json.loads(metadata.read_text())
    value={'schemaVersion':1,'createdAt':datetime.now(timezone.utc).isoformat(),'modelManifestSha256':sha256(model_dir/'model.json'),
        'filesSha256':manifest.get('filesSha256',{}),'architecture':manifest['architecture'],'version':manifest['version'],
        'threshold':manifest['threshold'],'acceptThreshold':manifest['acceptThreshold'],'classes':manifest['classes'],
        'validationReportSha256':sha256(report_path),'datasetMetadataSha256':sha256(metadata),
        'expectedTestAnnotationSha256':source['prepared_splits']['test']['annotation_sha256'],
        'selectionBasedOnValidationOnly':True,'testOpened':False,'testMayTuneThresholdOrModel':False}
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2)
    return value


def test_contract(dataset, expected_sha, classes):
    from PIL import Image
    dataset=Path(dataset);directory=dataset/'test';path=directory/'_annotations.coco.json'
    if sha256(path)!=expected_sha:raise ValueError('Sealed test annotation bytes differ from frozen metadata')
    coco=json.loads(path.read_text())
    categories=sorted(coco['categories'],key=lambda row:row['id'])
    if {str(i):row['name'] for i,row in enumerate(categories)}!=classes or [r['id'] for r in categories]!=list(range(1,len(classes)+1)):
        raise ValueError('Sealed test class mapping differs from frozen winner')
    prior_groups,prior_bytes=set(),set()
    for split in ('train','valid'):
        data=json.loads((dataset/split/'_annotations.coco.json').read_text())
        prior_groups.update(r['group_id'] for r in data['images'])
        prior_bytes.update(r.get('sha256') or sha256(dataset/split/r['file_name']) for r in data['images'])
    images={r['id']:r for r in coco['images']}
    if len(images)!=len(coco['images']) or not images:raise ValueError('Invalid test image IDs')
    digests=[]
    for record in images.values():
        path=(directory/record['file_name']).resolve()
        if not path.is_relative_to(directory.resolve()) or not record.get('group_id') or record['group_id'] in prior_groups:
            raise ValueError('Unsafe test path or source-group leakage')
        with Image.open(path) as image:image.verify()
        with Image.open(path) as image:
            if image.size!=(record['width'],record['height']) or image.getexif().get(274,1)!=1:raise ValueError('Test dimensions/EXIF differ')
        digest=sha256(path)
        if record.get('sha256')!=digest or digest in prior_bytes:raise ValueError('Test image bytes changed or leaked across splits')
        digests.append((str(record['id']),digest))
    seen=set()
    for annotation in coco['annotations']:
        if annotation['id'] in seen or annotation['image_id'] not in images or annotation['category_id'] not in range(1,len(classes)+1) or annotation.get('iscrowd',0):
            raise ValueError('Invalid test annotation')
        seen.add(annotation['id']);r=images[annotation['image_id']];x,y,w,h=annotation['bbox']
        if not all(math.isfinite(v) for v in (x,y,w,h)) or min(w,h)<=0 or min(x,y)<0 or x+w>r['width']+1e-5 or y+h>r['height']+1e-5:
            raise ValueError('Invalid test box')
    return {'annotationSha256':expected_sha,'images':len(images),'annotations':len(coco['annotations']),
        'groups':len({r['group_id'] for r in images.values()}),
        'imageTreeSha256':hashlib.sha256(json.dumps(sorted(digests)).encode()).hexdigest()}


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    freeze=commands.add_parser('lock');freeze.add_argument('--model-dir',type=Path,required=True);freeze.add_argument('--metadata',type=Path,required=True);freeze.add_argument('--out',type=Path,required=True)
    test=commands.add_parser('test')
    for name in ('model-dir','metadata','lock','dataset','out','ledger'):test.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.command=='lock':print(json.dumps(freeze_lock(args.model_dir,args.metadata,args.out),indent=2));return
    lock=json.loads(args.lock.read_text());manifest=inspect_model(args.model_dir)
    if sha256(args.model_dir/'model.json')!=lock['modelManifestSha256'] or sha256(args.metadata)!=lock['datasetMetadataSha256'] or manifest['threshold']!=lock['threshold'] or manifest['acceptThreshold']!=lock['acceptThreshold']:
        raise ValueError('Frozen winner/configuration changed after locking')
    if args.out.exists():raise FileExistsError('Use an immutable fresh test output directory')
    os.environ.update(YOLO_OFFLINE='true',YOLO_AUTOINSTALL='false',YOLO_CONFIG_DIR=str(args.out.parent/'sealed-test-settings'),NO_ALBUMENTATIONS_UPDATE='1')
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Final sealed evaluation requires the GPU serving path')
    torch.set_num_threads(2)
    predictor=load_predictor(args.model_dir,device='cuda')
    args.ledger.parent.mkdir(parents=True,exist_ok=True)
    # Reserve before reading labels; another model or threshold cannot reopen it.
    with args.ledger.open('x',encoding='utf-8') as stream:
        json.dump({'winnerLockSha256':sha256(args.lock),'modelManifestSha256':lock['modelManifestSha256'],
            'openedAt':datetime.now(timezone.utc).isoformat(),'testMayTuneThresholdOrModel':False},stream,indent=2)
    contract=dataset_contract(args.dataset,verify_images=True)
    sealed=test_contract(args.dataset,lock['expectedTestAnnotationSha256'],manifest['classes'])
    from ml.eval.yolo_evaluate import evaluate
    metadata={'architecture':manifest['architecture'],'resolution':manifest['resolution'],'version':manifest['version'],
        'dataContract':contract,'testContract':sealed,'winnerLockSha256':sha256(args.lock),'modelManifestSha256':lock['modelManifestSha256'],
        'filesSha256':manifest.get('filesSha256',{}),'evaluationDevice':'cuda','gpuName':torch.cuda.get_device_name(0),
        'inferenceCodeSha256':sha256(Path(__file__).resolve().parents[1]/'inference/cascade_model.py'),
        'testCodeSha256':sha256(__file__),'localizationScope':'predicted_detector_boxes','productionValidated':False}
    report=evaluate(predictor,args.dataset,args.out,metadata,thresholds=(manifest['threshold'],),accept_threshold=manifest['acceptThreshold'],
        split='test',select_threshold=False,expected_annotation_sha256=lock['expectedTestAnnotationSha256'])
    report.update(releaseDecisionRetunedOnTest=False,modelAndThresholdFrozenBeforeTest=True)
    atomic_json(args.out/'report.json',report)


if __name__=='__main__':main()
