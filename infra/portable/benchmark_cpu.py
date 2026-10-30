"""Bounded validation-only portable CPU benchmark; --run-cloud explicitly starts one CPU job."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import platform
import sys
import time


def benchmark(payload, annotation, threads=2,repeats=2):
    from PIL import Image
    import onnxruntime as ort
    from ml.inference.onnx_runtime import OnnxCascadePredictor
    from ml.inference.postprocess import normalize_tiles
    from ml.eval.metrics import Tile, score_image, summarize
    start=time.perf_counter(); model=OnnxCascadePredictor('/models/portable',threads,4)
    loaded=(time.perf_counter()-start)*1000
    truth={}
    for a in annotation['annotations']:
        truth.setdefault(a['image_id'],[]).append(Tile(model.manifest['classes'][str(a['category_id']-1)],tuple(a['bbox'])))
    rows=[]
    for item in payload:
        started=time.perf_counter()
        image=Image.open(BytesIO(item['data'])).convert('RGB')
        decoded=time.perf_counter()
        timings=[];raw=[]
        for _ in range(repeats):
            raw,timing=model.predict_with_timings(image);timings.append(timing)
        retained=[t for t in raw if t['score']>.35]
        tiles,reasons=normalize_tiles(retained,image.width,image.height,.7)
        score=score_image(truth[item['id']],[Tile(t['classId'],tuple(t['bbox'])) for t in tiles],accepted=not reasons)
        rows.append({'imageId':item['id'],'imageSha256':hashlib.sha256(item['data']).hexdigest(),
            'decodeMs':(decoded-started)*1000,'timings':timings,'metrics':score,'tiles':tiles,'reasons':reasons})
        print(json.dumps({'completed':len(rows),'total':len(payload),'lastMs':timings[-1]['totalMs']}),flush=True)
    def summary(selected):
        values=sorted(r['timings'][-1]['totalMs'] for r in selected)
        def percentile(q):
            if not values:return None
            import math
            n=(len(values)-1)*q;i=math.floor(n);return values[i]+(values[math.ceil(n)]-values[i])*(n-i)
        return {**summarize([r['metrics'] for r in selected]),'warmP50Ms':percentile(.5),'warmP95Ms':percentile(.95)}
    cpu=Path('/proc/cpuinfo').read_text() if Path('/proc/cpuinfo').exists() else platform.processor()
    return {'createdAt':datetime.now(timezone.utc).isoformat(),'runtime':ort.__version__,'provider':'CPUExecutionProvider',
        'hardware':platform.platform(),'cpuModel':next((line.split(':',1)[1].strip() for line in cpu.splitlines() if line.startswith('model name')),cpu[:160]),
        'threads':threads,'classifierBatch':4,'loadMs':loaded,'trainingStarted':False,'testOpened':False,
        'scope':'Frozen validation native pixels; bounded sequential runs per image; last run statistics; decode/load/network excluded',
        'repeatsPerImage':repeats,'runtimeSourceSha256':hashlib.sha256(Path('/app/ml/inference/onnx_runtime.py').read_bytes()).hexdigest(),
        'productionPoint':{'detectorFloor':.05,'fusedThreshold':.35,'acceptThreshold':.7},'browserCanvasTransformApplied':False,
        'sourceModelFilesSha256':model.manifest['sourceModelFilesSha256'],'onnxSha256':model.manifest['filesSha256'],
        'all':summary(rows),'hands10plus':summary([r for r in rows if r['metrics']['truth_count']>=10]),
        'hands13or14':summary([r for r in rows if r['metrics']['truth_count'] in (13,14)]),'perImage':rows}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--valid-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--run-cloud',action='store_true')
    p.add_argument('--limit',type=int,default=150);p.add_argument('--threads',type=int,choices=(2,4),default=2)
    p.add_argument('--repeats',type=int,choices=(1,2,3),default=2)
    a=p.parse_args()
    if a.valid_dir.name!='valid' or not 1<=a.limit<=150:raise ValueError('Only bounded frozen valid is allowed')
    if a.out.exists():raise FileExistsError('Preserve previous CPU benchmark evidence')
    annotation_bytes=(a.valid_dir/'_annotations.coco.json').read_bytes();annotation=json.loads(annotation_bytes)
    if hashlib.sha256(annotation_bytes).hexdigest()!='42758d7a3a7bec0a4f6fd415c0de5b5a76482a33de1ce45d91c37df4be1abc30':raise ValueError('Frozen valid annotation changed')
    # Representative hand and crowded-scene smoke first, then remaining immutable IDs.
    priority={16:0,40:1,76:2,39:3,125:4}
    selected=sorted(annotation['images'],key=lambda r:(priority.get(r['id'],5),r['id']))[:a.limit]
    payload=[]
    for r in selected:
        path=(a.valid_dir/r['file_name']).resolve()
        if not path.is_relative_to(a.valid_dir.resolve()):raise ValueError('Unsafe path')
        data=path.read_bytes()
        if hashlib.sha256(data).hexdigest()!=r['sha256']:raise ValueError('Validation pixels changed')
        payload.append({'id':r['id'],'data':data})
    if not a.run_cloud:
        print(json.dumps({'ready':True,'images':len(payload),'cloudStarted':False}));return
    import modal
    from importlib import import_module
    root=Path(__file__).resolve().parents[2]
    image=(modal.Image.debian_slim(python_version='3.11')
        .pip_install('onnxruntime==1.23.2','numpy==1.26.4','opencv-python-headless==4.11.0.86','Pillow==11.3.0')
        .add_local_dir(root/'ml','/app/ml',copy=True,ignore=['__pycache__'])
        .add_local_file(Path(__file__),'/app/infra/portable/benchmark_cpu.py',copy=True)
        .add_local_dir(a.model_dir,'/models/portable',copy=True)
        .env({'PYTHONPATH':'/app'}))
    app=modal.App('mahjong-portable-cpu-benchmark')
    remote=app.function(image=image,cpu=(a.threads,a.threads),memory=4096,timeout=1800,
        min_containers=0,max_containers=1,scaledown_window=2)(import_module('infra.portable.benchmark_cpu').benchmark)
    with modal.enable_output(),app.run():result=remote.remote(payload,annotation,a.threads,a.repeats)
    result.update(annotationSha256=hashlib.sha256(annotation_bytes).hexdigest(),partial=len(payload)!=150)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('all','hands10plus','hands13or14','loadMs','cpuModel')}))


if __name__=='__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]));main()
