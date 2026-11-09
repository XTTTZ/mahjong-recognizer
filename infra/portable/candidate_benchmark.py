"""Paired FP32 baseline/candidate CPU check. Default is offline preflight, never execution."""
import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import platform
import time
from infra.portable.quality_gate import paired_quality_gate, percentile, stages

ANNOTATION_SHA = '42758d7a3a7bec0a4f6fd415c0de5b5a76482a33de1ce45d91c37df4be1abc30'
STRESS = {17,39,49,100,110,121}


def validation_payload(valid_dir):
    valid_dir = Path(valid_dir)
    if valid_dir.name != 'valid': raise ValueError('Frozen valid split only; never sealed test')
    data = (valid_dir/'_annotations.coco.json').read_bytes()
    if hashlib.sha256(data).hexdigest()!=ANNOTATION_SHA: raise ValueError('Frozen annotation changed')
    annotation=json.loads(data); records=sorted(annotation['images'],key=lambda r:r['id'])
    if len(records)!=150 or len({r['id'] for r in records})!=150: raise ValueError('Require all150 unique images')
    payload=[]
    for row in records:
        path=(valid_dir/row['file_name']).resolve()
        if not path.is_relative_to(valid_dir.resolve()): raise ValueError('Image path escape')
        pixels=path.read_bytes()
        if hashlib.sha256(pixels).hexdigest()!=row['sha256']: raise ValueError('Frozen pixels changed')
        payload.append({'id':row['id'],'data':pixels})
    return payload,annotation


def benchmark(baseline_dir, candidate_dir, payload, annotation, threads=2, batch=4, repeats=2):
    from PIL import Image
    import onnxruntime as ort
    from ml.inference.onnx_runtime import OnnxCascadePredictor
    from ml.inference.candidate_onnx_runtime import CandidateOnnxPredictor
    from ml.inference.postprocess import normalize_tiles
    from ml.eval.metrics import Tile,summarize,score_image
    if threads not in (1,2) or batch not in (1,4,8) or repeats not in (1,2,3): raise ValueError('Bound CPU/batch/repeats')
    started=time.perf_counter(); baseline=OnnxCascadePredictor(baseline_dir,threads,batch); first=time.perf_counter()
    candidate=CandidateOnnxPredictor(candidate_dir,threads,batch); loaded=time.perf_counter()
    if baseline.manifest['classes']!=candidate.manifest['classes']: raise ValueError('Class order changed')
    models={'baseline':baseline,'candidate':candidate}; truth={}; labels=baseline.manifest['classes']
    for a in annotation['annotations']: truth.setdefault(a['image_id'],[]).append(Tile(labels[str(a['category_id']-1)],tuple(a['bbox'])))
    warm=Image.open(BytesIO(payload[0]['data'])).convert('RGB'); warmup={}
    for name,model in models.items(): _,warmup[name]=model.predict_with_timings(warm)
    rows={name:[] for name in models}; timing_rows={name:[] for name in models}
    for index,item in enumerate(payload):
        start=time.perf_counter(); image=Image.open(BytesIO(item['data'])).convert('RGB'); decode=(time.perf_counter()-start)*1000
        first_rows={}
        for repeat in range(repeats):
            order=('baseline','candidate') if (index+repeat)%2==0 else ('candidate','baseline')
            for name in order:
                raw=[]; error=None; timing=None
                try:
                    raw,timing=models[name].predict_with_timings(image)
                    normalize_started=time.perf_counter()
                    kept,reasons=normalize_tiles([t for t in raw if t['score']>.35],image.width,image.height,.7)
                    timing=dict(timing, modelPipelineMs=timing['totalMs'], normalizationMs=(time.perf_counter()-normalize_started)*1000)
                    timing['totalMs']+=timing['normalizationMs']
                    metric=score_image(truth[item['id']],[Tile(t['classId'],tuple(t['bbox'])) for t in kept],accepted=not reasons)
                    stage=stages(truth[item['id']],[Tile(t['classId'],tuple(t['bbox'])) for t in raw])
                except Exception as e:
                    error=type(e).__name__; reasons=['inference_failed']; kept=[]
                    metric=score_image(truth[item['id']],[],accepted=False,failed=True)
                    stage={'truth':len(truth[item['id']]),'proposals':0,'localized':0,'correctClassesOnLocalizedPairs':0}
                row={'imageId':item['id'],'metrics':metric,'stages':stage,'timings':timing,'decodeMs':decode,'errorType':error}
                if repeat==0:
                    first_rows[name]=row; rows[name].append(row)
                elif metric!=first_rows[name]['metrics'] or error:
                    # Repeated inference instability fails the image, rather than only reporting the best run.
                    first_rows[name]['metrics']['failed']=True
                    first_rows[name]['metrics']['accepted']=False
                    first_rows[name]['metrics']['hand_exact_match']=False
                    first_rows[name]['metrics']['model_exact']=False
                    first_rows[name]['metrics']['multiset_exact']=False
                    first_rows[name]['repeatInstability']=True
                timing_rows[name].append({'imageId':item['id'],'repeat':repeat,'timings':timing})
        print(json.dumps({'evaluated':index+1,'total':len(payload)}),flush=True)
    def summary(name, selected):
        ids={r['imageId'] for r in selected}; result=summarize([r['metrics'] for r in selected])
        # Per-image medians across bounded repeated warm runs; every failed request remains in quality denominators.
        per_image=[]
        for ident in ids:
            values=[r['timings']['totalMs'] for r in timing_rows[name] if r['imageId']==ident and r['timings'] is not None]
            if len(values)==repeats: per_image.append(percentile(values,.5))
        result.update(wrongAccepts=sum(r['metrics']['accepted'] and not r['metrics']['model_exact'] for r in selected),
                      timedImages=len(per_image),p50Ms=percentile(per_image,.5) if per_image else None,
                      p95Ms=percentile(per_image,.95) if per_image else None)
        counts={k:sum(r['stages'][k] for r in selected) for k in ('truth','proposals','localized','correctClassesOnLocalizedPairs')}
        result['stages']={**counts,'localizationRecall':counts['localized']/counts['truth'] if counts['truth'] else None,
                          'localizationPrecision':counts['localized']/counts['proposals'] if counts['proposals'] else None,
                          'classAccuracyOnMatchedProposals':counts['correctClassesOnLocalizedPairs']/counts['localized'] if counts['localized'] else None}
        return result
    summaries={}
    for name,values in rows.items():
        groups={'all150':values,'primary82':[r for r in values if r['metrics']['truth_count']>=10],
                'thirteenFourteen45':[r for r in values if r['metrics']['truth_count'] in (13,14)],
                'posthocDaily76':[r for r in values if r['metrics']['truth_count']>=10 and r['imageId'] not in STRESS]}
        summaries[name]={group:summary(name, selected) for group,selected in groups.items()}
    gate=paired_quality_gate(rows['baseline'],rows['candidate'])
    speed=summaries['candidate']['thirteenFourteen45']
    return {'createdAt':datetime.now(timezone.utc).isoformat(),'partial':False,'trainingStarted':False,'testOpened':False,
            'annotationSha256':ANNOTATION_SHA,'hardware':platform.platform(),'ort':ort.__version__,'threads':threads,'batch':batch,'repeats':repeats,
            'thresholds':{'detector':.05,'fused':.35,'accept':.7},'qualityGate':gate,
            'warmLatencyTargetPassed':speed['timedImages']==45 and speed['p95Ms']<=2000 and not any(r['metrics']['failed'] for r in rows['candidate']),
            'modelBytes':{n:sum(m.manifest['filesBytes'].values()) for n,m in models.items()},
            'filesSha256':{n:m.manifest['filesSha256'] for n,m in models.items()},
            'loadMs':{'baseline':(first-started)*1000,'candidate':(loaded-first)*1000},'warmup':warmup,
            'summaries':summaries,'perImage':rows,'timingRuns':timing_rows,
            'timingScope':'warm models, paired alternating order; pipeline excludes decode/load/network/cold start; per-image median then cross-image p50/p95',
            'stageScope':'First pass only; all raw detector proposals >.05 before fused threshold; class-blind maximum-cardinality IoU>=.5 matching then class accuracy on same pairs; repeated-run failures separately invalidate image quality',
            'productionPromotion':False,'phoneHardwareTested':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('baseline-dir','candidate-dir','valid-dir','out'): p.add_argument('--'+arg,type=Path,required=True)
    p.add_argument('--threads',type=int,choices=(1,2),default=2);p.add_argument('--batch',type=int,choices=(1,4,8),default=4)
    p.add_argument('--repeats',type=int,choices=(1,2,3),default=2)
    modes=p.add_mutually_exclusive_group();modes.add_argument('--run-local',action='store_true');modes.add_argument('--run-cloud',action='store_true')
    a=p.parse_args()
    from ml.inference.onnx_runtime import verify_portable_directory
    from ml.inference.candidate_onnx_runtime import verify_candidate_directory
    verify_portable_directory(a.baseline_dir);verify_candidate_directory(a.candidate_dir)
    payload,annotation=validation_payload(a.valid_dir)
    if a.out.exists(): raise FileExistsError('Use a fresh immutable report')
    if not(a.run_local or a.run_cloud):
        print(json.dumps({'ready':True,'images':len(payload),'trainingStarted':False,'inferenceStarted':False,'cloudStarted':False}));return
    if a.run_cloud:
        import modal
        from importlib import import_module
        root=Path(__file__).resolve().parents[2]
        image=(modal.Image.debian_slim(python_version='3.11')
               .pip_install('onnxruntime==1.23.2','numpy==1.26.4','opencv-python-headless==4.11.0.86','Pillow==11.3.0')
               .add_local_dir(root/'ml','/app/ml',copy=True,ignore=['__pycache__'])
               .add_local_dir(root/'infra/portable','/app/infra/portable',copy=True,ignore=['__pycache__'])
               .add_local_dir(a.baseline_dir,'/models/baseline',copy=True)
               .add_local_dir(a.candidate_dir,'/models/candidate',copy=True).env({'PYTHONPATH':'/app'}))
        app=modal.App('mahjong-v9-candidate-cpu-benchmark')
        function=app.function(image=image,cpu=(a.threads,a.threads),memory=4096,timeout=1800,min_containers=0,max_containers=1,scaledown_window=2)(import_module('infra.portable.candidate_benchmark').benchmark)
        with modal.enable_output(),app.run():
            result=function.remote('/models/baseline','/models/candidate',payload,annotation,a.threads,a.batch,a.repeats)
        result.update(execution='modal-cpu',allocatedPhysicalCores=a.threads,budgetSeconds=1800)
    else:
        result=benchmark(a.baseline_dir,a.candidate_dir,payload,annotation,a.threads,a.batch,a.repeats)
        result.update(execution='local-cpu',allocatedPhysicalCores=None)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'qualityGate':result['qualityGate'],'latencyPassed':result['warmLatencyTargetPassed'],'report':str(a.out)}))


if __name__=='__main__':main()
