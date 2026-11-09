"""One bounded CPU job compares multiple signed FP32 profiles on the same150 photos."""
import argparse
from datetime import datetime,timezone
from io import BytesIO
import json
from pathlib import Path
import platform
import time
from infra.portable.candidate_benchmark import validation_payload
from infra.portable.quality_gate import paired_quality_gate,percentile


def measure(profiles,payload,annotation,threads=2,batch=4):
    from PIL import Image
    import onnxruntime as ort
    from ml.inference.onnx_runtime import OnnxCascadePredictor
    from ml.inference.candidate_onnx_runtime import CandidateOnnxPredictor
    from ml.inference.postprocess import normalize_tiles
    from ml.eval.metrics import Tile,score_image,summarize
    if len(profiles)<2 or len(profiles)>6 or profiles[0]['name']!='baseline':raise ValueError('Baseline plus1..5 profiles required')
    if threads not in (1,2) or batch not in (1,4,8):raise ValueError('Bound CPU and batch')
    models={};loads={};warmup={}
    for row in profiles:
        start=time.perf_counter();constructor=OnnxCascadePredictor if row['name']=='baseline' else CandidateOnnxPredictor
        models[row['name']]=constructor(row['directory'],threads,batch);loads[row['name']]=(time.perf_counter()-start)*1000
    classes=models['baseline'].manifest['classes']
    if any(m.manifest['classes']!=classes for m in models.values()):raise ValueError('Profile class mapping mismatch')
    truth={}
    for a in annotation['annotations']:truth.setdefault(a['image_id'],[]).append(Tile(classes[str(a['category_id']-1)],tuple(a['bbox'])))
    warm=Image.open(BytesIO(payload[0]['data'])).convert('RGB')
    for name,m in models.items():_,warmup[name]=m.predict_with_timings(warm)
    rows={name:[] for name in models};names=list(models)
    for i,item in enumerate(payload):
        decode=time.perf_counter();image=Image.open(BytesIO(item['data'])).convert('RGB');decode=(time.perf_counter()-decode)*1000
        # Rotate execution order instead of giving one model a permanentfirst/last position.
        order=names[i%len(names):]+names[:i%len(names)]
        for name in order:
            timing=None;error=None
            try:
                raw,timing=models[name].predict_with_timings(image);start=time.perf_counter()
                tiles,reasons=normalize_tiles([r for r in raw if r['score']>.35],image.width,image.height,.7)
                timing=dict(timing,normalizationMs=(time.perf_counter()-start)*1000)
                timing['totalMs']+=timing['normalizationMs']
                metric=score_image(truth[item['id']],[Tile(r['classId'],tuple(r['bbox']))for r in tiles],accepted=not reasons)
            except Exception as e:
                metric=score_image(truth[item['id']],[],accepted=False,failed=True);error=type(e).__name__
            rows[name].append({'imageId':item['id'],'metrics':metric,'timings':timing,'decodeMs':decode,'errorType':error})
        if(i+1)%10==0:print(json.dumps({'evaluated':i+1,'total':len(payload)}),flush=True)
    def summary(values):
        r=summarize([v['metrics']for v in values]);times=[v['timings']['totalMs']for v in values if v['timings'] is not None]
        r.update(modelExactCount=sum(v['metrics']['model_exact']for v in values),wrongAccepts=sum(v['metrics']['accepted']and not v['metrics']['model_exact']for v in values),
                 timedImages=len(times),p50Ms=percentile(times,.5)if times else None,p95Ms=percentile(times,.95)if times else None)
        return r
    scopes={}
    for name,values in rows.items():
        scopes[name]={'all150':summary(values),'primary82':summary([v for v in values if v['metrics']['truth_count']>=10]),
            'thirteenFourteen45':summary([v for v in values if v['metrics']['truth_count']in(13,14)])}
    return {'createdAt':datetime.now(timezone.utc).isoformat(),'hardware':platform.platform(),'ort':ort.__version__,'threads':threads,'batch':batch,
        'profileNames':names,'modelBytes':{n:sum(m.manifest['filesBytes'].values())for n,m in models.items()},
        'filesSha256':{n:m.manifest['filesSha256']for n,m in models.items()},'loadsMs':loads,'warmupTimings':warmup,
        'summaries':scopes,'perImage':rows,'strictHistoricalGate':{n:paired_quality_gate(rows['baseline'],rows[n])for n in names[1:]},
        'timingScope':'warm-model single-pass rotating order; fullpipeline including normalization, excluding decode/load/network/coldstart',
        'thresholds':{'detector':.05,'fused':.35,'accept':.7},'pairedOnSameHardware':True,
        'trainingStarted':False,'testOpened':False,'partial':len(payload)!=150,'phoneHardwareMeasured':False,'productionPromotion':False}


def measure_uploaded(profiles,threads=2,batch=4):
    payload,annotation=validation_payload(Path('/work/validation'))
    return measure(profiles,payload,annotation,threads,batch)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profiles',type=Path,required=True);p.add_argument('--valid-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--threads',type=int,choices=(1,2),default=2);p.add_argument('--batch',type=int,choices=(1,4,8),default=4)
    mode=p.add_mutually_exclusive_group();mode.add_argument('--run-local',action='store_true');mode.add_argument('--run-cloud',action='store_true');a=p.parse_args()
    if a.out.exists():raise FileExistsError('Preserve previous report')
    profiles=json.loads(a.profiles.read_text());payload,annotation=validation_payload(a.valid_dir)
    from ml.inference.onnx_runtime import verify_portable_directory
    from ml.inference.candidate_onnx_runtime import verify_candidate_directory
    names=set()
    for i,row in enumerate(profiles):
        if row['name'] in names or not row['name'].replace('-','').isalnum():raise ValueError('Unique safe profile names required')
        names.add(row['name']);(verify_portable_directory if i==0 else verify_candidate_directory)(row['directory'])
    if not(a.run_local or a.run_cloud):print(json.dumps({'ready':True,'profiles':len(profiles),'images':len(payload),'cloudStarted':False}));return
    if a.run_cloud:
        import modal
        from importlib import import_module
        root=Path(__file__).resolve().parents[2]
        image=(modal.Image.debian_slim(python_version='3.11').pip_install('onnxruntime==1.23.2','numpy==1.26.4','opencv-python-headless==4.11.0.86','Pillow==11.3.0')
            .add_local_dir(root/'ml','/app/ml',copy=True,ignore=['__pycache__']).add_local_dir(root/'infra/portable','/app/infra/portable',copy=True,ignore=['__pycache__']).env({'PYTHONPATH':'/app'}))
        from infra.portable.matrix_upload import prepare
        upload=prepare(profiles,a.valid_dir,root/'artifacts/models/matrix-upload')
        image=image.add_local_dir(upload,'/matrix-upload',copy=True).run_commands('python /matrix-upload/assemble.py && rm -rf /matrix-upload')
        remote=[dict(row,directory='/work/models/'+row['name'])for row in profiles]
        app=modal.App('mahjong-compact-cpu-matrix-v10')
        fn=app.function(image=image,cpu=(a.threads,a.threads),memory=4096,timeout=1800,min_containers=0,max_containers=1,scaledown_window=2)(import_module('infra.portable.profile_matrix').measure_uploaded)
        with modal.enable_output(),app.run():result=fn.remote(remote,a.threads,a.batch)
        result.update(execution='modal-cpu',allocatedPhysicalCores=a.threads,maximumSeconds=1800)
    else:result=measure(profiles,payload,annotation,a.threads,a.batch);result.update(execution='local-cpu')
    result.update(annotationSha256='42758d7a3a7bec0a4f6fd415c0de5b5a76482a33de1ce45d91c37df4be1abc30')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result['summaries'],indent=2))
if __name__=='__main__':main()
