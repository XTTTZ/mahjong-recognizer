"""Bounded same-container CPU comparison; public weights, private one-photo-at-a-time inputs."""
import argparse
from datetime import datetime,timezone
from io import BytesIO
import json,os,platform,sys,time
from pathlib import Path
import modal

CORES=int(os.environ.get('MAHJONG_STREAM_CORES','2'))
if CORES not in (1,2):raise ValueError('Use1or2physicalcores')
app=modal.App('mahjong-compact-stream-benchmark-v10')
image=None
if modal.is_local():
    ROOT=Path(__file__).resolve().parents[2]
    from infra.modal.package import prepare_source
    from ml.inference.manifest import file_sha256
    from ml.inference.onnx_runtime import verify_portable_directory
    from ml.inference.candidate_onnx_runtime import verify_candidate_directory
    base=ROOT/'artifacts/models/releases/mahjong-onnx-fp32-public-20261030'
    candidate=Path(os.environ.get('MAHJONG_STREAM_CANDIDATE',ROOT/'artifacts/models/releases/v10-nano800-kd224-fp32-20261109'))
    verify_portable_directory(base);verify_candidate_directory(candidate)
    specs=[]
    for name,directory,tag in [('baseline',base,'runtime-onnx-fp32-20261030'),('compact',candidate,'runtime-compact-fp32-20261109')]:
        specs.append({'name':name,'baseUrl':'https://github.com/XTTTZ/mahjong-recognizer/releases/download/'+tag+'/',
            'files':{n:{'sha256':file_sha256(directory/n),'bytes':(directory/n).stat().st_size}for n in ('manifest.json','detector.onnx','classifier.onnx')}})
    source=prepare_source(ROOT/'ml',ROOT/'artifacts/models/stream-source')
    portable=prepare_source(ROOT/'infra/portable',ROOT/'artifacts/models/stream-tools')
    image=(modal.Image.debian_slim(python_version='3.11').pip_install('onnxruntime==1.23.2','numpy==1.26.4','opencv-python-headless==4.11.0.86','Pillow==11.3.0')
        .add_local_dir(source,'/app/ml',copy=True).add_local_dir(portable,'/app/infra/portable',copy=True)
        .env({'PYTHONPATH':'/app','PUBLIC_MATRIX_SPEC':json.dumps(specs)})
        .run_commands("python -c \"import os,json;from infra.portable.fetch_public_model import fetch_package;[fetch_package(s['baseUrl'],s['files'],'/models/'+s['name'])for s in json.loads(os.environ['PUBLIC_MATRIX_SPEC'])]\"")
        .env({'MAHJONG_STREAM_CORES':str(CORES)}))

@app.cls(image=image,cpu=(CORES,CORES),memory=4096,min_containers=0,max_containers=1,scaledown_window=10,timeout=90)
class MatrixWorker:
    @modal.enter()
    def load(self):
        sys.path.insert(0,'/app')
        import onnxruntime as ort
        from ml.inference.onnx_runtime import OnnxCascadePredictor
        from ml.inference.candidate_onnx_runtime import CandidateOnnxPredictor
        self.models={};loads={}
        for name,constructor in [('baseline',OnnxCascadePredictor),('compact',CandidateOnnxPredictor)]:
            start=time.perf_counter();self.models[name]=constructor('/models/'+name,CORES,4);loads[name]=(time.perf_counter()-start)*1000
        if self.models['baseline'].manifest['classes']!=self.models['compact'].manifest['classes']:raise ValueError('Classesdiffer')
        self.info={'hardware':platform.platform(),'ort':ort.__version__,'threads':CORES,'batch':4,'loadsMs':loads,
            'modelBytes':{n:sum(m.manifest['filesBytes'].values())for n,m in self.models.items()},
            'filesSha256':{n:m.manifest['filesSha256']for n,m in self.models.items()}}
    @modal.method()
    def warm(self,data):
        from PIL import Image
        photo=Image.open(BytesIO(data)).convert('RGB');warm={}
        for name,m in self.models.items():_,warm[name]=m.predict_with_timings(photo)
        return dict(self.info,warmupTimings=warm)
    @modal.method()
    def score(self,item,truth_rows,index):
        from PIL import Image
        from ml.eval.metrics import Tile,score_image
        from ml.inference.postprocess import normalize_tiles
        start=time.perf_counter();photo=Image.open(BytesIO(item['data'])).convert('RGB');decode=(time.perf_counter()-start)*1000
        truth=[Tile(r['classId'],tuple(r['bbox']))for r in truth_rows];out={}
        for name in (['baseline','compact']if index%2==0 else['compact','baseline']):
            timing=None;error=None
            try:
                raw,timing=self.models[name].predict_with_timings(photo);start=time.perf_counter()
                tiles,reasons=normalize_tiles([r for r in raw if r['score']>.35],photo.width,photo.height,.7)
                timing=dict(timing,normalizationMs=(time.perf_counter()-start)*1000);timing['totalMs']+=timing['normalizationMs']
                metric=score_image(truth,[Tile(r['classId'],tuple(r['bbox']))for r in tiles],accepted=not reasons)
            except Exception as e:
                error=type(e).__name__;metric=score_image(truth,[],accepted=False,failed=True)
            out[name]={'imageId':item['id'],'metrics':metric,'timings':timing,'decodeMs':decode,'errorType':error}
        return out

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--valid-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--run-cloud',action='store_true');a=p.parse_args()
    from infra.portable.candidate_benchmark import validation_payload
    from infra.portable.quality_gate import paired_quality_gate,percentile
    from ml.eval.metrics import summarize
    payload,annotation=validation_payload(a.valid_dir)
    if a.out.exists():raise FileExistsError('Choosefreshreport')
    if not a.run_cloud:print(json.dumps({'ready':True,'cloudStarted':False,'images':len(payload)}));return
    tokens={c['id']:c['name']for c in annotation['categories']};truth={}
    for r in annotation['annotations']:truth.setdefault(r['image_id'],[]).append({'classId':tokens[r['category_id']],'bbox':r['bbox']})
    rows={'baseline':[],'compact':[]};deadline=time.monotonic()+1500
    with modal.enable_output(),app.run():
        worker=MatrixWorker();meta=worker.warm.remote(payload[0]['data'])
        for i,item in enumerate(payload):
            if time.monotonic()>deadline:raise TimeoutError('Global25minutecomparisonbudget')
            out=worker.score.remote(item,truth[item['id']],i)
            for name in rows:rows[name].append(out[name])
            if(i+1)%10==0:print(json.dumps({'completed':i+1,'total':150,'compactMs':out['compact']['timings']['totalMs']if out['compact']['timings']else None}),flush=True)
    def summary(values):
        result=summarize([r['metrics']for r in values]);times=[r['timings']['totalMs']for r in values if r['timings']]
        return dict(result,wrongAccepts=sum(r['metrics']['accepted']and not r['metrics']['model_exact']for r in values),modelExactCount=sum(r['metrics']['model_exact']for r in values),timedImages=len(times),p50Ms=percentile(times,.5)if times else None,p95Ms=percentile(times,.95)if times else None)
    scopes={name:{'all150':summary(values),'primary82':summary([r for r in values if r['metrics']['truth_count']>=10]),'thirteenFourteen45':summary([r for r in values if r['metrics']['truth_count']in(13,14)])}for name,values in rows.items()}
    report=dict(meta,createdAt=datetime.now(timezone.utc).isoformat(),execution='modal-cpu',allocatedPhysicalCores=CORES,partial=False,pairedOnSameHardware=True,
        annotationSha256='42758d7a3a7bec0a4f6fd415c0de5b5a76482a33de1ce45d91c37df4be1abc30',thresholds={'detector':.05,'fused':.35,'accept':.7},
        summaries=scopes,perImage=rows,strictHistoricalGate={'compact':paired_quality_gate(rows['baseline'],rows['compact'])},
        timingScope='samewarmcontainer alternatingorder onepassperphoto; pipelineincludingnormalization excludesdecode/network/load/coldstart',trainingStarted=False,testOpened=False,phoneHardwareMeasured=False)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(scopes,indent=2))
if __name__=='__main__':main()
