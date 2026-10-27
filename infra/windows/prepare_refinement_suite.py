"""Write a bounded v6 twelve-hour fine-tune plan and cold bundle; never run training."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib,json,stat,zipfile,tempfile
from pathlib import Path
from infra.windows.experiment_queue import config_contract,signature
from infra.windows.offline_bundle import build,MANIFEST,digest,safe_name,verify

CODE='D:/mahjong/code-v6-ready-20261028'
PYTHON='C:/Users/TRAINING_USER/miniforge3/envs/mahjong-v6/python.exe'
BASE_PYTHON='C:/Users/TRAINING_USER/miniforge3/envs/mahjong-yolo/python.exe'
PREFIX='D:/mahjong/runs/v5-breadth-12h-20261026-'


def make_suite(root:Path,code_root=CODE,queue_id='v6-refine-12h-20261028',allowed_gpu_paths=(),*,expected_environment=None,parent_assets=None,parent_roots=None):
    root=Path(root);code=code_root.rstrip('/\\');data='D:/mahjong/data/prepared-v4-main'
    generic=data+'-yolo1';crops=data+'-crops42';background='D:/mahjong/data/v6-background-reviewed';cache='D:/mahjong/data/v6-crop-cache'
    parents=parent_roots or {}
    cls=parents.get('classifier',PREFIX+'efficientnetv2-s-320');v8=parents.get('v8',PREFIX+'tile-detector-v8-960');v26=parents.get('v26',PREFIX+'tile-detector-v26-960')
    jobs=[];checks=[]
    def evaluation(det_run,det_checkpoint,cls_run,cls_checkpoint,out,terminal=None):
        argv=[PYTHON,'-m','ml.eval.recover_terminal','--dataset',data,'--run',det_run,'--checkpoint',det_checkpoint,
              '--classifier-run',cls_run,'--classifier',cls_checkpoint,'--out',out,'--max-candidates','2','--threads','2']
        if terminal:argv+=['--classifier-terminal-receipt',terminal]
        return argv
    def add(name,argv,eval_argv,estimate,timeout,soft):
        identifier=queue_id+'-'+name;run='D:/mahjong/runs/'+identifier
        argv=[str(v).replace('{run}',run)for v in argv]
        job={'id':identifier,'run_dir':run,'cwd':code,'estimated_minutes':estimate,'timeout_minutes':timeout,
             'steps':[{'name':'train','argv':argv,'timeout_minutes':soft+10},
                      {'name':'validate-complete-hands','argv':[v.replace('{run}',run)for v in eval_argv],
                       'timeout_minutes':25,'env':{'CUDA_VISIBLE_DEVICES':'-1'}}],
             'report_paths':[run+'/validation/report.json']}
        jobs.append(job);checks.append({'name':name,'argv':argv+['--preflight-only'],'timeout_seconds':600,'requireDerivativeReady':True})
    # All branches within an architecture use exactly the same signed parent.
    detector_rows=[('v8-control',v8,'best.pt',generic,'none',960,8,40,40,65),
                   ('v8-background',v8,'best.pt',background,'none',960,8,45,42,68),
                   ('v8-background-geometry',v8,'best.pt',background,'mild',960,8,45,45,70),
                   ('v26-background-geometry-960',v26,'last.pt',background,'mild',960,8,60,50,85),
                   ('v26-background-geometry-1280',v26,'last.pt',background,'mild',1280,4,85,80,120)]
    for name,parent,checkpoint,prepared,geometry,resolution,batch,soft,estimate,timeout in detector_rows:
        train=[PYTHON,'-m','ml.train.yolo_train_v6','--dataset',data,'--prepared',prepared,
               '--parent-run',parent+'/run.json','--parent-checkpoint',parent+'/train/weights/'+checkpoint,
               '--output','{run}','--epochs','12','--resolution',str(resolution),'--batch',str(batch),
               '--geometry',geometry,'--max-train-minutes',str(soft),'--seed','2026']
        evaluate=evaluation('{run}/run.json','{run}/train/weights/best.pt',cls+'/run.json',cls+'/last.pt','{run}/validation',cls+'/status.json')
        add(name,train,evaluate,estimate,timeout,soft)
    for name,camera,context,soft,estimate,timeout in [('classifier-control','none',.08,80,75,110),
            ('classifier-mild-camera','mild',.08,85,80,115),('classifier-context002','none',.02,80,75,110)]:
        train=[PYTHON,'-m','ml.train.cascade_train_v6','--dataset',data,'--crops',crops,'--cache',cache,
               '--parent-run',cls+'/run.json','--parent-checkpoint',cls+'/last.pt','--parent-terminal-receipt',cls+'/status.json',
               '--output','{run}','--resolution','320','--epochs','12','--batch','16','--gradient-accumulation','2',
               '--rotation','quarter-turn','--camera-profile',camera,'--context',str(context),'--max-train-minutes',str(soft),'--seed','2026']
        evaluate=evaluation(v26+'/run.json',v26+'/train/weights/last.pt','{run}/run.json','{run}/best.pt','{run}/validation')
        add(name,train,evaluate,estimate,timeout,soft)
    config={'schemaVersion':1,'queue_id':queue_id,'queue_dir':'D:/mahjong/queues/'+queue_id,
            'deadline_hours':12,'reserve_minutes':60,'poll_seconds':2,'telemetry_seconds':60,'wait_timeout_minutes':15,
            'gpu_policy':'exclusive','allowed_gpu_process_paths':list(allowed_gpu_paths),'wait_for_status':[],
            'env':{'HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','YOLO_OFFLINE':'true','YOLO_AUTOINSTALL':'false','WANDB_MODE':'disabled','PYTHONUTF8':'1','OMP_NUM_THREADS':'8'},'experiments':jobs}
    config_contract(config)
    preparations=[{'name':'reviewed-train-backgrounds','argv':[PYTHON,'-m','ml.data.prepare_background_v6','--dataset',data,
            '--base-prepared',generic,'--review',code+'/docs/data/background-v6-review.json','--out',background]},
            {'name':'lossless-crop-cache','argv':[PYTHON,'-m','ml.data.prepare_crop_cache_v6','--dataset',data,'--crops',crops,'--out',cache]}]
    derivative_checks=[{'name':p['name'],'argv':p['argv']+['--preflight-only'],'timeout_seconds':900,'requireDerivativeReady':True}for p in preparations]
    paths=sorted(set(root.glob('ml/**/*.py'))|set(root.glob('infra/windows/*.py')))
    paths += [root/p for p in ['docs/data/class-map.json','docs/data/final-dataset-v4.json','docs/data/split-v4.jsonl',
              'docs/data/background-v6-review.json']]
    if expected_environment is None:paths.append(root/'infra/windows/verified/v6/expected-environment.json')
    if parent_assets is None:paths.append(root/'infra/windows/verified/v6/parent-assets.json')
    source=[{'path':code+'/'+p.relative_to(root).as_posix(),'sha256':digest(p)}for p in paths]
    expected=expected_environment if expected_environment is not None else json.loads((root/'infra/windows/verified/v6/expected-environment.json').read_text())
    assets=parent_assets if parent_assets is not None else json.loads((root/'infra/windows/verified/v6/parent-assets.json').read_text())['assets']
    return {'schemaVersion':1,'kind':'prepared_experiment_suite','preparationStatus':'offline_frozen_remote_v6_unverified',
            'trainingStarted':False,'codeRoot':code,'queueConfig':config,'sourceFiles':source,'pythonPaths':[PYTHON],
            'environmentSnapshots':[{'python':PYTHON,'version':expected['python'],'packages':expected['packages'],
                'evidence':'Expected clone source package pins; live v6 verification pending and mandatory before submission.'}],
            'assetsVerifiedByPreparation':True,'assetEvidence':'prior_verified_v5_hashes_not_live_v6',
            'assetReferenceHashesFrozen':True,'remoteV6AssetsChecked':False,'requiredFiles':assets,'preparationCommands':preparations,
            'cpuPreflightCommands':derivative_checks+checks,'basePython':BASE_PYTHON,
            'cloneCommand':['C:/Users/TRAINING_USER/miniforge3/Scripts/conda.exe','create','--name','mahjong-v6','--clone','mahjong-yolo','--yes','--offline'],
            'note':'Frozen references are not live readiness. Only prepare/check on the powered-on host can verify v6. Generator never submits. All branches are independent new warm starts, not exact optimizer resume.'}


def write_plan(root,out,code_root=CODE,queue_id='v6-refine-12h-20261028',allowed_gpu_paths=(),**configuration):
    out=Path(out)
    if out.exists():raise FileExistsError('Use a fresh immutable plan directory')
    suite=make_suite(root,code_root,queue_id,allowed_gpu_paths,**configuration);out.mkdir(parents=True)
    for name,data in [('suite.json',suite),('queue.json',suite['queueConfig'])]:
        (out/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    remote=code_root+'/prepared-suites/'+out.name+'/suite.json'
    launch='"'+PYTHON+'" "'+code_root+'/infra/windows/launch_prepared_queue.py" --suite "'+remote+'"'
    (out/'check.cmd').write_bytes(('@echo off\r\nsetlocal\r\nset CUDA_VISIBLE_DEVICES=-1\r\n'+launch+'\r\nexit /b %errorlevel%\r\n').encode())
    (out/'submit.cmd').write_bytes(('@echo off\r\n'+launch+' --start\r\n').encode())
    (out/'prepare.cmd').write_bytes(('@echo off\r\nsetlocal\r\nset CUDA_VISIBLE_DEVICES=-1\r\n"'+BASE_PYTHON+'" "'+code_root+'/infra/windows/prepare_v6_host.py" --suite "'+remote+'"\r\nexit /b %errorlevel%\r\n').encode())
    (out/'watch.cmd').write_bytes(('@echo off\r\n"'+PYTHON+'" "'+code_root+'/infra/windows/watch_queue.py" --queue "'+suite['queueConfig']['queue_dir']+'" --follow\r\n').encode())
    return suite


def cold_bundle(root,plan,out):
    """Extend the verified source bundle with one prepared suite, preserving its manifest."""
    root,plan,out=map(Path,(root,plan,out))
    if out.exists():raise FileExistsError('Use a fresh bundle path')
    with tempfile.TemporaryDirectory()as tmp:
        result=build(root,Path(tmp)/'source.zip',root/'docs/data/final-dataset-v4.json',root/'docs/data/split-v4.jsonl')
        with zipfile.ZipFile(result['bundle'])as source:contents={n:source.read(n)for n in source.namelist()}
    manifest=json.loads(contents.pop(MANIFEST))
    # Cold v6 execution needs only public expected pins and parent references,
    # not historical private machine snapshots or experiment logs.
    public_configs={'infra/windows/verified/v6/expected-environment.json','infra/windows/verified/v6/parent-assets.json'}
    contents={name:value for name,value in contents.items()if not name.startswith('infra/windows/verified/')or name in public_configs}
    for p in sorted(plan.iterdir()):
        if p.is_file():contents[safe_name('prepared-suites/'+plan.name+'/'+p.name)]=p.read_bytes()
    doc=root/'docs/models/V6_REFINEMENT.md'
    if doc.is_file():contents['docs/models/V6_REFINEMENT.md']=doc.read_bytes()
    manifest.update(preparationStatus='offline_frozen_remote_v6_unverified',training_started=False)
    manifest['files']={n:{'sha256':hashlib.sha256(v).hexdigest(),'bytes':len(v)}for n,v in sorted(contents.items())}
    contents[MANIFEST]=json.dumps(manifest,indent=2).encode();out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_DEFLATED)as target:
        for name,data in sorted(contents.items()):
            info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=(stat.S_IFREG|0o644)<<16
            target.writestr(info,data)
    checksum=digest(out);verify(out,checksum)
    verifier=out.with_suffix('.verify.py');verifier.write_bytes((root/'infra/windows/offline_bundle.py').read_bytes())
    receipt={'bundle':str(out),'sha256':checksum,'bytes':out.stat().st_size,'payloadFiles':len(manifest['files']),
             'verifier':str(verifier),'verifierSha256':digest(verifier),'localBundleVerificationPassed':True,
             'remoteV6Prepared':False,'remoteV6CheckPassed':False,'trainingStarted':False}
    out.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');out.with_suffix(out.suffix+'.sha256').write_text(checksum+'\n')
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--out',type=Path,required=True);p.add_argument('--bundle-out',type=Path)
    p.add_argument('--code-root',default=CODE);p.add_argument('--queue-id',default='v6-refine-12h-20261028')
    p.add_argument('--gpu-background-paths',type=Path)
    p.add_argument('--expected-environment',type=Path,help='Your expected Python/package pins JSON; defaults to the public v6 template')
    p.add_argument('--parent-assets',type=Path,help='Your signed parent asset paths/SHA256 JSON')
    p.add_argument('--v8-parent');p.add_argument('--v26-parent');p.add_argument('--classifier-parent');a=p.parse_args()
    allowed=json.loads(a.gpu_background_paths.read_text(encoding='utf-8-sig'))if a.gpu_background_paths else []
    if isinstance(allowed,dict):allowed=allowed.get('allowed_gpu_process_paths',allowed.get('paths',[]))
    configuration={}
    if a.expected_environment:configuration['expected_environment']=json.loads(a.expected_environment.read_text())
    if a.parent_assets:configuration['parent_assets']=json.loads(a.parent_assets.read_text())['assets']
    parents={k:v for k,v in [('v8',a.v8_parent),('v26',a.v26_parent),('classifier',a.classifier_parent)]if v}
    if parents:configuration['parent_roots']=parents
    suite=write_plan(a.root,a.out,a.code_root,a.queue_id,allowed,**configuration)
    result={'state':'offline_plan_written','trainingStarted':False,'remoteV6Verified':False,'experiments':len(suite['queueConfig']['experiments']),
            'estimatedMinutes':sum(j['estimated_minutes']for j in suite['queueConfig']['experiments']),'hardBudgetHours':12}
    if a.bundle_out:result['bundle']=cold_bundle(a.root,a.out,a.bundle_out)
    print(json.dumps(result))

if __name__=='__main__':main()
