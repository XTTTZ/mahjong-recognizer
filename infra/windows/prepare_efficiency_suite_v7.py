"""Prepare remaining V6 ablations + small-classifier/KD/PTQ in a fresh12h window; never launch."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import stat
import tempfile
import zipfile
from infra.windows.prepare_refinement_suite import make_suite as refinement_suite, PYTHON, BASE_PYTHON, PREFIX
from infra.windows.experiment_queue import config_contract, signature
from infra.windows.offline_bundle import MANIFEST, digest, verify, safe_name
from infra.windows.prepare_refinement_suite import cold_bundle

CODE='D:/mahjong/code-v7-ready-20261029'
QUEUE='v7-efficiency-12h-20261029'


def make_suite(root, code_root=CODE, queue_id=QUEUE, allowed_gpu_paths=(), *, expected_environment=None, parent_assets=None, student_assets=None):
    root=Path(root)
    options={}
    if expected_environment is not None:options['expected_environment']=expected_environment
    if parent_assets is not None:options['parent_assets']=parent_assets
    suite=refinement_suite(root,code_root,queue_id,allowed_gpu_paths,**options)
    config=suite['queueConfig'];old={job['id'].removeprefix(queue_id+'-'):job for job in config['experiments']}
    data='D:/mahjong/data/prepared-v4-main';crops=data+'-crops42';cache='D:/mahjong/data/v6-crop-cache'
    classifier=PREFIX+'efficientnetv2-s-320';detector=PREFIX+'tile-detector-v26-960'
    assets=student_assets if student_assets is not None else json.loads((root/'docs/models/student-assets-v7.json').read_text())['assets']
    if len(assets)!=1 or assets[0]['path'].rsplit('/',1)[-1]!='mobilenet_v3_large-5c1a4163.pth' or len(assets[0].get('sha256',''))!=64:
        raise ValueError('Freeze the official student pretrained asset first')
    weights=assets[0]['path'];new={};checks=[]
    for name,alpha,soft,estimate in [('student-ce',0,40,40),('student-kd',.5,55,55)]:
        ident=queue_id+'-'+name;run='D:/mahjong/runs/'+ident
        train=[PYTHON,'-m','ml.train.student_train_v7','--dataset',data,'--crops',crops,'--cache',cache,'--weights',weights,
            '--teacher-run',classifier+'/run.json','--teacher-checkpoint',classifier+'/last.pt','--teacher-terminal-receipt',classifier+'/status.json',
            '--output',run,'--resolution','224','--context','.08','--epochs','20','--batch','16','--gradient-accumulation','2','--workers','2',
            '--lr','.0003','--seed','2026','--distillation-alpha',str(alpha),'--temperature','2','--teacher-min-confidence','.7','--max-train-minutes',str(soft)]
        evaluation=[PYTHON,'-m','ml.eval.student_evaluate_v7','--dataset',data,'--run',run+'/run.json','--checkpoint',run+'/best.pt',
            '--detector-run',detector+'/run.json','--detector-checkpoint',detector+'/train/weights/last.pt','--out',run+'/validation','--threads','2']
        new[name]={'id':ident,'run_dir':run,'cwd':code_root,'estimated_minutes':estimate,'timeout_minutes':soft+25,
            'steps':[{'name':'train','argv':train,'timeout_minutes':soft+8},
                     {'name':'validate-complete-hands','argv':evaluation,'timeout_minutes':17,'env':{'CUDA_VISIBLE_DEVICES':'-1'}}],
            'report_paths':[run+'/validation/report.json']}
        checks.append({'name':name,'argv':train+['--preflight-only'],'timeout_seconds':900,'requireDerivativeReady':True})
        qname=name+'-int8';qrun='D:/mahjong/runs/'+queue_id+'-'+qname
        quant=[PYTHON,'-m','ml.train.student_quantize_v7','--dataset',data,'--crops',crops,'--cache',cache,
            '--run',run+'/run.json','--selection',run+'/validation/report.json','--detector-run',detector+'/run.json',
            '--detector-checkpoint',detector+'/train/weights/last.pt','--output',qrun,'--calibration-limit','672','--threads','2','--seed','2026']
        new[qname]={'id':queue_id+'-'+qname,'run_dir':qrun,'cwd':code_root,'estimated_minutes':8,'timeout_minutes':18,
            'depends_on':[ident],'steps':[{'name':'calibrate-export-and-locked-validation','argv':quant,'timeout_minutes':18,'env':{'CUDA_VISIBLE_DEVICES':'-1'}}],
            'report_paths':[qrun+'/report.json']}
        checks.append({'name':qname+'-configuration','argv':quant+['--preflight-only'],'timeout_seconds':30})
    # Test both data-treatment controls and both small-model variants early. Long
    # resolution/context ablations remain independent and may be budget-skipped.
    order=['v8-background','v8-background-geometry','student-ce','student-kd','student-ce-int8','student-kd-int8',
           'v26-background-geometry-960','classifier-control','classifier-mild-camera','v26-background-geometry-1280','classifier-context002']
    config['experiments']=[(new.get(name)or old[name]) for name in order]
    suite['cpuPreflightCommands']=[c for c in suite['cpuPreflightCommands']if c['name']!='v8-control']+checks
    suite['requiredFiles'] += [{'path':asset['path'],'sha256':asset['sha256']} for asset in assets]
    suite.update(preparationStatus='offline_v7_prepared_windows_unverified',trainingStarted=False,newWindowAuthorized=False,submitted=False,
        smallModelStage='classifier_only; localization fixed to previously selected V26-last detector',
        distillationScope='frozen EfficientNetV2-S classifier teacher; train GT crops, not pseudo-label replacement; only correct/confident teacher targets',
        calibrationScope='train-only672 class-balanced complete crops; valid used only for full-image locked-point comparison; sealed test untouched',
        priorControlNotRepeated='v6-refine-12h-20261028-v8-control',
        requiredHumanActionBeforeStart='Windows must be on, new snapshot/assets copied, prepare.cmd and check.cmd passed, then explicit submission. This generator never calls them.',
        budget={'hardHours':12,'reserveMinutes':60,'estimatedMinutes':sum(j['estimated_minutes']for j in config['experiments']),
                'epochCaps':{'remainingV6Branches':12,'studentCEAndKD':20,'postTrainingQuantization':0},
                'measuredReference':{'run':'v6-refine-12h-20261028-v8-control','completedEpochs':12,'trainSeconds':1504.8439999999998,'secondsPerEpoch':1504.8439999999998/12,
                    'limitation':'Measured V8 detector only; no measured runtime claim for students, KD, INT8, V26-1280 or classifier variants.'},
                'policy':'wall+monotonic deadline, bounded owned subprocesses, incomplete epochs never marked finished, independent next jobs after failure/timeout, dependent PTQ skipped after failed student'},
        limitations=['Runtime and VRAM for new students/FX INT8 remain unmeasured on Windows.','FX PTQ may fail or reduce accuracy; no fallback silently changes method.','Small classifier does not remove GPU detector or Modal cold start.','Desktop executable allowlist is a historical reference; recheck current signed paths on host.'])
    if suite['budget']['estimatedMinutes']+60>720:raise ValueError('Plan estimates exceed12h including reserve')
    for relative in ('docs/models/student-assets-v7.json','docs/models/V7_EFFICIENCY.md','docs/models/licenses/torchvision-v0.23.0-LICENSE.txt'):
        path=root/relative
        if path.is_file():suite['sourceFiles'].append({'path':code_root+'/'+relative,'sha256':digest(path)})
    config_contract(config)
    return suite


def write_plan(root,out,**options):
    out=Path(out)
    if out.exists():raise FileExistsError('Use a new immutable prepared plan')
    suite=make_suite(root,**options);out.mkdir(parents=True)
    (out/'suite.json').write_text(json.dumps(suite,ensure_ascii=False,indent=2)+'\n')
    (out/'queue.json').write_text(json.dumps(suite['queueConfig'],ensure_ascii=False,indent=2)+'\n')
    remote=suite['codeRoot']+'/prepared-suites/'+out.name+'/suite.json'
    launch='"'+PYTHON+'" "'+suite['codeRoot']+'/infra/windows/launch_prepared_queue.py" --suite "'+remote+'"'
    scripts={
        'prepare.cmd':'@echo off\r\nsetlocal\r\nset CUDA_VISIBLE_DEVICES=-1\r\n"'+BASE_PYTHON+'" "'+suite['codeRoot']+'/infra/windows/prepare_v6_host.py" --suite "'+remote+'"\r\nexit /b %errorlevel%\r\n',
        'check.cmd':'@echo off\r\nsetlocal\r\nset CUDA_VISIBLE_DEVICES=-1\r\n'+launch+'\r\nexit /b %errorlevel%\r\n',
        'submit.cmd':'@echo off\r\n'+launch+' --start\r\n',
        'watch.cmd':'@echo off\r\n"'+PYTHON+'" "'+suite['codeRoot']+'/infra/windows/watch_queue.py" --queue "'+suite['queueConfig']['queue_dir']+'" --follow\r\n'}
    for name,text in scripts.items():(out/name).write_bytes(text.encode())
    return suite


def bundle(root,plan,out):
    root,plan,out=Path(root),Path(plan),Path(out)
    if out.exists():raise FileExistsError('Use a fresh immutable bundle')
    with tempfile.TemporaryDirectory()as temporary:
        source=Path(temporary)/'base.zip';cold_bundle(root,plan,source)
        with zipfile.ZipFile(source)as archive:contents={n:archive.read(n)for n in archive.namelist()}
    manifest=json.loads(contents.pop(MANIFEST))
    for name in ('docs/models/student-assets-v7.json','docs/models/V7_EFFICIENCY.md','docs/models/licenses/torchvision-v0.23.0-LICENSE.txt'):
        if (root/name).is_file():contents[safe_name(name)]=(root/name).read_bytes()
    manifest.update(preparationStatus='offline_v7_prepared_windows_unverified',training_started=False)
    manifest['files']={n:{'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}for n,data in sorted(contents.items())}
    contents[MANIFEST]=json.dumps(manifest,indent=2).encode();out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_DEFLATED)as archive:
        for name,data in sorted(contents.items()):
            info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=(stat.S_IFREG|0o644)<<16
            archive.writestr(info,data)
    checksum=digest(out);verify(out,checksum)
    verifier=out.with_suffix('.verify.py');verifier.write_bytes((root/'infra/windows/offline_bundle.py').read_bytes())
    receipt={'bundle':str(out),'sha256':checksum,'bytes':out.stat().st_size,'payloadFiles':len(manifest['files']),
        'verifier':str(verifier),'verifierSha256':digest(verifier),'localBundleVerificationPassed':True,
        'windowsV7Prepared':False,'windowsV7Checked':False,'trainingStarted':False,'modelWeightsIncluded':False}
    out.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');out.with_suffix(out.suffix+'.sha256').write_text(checksum+'\n')
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--out',type=Path,required=True);p.add_argument('--bundle-out',type=Path);p.add_argument('--code-root',default=CODE);p.add_argument('--queue-id',default=QUEUE)
    p.add_argument('--gpu-background-paths',type=Path);a=p.parse_args()
    historical=a.root/'docs/models/runs/v6-remaining-suite-20261028.json'
    allowed=json.loads(a.gpu_background_paths.read_text())if a.gpu_background_paths else json.loads(historical.read_text())['queueConfig']['allowed_gpu_process_paths']
    if isinstance(allowed,dict):allowed=allowed.get('allowed_gpu_process_paths',allowed.get('paths',[]))
    suite=write_plan(a.root,a.out,code_root=a.code_root,queue_id=a.queue_id,allowed_gpu_paths=allowed)
    result={'trainingStarted':False,'windowsConnected':False,'experiments':len(suite['queueConfig']['experiments']),
        'steps':sum(len(j['steps'])for j in suite['queueConfig']['experiments']),'budget':suite['budget']}
    if a.bundle_out:result['bundle']=bundle(a.root,a.out,a.bundle_out)
    print(json.dumps(result))


if __name__=='__main__':main()
