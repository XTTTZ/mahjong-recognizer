"""Prepare a fresh efficiency-focused 12h queue offline; never connect or submit."""
import argparse
import copy
import json
from pathlib import Path
from infra.windows.prepare_efficiency_suite_v7 import make_suite as previous_suite
from infra.windows.prepare_refinement_suite import PYTHON, BASE_PYTHON
from infra.windows.experiment_queue import config_contract

CODE='D:/mahjong/code-v8-portable-20261030'
QUEUE='v8-portable-12h-20261030'


def make_suite(root,code_root=CODE,queue_id=QUEUE):
    suite=previous_suite(root,code_root,queue_id)
    config=suite['queueConfig'];old={j['id'].removeprefix(queue_id+'-'):j for j in config['experiments']}
    old_checks={c['name']:c for c in suite['cpuPreflightCommands']}
    jobs=[];checks=[copy.deepcopy(old_checks[n]) for n in ('reviewed-train-backgrounds','lossless-crop-cache')]
    experiments=[
        ('v26-background-geometry-960','v26s-640',{'--resolution':'640'},90),
        ('v26-background-geometry-960','v26s-800',{'--resolution':'800'},110),
        ('v8-background-geometry','v8s-640',{'--resolution':'640'},75),
        ('v8-background-geometry','v8s-800',{'--resolution':'800'},90),
        ('student-ce','student-ce-context008',{'--context':'.08'},60),
        ('student-ce','student-ce-context002',{'--context':'.02'},60),
        ('student-ce','student-ce-context012',{'--context':'.12'},60),
    ]
    for parent,name,updates,estimate in experiments:
        old_id=queue_id+'-'+parent;new_id=queue_id+'-'+name
        job=json.loads(json.dumps(old[parent]).replace(old_id,new_id));check=json.loads(json.dumps(old_checks[parent]).replace(old_id,new_id))
        job.update(id=new_id,estimated_minutes=estimate,timeout_minutes=estimate+30)
        updates={**updates,'--epochs':'30','--max-train-minutes':str(estimate)}
        for argv in (job['steps'][0]['argv'],check['argv']):
            for flag,value in updates.items():argv[argv.index(flag)+1]=value
        job['steps'][0]['timeout_minutes']=estimate+5
        job['steps'][1]['timeout_minutes']=25
        check['name']=name;jobs.append(job);checks.append(check)
    config.update(experiments=jobs,deadline_hours=12,reserve_minutes=60)
    config_contract(config)
    suite.update(cpuPreflightCommands=checks,preparationStatus='offline_v8_prepared_windows_unverified',
        trainingStarted=False,submitted=False,newWindowAuthorized=False,
        budget={'hardHours':12,'reserveMinutes':60,'estimatedMinutes':sum(j['estimated_minutes'] for j in jobs),
                'estimatesAreProvisional':True,'epochCap':30,'policy':'Existing fail-next serial coordinator; hard deadline includes evaluation; no automatic production promotion'},
        priorControlNotRepeated='V7 detector960 and saved MobileNet224 remain comparison controls; student context008 is a matched30epoch control',
        calibrationScope='No automatic PTQ jobs: prior CPU INT8 quality regression must be diagnosed first',
        limitations=['Windows is off; host, files, GPU and actual batch memory remain unverified.',
                    'Export and benchmark existing V7 student weights before deciding to submit retraining.',
                    'No new independent photos or annotations are claimed.',
                    'Queue validation may explore operating points; promotion must recompute fixed production .35/.7.',
                    '12h is a ceiling, not a target to consume regardless of usefulness.'],
        nextStep='Transfer fresh bundle, run CPU preparation/check, recheck GPU process allowlist, then obtain explicit new-window submission authorization.')
    return suite


def write_plan(root,out):
    out=Path(out)
    if out.exists():raise FileExistsError('Use a fresh immutable directory')
    suite=make_suite(root);out.mkdir(parents=True)
    for name,value in [('suite.json',suite),('queue.json',suite['queueConfig'])]:
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    remote=CODE+'/prepared-suites/'+out.name+'/suite.json'
    launch=f'"{PYTHON}" "{CODE}/infra/windows/launch_prepared_queue.py" --suite "{remote}"'
    scripts={
        'prepare.cmd':f'@echo off\r\nsetlocal\r\nset CUDA_VISIBLE_DEVICES=-1\r\n"{BASE_PYTHON}" "{CODE}/infra/windows/prepare_v6_host.py" --suite "{remote}"\r\nexit /b %errorlevel%\r\n',
        'check.cmd':'@echo off\r\nsetlocal\r\nset CUDA_VISIBLE_DEVICES=-1\r\n'+launch+'\r\nexit /b %errorlevel%\r\n',
        'submit.cmd':'@echo off\r\n'+launch+' --start\r\n',
        'watch.cmd':f'@echo off\r\n"{PYTHON}" "{CODE}/infra/windows/watch_queue.py" --queue "{suite["queueConfig"]["queue_dir"]}" --follow\r\n',
    }
    for name,value in scripts.items():(out/name).write_bytes(value.encode())
    return suite


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    s=write_plan(Path(__file__).resolve().parents[2],a.out)
    print(json.dumps({'trainingStarted':False,'windowsConnected':False,'jobs':len(s['queueConfig']['experiments']),'budget':s['budget']}))
