"""Complete an offline v6 plan on its Windows host using CPU only; never submit."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json,os,subprocess,sys
from pathlib import Path
if __package__ in {None,''}:sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from infra.windows.launch_prepared_queue import last_json,validate_suite
from infra.windows.experiment_queue import read_json,write_json


def prepare(suite):
    if os.name!='nt':raise RuntimeError('Run host preparation on the Windows training host')
    environment=os.environ.copy();environment.update(CUDA_VISIBLE_DEVICES='-1',PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
    if not Path(suite['pythonPaths'][0]).is_file():
        subprocess.run(suite['cloneCommand'],check=True,env=environment,timeout=1200)
    # Check sources, parent assets, environment pins and pip before writing data.
    validate_suite(suite,run_commands=False)
    label=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory=Path('D:/mahjong/readiness')/('v6-prepare-'+label);directory.mkdir(parents=True,exist_ok=False)
    status={'trainingStarted':False,'state':'preparing','steps':[]};write_json(directory/'status.json',status)
    try:
        for command in suite['preparationCommands']:
            argv=command['argv'];target=Path(argv[argv.index('--out')+1])
            if target.exists():
                if not(target/'prepared.json').is_file():raise RuntimeError('Partial derivative retained; inspect before choosing a fresh path: '+str(target))
                argv=argv+['--preflight-only']
            with (directory/(command['name']+'.stdout.log')).open('wb')as output,(directory/(command['name']+'.stderr.log')).open('wb')as error:
                result=subprocess.run(argv,cwd=suite['codeRoot'],env=environment,stdout=output,stderr=error,timeout=3600,check=False)
            if result.returncode:raise RuntimeError('CPU derivative failed: '+command['name'])
            status['steps'].append({'name':command['name'],'state':'succeeded','argv':argv});write_json(directory/'status.json',status)
        # Force CPU visibility for subprocess-only preflights; no global/system env change.
        original=os.environ.get('CUDA_VISIBLE_DEVICES');os.environ['CUDA_VISIBLE_DEVICES']='-1'
        try:receipt=validate_suite(suite)
        finally:
            if original is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
            else:os.environ['CUDA_VISIBLE_DEVICES']=original
        write_json(directory/'check-receipt.json',receipt)
        status.update(state='cpu_host_prepared_and_checked',finishedAt=datetime.now(timezone.utc).isoformat());write_json(directory/'status.json',status)
        return {'state':status['state'],'trainingStarted':False,'receiptDirectory':str(directory),'next':'User may later explicitly run submit.cmd; preparation never calls --start.'}
    except Exception as error:
        status.update(state='failed',error=str(error));write_json(directory/'status.json',status);raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--suite',type=Path,required=True);a=p.parse_args()
    print(json.dumps(prepare(read_json(a.suite))))

if __name__=='__main__':main()
