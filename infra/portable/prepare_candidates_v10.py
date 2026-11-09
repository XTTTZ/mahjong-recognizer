"""Fetch byte-verified inactive source runs and export existing V10 pairs; never train."""
import argparse
import hashlib
import json
from pathlib import Path, PureWindowsPath
import shutil
import subprocess

from infra.portable.candidate_export import export

PAIRS = (
    ('nano800-v7ce224', 'nano800', 'v7ce224'),
    ('v26s640-eff320', 'v26s640', 'eff320'),
    ('nano800-eff320', 'nano800', 'eff320'),
    ('nano800-kd224', 'nano800', 'kd224'),
    ('nano800-ce192', 'nano800', 'ce192'),
)


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def fetch_sources(root, work, host='win'):
    """Separate source bytes from export outputs, with selected-report checkpoint hash pins."""
    root, work = Path(root), Path(work)
    source_root = work/'sources'
    if source_root.exists(): raise FileExistsError('Use a fresh immutable sources directory')
    source_root.mkdir(parents=True)
    final = json.loads((root/'docs/models/runs/v9-final-20261109.json').read_text())
    rows = {r['name']:r for r in final['experiments']}
    requested = [('nano800','v8n-800','selectedCheckpoint','checkpointSha256'),
                 ('v26s640','v26s-640','selectedCheckpoint','checkpointSha256'),
                 ('kd224','student-kd224-a02','selectedClassifier','classifierCheckpointSha256'),
                 ('ce192','student-ce192','selectedClassifier','classifierCheckpointSha256'),
                 ('eff320','v8n-800','selectedClassifier','classifierCheckpointSha256')]
    sources=[]
    for name, key, checkpoint_key, hash_key in requested:
        checkpoint=PureWindowsPath(rows[key][checkpoint_key])
        original=checkpoint.parent.parent.parent if checkpoint.parent.name=='weights' else checkpoint.parent
        relative=checkpoint.relative_to(original).as_posix(); local=source_root/name; local.mkdir()
        files=['run.json','status.json',relative,relative+'.receipt.json']
        for file in files:
            target=local/file; target.parent.mkdir(parents=True,exist_ok=True)
            subprocess.run(['scp',host+':'+(original/PureWindowsPath(file)).as_posix(),str(target)],check=True)
        if sha(local/relative)!=rows[key][hash_key]: raise ValueError('Fetched checkpoint differs from completed selected validation')
        selected=json.loads((root/'artifacts/runs/v9-final-20261109/metadata'/('v9-portable-12h-20261109-'+key)/'validation/report.json').read_text())
        run_hash=selected['runSha256'] if checkpoint_key=='selectedCheckpoint' else selected['classifierRunSha256']
        if sha(local/'run.json')!=run_hash:raise ValueError('Fetched run metadata differs from completed selected validation')
        sources.append({'name':name,'localDirectory':str(local.resolve()),'originalDirectory':original.as_posix(),
                        'checkpointRelative':relative,'filesSha256':{n:sha(local/n)for n in files}})
        print(json.dumps({'sourceFetched':name,'checkpointHashMatched':True}),flush=True)
    old=root/'artifacts/portable-students/20261030/student-ce'; local=source_root/'v7ce224'; local.mkdir()
    files=['run.json','status.json','last.pt','last.pt.receipt.json']
    for name in files: shutil.copyfile(old/name,local/name)
    selected=json.loads((old/'validation/report.json').read_text())
    if sha(local/'run.json')!=selected['classifierRunSha256'] or sha(local/'last.pt')!=selected['classifierCheckpointSha256']:
        raise ValueError('Existing V7 CE source differs from its signed full-image selection')
    sources.append({'name':'v7ce224','localDirectory':str(local.resolve()),'originalDirectory':'D:/mahjong/runs/v7-efficiency-12h-20261029-r2-student-ce',
                    'checkpointRelative':'last.pt','filesSha256':{n:sha(local/n)for n in files}})
    transport={'schemaVersion':1,'sources':sources,'originalBytesPreserved':True,'fetchedFrom':'ssh '+host,'trainingStarted':False}
    (work/'source-transport.json').write_text(json.dumps(transport,indent=2)+'\n')
    return transport


def verify_sources(work):
    transport=Path(work)/'source-transport.json'; value=json.loads(transport.read_text())
    if value.get('schemaVersion')!=1 or not value.get('originalBytesPreserved'): raise ValueError('Original source byte map required')
    sources={r['name']:r for r in value['sources']}
    if set(sources)!={'nano800','v26s640','kd224','ce192','eff320','v7ce224'}: raise ValueError('Use the frozen six V10 source stages')
    for source in sources.values():
        directory=Path(source['localDirectory']).resolve()
        for relative, expected in source['filesSha256'].items():
            path=(directory/relative).resolve()
            if not path.is_relative_to(directory) or sha(path)!=expected: raise ValueError('Frozen source changed')
    return sources


def prepare(root, work, output_root, *, run_exports=False):
    root,work,output_root=map(Path,(root,work,output_root))
    sources=verify_sources(work); outputs={}; ready=[]
    from ml.inference.candidate_onnx_runtime import verify_candidate_directory
    for name,det,cls in PAIRS:
        out=output_root/('v10-'+name+'-fp32-20261109')
        detector=sources[det];classifier=sources[cls]
        dr=Path(detector['localDirectory']);cr=Path(classifier['localDirectory'])
        args=dict(dataset=root/'artifacts/data/roboflow/mahjong-lfhxf-v2/prepared-v4-main',detector_run=dr/'run.json',
                  detector_checkpoint=dr/detector['checkpointRelative'],classifier_run=cr/'run.json',
                  classifier_checkpoint=cr/classifier['checkpointRelative'],out=out,source_transport=work/'source-transport.json')
        if not json.loads((cr/'run.json').read_text()).get('finishedAt'): args['classifier_terminal_receipt']=cr/'status.json'
        detector_reuse=next((path for pair,path in outputs.items()if next(p[1]for p in PAIRS if p[0]==pair)==det),None)
        classifier_reuse=next((path for pair,path in outputs.items()if next(p[2]for p in PAIRS if p[0]==pair)==cls),None)
        if out.exists():
            _,manifest=verify_candidate_directory(out)
            # Reusing an immutable directory requires exactly the same original run/checkpoint identity.
            for stage,source in [('detector',detector),('classifier',classifier)]:
                evidence=manifest['sourceEvidence'][stage]
                if evidence['runSha256']!=source['filesSha256']['run.json'] or evidence['checkpointSha256']!=source['filesSha256'][source['checkpointRelative']]:
                    raise ValueError('Existing profile belongs to another source')
        elif run_exports:
            manifest=export(**args,reuse_detector=detector_reuse,reuse_classifier=classifier_reuse)
        else:
            ready.append({'name':name,'out':str(out),'state':'not_exported'});continue
        outputs[name]=out
        row={'name':name,'out':str(out),'state':'exported_verified','filesBytes':manifest['filesBytes'],
             'filesSha256':manifest['filesSha256'],'sourceEvidence':manifest['sourceEvidence'],'reusedStages':manifest.get('reusedStages',{})}
        ready.append(row)
        (work/'candidate-index.json').write_text(json.dumps({'schemaVersion':1,'trainingStarted':False,'profiles':ready},indent=2)+'\n')
        print(json.dumps({k:v for k,v in row.items()if k!='sourceEvidence'}),flush=True)
    return ready


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-dir',type=Path,default=Path('artifacts/portable-v10-20261109'))
    p.add_argument('--output-root',type=Path,default=Path('artifacts/models/releases'))
    p.add_argument('--fetch-win',action='store_true');p.add_argument('--export',action='store_true')
    a=p.parse_args();root=Path(__file__).resolve().parents[2]
    if a.fetch_win:fetch_sources(root,a.work_dir)
    result=prepare(root,a.work_dir,a.output_root,run_exports=a.export)
    print(json.dumps({'trainingStarted':False,'profiles':len(result),'exportsRequested':a.export}))

if __name__=='__main__':main()
