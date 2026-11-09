"""Stage bounded4MiB chunks for model/data benchmark uploads, preserving every source byte."""
import hashlib,json
from pathlib import Path

ASSEMBLER='''from pathlib import Path
import hashlib,json
source=Path('/matrix-upload');target=Path('/work');target.mkdir(exist_ok=True)
for item in json.loads((source/'index.json').read_text())['files']:
    out=target/item['name'];out.parent.mkdir(parents=True,exist_ok=True);digest=hashlib.sha256()
    with out.open('wb')as f:
        for sha in item['parts']:
            data=(source/(sha+'.chunk')).read_bytes()
            if hashlib.sha256(data).hexdigest()!=sha:raise ValueError('Matrix chunk changed')
            f.write(data);digest.update(data)
    if out.stat().st_size!=item['bytes']or digest.hexdigest()!=item['sha256']:raise ValueError('Matrix assembled file changed')
print('Matrix models and150validation photos assembled with exactSHA256 checks')
'''

def prepare(profiles,valid_dir,cache):
    sources=[]
    for row in profiles:
        for name in ('manifest.json','detector.onnx','classifier.onnx'):
            sources.append(('models/'+row['name']+'/'+name,Path(row['directory'])/name))
    sources.extend(('validation/'+p.name,p)for p in sorted(Path(valid_dir).iterdir())if p.is_file())
    files=[]
    for name,path in sources:
        if path.is_symlink()or '..' in Path(name).parts:raise ValueError('Unsafe upload source')
        with path.open('rb')as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
        files.append({'name':name,'path':path,'sha256':sha,'bytes':path.stat().st_size})
    identity=hashlib.sha256(json.dumps([{k:v for k,v in r.items()if k!='path'}for r in files],sort_keys=True).encode()).hexdigest()
    out=Path(cache)/identity;out.mkdir(parents=True,exist_ok=True)
    for row in files:
        parts=[]
        with row.pop('path').open('rb')as f:
            while data:=f.read(4*1024*1024):
                sha=hashlib.sha256(data).hexdigest();p=out/(sha+'.chunk')
                if p.exists()and hashlib.sha256(p.read_bytes()).hexdigest()!=sha:raise ValueError('Cachedchunk tampered')
                if not p.exists():p.write_bytes(data)
                parts.append(sha)
        row['parts']=parts
    (out/'index.json').write_text(json.dumps({'files':files},indent=2)+'\n');(out/'assemble.py').write_text(ASSEMBLER)
    return out
