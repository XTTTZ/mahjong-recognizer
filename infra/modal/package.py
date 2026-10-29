"""Chunk immutable model files for bounded parallel uploads, preserving exact bytes."""
import hashlib
import json
from pathlib import Path
import shutil


def prepare_source(source, cache_root):
    """Freeze source bytes before a long GPU image build, outside live edits."""
    source = Path(source)
    files = {str(p.relative_to(source)): p.read_bytes() for p in sorted(source.rglob('*'))
             if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    digest = hashlib.sha256()
    for name, data in files.items():
        digest.update(name.encode()); digest.update(b'\0'); digest.update(hashlib.sha256(data).digest())
    out = Path(cache_root)/digest.hexdigest()
    if out.exists():
        if any(not (out/name).is_file() or (out/name).read_bytes() != data for name, data in files.items()):
            raise ValueError('Frozen source cache changed')
        return out
    out.mkdir(parents=True)
    for name, data in files.items():
        path = out/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
    return out

ASSEMBLER = '''from pathlib import Path
import hashlib,json,shutil
root=Path('/model-upload'); target=Path('/model'); target.mkdir(exist_ok=True)
for entry in json.loads((root/'upload-index.json').read_text())['files']:
    destination=target/entry['name']; temporary=destination.with_suffix(destination.suffix+'.assembling')
    digest=hashlib.sha256(); count=0
    with temporary.open('wb') as stream:
        for part in entry['parts']:
            data=(root/part['name']).read_bytes()
            if hashlib.sha256(data).hexdigest()!=part['sha256']: raise ValueError('Upload chunk hash mismatch')
            stream.write(data); digest.update(data); count+=len(data)
    if count!=entry['bytes'] or digest.hexdigest()!=entry['sha256']: raise ValueError('Assembled model hash mismatch')
    temporary.replace(destination)
print('All frozen model files assembled and SHA256 verified')
'''


def prepare_upload(model_dir, cache_root, chunk_bytes=4*1024*1024):
    source, cache = Path(model_dir).resolve(), Path(cache_root)
    if chunk_bytes < 1024:
        raise ValueError('Chunk size too small')
    files = []
    for file in sorted(source.iterdir()):
        if not file.is_file() or file.is_symlink():
            raise ValueError('A frozen model package must contain regular model-local files only')
        with file.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        files.append({'name': file.name, 'bytes': file.stat().st_size, 'sha256': digest})
    if not files:
        raise ValueError('Empty model package')
    identity = hashlib.sha256(json.dumps({'files': files, 'chunkBytes': chunk_bytes}, sort_keys=True).encode()).hexdigest()
    out = cache / identity
    index_file = out/'upload-index.json'
    if index_file.exists():
        index = json.loads(index_file.read_text())
        if [{k: entry[k] for k in ('name','bytes','sha256')} for entry in index['files']] != files:
            raise ValueError('Cached upload index differs from its model package')
        for entry in index['files']:
            for part in entry['parts']:
                path = out/part['name']
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != part['sha256']:
                    raise ValueError('Cached upload chunk changed; rebuild its cache directory')
        if (out/'assemble.py').read_text() != ASSEMBLER:
            raise ValueError('Cached assembler changed; rebuild its cache directory')
        return out
    if out.exists():
        # Only a prior incomplete cache in this task-owned namespace is removed.
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for number, entry in enumerate(files):
        parts=[]
        with (source/entry['name']).open('rb') as stream:
            while block := stream.read(chunk_bytes):
                name=f'{number:03d}-{len(parts):05d}.chunk'
                (out/name).write_bytes(block)
                parts.append({'name':name,'sha256':hashlib.sha256(block).hexdigest()})
        entry['parts']=parts
    (out/'assemble.py').write_text(ASSEMBLER)
    index_file.write_text(json.dumps({'schemaVersion':1,'files':files},indent=2))
    return out
