"""Fetch hash-pinned public GitHub model assets atomically, without credentials or SDK uploads."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from urllib.parse import urlsplit
from urllib.request import Request, HTTPRedirectHandler, build_opener

FILES={'manifest.json','detector.onnx','classifier.onnx'}
REPOSITORY='/XTTTZ/mahjong-recognizer/releases/download/'
REDIRECT_HOSTS={'github.com','release-assets.githubusercontent.com','objects.githubusercontent.com'}
MAX_BYTES=256*1024*1024
MAX_SECONDS=300


def validate_spec(base_url, files):
    parsed=urlsplit(base_url)
    if (parsed.scheme!='https' or parsed.netloc!='github.com' or parsed.query or parsed.fragment or
        not parsed.path.startswith(REPOSITORY) or not re.fullmatch(r'[A-Za-z0-9_.-]+/?',parsed.path[len(REPOSITORY):]) or
        parsed.path[len(REPOSITORY):].strip('/') in {'.','..'}):
        raise ValueError('Use a public HTTPS release in the approved Mahjong GitHub repository')
    if not isinstance(files,dict) or set(files)!=FILES:
        raise ValueError('Exactly the manifest and both ONNX stage files must be pinned')
    normalized={}
    for name,item in files.items():
        if (not isinstance(item,dict) or not isinstance(item.get('sha256'),str) or
            not re.fullmatch('[0-9a-f]{64}',item['sha256']) or type(item.get('bytes'))is not int or
            not 0<item['bytes']<=MAX_BYTES):
            raise ValueError('Each public asset needs a full SHA256 and bounded positive byte count')
        normalized[name]={'sha256':item['sha256'],'bytes':item['bytes']}
    if sum(item['bytes']for item in normalized.values())>MAX_BYTES:
        raise ValueError('Model package exceeds public transfer budget')
    return base_url.rstrip('/')+'/',normalized


class PublicRedirects(HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        parsed=urlsplit(newurl)
        if parsed.scheme!='https' or parsed.hostname not in REDIRECT_HOSTS or parsed.username or parsed.password or parsed.port not in (None,443):
            raise ValueError('Public model download redirected outside HTTPS GitHub asset hosts')
        return super().redirect_request(req,fp,code,msg,headers,newurl)


def _open_url(request, timeout):
    return build_opener(PublicRedirects()).open(request,timeout=timeout)


def _sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def fetch_package(base_url, files, destination):
    """Verify all3 exact bytes, then publish one complete directory; never overwrite a partial target."""
    base_url,files=validate_spec(base_url,files)
    destination=Path(destination)
    if destination.exists():
        if destination.is_symlink() or not destination.is_dir() or set(p.name for p in destination.iterdir())!=FILES:
            raise FileExistsError('Existing model directory is not this exact frozen package')
        for name,item in files.items():
            p=destination/name
            if p.is_symlink() or not p.is_file() or p.stat().st_size!=item['bytes'] or _sha(p)!=item['sha256']:
                raise FileExistsError('Existing model directory differs from pinned bytes')
        return {'state':'verified_existing','files':files,'bytes':sum(x['bytes']for x in files.values()),'credentialsUsed':False}
    destination.parent.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    temporary=Path(tempfile.mkdtemp(prefix='.'+destination.name+'.fetch-',dir=destination.parent))
    try:
        for name,item in sorted(files.items()):
            request=Request(base_url+name,headers={'User-Agent':'MahjongPortableModel/1','Accept':'application/octet-stream'})
            with _open_url(request,timeout=30)as response, (temporary/name).open('xb')as output:
                final=urlsplit(response.geturl())
                if final.scheme!='https' or final.hostname not in REDIRECT_HOSTS or final.username or final.password or final.port not in (None,443):
                    raise ValueError('Public model response came from an unapproved origin')
                digest=hashlib.sha256();count=0
                while True:
                    if time.monotonic()-started>MAX_SECONDS:raise TimeoutError('Public model transfer exceeded its deadline')
                    block=response.read(1024*1024)
                    if not block:break
                    count+=len(block)
                    if count>item['bytes']:raise ValueError('Public asset exceeds pinned byte count: '+name)
                    output.write(block);digest.update(block)
                if count!=item['bytes'] or digest.hexdigest()!=item['sha256']:
                    raise ValueError('Public asset SHA256/byte count mismatch: '+name)
        if destination.exists():raise FileExistsError('Model destination appeared during transfer')
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary,ignore_errors=True)
        raise
    return {'state':'downloaded_verified','files':files,'bytes':sum(x['bytes']for x in files.values()),
            'elapsedSeconds':round(time.monotonic()-started,3),'credentialsUsed':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec-env',default='MAHJONG_CPU_PUBLIC_MODEL_SPEC')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();spec=json.loads(os.environ[a.spec_env])
    print(json.dumps(fetch_package(spec['baseUrl'],spec['files'],a.out),sort_keys=True))

if __name__=='__main__':main()
