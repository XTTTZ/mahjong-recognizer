"""Cache official RF-DETR1.11.2 assets with registry MD5 and recorded SHA256."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from urllib.request import Request,urlopen

ASSETS = {
 'medium':('rf-detr-medium.pth','https://storage.googleapis.com/rfdetr/medium_coco/checkpoint_best_regular.pth',404992918,'7223f764a87b863f02eb8d52bf0ce2ee'),
 'large':('rf-detr-large-2026.pth','https://storage.googleapis.com/rfdetr/rf-detr-large-2026.pth',135954129,'5cb72153541cbcb9aa6efa26222acc75'),
}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);p.add_argument('--architecture',choices=['medium','large','all'],default='all');a=p.parse_args();a.directory.mkdir(parents=True,exist_ok=True)
 for architecture in ASSETS if a.architecture=='all' else [a.architecture]:
  name,url,size,md5=ASSETS[architecture];dest=a.directory/name
  if dest.exists() and dest.stat().st_size==size and hashlib.md5(dest.read_bytes()).hexdigest()==md5:
   print(json.dumps({'event':'verified_cached','file':name}),flush=True);continue
  pieces=a.directory/(name+'.parts');pieces.mkdir(exist_ok=True);stride=8*1024*1024
  def get(start):
   end=min(size-1,start+stride-1);path=pieces/str(start)
   if path.exists() and path.stat().st_size==end-start+1:return path
   for retry in range(3):
    try:
     with urlopen(Request(url,headers={'Range':f'bytes={start}-{end}'}),timeout=90) as r:
      if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{size}':raise ValueError('Invalid range response')
      data=r.read(end-start+2)
     if len(data)!=end-start+1:raise ValueError('Truncated response')
     path.write_bytes(data);return path
    except Exception:
     if retry==2:raise
  with ThreadPoolExecutor(max_workers=8) as pool:chunks=list(pool.map(get,range(0,size,stride)))
  md=hashlib.md5();sh=hashlib.sha256();temporary=dest.with_suffix('.assembling')
  with temporary.open('wb') as out:
   for piece in chunks:
    data=piece.read_bytes();md.update(data);sh.update(data);out.write(data)
  if temporary.stat().st_size!=size or md.hexdigest()!=md5:raise ValueError('Registry checksum mismatch')
  temporary.replace(dest);record={'architecture':architecture,'file':name,'source':url,'registry':'rfdetr==1.11.2','bytes':size,'md5':md5,'sha256':sh.hexdigest()};(a.directory/(name+'.json')).write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)

if __name__=='__main__':main()
