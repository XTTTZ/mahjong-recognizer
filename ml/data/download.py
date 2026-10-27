"""Download a signed existing-version export; never print/store the secret URL.

Set MAHJONG_DATASET_URL in the invoking process, outside repository/config logs.
The archive is downloaded atomically, checked as a ZIP, then safely extracted.
"""
from __future__ import annotations
import argparse,hashlib,json,os,time,urllib.request,zipfile
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--source',default='roboflow:mahjong-anaun/mahjong-lfhxf/2');args=p.parse_args()
    url=os.environ.get('MAHJONG_DATASET_URL')
    if not url:raise SystemExit('Set MAHJONG_DATASET_URL securely before invocation')
    args.out.mkdir(parents=True,exist_ok=True);archive=args.out/'coco.zip';partial=args.out/'coco.zip.part'
    if not archive.exists():
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'mahjong-data-audit/1.0'}),timeout=120) as r,partial.open('wb') as f:
                total=0;last=time.monotonic()
                while b:=r.read(1024*1024):
                    f.write(b);total+=len(b)
                    if time.monotonic()-last>30:print(f'Downloaded {total//1024**2} MiB',flush=True);last=time.monotonic()
            partial.rename(archive)
        except Exception as exc:
            # urllib exceptions may contain URL/credentials; redact them.
            raise SystemExit(f'Download failed ({type(exc).__name__}); renew export link or check network, then retry') from None
    digest=hashlib.sha256()
    with archive.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):digest.update(b)
    dest=(args.out/'original').resolve()
    with zipfile.ZipFile(archive) as z:
        bad=z.testzip()
        if bad:raise SystemExit('ZIP checksum failed')
        for info in z.infolist():
            if not (dest/info.filename).resolve().is_relative_to(dest):raise SystemExit('Unsafe ZIP member')
        z.extractall(dest)
    metadata={'source':args.source,'format':'coco','bytes':archive.stat().st_size,'sha256':digest.hexdigest()}
    (args.out/'archive.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata))
if __name__=='__main__':main()
