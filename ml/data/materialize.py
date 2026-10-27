"""Materialize a grouped COCO manifest with hardlinks/copies, without augmentation."""
from __future__ import annotations
import argparse, collections, hashlib, json, os, shutil
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--train-limit',type=int);p.add_argument('--valid-limit',type=int);p.add_argument('--test-limit',type=int);args=p.parse_args()
    records=[json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    cats={a['class_id']:a['class_name'] for r in records for a in r['annotations']};stats={}
    for split in ('train','valid','test'):
        dest=args.out/split;dest.mkdir(parents=True,exist_ok=True);images=[];annotations=[]
        limit=getattr(args,split+'_limit');chosen=sorted((r for r in records if r['research_split']==split),key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest())
        if limit is not None:chosen=chosen[:limit]
        for r in chosen:
            source=(args.root/r['relative_path']).resolve()
            if not source.is_relative_to(args.root.resolve()):raise ValueError('Path traversal')
            filename=hashlib.sha256(r['id'].encode()).hexdigest()[:16]+source.suffix.lower()
            target=dest/filename
            if not target.exists():
                try:os.link(source,target)
                except OSError:shutil.copy2(source,target)
            idx=len(images);images.append({'id':idx,'file_name':filename,'width':r['width'],'height':r['height'],'source_id':r['id'],'group_id':r['group_id']})
            for a in r['annotations']:
                b=a['bbox'];annotations.append({'id':len(annotations),'image_id':idx,'category_id':a['class_id'],'bbox':b,'area':b[2]*b[3],'iscrowd':a.get('iscrowd',0)})
        (dest/'_annotations.coco.json').write_text(json.dumps({'images':images,'annotations':annotations,'categories':[{'id':k,'name':v,'supercategory':'mahjong'} for k,v in sorted(cats.items())]},ensure_ascii=False))
        stats[split]={'images':len(images),'annotations':len(annotations)}
    print(json.dumps(stats))
if __name__=='__main__':main()
