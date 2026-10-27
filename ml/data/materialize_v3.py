"""Rebuild reviewed oracle-ROI COCO data directly from immutable source ZIP.

No training starts here. Retained images must have fixed cropPixelsXYWH and
byte/pixel hashes. Raw source photographs and annotations are never modified.
"""
from __future__ import annotations
import argparse,collections,hashlib,io,json,zipfile
from pathlib import Path,PurePosixPath
from PIL import Image,__version__ as pillow_version

def sha(data):return hashlib.sha256(data).hexdigest()
def bounds(box,width,height):
    x,y,w,h=box
    return w>0 and h>0 and x>=0 and y>=0 and x+w<=width and y+h<=height

def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--split-manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.out.exists() and any(args.out.iterdir()):raise SystemExit('Output must be a new empty directory; never mix data revisions')
    rows=[json.loads(l) for l in args.split_manifest.read_text().splitlines() if l.strip()];selected=[r for r in rows if r['split'] in ('train','valid','test')]
    if not selected:raise SystemExit('No retained rows')
    if len({r['source_id'] for r in rows})!=len(rows):raise SystemExit('Duplicate source ID')
    with zipfile.ZipFile(args.archive) as z:
        source={};cats=None
        for s in ('train','valid','test'):
            d=json.loads(z.read(s+'/_annotations.coco.json'));annotations=collections.defaultdict(list)
            for a in d['annotations']:annotations[a['image_id']].append(a)
            current=[c for c in d['categories'] if c['id']!=0]
            if cats is not None and current!=cats:raise ValueError('Category mismatch across original splits')
            cats=current
            for im in d['images']:source[s+'/'+im['file_name']]=(im,annotations[im['id']])
        if len(cats)!=43 or {c['id'] for c in cats}!=set(range(1,44)):raise ValueError('Expected explicit 43-category mapping, IDs 1..43')
        result={};groups=collections.defaultdict(set)
        for s in ('train','valid','test'):
            out=args.out/s;out.mkdir(parents=True,exist_ok=True);images=[];anns=[]
            for r in sorted((r for r in selected if r['split']==s),key=lambda r:sha(r['source_id'].encode())):
                uid=r['source_id'];member=PurePosixPath(uid)
                if member.is_absolute() or '..' in member.parts or '\\' in uid:raise ValueError('Unsafe source ZIP member')
                if not r.get('oracle_derived') or not r.get('group_id'):raise ValueError('Missing oracle/group provenance')
                if r.get('review_status')!='accepted_for_oracle_roi_research':raise ValueError('Retained row lacks accepted visual scope review')
                if uid not in source:raise ValueError('Unknown source ID')
                imrec,rawanns=source[uid];raw=z.read(uid)
                if sha(raw)!=r['sha256']:raise ValueError('Raw source SHA256 mismatch')
                x,y,w,h=r['cropPixelsXYWH']
                if any(isinstance(v,bool) or not isinstance(v,int) for v in (x,y,w,h)):raise ValueError('Crop must use integers')
                with Image.open(io.BytesIO(raw)) as im:
                    im=im.convert('RGB')
                    if im.size!=(imrec['width'],imrec['height']):raise ValueError('Unrepaired source dimensions mismatch')
                    if not bounds([x,y,w,h],im.width,im.height):raise ValueError('Crop out of bounds')
                    crop=im.crop((x,y,x+w,y+h));rgb_sha=sha(crop.tobytes());buffer=io.BytesIO();crop.save(buffer,format='PNG',compress_level=6);encoded=buffer.getvalue()
                if rgb_sha!=r['roi_rgb_sha256']:raise ValueError('ROI RGB pixel hash mismatch; inspect decoder/version, never silently accept')
                if sha(encoded)!=r['roi_sha256']:raise ValueError('ROI PNG hash mismatch; reproduce pinned Pillow/codec or explicitly version new bytes')
                name=sha(uid.encode())[:16]+'.png';(out/name).write_bytes(encoded);idx=len(images)
                images.append({'id':idx,'file_name':name,'width':w,'height':h,'source_id':uid,'group_id':r['group_id'],'source_sha256':r['sha256'],'roi_sha256':r['roi_sha256'],'roi_rgb_sha256':rgb_sha,'cropPixelsXYWH':r['cropPixelsXYWH'],'oracle_derived':True})
                for a in rawanns:
                    if not bounds(a['bbox'],imrec['width'],imrec['height']):raise ValueError('Source bbox outside exact bounds; quarantine required, no silent clipping')
                    bx,by,bw,bh=a['bbox'];box=[bx-x,by-y,bw,bh]
                    if not bounds(box,w,h):raise ValueError('Translated bbox outside crop; no silent clipping')
                    anns.append({'id':len(anns),'image_id':idx,'category_id':a['category_id'],'bbox':box,'area':bw*bh,'iscrowd':a.get('iscrowd',0)})
                groups[r['group_id']].add(s)
            path=out/'_annotations.coco.json';path.write_text(json.dumps({'images':images,'annotations':anns,'categories':cats},ensure_ascii=False),encoding='utf-8')
            result[s]={'images':len(images),'annotations':len(anns),'whole_hand_images_ge10':sum(sum(a['image_id']==im['id'] for a in anns)>=10 for im in images),'annotation_sha256':sha(path.read_bytes())}
        if any(len(s)>1 for s in groups.values()):raise ValueError('Group crosses retained splits')
    receipt={'oracle_derived':True,'scope':'research_training_and_auxiliary_validation_not_end_to_end','pillow_version':pillow_version,'split_manifest_sha256':sha(args.split_manifest.read_bytes()),'splits':result}
    (args.out/'materialization.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
