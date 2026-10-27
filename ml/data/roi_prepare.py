"""Create annotation-derived target-region candidates for explicit visual review.

These crops use ground-truth boxes (an oracle) and must never be scored as real
end-to-end user capture/ROI detection performance. The crop alone is not proof
that every visible tile inside it was annotated.
"""
from __future__ import annotations
import argparse,json,math,statistics,hashlib
from pathlib import Path
from PIL import Image,ImageDraw,__version__ as pillow_version

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--candidates',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--margin-fraction',type=float,default=.1);args=p.parse_args()
    if not 0<=args.margin_fraction<=.1:raise SystemExit('Margin must be <=10% of median tile shorter side')
    args.out.mkdir(parents=True,exist_ok=True);rows={r['id']:r for r in map(json.loads,args.manifest.read_text().splitlines())};candidates=json.loads(args.candidates.read_text());records=[]
    for c in candidates:
        r=rows[c['source_id']];b=[a['bbox'] for a in r['annotations']];margin=args.margin_fraction*statistics.median(min(a[2],a[3]) for a in b)
        x0=max(0,math.floor(min(a[0] for a in b)-margin));y0=max(0,math.floor(min(a[1] for a in b)-margin));x1=min(r['width'],math.ceil(max(a[0]+a[2] for a in b)+margin));y1=min(r['height'],math.ceil(max(a[1]+a[3] for a in b)+margin));crop=[x0,y0,x1,y1]
        name=f'{c["index"]:03d}-'+hashlib.sha256(r['id'].encode()).hexdigest()[:12]+'.png';dest=args.out/name
        with Image.open(args.root/r['relative_path']) as im:
            im=im.convert('RGB').crop(crop);im.save(dest,compress_level=6)
        record={**c,'roi_xyxy':crop,'cropPixelsXYWH':[x0,y0,x1-x0,y1-y0],'roi_margin_fraction':args.margin_fraction,'roi_margin_pixels':margin,'roi_file':name,'roi_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'roi_width':x1-x0,'roi_height':y1-y0,'oracle_derived':True,'pillow_version':pillow_version}
        records.append(record)
    (args.out/'roi-candidates.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
    for page in range((len(records)+29)//30):
        sheet=Image.new('RGB',(1400,1500),'#eee');draw=ImageDraw.Draw(sheet)
        for slot,c in enumerate(records[page*30:(page+1)*30]):
            r=rows[c['source_id']];x=slot%5*280;y=slot//5*250;x0,y0,_,_=c['roi_xyxy']
            with Image.open(args.out/c['roi_file']) as im:
                scale=min(270/im.width,205/im.height);im=im.convert('RGB').resize((max(1,round(im.width*scale)),max(1,round(im.height*scale))));d=ImageDraw.Draw(im)
                for a in r['annotations']:
                    bx,by,w,h=a['bbox'];d.rectangle(((bx-x0)*scale,(by-y0)*scale,(bx+w-x0)*scale,(by+h-y0)*scale),outline='#ff2222',width=1)
                sheet.paste(im,(x+5,y+5))
            draw.text((x+5,y+212),f'{c["index"]:03d} {c["source_identity"]} n={c["tile_count"]}',fill='black');draw.text((x+5,y+228),f'{c["v2_split"]} ORACLE ROI',fill='black')
        sheet.save(args.out/f'roi-review-{page+1:02d}.jpg',quality=92)
    print(json.dumps({'roi_candidates':len(records),'ground_truth_derived':True,'review_required':True}))
if __name__=='__main__':main()
