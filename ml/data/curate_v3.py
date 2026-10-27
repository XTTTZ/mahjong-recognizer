"""Conservative source-whitelist/de-augmentation candidates, preserving v2 splits.

This command first emits candidates/contact sheets. A tracked human review file
then explicitly chooses pass/quarantine. It never equates filename or box-union
geometry with exhaustive annotation validity.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,math,re
from pathlib import Path
from PIL import Image,ImageDraw

def load(path):return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
def family(r):
    n=r['file_name'].lower()
    if n.startswith('s__'):return 'phone_blue_S__'
    if re.match(r'\d+_\d{3}_',n):return 'phone_keyboard_rotated'
    if n.startswith('scene'):return 'broadcast_scene'
    if 'screenshot' in n:return 'screenshot'
    if 'home-mj' in n:return 'home_mj'
    if n.startswith('hand_'):return 'mixed_web_hand'
    if re.match(r'r?[1-9][mpsz]_(?:add|sub|\d)',n) or n.startswith('back_') or n.startswith('all_'):return 'preaugmented_single_or_layout'
    if re.match(r'img[_-]?\d+',n):return 'unreviewed_camera'
    return 'unreviewed_web_or_other'
def source_identity(r):
    if family(r)=='phone_keyboard_rotated':return 'keyboard_'+re.match(r'(\d+)_',r['file_name']).group(1)
    return r['source_key']
def image_rank(r,root):
    n=r['file_name'];angle=re.match(r'\d+_(\d{3})_',n)
    angle_penalty=min(int(angle.group(1)),360-int(angle.group(1))) if angle else 0
    with Image.open(root/r['relative_path']) as im:
        im.draft('RGB',(256,256));im=im.convert('RGB');im.thumbnail((160,160));pixels=list(im.getdata());white=sum(min(px)>245 for px in pixels)/len(pixels)
    return angle_penalty,round(white,3),hashlib.sha256(r['id'].encode()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True);rows=load(args.audit/'manifest.jsonl');candidates=collections.defaultdict(list);familycounts=collections.Counter();risky_groups=set()
    for r in rows:
        f=family(r);familycounts[f]+=1
        if f in {'broadcast_scene','screenshot','mixed_web_hand','unreviewed_web_or_other'} or r['tile_count']>18:risky_groups.add(r['group_id'])
        if r['exclusion_reason'] is None and f in {'phone_blue_S__','phone_keyboard_rotated'} and 1<=r['tile_count']<=18:candidates[source_identity(r)].append(r)
    selected=[]
    for source,items in sorted(candidates.items()):
        ranked=sorted(items,key=lambda r:image_rank(r,args.root));r=ranked[0]
        selected.append({'index':len(selected),'source_id':r['id'],'source_identity':source,'sha256':r['sha256'],'group_id':r['group_id'],'v2_split':r['research_split'],'family':family(r),'tile_count':r['tile_count'],'variant_count':len(items),'group_has_risky_source':r['group_id'] in risky_groups,'review_status':'pending_contact_scope_review','review_note':None})
    (args.out/'candidates.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2)+'\n');(args.out/'source_family_counts.json').write_text(json.dumps(familycounts,indent=2)+'\n')
    lookup={r['id']:r for r in rows}
    for page in range((len(selected)+29)//30):
        sheet=Image.new('RGB',(1400,1500),'#eeeeee');draw=ImageDraw.Draw(sheet)
        for slot,c in enumerate(selected[page*30:(page+1)*30]):
            r=lookup[c['source_id']];x=slot%5*280;y=slot//5*250
            with Image.open(args.root/r['relative_path']) as im:
                im=im.convert('RGB');scale=min(270/im.width,205/im.height);im=im.resize((max(1,round(im.width*scale)),max(1,round(im.height*scale))))
                d=ImageDraw.Draw(im)
                for a in r['annotations']:
                    bx,by,w,h=a['bbox'];d.rectangle((bx*scale,by*scale,(bx+w)*scale,(by+h)*scale),outline='#ff2222',width=1)
                sheet.paste(im,(x+5,y+5))
            draw.text((x+5,y+212),f'{c["index"]:03} {c["source_identity"]} n={c["tile_count"]}',fill='black');draw.text((x+5,y+228),f'{c["v2_split"]} riskyGroup={c["group_has_risky_source"]}',fill='black')
        sheet.save(args.out/f'candidate-{page+1:02d}.jpg',quality=92)
    print(json.dumps({'candidates':len(selected),'candidate_families':dict(collections.Counter(c['family'] for c in selected)),'candidate_splits':dict(collections.Counter(c['v2_split'] for c in selected)),'risky_group_candidates':sum(c['group_has_risky_source'] for c in selected)}))
if __name__=='__main__':main()
