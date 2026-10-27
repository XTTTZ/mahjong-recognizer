"""Audit an immutable Roboflow COCO export and make a source-grouped split.

Never edits the export. Dependencies: Pillow, numpy. The split is a local research
holdout, not proof of generalization to independent real-world phone photos.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, math, re, statistics
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np

SPLITS = ('train', 'valid', 'test')

class UnionFind:
    def __init__(self, n): self.p=list(range(n))
    def find(self, a):
        while self.p[a]!=a:
            self.p[a]=self.p[self.p[a]]; a=self.p[a]
        return a
    def union(self, a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.p[max(a,b)]=min(a,b)

class BKTree:
    def __init__(self):self.root=None
    def add(self,value,idx):
        if self.root is None:self.root=[value,[idx],{}];return
        node=self.root
        while True:
            d=(value^node[0]).bit_count()
            if d==0:node[1].append(idx);return
            if d not in node[2]:node[2][d]=[value,[idx],{}];return
            node=node[2][d]
    def query(self,value,radius):
        stack=[self.root] if self.root else []
        while stack:
            n=stack.pop(); d=(value^n[0]).bit_count()
            if d<=radius:
                for idx in n[1]:yield idx,d
            stack.extend(v for k,v in n[2].items() if d-radius<=k<=d+radius)

def source_key(name):
    # Roboflow format suffixes do not identify an independent original photo.
    base=Path(name).name.lower()
    base=re.sub(r'\.rf\.[a-z0-9]+(?:\.[a-z0-9]+)?$','',base)
    base=re.sub(r'[_\.]?(jpe?g|png|webp)$','',base)
    return base

def pixel_info(path):
    raw=path.read_bytes()
    with Image.open(path) as im:
        im.load(); size=im.size; orientation=im.getexif().get(274); im=ImageOps.exif_transpose(im).convert('RGB')
        tiny=np.asarray(im.resize((9,8)).convert('L'),dtype=np.int16)
        bits=tiny[:,1:]>tiny[:,:-1]
        dh=0
        for bit in bits.flat:dh=(dh<<1)|int(bit)
        color=np.asarray(im.resize((4,4)),dtype=np.float32)/255
        mean=np.asarray(im.resize((32,32)),dtype=np.float32).mean(axis=(0,1)).tolist()
        return {'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'actual_width':size[0], 'actual_height':size[1], 'dhash':format(dh,'016x'),'color':color.flatten().round(4).tolist(),'mean_rgb':mean,'exif_orientation':orientation}

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--near-radius',type=int,default=4);p.add_argument('--sample',type=int,default=240);args=p.parse_args()
    root=args.dataset.resolve();out=args.out;out.mkdir(parents=True,exist_ok=True)
    records=[];issues=[];cats={};annotation_hashes={}
    for split in SPLITS:
        annotation=root/split/'_annotations.coco.json'
        if not annotation.exists():continue
        annotation_hashes[split]=hashlib.sha256(annotation.read_bytes()).hexdigest()
        data=json.loads(annotation.read_text());localcats={c['id']:c['name'] for c in data['categories']}
        if cats and cats!=localcats:raise ValueError('Category maps differ between splits')
        cats=localcats;byimage=collections.defaultdict(list)
        for ann in data['annotations']:byimage[ann['image_id']].append(ann)
        for im in data['images']:
            path=root/split/im['file_name']; ann=byimage[im['id']];uid=f'{split}/{im["file_name"]}'
            rec={'id':uid,'original_split':split,'file_name':im['file_name'],'relative_path':path.relative_to(root).as_posix(),'width':im['width'],'height':im['height'],'source_key':source_key(im['file_name']),'annotations':[],'issues':[]}
            for a in ann:
                box=a.get('bbox',[]);cls=a['category_id'];reason=[]
                if cls not in cats:reason.append('unknown_class')
                if len(box)!=4 or not all(math.isfinite(x) for x in box):reason.append('invalid_box')
                elif box[2]<=0 or box[3]<=0:reason.append('nonpositive_box')
                elif box[0]<-1 or box[1]<-1 or box[0]+box[2]>im['width']+1 or box[1]+box[3]>im['height']+1:reason.append('out_of_bounds_box')
                rec['issues'].extend(reason);rec['annotations'].append({'class_id':cls,'class_name':cats.get(cls),'bbox':box,'iscrowd':a.get('iscrowd',0)})
            sig=[(a['class_id'],*(round(float(x),3) for x in a['bbox'])) for a in rec['annotations']]
            rec['annotation_signature']=hashlib.sha256(json.dumps(sorted(sig)).encode()).hexdigest()
            if len(sig)!=len(set(sig)):rec['issues'].append('duplicate_annotation')
            try:
                rec.update(pixel_info(path))
                if (rec['actual_width'],rec['actual_height'])!=(im['width'],im['height']):rec['issues'].append('dimension_mismatch')
            except Exception as exc:rec['issues'].append('unreadable_image');rec['error']=str(exc)
            rec['tile_count']=len(ann);records.append(rec)
            if rec['issues']:issues.append({'id':uid,'issues':sorted(set(rec['issues']))})
    uf=UnionFind(len(records));bysha={};sha_members=collections.defaultdict(list);bysource={};tree=BKTree();near_pairs=[]
    for i,r in enumerate(records):
        if 'sha256' not in r:continue
        sha_members[r['sha256']].append(i)
        if r['sha256'] in bysha:uf.union(i,bysha[r['sha256']])
        else:bysha[r['sha256']]=i
        if r['source_key'] in bysource:uf.union(i,bysource[r['source_key']])
        else:bysource[r['source_key']]=i
        value=int(r['dhash'],16);aspect=r['width']/r['height'];color=np.asarray(r['color'])
        for j,d in tree.query(value,args.near_radius):
            q=records[j];aspect_delta=abs(math.log(aspect/(q['width']/q['height'])))
            color_delta=float(np.abs(color-np.asarray(q['color'])).mean())
            # Conservative geometric similarity groups; every candidate is auditable.
            if aspect_delta<0.05 and color_delta<0.05:
                same_labels=collections.Counter(a['class_id'] for a in r['annotations'])==collections.Counter(a['class_id'] for a in q['annotations'])
                uf.union(i,j)
                near_pairs.append({'a':j,'b':i,'hamming':d,'color_delta':round(color_delta,5),'grouped':True,'label_disagreement':not same_labels})
        tree.add(value,i)
    groups=collections.defaultdict(list)
    for i,r in enumerate(records):groups[uf.find(i)].append(i)
    crossing=[];conflicts=[];dedup=[];excluded=[]
    group_ids={}
    for members in groups.values():
        gid=hashlib.sha256('\n'.join(sorted(records[i]['id'] for i in members)).encode()).hexdigest()[:20]
        for i in members:records[i]['group_id']=gid
        group_ids[gid]=members
        if len({records[i]['original_split'] for i in members})>1:crossing.append(gid)
    for sha,members in sha_members.items():
        if len({records[i]['annotation_signature'] for i in members})>1:
            conflicts.append([records[i]['id'] for i in members]);excluded.extend(members)
        elif len(members)>1:dedup.extend(members[1:])
    excluded_set=set(excluded)|{i for i,r in enumerate(records) if set(r['issues'])&{'unreadable_image','invalid_box','nonpositive_box','unknown_class','dimension_mismatch','out_of_bounds_box','duplicate_annotation'}}
    keep_set=set(range(len(records)))-excluded_set-set(dedup)
    split_counts=collections.Counter();class_counts=collections.defaultdict(collections.Counter)
    for gid,members in sorted(group_ids.items()):
        # Deterministic source-group hash. Whole source groups, not individual images.
        val=int(hashlib.sha256(('mahjong-v2-local-v1:'+gid).encode()).hexdigest()[:8],16)/2**32
        split='train' if val<0.80 else ('valid' if val<0.90 else 'test')
        for i in members:
            r=records[i];r['research_split']=split if i in keep_set else 'excluded'
            r['exclusion_reason']='conflicting_or_invalid' if i in excluded_set else ('exact_duplicate' if i in set(dedup) else None)
            if i in keep_set:
                split_counts[split]+=1;class_counts[split].update(a['class_name'] for a in r['annotations'])
    counter=collections.Counter(a['class_name'] for r in records for a in r['annotations'])
    report={'dataset':str(root),'images':len(records),'annotations':sum(r['tile_count'] for r in records),'categories':cats,'class_instances':dict(sorted(counter.items())),'original_splits':dict(collections.Counter(r['original_split'] for r in records)),'tile_count_histogram':dict(sorted(collections.Counter(r['tile_count'] for r in records).items())),'dimension_histogram':dict(collections.Counter(f'{r["width"]}x{r["height"]}' for r in records).most_common()),'issue_count':len(issues),'exact_unique_images':len(bysha),'exact_duplicates_removed':len(dedup),'conflicting_annotation_groups':len(conflicts),'groups':len(groups),'cross_original_split_groups':len(crossing),'near_pairs':len(near_pairs),'near_pairs_label_disagreement':sum(p.get('label_disagreement',False) for p in near_pairs),'largest_group_images':max(map(len,groups.values()),default=0),'research_splits':dict(split_counts),'research_class_instances':{s:dict(c) for s,c in class_counts.items()},'excluded_images':len(excluded_set),'annotation_sha256':annotation_hashes,'warnings':['Research holdout only; capture-session identity and unseen-room coverage unknown.','Near duplicate candidates use dHash/aspect/color checks, not a proof of scene independence.','Original split metrics may be optimistic when source groups cross splits.']}
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    with (out/'manifest.jsonl').open('w') as f:
        for r in records:f.write(json.dumps({k:v for k,v in r.items() if k not in {'color','mean_rgb'}},ensure_ascii=False)+'\n')
    (out/'issues.json').write_text(json.dumps(issues,ensure_ascii=False,indent=2)+'\n')
    (out/'duplicate_groups.json').write_text(json.dumps({'cross_original_split_groups':crossing,'exact_conflicts':conflicts,'near_pairs':near_pairs,'groups':{g:[records[i]['id'] for i in m] for g,m in group_ids.items() if len(m)>1}},ensure_ascii=False,indent=2)+'\n')
    # Research COCO splits reference original files with relative paths, never move originals.
    for split in SPLITS:
        ims=[];anns=[]
        for i,r in enumerate(records):
            if r['research_split']!=split:continue
            ims.append({'id':i,'file_name':r['relative_path'],'width':r['width'],'height':r['height']})
            for a in r['annotations']:anns.append({'id':len(anns),'image_id':i,'category_id':a['class_id'],'bbox':a['bbox'],'area':a['bbox'][2]*a['bbox'][3],'iscrowd':a['iscrowd']})
        (out/f'{split}.coco.json').write_text(json.dumps({'images':ims,'annotations':anns,'categories':[{'id':k,'name':v} for k,v in cats.items()]},ensure_ascii=False)+'\n')
    # Crop exemplars make category semantics auditable; no training transformation.
    examples=collections.defaultdict(list)
    for r in records:
        for a in r['annotations']:
            if len(examples[a['class_id']])<4 and not r['issues']:
                examples[a['class_id']].append((r,a))
    sheet=Image.new('RGB',(900, max(1,len(examples))*90),'#f0f0f0');draw=ImageDraw.Draw(sheet)
    for row,cls in enumerate(sorted(examples)):
        draw.text((5,row*90+35),cats[cls],fill='black')
        for col,(r,a) in enumerate(examples[cls]):
            with Image.open(root/r['relative_path']) as im:
                x,y,w,h=a['bbox'];crop=im.convert('RGB').crop((max(0,x),max(0,y),min(im.width,x+w),min(im.height,y+h)))
                crop.thumbnail((150,82));sheet.paste(crop,(80+col*200,row*90+4))
    sheet.save(out/'class-exemplars.jpg',quality=92)
    # Deterministic, class/count/dimension-diverse contact sheets for visual audit.
    ranked=sorted(range(len(records)),key=lambda i:hashlib.sha256(records[i]['id'].encode()).hexdigest())
    selected=[];seen=set()
    for cls in cats:
        for i in ranked:
            if i not in seen and any(a['class_id']==cls for a in records[i]['annotations']):selected.append(i);seen.add(i);break
    for count in sorted(set(r['tile_count'] for r in records)):
        for i in ranked:
            if i not in seen and records[i]['tile_count']==count:selected.append(i);seen.add(i);break
    selected += [i for i in ranked if i not in seen][:max(0,args.sample-len(selected))]
    (out/'visual_audit_sample.json').write_text(json.dumps([{'index':n,'id':records[i]['id'],'tile_count':records[i]['tile_count']} for n,i in enumerate(selected)],indent=2)+'\n')
    for page in range((len(selected)+39)//40):
        sheet=Image.new('RGB',(1200,1600),'#eeeeee');draw=ImageDraw.Draw(sheet)
        for slot,i in enumerate(selected[page*40:(page+1)*40]):
            r=records[i];x=(slot%5)*240;y=(slot//5)*200
            try:
                with Image.open(root/r['relative_path']) as im:
                    im=im.convert('RGB');scale=min(230/im.width,162/im.height);im=im.resize((max(1,round(im.width*scale)),max(1,round(im.height*scale))))
                    d=ImageDraw.Draw(im)
                    for a in r['annotations']:
                        bx,by,bw,bh=a['bbox'];d.rectangle((bx*scale,by*scale,(bx+bw)*scale,(by+bh)*scale),outline='#f22',width=1)
                    sheet.paste(im,(x+5,y+5))
                draw.text((x+5,y+168),f'{page*40+slot:03d} tiles={r["tile_count"]} {r["original_split"]}',fill='black');draw.text((x+5,y+182),r['file_name'][:30],fill='black')
            except Exception:pass
        sheet.save(out/f'contact-{page+1:02d}.jpg',quality=90)
    print(json.dumps({k:report[k] for k in ['images','annotations','issue_count','exact_duplicates_removed','conflicting_annotation_groups','cross_original_split_groups','research_splits']},ensure_ascii=False))
if __name__=='__main__':main()
