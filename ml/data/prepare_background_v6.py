"""Train-only, manually reviewed real-background crops for generic tile detection."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
import re
from pathlib import Path
import shutil
from PIL import Image,ImageDraw,ImageStat
from ml.data.prepare_yolo import verify_prepared
from ml.train.provenance import atomic_json,dataset_contract,sha256


def label_tree_sha256(records):
    # v5 verifier signs native relative-path strings (Windows uses backslashes).
    # Preserve that protocol rather than normalizing it to POSIX in this fork.
    return hashlib.sha256(json.dumps(sorted((str(path),digest)for path,digest in records)).encode()).hexdigest()


def intersects(rect,box,margin=4):
    x,y,w,h=box;l,t,r,b=rect
    return min(r,x+w+margin)>max(l,x-margin) and min(b,y+h+margin)>max(t,y-margin)


def propose(dataset,out,limit=72):
    dataset,out=Path(dataset),Path(out);contract=dataset_contract(dataset,verify_images=True)
    coco=json.loads((dataset/'train/_annotations.coco.json').read_text());anns=defaultdict(list);groups=defaultdict(list)
    for a in coco['annotations']:anns[a['image_id']].append(a['bbox'])
    for image in coco['images']:groups[image['group_id']].append(image)
    def family(image):
        name=image['source_id'].split('/')[-1]
        return 0 if name[0].isdigit() else 1 if name.startswith('IMG') else 2
    candidates=[];out.mkdir(parents=True,exist_ok=False)
    ordered=sorted(groups,key=lambda g:(min(family(i)for i in groups[g]),g))
    for group in ordered:
        images=sorted(groups[group],key=lambda i:(family(i),'_000_'not in i['source_id'],i['source_id']))
        chosen=[]
        for im in images[:3]:
            with Image.open(dataset/'train'/im['file_name'])as source:
                source=source.convert('RGB');choices=[]
                for fraction in (.45,.35):
                    cw,ch=int(source.width*fraction),int(source.height*fraction)
                    for fy in (0,.33,.66,1):
                        for fx in (0,.33,.66,1):
                            l,t=round((source.width-cw)*fx),round((source.height-ch)*fy);rect=[l,t,l+cw,t+ch]
                            if any(intersects(rect,box)for box in anns[im['id']]):continue
                            crop=source.crop(rect);texture=ImageStat.Stat(crop.resize((128,128)).convert('L')).var[0]
                            if texture>180:choices.append((texture,rect))
                for texture,rect in sorted(choices,reverse=True):
                    if any(min(rect[2],r[2])>max(rect[0],r[0])and min(rect[3],r[3])>max(rect[1],r[1])for r in chosen):continue
                    identifier='bg-'+f'{len(candidates):03}'
                    source.crop(rect).save(out/(identifier+'.png'),compress_level=1)
                    candidates.append({'id':identifier,'sourceFile':im['file_name'],'sourceId':im['source_id'],'sourceSha256':sha256(dataset/'train'/im['file_name']),'sourceGroup':group,'cropXYXY':rect,'texture':texture})
                    chosen.append(rect)
                    if len(chosen)>=2 or len(candidates)>=limit:break
            if chosen or len(candidates)>=limit:break
        if len(candidates)>=limit:break
    atomic_json(out/'candidates.json',{'dataContract':contract,'candidates':candidates,'testOpened':False,'reviewRequired':True})
    for start in range(0,len(candidates),12):
        sheet=Image.new('RGB',(960,880),'#eeeeee');draw=ImageDraw.Draw(sheet)
        for index,row in enumerate(candidates[start:start+12]):
            crop=Image.open(out/(row['id']+'.png')).convert('RGB');crop.thumbnail((312,190))
            col,line=index%3,index//3;x,y=col*320,line*220
            sheet.paste(crop,(x+(312-crop.width)//2,y+22));draw.text((x+4,y+3),row['id']+' '+row['sourceId'].split('/')[-1][:22],fill='black')
        sheet.save(out/f'contact-{start//12:02}.jpg',quality=90)
    return {'candidates':len(candidates),'groups':len({r['sourceGroup']for r in candidates}),'trainingStarted':False}


def prepare(dataset,base,review,out,preflight_only=False):
    dataset,base,review,out=map(Path,(dataset,base,review,out));contract=dataset_contract(dataset,verify_images=True)
    original=verify_prepared(base,contract,True);checked=json.loads(review.read_text());accepted=checked['accepted']
    if checked['trainAnnotationSha256']!=contract['splits']['train']['annotationSha256']or checked['trainImageTreeSha256']!=contract['splits']['train']['imageTreeSha256']:
        raise ValueError('Manual background review belongs to another training set')
    if not checked.get('contactSheetsVisuallyReviewed')or not accepted:raise ValueError('Explicit visual review is required')
    coco=json.loads((dataset/'train/_annotations.coco.json').read_text());images={r['file_name']:r for r in coco['images']};anns=defaultdict(list)
    for a in coco['annotations']:anns[a['image_id']].append(a['bbox'])
    counts=defaultdict(int);seen=set()
    for row in accepted:
        image=images.get(row['sourceFile']);rect=row['cropXYXY'];key=(row['sourceFile'],tuple(rect))
        if not re.fullmatch(r'bg-[0-9]{3}',row.get('id','')):raise ValueError('Unsafe reviewed background ID')
        if image is None or image['group_id']!=row['sourceGroup']or sha256(dataset/'train'/row['sourceFile'])!=row['sourceSha256']:raise ValueError('Background source must be a byte-verified train image')
        if key in seen or any(intersects(rect,b)for b in anns[image['id']]):raise ValueError('Background is duplicated or intersects a ground-truth face')
        if len(rect)!=4 or not all(isinstance(v,int)for v in rect)or not(0<=rect[0]<rect[2]<=image['width']and 0<=rect[1]<rect[3]<=image['height']):raise ValueError('Invalid reviewed crop')
        seen.add(key);counts[row['sourceGroup']]+=1
    if max(counts.values())>2 or len(accepted)>120:raise ValueError('Limit review to2 crops per group and120 total')
    if preflight_only:
        ready=(out/'prepared.json').is_file()
        if ready:
            result=verify_prepared(out,contract,True)
            if result.get('backgroundReviewSha256')!=sha256(review):raise ValueError('Background review changed')
        return {'derivativeReady':ready,'backgrounds':len(accepted),'testOpened':False,'trainingStarted':False}
    if out.exists():raise FileExistsError('Use a fresh reviewed derivative directory')
    shutil.copytree(base,out,copy_function=lambda src,dst:os.link(src,dst))
    metadata=json.loads((out/'prepared.json').read_text());extra=[]
    for row in accepted:
        target=out/'images/train'/('negative-'+row['id']+'.png')
        with Image.open(dataset/'train'/row['sourceFile'])as source:source.convert('RGB').crop(row['cropXYXY']).save(target,compress_level=1)
        (out/'labels/train'/('negative-'+row['id']+'.txt')).write_text('')
        extra.append((target.relative_to(out).as_posix(),sha256(target)))
    # Never rewrite a hardlinked base metadata/config file in place.
    (out/'data.yaml').unlink();atomic_json(out/'data.yaml',{'path':out.resolve().as_posix(),'train':'images/train','val':'images/valid','names':['tile']})
    metadata.update(images=metadata['images']+extra,dataYamlSha256=sha256(out/'data.yaml'),
        labelTreeSha256=label_tree_sha256((p.relative_to(out),sha256(p))for p in (out/'labels').rglob('*.txt')),
        backgroundReviewSha256=sha256(review),backgrounds=accepted,backgroundGroups=len(counts),preparationCodeSha256=sha256(__file__))
    (out/'prepared.json').unlink();atomic_json(out/'prepared.json',metadata)
    verify_prepared(out,contract,True)
    return {'derivativeReady':True,'backgrounds':len(accepted),'groups':len(counts),'testOpened':False,'trainingStarted':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--base-prepared',type=Path);p.add_argument('--review',type=Path);p.add_argument('--propose',action='store_true');p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
    if a.propose:result=propose(a.dataset,a.out)
    else:
        if not a.base_prepared or not a.review:p.error('--base-prepared and --review are required')
        result=prepare(a.dataset,a.base_prepared,a.review,a.out,a.preflight_only)
    print(json.dumps(result))

if __name__=='__main__':main()
