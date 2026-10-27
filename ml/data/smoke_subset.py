"""Create a bounded representative train/valid probe; never opens test."""
import argparse,collections,hashlib,json,os,shutil
from pathlib import Path

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--train-limit',type=int,default=96);p.add_argument('--valid-limit',type=int,default=24);a=p.parse_args()
    if a.out.exists():raise ValueError('Use a fresh output')
    stats={}
    for split,limit in [('train',a.train_limit),('valid',a.valid_limit)]:
        root=a.dataset/split;data=json.loads((root/'_annotations.coco.json').read_text());byim=collections.defaultdict(list)
        for ann in data['annotations']:byim[ann['image_id']].append(ann)
        strata=collections.defaultdict(list)
        for image in data['images']:
            count=len(byim[image['id']]);size='negative' if count==0 else 'single' if count==1 else 'hand' if count>=10 else 'few'
            strata[(image.get('source_family','unknown'),size)].append(image)
        for key in strata:strata[key].sort(key=lambda r:hashlib.sha256((r.get('source_identity','')+r['file_name']).encode()).hexdigest())
        selected=[];seen_origins=set()
        # Prefer unique source origins, then allow additional existing views if
        # the requested probe is larger than the pool of unique origins.
        for unique in (True,False):
            while len(selected)<limit:
                progress=False
                for key in sorted(strata):
                    candidate=next((r for r in strata[key] if r not in selected and (not unique or r.get('source_identity',r['file_name']) not in seen_origins)),None)
                    if candidate:
                        selected.append(candidate);seen_origins.add(candidate.get('source_identity',candidate['file_name']));progress=True
                    if len(selected)>=limit:break
                if not progress:break
        ids={r['id'] for r in selected};anns=[x for x in data['annotations'] if x['image_id'] in ids];dest=a.out/split;dest.mkdir(parents=True)
        for r in selected:
            try:os.link(root/r['file_name'],dest/r['file_name'])
            except OSError:shutil.copy2(root/r['file_name'],dest/r['file_name'])
        path=dest/'_annotations.coco.json';path.write_text(json.dumps({'images':selected,'annotations':anns,'categories':data['categories']},ensure_ascii=False))
        stats[split]={'images':len(selected),'annotations':len(anns),'whole_hand_images_ge10':sum(len(byim[r['id']])>=10 for r in selected),'negative_images':sum(not byim[r['id']] for r in selected),'annotation_sha256':sha(path)}
    receipt={'dataset':str(a.dataset),'scope':'small GPU performance/contract probe, not model selection','testOpened':False,'splits':stats};(a.out/'subset.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
