"""Rebuild the materialization manifest from immutable COCO + tracked split IDs.

This is the portable reproduction path after downloading the fixed archive.
No hidden cached dHash fields or machine-specific original paths are required.
"""
from __future__ import annotations
import argparse,collections,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--split-manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    splitrows=[json.loads(line) for line in args.split_manifest.read_text().splitlines() if line.strip()]
    assignments={r['source_id']:r for r in splitrows}
    if len(assignments)!=len(splitrows):raise ValueError('Duplicate source IDs in tracked split manifest')
    output=[];seen=set()
    for split in ('train','valid','test'):
        d=json.loads((args.root/split/'_annotations.coco.json').read_text());cats={c['id']:c['name'] for c in d['categories']};anns=collections.defaultdict(list)
        for a in d['annotations']:anns[a['image_id']].append(a)
        for im in d['images']:
            uid=split+'/'+im['file_name'];a=assignments.get(uid)
            if a is None:raise ValueError('Source image missing from tracked split manifest')
            seen.add(uid)
            output.append({'id':uid,'relative_path':uid,'original_split':split,'file_name':im['file_name'],'sha256':a['sha256'],'width':im['width'],'height':im['height'],'group_id':a['group_id'],'research_split':a['split'],'exclusion_reason':a['exclusion_reason'],'annotations':[{'class_id':b['category_id'],'class_name':cats[b['category_id']],'bbox':b['bbox'],'iscrowd':b.get('iscrowd',0)} for b in anns[im['id']]]})
    if seen!=set(assignments):raise ValueError('Tracked split manifest contains unknown source images')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('w') as f:
        for r in output:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    print(json.dumps({'images':len(output),'splits':dict(collections.Counter(r['research_split'] for r in output))}))
if __name__=='__main__':main()
