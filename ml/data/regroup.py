"""Repair cached near-duplicate grouping without decoding source photos again.

All existing source/exact groups and every recorded strong perceptual candidate
edge are unioned regardless of annotation labels. Similar-looking but distinct
single tiles can therefore be overgrouped; keeping such components together is
conservative for a research holdout. This cannot prove capture-session isolation.
"""
from __future__ import annotations
import argparse, collections, hashlib, json
from pathlib import Path
from audit import UnionFind

RATIOS={'train':0.8,'valid':0.1,'test':0.1}

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.out.exists() and any(args.out.iterdir()):raise SystemExit('Output must be new and empty; freeze each audit revision')
    args.out.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(line) for line in (args.audit/'manifest.jsonl').read_text().splitlines() if line.strip()]
    old=json.loads((args.audit/'summary.json').read_text());dup=json.loads((args.audit/'duplicate_groups.json').read_text());pairs=dup['near_pairs'];uf=UnionFind(len(rows));known={}
    for i,r in enumerate(rows):
        if r['group_id'] in known:uf.union(i,known[r['group_id']])
        else:known[r['group_id']]=i
    for edge in pairs:
        if not (0<=edge['a']<len(rows) and 0<=edge['b']<len(rows)):raise ValueError('Cached near-pair index out of bounds')
        uf.union(edge['a'],edge['b'])
    groups=collections.defaultdict(list)
    for i in range(len(rows)):groups[uf.find(i)].append(i)
    kept={i for i,r in enumerate(rows) if r['exclusion_reason'] is None}
    total_cls=collections.Counter(a['class_name'] for i in kept for a in rows[i]['annotations'])
    assigned_n=collections.Counter();assigned_cls=collections.defaultdict(collections.Counter);assignments={};group_info=[]
    for members in groups.values():
        gid=hashlib.sha256('\n'.join(sorted(rows[i]['id'] for i in members)).encode()).hexdigest()[:20]
        use=[i for i in members if i in kept]
        hist=collections.Counter(a['class_name'] for i in use for a in rows[i]['annotations'])
        group_info.append((gid,members,use,hist))
    for gid,members,use,hist in sorted(group_info,key=lambda g:(-len(g[2]),g[0])):
        if not use:split='excluded'
        else:
            def cost(s):
                target=len(kept)*RATIOS[s];old_n=assigned_n[s];n=len(use)
                size=((old_n+n-target)**2-(old_n-target)**2)/max(1,target)
                cls=0.0
                for name,number in hist.items():
                    goal=total_cls[name]*RATIOS[s];before=assigned_cls[s][name]
                    cls+=((before+number-goal)**2-(before-goal)**2)/max(1,goal)
                return size+0.1*cls
            split=min(RATIOS,key=lambda s:(cost(s),s));assigned_n[split]+=len(use);assigned_cls[split].update(hist)
        assignments[gid]=split
        for i in members:
            rows[i]['previous_group_id']=rows[i]['group_id'];rows[i]['group_id']=gid
            rows[i]['research_split']=split if i in kept else 'excluded'
    crossing_original=[gid for gid,members,_,_ in group_info if len({rows[i]['original_split'] for i in members})>1]
    crossing_new=[edge for edge in pairs if rows[edge['a']]['research_split']!='excluded' and rows[edge['b']]['research_split']!='excluded' and rows[edge['a']]['research_split']!=rows[edge['b']]['research_split']]
    assert not crossing_new,'Known near candidate crosses research split'
    sizes=sorted((len(use) for _,_,use,_ in group_info),reverse=True)
    updated={**old,'audit_revision':2,'group_policy':'Union source/exact groups and every stored dHash/aspect/color candidate irrespective of class labels; deterministic greedy 80/10/10 size+class balance.','groups':len(groups),'largest_group_images':max(map(len,groups.values()),default=0),'largest_retained_group_images':sizes[0] if sizes else 0,'largest_retained_group_sizes':sizes[:20],'cross_original_split_groups':len(crossing_original),'known_near_edges_cross_research_splits':0,'research_splits':dict(assigned_n),'research_class_instances':{s:dict(c) for s,c in assigned_cls.items()},'research_missing_classes':{s:sorted(set(total_cls)-set(assigned_cls[s])) for s in RATIOS},'lineage':{'previous_manifest_sha256':digest(args.audit/'manifest.jsonl'),'cached_duplicate_groups_sha256':digest(args.audit/'duplicate_groups.json')},'warnings':['Research holdout only; capture-session identity and unseen-room/device/tile-set coverage unknown.','All known candidate edges are kept together regardless of labels, avoiding the previous annotation-dependent split risk.','Perceptual candidates may overgroup similar-looking distinct tiles; large connected components are reported, not treated as proven true duplicates.','Zero known cached edges crossing research splits does not prove there is no unobserved near duplicate or same-session leakage.','Do not use obsolete audit/prepared/smoke directories for experiments after audit revision 2.']}
    (args.out/'summary.json').write_text(json.dumps(updated,ensure_ascii=False,indent=2)+'\n')
    with (args.out/'manifest.jsonl').open('w') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    (args.out/'group_assignments.json').write_text(json.dumps({'assignments':assignments,'cross_original_split_groups':crossing_original,'largest_groups':[{'group_id':gid,'images':len(members),'retained':len(use),'split':assignments[gid],'class_instances':dict(hist)} for gid,members,use,hist in sorted(group_info,key=lambda g:-len(g[2]))[:20]]},ensure_ascii=False,indent=2)+'\n')
    for split in RATIOS:
        images=[];annotations=[]
        for i,r in enumerate(rows):
            if r['research_split']!=split:continue
            images.append({'id':i,'file_name':r['relative_path'],'width':r['width'],'height':r['height'],'group_id':r['group_id']})
            for a in r['annotations']:
                b=a['bbox'];annotations.append({'id':len(annotations),'image_id':i,'category_id':a['class_id'],'bbox':b,'area':b[2]*b[3],'iscrowd':a.get('iscrowd',0)})
        cats={a['class_id']:a['class_name'] for r in rows for a in r['annotations']}
        (args.out/f'{split}.coco.json').write_text(json.dumps({'images':images,'annotations':annotations,'categories':[{'id':k,'name':v,'supercategory':'mahjong'} for k,v in sorted(cats.items())]},ensure_ascii=False)+'\n')
    print(json.dumps({k:updated[k] for k in ['groups','largest_group_images','cross_original_split_groups','known_near_edges_cross_research_splits','research_splits','research_missing_classes']},ensure_ascii=False))
if __name__=='__main__':main()
