"""Expand face-only data without treating unreviewed v3 quarantine as bad labels.

Deterministic grouped research split. Existing augmentations are allowed only in
train, bounded per original; validation/test contain canonical source views.
Source back faces are background, never a predicted Mahjong face class.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, math, re
from pathlib import Path
import numpy as np
from PIL import Image
from ml.data.audit import UnionFind
from ml.data.curate_v3 import family

ONTOLOGY = 'mahjong_faces_42'
SPLITS = ('train', 'valid', 'test')
SAFE = {'phone_blue_S__', 'phone_keyboard_rotated', 'preaugmented_single_or_layout', 'unreviewed_camera'}

def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def source_identity(r):
    name = r['file_name'].lower()
    numeric = re.match(r'^(\d{1,3})(?:_(\d{3}))?_(?:jpg|jpeg|png)(?:\.|$)', name)
    if numeric and int(numeric[1]) <= 120:return 'keyboard:' + numeric[1]
    f = family(r)
    if f == 'phone_blue_S__':return 'blue:' + r['source_key']
    single = re.match(r'^r?([1-9][mpsz])_(?:add|sub|\d)', name)
    if single:return 'green_single:' + single[1]
    if name.startswith('back_'):return 'green_back'
    if name.startswith('all_'):return 'green_layout'
    cam = re.match(r'^img[_-]?(\d+)', name)
    if cam:
        number = int(cam[1])
        # Conservative short capture-burst grouping, not a claim of known sessions.
        return f'camera_burst8:{number//8}'
    return 'other:' + r['source_key']

def family_v4(r):
    if source_identity(r).startswith('keyboard:'):return 'phone_keyboard_rotated'
    return family(r)

def repair_annotations(r):
    result=[];repairs=[]
    for i,a in enumerate(r['annotations']):
        if a['class_id'] == 43:continue
        x,y,w,h=map(float,a['bbox']);values=(x,y,w,h)
        if not all(math.isfinite(v) for v in values) or min(w,h)<=0:raise ValueError('invalid_truth')
        overshoot=max(0,-x,-y,x+w-r['width'],y+h-r['height'])
        if overshoot > .020001:raise ValueError('truth_outside_bounds_gt_0.02px')
        x1,y1=max(0,x),max(0,y);x2,y2=min(r['width'],x+w),min(r['height'],y+h)
        box=[round(x1,6),round(y1,6),round(x2-x1,6),round(y2-y1,6)]
        if box[2]<=0 or box[3]<=0:raise ValueError('nonpositive_repaired_truth')
        if any(abs(q-v)>1e-7 for q,v in zip(box,values)):repairs.append({'annotation_index':i,'before':list(values),'after':box,'reason':'explicit_subpixel_rounding_clamp_le_0.02px'})
        if a.get('iscrowd',0):raise ValueError('crowd_truth')
        result.append({**a,'bbox':box})
    return result,repairs

def normalized_layout(r):
    return sorted((a['class_id'],*(round(float(v)/d,3) for v,d in zip(a['bbox'],(r['width'],r['height'],r['width'],r['height'])))) for a in r['annotations'] if a['class_id']!=43)

def plausible_near(a,b):
    aa,bb=normalized_layout(a),normalized_layout(b)
    if len(aa)!=len(bb) or not aa:return False
    if any(x[0]!=y[0] for x,y in zip(aa,bb)):return False
    if max(abs(x-y) for ra,rb in zip(aa,bb) for x,y in zip(ra[1:],rb[1:]))>.015:return False
    return abs(math.log((a['width']/a['height'])/(b['width']/b['height'])))<.02

def canonical_rank(r,root,cache):
    key=r['id']
    if key not in cache:
        with Image.open(root/r['relative_path']) as source:
            image=source.convert('RGB');image.thumbnail((128,128))
            array=np.asarray(image,dtype=np.int16)
            border=np.concatenate((array[:4].reshape(-1,3),array[-4:].reshape(-1,3),array[:,:4].reshape(-1,3),array[:,-4:].reshape(-1,3)))
            padding=float(((border.max(axis=1)<8)|(border.min(axis=1)>247)).mean())
            cache[key]=(padding,abs(float(array.mean())-145))
    match=re.match(r'^\d{1,3}_(\d{3})_',r['file_name'])
    angle=min(int(match[1]),360-int(match[1])) if match else 0
    return angle,*cache[key],hashlib.sha256(r['id'].encode()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--metadata',type=Path,required=True)
    p.add_argument('--max-train-variants',type=int,default=12);p.add_argument('--validation-target',type=int,default=150)
    p.add_argument('--archive',type=Path,required=True);p.add_argument('--review',type=Path,default=Path('docs/data/review-v4.json'));args=p.parse_args()
    if args.manifest.exists() or args.metadata.exists():raise ValueError('Use fresh immutable output paths')
    rows=[json.loads(l) for l in (args.audit/'manifest.jsonl').read_text().splitlines() if l.strip()]
    byidentity={};uf=UnionFind(len(rows));bysha={};reasons={};review={};fixes={}
    # Known uncertain/crowded full-table cases from the complete 322-image camera
    # contact review. The image is quarantined, not called a proven mislabel.
    review_decisions=json.loads(args.review.read_text(encoding='utf-8'))
    uncertain=set(review_decisions['camera_scope_quarantined_source_ids'])
    for i,r in enumerate(rows):
        identity=source_identity(r);byidentity.setdefault(identity,[]).append(i)
        if r.get('sha256'):
            if r['sha256'] in bysha:uf.union(i,bysha[r['sha256']])
            else:bysha[r['sha256']]=i
        if 'dimension_mismatch' in r.get('issues',[]) or r.get('actual_width')!=r['width'] or r.get('actual_height')!=r['height']:
            reasons[i]='encoded_annotation_dimensions_mismatch';continue
        if r.get('exif_orientation') not in (None,1):reasons[i]='unnormalized_exif';continue
        if family_v4(r) not in SAFE:reasons[i]='outside_this_full_face_scope_pass';continue
        if r['id'] in uncertain:reasons[i]='camera_crowded_scope_uncertain_after_contact_review';continue
        try:
            faces,fix=repair_annotations(r)
        except ValueError as e:reasons[i]=str(e);continue
        if not faces:reasons[i]='back_only_not_admitted_without_explicit_negative_review';continue
        fixes[i]=fix;review[i]='full_front_face_scope_source_review_no_complete_per_tile_reannotation'
    for members in byidentity.values():
        for i in members[1:]:uf.union(members[0],i)
    # Refine cached low-resolution dHash candidates with label/layout agreement,
    # then require high-resolution thumbnail agreement. This avoids grouping all
    # blue/green empty backgrounds and unrelated single-tile classes together.
    duplicate_path=args.audit.parent/'audit/duplicate_groups.json';cached=json.loads(duplicate_path.read_text())['near_pairs']
    thumbs={};accepted_edges=0
    for edge in cached:
        i,j=edge['a'],edge['b'];a,b=rows[i],rows[j]
        if uf.find(i)==uf.find(j) or i in reasons or j in reasons or not plausible_near(a,b):continue
        for k in (i,j):
            if k not in thumbs:
                with Image.open(args.root/rows[k]['relative_path']) as image:thumbs[k]=np.asarray(image.convert('RGB').resize((64,64)),dtype=np.float32)/255
        if float(np.square(thumbs[i]-thumbs[j]).mean()) < .0004:
            uf.union(i,j);accepted_edges+=1
    components=collections.defaultdict(list)
    for i in range(len(rows)):components[uf.find(i)].append(i)
    groups=[];rank_cache={};exact_selected=set();old_test_sources=0
    for members in components.values():
        candidates=[i for i in members if i not in reasons]
        if not candidates:continue
        gid=hashlib.sha256('\n'.join(sorted(rows[i]['id'] for i in members)).encode()).hexdigest()[:20]
        protected=any(rows[i]['research_split']=='test' for i in members)
        # Canonical views per original stem, not one random augmented view per group.
        stems=collections.defaultdict(list)
        for i in candidates:stems[rows[i]['source_key']].append(i)
        canon=[min(items,key=lambda i:canonical_rank(rows[i],args.root,rank_cache)) for items in stems.values()]
        # Existing original photo identity is stronger than arbitrary suffix stems.
        # For S__/keyboard/green-single source families choose one canonical view.
        if all(source_identity(rows[i]).startswith(('blue:','keyboard:','green_single:','green_layout')) for i in candidates):
            canon=[min(candidates,key=lambda i:canonical_rank(rows[i],args.root,rank_cache))]
        if protected:
            old_test_sources+=1
            canon=[i for i in canon if rows[i]['research_split']=='test'] or [min((i for i in candidates if rows[i]['research_split']=='test'),key=lambda i:canonical_rank(rows[i],args.root,rank_cache))]
        groups.append({'id':gid,'members':members,'candidates':candidates,'canonical':canon,'protected':protected,'family':family_v4(rows[canon[0]]),
                       'whole':sum(sum(a['class_id']!=43 for a in rows[i]['annotations'])>=10 for i in canon)})
    available=[g for g in groups if not g['protected']]
    # Select whole source groups before training, stratifying family and prioritizing
    # whole hands. New models must start from public pretrained weights, not v2/v3.
    valid_groups=set();valid_n=0
    buckets=collections.defaultdict(list)
    for g in available:buckets[(g['family'],bool(g['whole']))].append(g)
    for key,items in sorted(buckets.items()):
        items.sort(key=lambda g:hashlib.sha256(('v4-valid-seed42:'+g['id']).encode()).hexdigest())
        quota=max(1,round(len(items)*.22))
        for g in items[:quota]:valid_groups.add(g['id']);valid_n+=len(g['canonical'])
    remaining=sorted((g for g in available if g['id'] not in valid_groups),key=lambda g:(not bool(g['whole']),hashlib.sha256(('v4-fill:'+g['id']).encode()).hexdigest()))
    for g in remaining:
        if valid_n>=args.validation_target:break
        if g['family']=='preaugmented_single_or_layout':continue
        valid_groups.add(g['id']);valid_n+=len(g['canonical'])
    selected={};group_assignment={};group_lookup={}
    for g in groups:
        split='test' if g['protected'] else 'valid' if g['id'] in valid_groups else 'train'
        group_assignment[g['id']]=split
        for i in g['members']:group_lookup[i]=g['id']
        if split in ('valid','test'):chosen=g['canonical']
        else:
            byorigin=collections.defaultdict(list)
            for i in g['candidates']:byorigin[source_identity(rows[i])].append(i)
            chosen=[]
            for identity,members in sorted(byorigin.items()):
                members.sort(key=lambda i:canonical_rank(rows[i],args.root,rank_cache))
                # One canonical and bounded existing variants, not thousands of
                # freshly generated copies presented as new photos.
                chosen.extend(members[:args.max_train_variants])
        for i in chosen:
            if rows[i]['sha256'] in exact_selected:reasons[i]='exact_byte_duplicate';continue
            exact_selected.add(rows[i]['sha256']);selected[i]=split
    output=[];counts=collections.Counter();originals=collections.defaultdict(set);classcounts=collections.defaultdict(collections.Counter);whole=collections.Counter();families=collections.defaultdict(collections.Counter)
    for i,r in enumerate(rows):
        split=selected.get(i,'excluded');gid=group_lookup.get(i,hashlib.sha256(source_identity(r).encode()).hexdigest()[:20])
        if i not in selected:
            reason=reasons.get(i,'protected_group_noncanonical' if group_assignment.get(gid)=='test' else 'noncanonical_holdout_or_bounded_train_variant')
        else:reason=None
        row={'source_id':r['id'],'sha256':r.get('sha256'),'source_identity':source_identity(r),'source_family':family_v4(r),'group_id':gid,'split':split,
             'v2_group_id':r['group_id'],'v2_split':r['research_split'],'exclusion_reason':reason,'mode':'full_native_image','ontology':ONTOLOGY,
             'review_status':review.get(i,'not_admitted_in_this_scope_pass'),'bbox_repairs':fixes.get(i,[]),'oracle_derived':False}
        output.append(row)
        if split in SPLITS:
            faces,_=repair_annotations(r);counts[split]+=1;originals[split].add(source_identity(r));classcounts[split].update(a['class_name'] for a in faces);whole[split]+=len(faces)>=10;families[split][family_v4(r)]+=1
    args.manifest.parent.mkdir(parents=True,exist_ok=True)
    args.manifest.write_text(''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in output))
    metadata={'schema_version':1,'selection_revision':4,'ontology':ONTOLOGY,'category_ids':list(range(1,43)),'back_policy':'back43 removed from categories/annotations; face model treats backs as background',
      'source':'mahjong-anaun/mahjong-lfhxf/2','archive_bytes':args.archive.stat().st_size,'archive_sha256':digest(args.archive),'tracked_split_manifest':str(args.manifest),'tracked_split_sha256':digest(args.manifest),
      'source_audit_sha256':digest(args.audit/'manifest.jsonl'),'review_manifest':str(args.review),'review_manifest_sha256':digest(args.review),'selector_sha256':digest(__file__),'fresh_pretrained_required':True,'resume_from_v2_v3_allowed':False,'scope':'native full-image visible identifiable Mahjong front faces; research grouped holdout',
      'source_images':len(rows),'selected_images':dict(counts),'distinct_source_identities':{s:len(v) for s,v in originals.items()},'whole_hand_images_ge10':dict(whole),'class_instance_counts':{s:dict(c) for s,c in classcounts.items()},'source_family_images':{s:dict(c) for s,c in families.items()},
      'retained_groups':{s:sum(group_assignment.get(g['id'])==s for g in groups) for s in SPLITS},'old_test_source_groups_protected':old_test_sources,'cached_near_edges_total':len(cached),'strict_thumbnail_edges_unioned':accepted_edges,
      'exclusion_counts':dict(collections.Counter(r['exclusion_reason'] for r in output if r['split']=='excluded')),'retained_images_with_subpixel_repairs':sum(bool(r['bbox_repairs']) for r in output if r['split'] in SPLITS),'max_train_variants_per_source_identity':args.max_train_variants,
      'validation_policy':'Canonical source views only; full source groups held together; broad family-stratified22% plus deterministic fill to target. Former v2 test source families never enter train/valid. Some v2 train groups newly held out require fresh pretrained model, not v2/v3 warm start.',
      'limits':['The same room/device/tile-set can occur across source groups; capture sessions are inferred, not known.','Existing augmented views count as training samples, not independent original photographs.','Representative full-source contact review is not independent reannotation of every glyph or every variation.','No claim of real-room generalization or release readiness.','V4 validation is a new research split and cannot be compared numerically to tiny v3 validation as the same benchmark.'],
      'prepared_splits':{s:{'images':counts[s],'annotations':sum(classcounts[s].values()),'whole_hand_images_ge10':whole[s]} for s in SPLITS}}
    args.metadata.write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:metadata[k] for k in ('selected_images','distinct_source_identities','whole_hand_images_ge10','source_family_images','retained_groups','retained_images_with_subpixel_repairs')},ensure_ascii=False))

if __name__=='__main__':main()
