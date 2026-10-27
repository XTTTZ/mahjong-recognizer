"""Materialize frozen native-image face-only v4 data from the original ZIP."""
from __future__ import annotations
import argparse,collections,hashlib,json,os,shutil,zipfile
from pathlib import Path,PurePosixPath
from ml.data.curate_v4 import repair_annotations
SPLITS=('train','valid','test')
def sha(data):return hashlib.sha256(data).hexdigest()
def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--metadata',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source-root',type=Path);p.add_argument('--update-metadata',action='store_true')
    args=p.parse_args();metadata=json.loads(args.metadata.read_text(encoding='utf-8-sig'));rows=[json.loads(l) for l in args.manifest.read_text(encoding='utf-8-sig').splitlines() if l.strip()]
    if args.out.exists() and any(args.out.iterdir()):raise ValueError('Choose a new empty output directory')
    if digest(args.archive)!=metadata['archive_sha256'] or digest(args.manifest)!=metadata['tracked_split_sha256']:raise ValueError('Archive/selection provenance mismatch')
    retained=[r for r in rows if r['split'] in SPLITS]
    if len({r['source_id'] for r in retained})!=len(retained):raise ValueError('Duplicate retained source ID')
    groups=collections.defaultdict(set);hash_groups=collections.defaultdict(set);stats={};images_meta={};annotations=collections.defaultdict(list);cats=None
    with zipfile.ZipFile(args.archive) as zipped:
        for original_split in SPLITS:
            coco=json.loads(zipped.read(f'{original_split}/_annotations.coco.json'));current=[c for c in coco['categories'] if c['id'] not in (0,43)]
            if [c['id'] for c in sorted(current,key=lambda c:c['id'])]!=list(range(1,43)):raise ValueError('Original category IDs1..42 missing')
            if cats is not None and current!=cats:raise ValueError('Source split categories differ')
            cats=current
            for im in coco['images']:images_meta[original_split+'/'+im['file_name']]=im
            for a in coco['annotations']:annotations[(original_split,a['image_id'])].append(a)
        catnames={c['id']:c['name'] for c in cats}
        for split in SPLITS:
            dest=args.out/split;dest.mkdir(parents=True,exist_ok=True);ims=[];anns=[]
            for row in sorted((r for r in retained if r['split']==split),key=lambda r:sha(r['source_id'].encode())):
                uid=row['source_id'];member=PurePosixPath(uid)
                if member.is_absolute() or '..' in member.parts or '\\' in uid or ':' in uid:raise ValueError('Unsafe source member')
                record=images_meta[uid];raw_annotations=annotations[(member.parts[0],record['id'])]
                reconstructed={'width':record['width'],'height':record['height'],'annotations':[{'class_id':a['category_id'],'class_name':catnames.get(a['category_id'],'back'),'bbox':a['bbox'],'iscrowd':a.get('iscrowd',0)} for a in raw_annotations]}
                faces,repairs=repair_annotations(reconstructed)
                if repairs!=row['bbox_repairs']:raise ValueError(f'Repair differs from frozen selection: {uid}')
                filename=sha(uid.encode())[:20]+Path(uid).suffix.lower();target=dest/filename
                if args.source_root:
                    source=(args.source_root/uid).resolve()
                    if not source.is_relative_to(args.source_root.resolve()) or digest(source)!=row['sha256']:raise ValueError('Source bytes mismatch')
                    try:os.link(source,target)
                    except OSError:shutil.copy2(source,target)
                else:
                    raw=zipped.read(uid)
                    if sha(raw)!=row['sha256']:raise ValueError('Source ZIP bytes mismatch')
                    target.write_bytes(raw)
                idx=len(ims);ims.append({'id':idx,'file_name':filename,'width':record['width'],'height':record['height'],'source_id':uid,'source_identity':row['source_identity'],'source_family':row['source_family'],'group_id':row['group_id'],'sha256':row['sha256'],'mode':row['mode'],'ontology':'mahjong_faces_42','oracle_derived':False,'bbox_repair_count':len(repairs)})
                for a in faces:
                    x,y,w,h=a['bbox'];anns.append({'id':len(anns),'image_id':idx,'category_id':a['class_id'],'bbox':a['bbox'],'area':w*h,'iscrowd':0})
                groups[row['group_id']].add(split);hash_groups[row['sha256']].add(split)
            path=dest/'_annotations.coco.json';path.write_text(json.dumps({'images':ims,'annotations':anns,'categories':cats},ensure_ascii=False),encoding='utf-8')
            counts=collections.Counter(a['image_id'] for a in anns);stats[split]={'images':len(ims),'annotations':len(anns),'whole_hand_images_ge10':sum(counts[im['id']]>=10 for im in ims),'annotation_sha256':digest(path)}
            expected=metadata['prepared_splits'][split]
            if (expected['images'],expected['annotations'])!=(len(ims),len(anns)):raise ValueError('Materialized counts differ from selection')
            if expected.get('annotation_sha256') and expected['annotation_sha256']!=stats[split]['annotation_sha256']:raise ValueError('COCO differs from frozen metadata')
    if any(len(s)>1 for s in groups.values()) or any(len(s)>1 for s in hash_groups.values()):raise ValueError('Group/identical bytes cross splits')
    receipt={'schema_version':1,'selection_revision':4,'ontology':'mahjong_faces_42','scope':metadata['scope'],'archive_sha256':metadata['archive_sha256'],'tracked_split_sha256':metadata['tracked_split_sha256'],'split_groups_crossing':0,'exact_byte_hashes_crossing':0,'splits':stats,'productionValidated':False}
    (args.out/'materialization.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if args.update_metadata:
        metadata['prepared_splits']=stats;metadata['materializer_sha256']=digest(__file__);metadata['prepared_root']=str(args.out);args.metadata.write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(receipt))
if __name__=='__main__':main()
