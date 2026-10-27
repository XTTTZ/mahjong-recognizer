"""Regression checks for local face-only data recovery and source lineage."""
import hashlib,json,subprocess,sys,tempfile,unittest,zipfile
from pathlib import Path
from PIL import Image
from ml.data.curate_v4 import repair_annotations,source_identity

class RepairTests(unittest.TestCase):
    def test_subpixel_rounding_is_explicit_and_back_is_removed(self):
        record={'width':100,'height':80,'annotations':[{'class_id':1,'class_name':'1h','bbox':[-.01,1,10,20]}, {'class_id':43,'class_name':'back','bbox':[60,20,10,20]}]}
        faces,repairs=repair_annotations(record)
        self.assertEqual(len(faces),1);self.assertEqual(faces[0]['bbox'],[0,1,9.99,20])
        self.assertEqual(repairs[0]['before'],[-.01,1.0,10.0,20.0]);self.assertEqual(repairs[0]['after'],faces[0]['bbox'])
        record['annotations'][0]['bbox']=[-.03,1,10,20]
        with self.assertRaisesRegex(ValueError,'0.02'):repair_annotations(record)
    def test_keyboard_rotation_and_original_alias_share_one_source_identity(self):
        for name in ('107_210_jpg.rf.abcd.jpg','107_jpg.rf.efgh.jpg','107_000_jpg.rf.ijkl.jpg'):
            self.assertEqual(source_identity({'file_name':name,'source_key':name}),'keyboard:107')
    def test_camera_neighbor_capture_burst_is_not_treated_as_many_groups(self):
        self.assertEqual(source_identity({'file_name':'IMG_6104_jpg.rf.abc.jpg','source_key':'IMG_6104'}),source_identity({'file_name':'IMG_6107_jpg.rf.def.jpg','source_key':'IMG_6107'}))

class MaterializerTests(unittest.TestCase):
    def test_original_bytes_reused_face_schema_and_repair_match(self):
        class_map=json.loads(Path('docs/data/class-map.json').read_text())['classes'];categories=[{'id':c['coco_id'],'name':c['token']} for c in class_map]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);original=root/'original';rows=[]
            with zipfile.ZipFile(root/'source.zip','w') as archive:
                for index,split in enumerate(('train','valid','test')):
                    folder=original/split;folder.mkdir(parents=True);path=folder/'a.png';Image.new('RGB',(64,48),(index*70,100,120)).save(path)
                    annotations=[{'id':1,'image_id':0,'category_id':1,'bbox':[-.01,1,10,20]}, {'id':2,'image_id':0,'category_id':43,'bbox':[30,1,10,20]}]
                    coco={'images':[{'id':0,'file_name':'a.png','width':64,'height':48}],'annotations':annotations,'categories':categories}
                    archive.writestr(split+'/_annotations.coco.json',json.dumps(coco));archive.write(path,split+'/a.png')
                    _,fixes=repair_annotations({'width':64,'height':48,'annotations':[{'class_id':a['category_id'],'class_name':'fixture','bbox':a['bbox']} for a in annotations]})
                    rows.append({'source_id':split+'/a.png','source_identity':split,'source_family':'fixture','sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'group_id':split,'split':split,'bbox_repairs':fixes,'mode':'full_native_image'})
            manifest=root/'selection.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows));meta=root/'metadata.json';meta.write_text(json.dumps({'archive_sha256':hashlib.sha256((root/'source.zip').read_bytes()).hexdigest(),'tracked_split_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'scope':'fixture','prepared_splits':{s:{'images':1,'annotations':1} for s in ('train','valid','test')}}))
            result=subprocess.run([sys.executable,'-m','ml.data.materialize_v4','--archive',str(root/'source.zip'),'--manifest',str(manifest),'--metadata',str(meta),'--out',str(root/'prepared'),'--source-root',str(original)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            for row in rows:
                coco=json.loads((root/'prepared'/row['split']/'_annotations.coco.json').read_text());self.assertEqual(len(coco['categories']),42);self.assertEqual(len(coco['annotations']),1)
                self.assertEqual(coco['annotations'][0]['bbox'],[0,1,9.99,20]);photo=root/'prepared'/row['split']/coco['images'][0]['file_name'];self.assertEqual(hashlib.sha256(photo.read_bytes()).hexdigest(),row['sha256'])
if __name__=='__main__':unittest.main()
