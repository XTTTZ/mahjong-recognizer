"""V6 pure-CPU data, provenance, augmentation and budget tests; no model forward/backward."""
from collections import Counter
import hashlib,json,random,tempfile,unittest,subprocess,sys
from pathlib import Path,PureWindowsPath
from PIL import Image,ImageDraw
from ml.inference.manifest import FACE_TOKENS
from ml.inference.cascade_model import crop_face
from ml.train.provenance import dataset_contract,sha256
from ml.train.yolo_train import checkpoint_receipt
from ml.train.v6_common import completed_epoch_budget,parent_contract
from ml.train.v6_augment import camera_augment,cached_crop_face
from ml.data.prepare_yolo import prepare as yolo_prepare
from ml.data.prepare_crops import prepare as crop_prepare
from ml.data.prepare_background_v6 import prepare as bg_prepare,label_tree_sha256
from ml.data.prepare_crop_cache_v6 import prepare as cache_prepare,envelope,verify_cache


def fixture(root):
    classes={str(i):name for i,name in enumerate(sorted(FACE_TOKENS))}
    for number,split in enumerate(('train','valid')):
        path=root/split;path.mkdir(parents=True)
        image=Image.new('RGB',(100,120))
        for y in range(120):
            for x in range(100):image.putpixel((x,y),((x+number*20)%256,y%256,(x+y)%256))
        image.save(path/'source.png')
        data={'categories':[{'id':int(i)+1,'name':name}for i,name in classes.items()],
              'images':[{'id':1,'file_name':'source.png','width':100,'height':120,'group_id':split,'source_id':split+'/source.png'}],
              'annotations':[{'id':1,'image_id':1,'category_id':1,'bbox':[12.3,22.7,32.5,41.2]}]}
        (path/'_annotations.coco.json').write_text(json.dumps(data))
    (root/'test').mkdir();(root/'test/_annotations.coco.json').write_text('SEALED_DO_NOT_OPEN')
    return dataset_contract(root,True)


def review(root,contract):
    return {'trainAnnotationSha256':contract['splits']['train']['annotationSha256'],
            'trainImageTreeSha256':contract['splits']['train']['imageTreeSha256'],
            'contactSheetsVisuallyReviewed':True,
            'accepted':[{'id':'bg-000','sourceFile':'source.png','sourceGroup':'train',
                         'sourceSha256':sha256(root/'train/source.png'),'cropXYXY':[60,75,100,120]}]}


class BackgroundTests(unittest.TestCase):
    def test_windows_label_tree_keeps_the_native_v5_verifier_protocol(self):
        path=PureWindowsPath('labels/train/negative-bg-000.txt')
        expected=hashlib.sha256(json.dumps([['labels\\train\\negative-bg-000.txt','pixel-hash']]).encode()).hexdigest()
        posix=hashlib.sha256(json.dumps([['labels/train/negative-bg-000.txt','pixel-hash']]).encode()).hexdigest()
        self.assertEqual(label_tree_sha256([(path,'pixel-hash')]),expected)
        self.assertNotEqual(expected,posix)

    def test_reviewed_negative_is_train_only_empty_label_and_base_unchanged(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);contract=fixture(root/'data');yolo_prepare(root/'data',root/'base',generic=True)
            before=sha256(root/'base/prepared.json');r=root/'review.json';r.write_text(json.dumps(review(root/'data',contract)))
            result=bg_prepare(root/'data',root/'base',r,root/'out')
            self.assertEqual(result['backgrounds'],1);self.assertEqual(before,sha256(root/'base/prepared.json'))
            self.assertEqual((root/'out/labels/train/negative-bg-000.txt').read_text(),'')
            self.assertFalse((root/'out/images/test').exists())
            self.assertTrue(bg_prepare(root/'data',root/'base',r,root/'out',True)['derivativeReady'])

    def test_review_mismatch_unreviewed_outside_train_and_positive_intersection_are_rejected(self):
        for kind in ('mismatch','unreviewed','valid_pixels','valid_group','overlap','duplicate','unsafe_id'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory()as tmp:
                root=Path(tmp);contract=fixture(root/'data');yolo_prepare(root/'data',root/'base',True);r=review(root/'data',contract)
                if kind=='mismatch':r['trainImageTreeSha256']='bad'
                if kind=='unreviewed':r['contactSheetsVisuallyReviewed']=False
                if kind=='valid_pixels':r['accepted'][0]['sourceSha256']=sha256(root/'data/valid/source.png')
                if kind=='valid_group':r['accepted'][0]['sourceGroup']='valid'
                if kind=='overlap':r['accepted'][0]['cropXYXY']=[0,0,50,70]
                if kind=='duplicate':r['accepted']*=2
                if kind=='unsafe_id':r['accepted'][0]['id']='../escape'
                path=root/'review.json';path.write_text(json.dumps(r))
                with self.assertRaises(ValueError):bg_prepare(root/'data',root/'base',path,root/'out')
                self.assertFalse((root/'out').exists())

    def test_preflight_never_materializes_a_derivative(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);contract=fixture(root/'data');yolo_prepare(root/'data',root/'base',True)
            path=root/'review.json';path.write_text(json.dumps(review(root/'data',contract)))
            self.assertFalse(bg_prepare(root/'data',root/'base',path,root/'missing',True)['derivativeReady'])
            self.assertFalse((root/'missing').exists())


class CacheAndAugmentationTests(unittest.TestCase):
    def test_envelope_pixels_equal_original_crop_including_edges_context_and_jitter(self):
        image=Image.new('RGB',(200,160))
        for y in range(160):
            for x in range(200):image.putpixel((x,y),(x%256,y%256,(x+y)%256))
        boxes=[[12.3,22.7,32.5,41.2],[0,0,80,60],[150,110,50,50],[55,40,100,80]]
        for box in boxes:
            rect=envelope(box,image.width,image.height);cached=image.crop(rect)
            local=[box[0]-rect[0],box[1]-rect[1],box[2],box[3]]
            for context in (.02,.08,.16):
                for seed in range(8):
                    a=crop_face(image,box,96,context,random.Random(seed))
                    b=cached_crop_face(cached,{'bbox':box,'cacheEnvelopeXYXY':rect,'sourceImageSize':image.size},96,context,random.Random(seed))
                    self.assertEqual(a.tobytes(),b.tobytes())

    def test_real_annotation2641_global_rounding_regression(self):
        box=[1435.3,1609.5,1016.25,786.73];image=Image.new('RGB',(3500,2800));draw=ImageDraw.Draw(image)
        for x in range(image.width):draw.line((x,0,x,image.height),fill=(x%251,(x*7)%253,(x*11)%255))
        rect=envelope(box,image.width,image.height);cached=image.crop(rect)
        self.assertEqual(rect[0],1079)
        for context in (.02,.08,.16):
            for seed in (None,0,1,2,3):
                first=random.Random(seed)if seed is not None else None
                second=random.Random(seed)if seed is not None else None
                reference=crop_face(image,box,96,context,first)
                actual=cached_crop_face(cached,{'bbox':box,'cacheEnvelopeXYXY':rect,'sourceImageSize':image.size},96,context,second)
                self.assertEqual(reference.tobytes(),actual.tobytes())

    def test_cache_freezes_source_and_pixel_hashes_without_test_access(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);contract=fixture(root/'data');crop_prepare(root/'data',root/'indices')
            result=cache_prepare(root/'data',root/'indices',root/'cache');self.assertTrue(result['derivativeReady'])
            verify_cache(root/'cache',contract,root/'indices')
            (root/'cache/train/1.png').write_bytes(b'tamper')
            with self.assertRaises(ValueError):verify_cache(root/'cache',contract,root/'indices')

    def test_legacy_cache_schema_and_missing_global_fields_are_rejected(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);contract=fixture(root/'data');crop_prepare(root/'data',root/'indices');cache_prepare(root/'data',root/'indices',root/'cache')
            meta_path=root/'cache/prepared.json';meta=json.loads(meta_path.read_text());meta['schemaVersion']=1;meta_path.write_text(json.dumps(meta))
            with self.assertRaises(ValueError):verify_cache(root/'cache',contract,root/'indices')
            meta['schemaVersion']=2;index_path=root/'cache/train.json';index=json.loads(index_path.read_text());index['records'][0].pop('sourceImageSize')
            index_path.write_text(json.dumps(index));meta['indices']['train']['sha256']=sha256(index_path);meta_path.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'global-coordinate'):verify_cache(root/'cache',contract,root/'indices')

    def test_camera_augmentation_is_reproducible_and_keeps_square_complete_input(self):
        image=Image.new('RGB',(96,96),'black')
        a=camera_augment(image,random.Random(4),'mild');b=camera_augment(image,random.Random(4),'mild')
        self.assertEqual(a.size,(96,96));self.assertEqual(a.tobytes(),b.tobytes())
        self.assertIs(camera_augment(image,random.Random(4),'none'),image)
        with self.assertRaises(ValueError):camera_augment(image,random.Random(4),'unknown')


class PolicyTests(unittest.TestCase):
    def test_budget_is_checked_only_after_a_complete_epoch_and_reserves_next_epoch(self):
        self.assertIsNone(completed_epoch_budget(1,12,60,10,60))
        self.assertEqual(completed_epoch_budget(2,12,550,10,60),'budget_finished')
        self.assertEqual(completed_epoch_budget(12,12,550,10,60),'epoch_limit_reached')
        for args in [(0,12,20,10,10),(1,12,20,float('nan'),10)]:
            with self.assertRaises(ValueError):completed_epoch_budget(*args)

    def test_both_actual_preflight_entrypoints_finish_without_loading_a_model(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);contract=fixture(root/'data');yolo_prepare(root/'data',root/'yolo',True)
            crop_prepare(root/'data',root/'crops');cache_prepare(root/'data',root/'crops',root/'cache')
            for generic,module in [(True,'ml.train.yolo_train_v6'),(False,'ml.train.cascade_train_v6')]:
                folder=root/('detector'if generic else'classifier');folder.mkdir()
                signature={'data':contract['splits']};metadata={'classes':contract['classes'],'generic':generic,
                    'architecture':'yolov8s-generic'if generic else'efficientnet-v2-s','finishedAt':'2026-10-27T00:00:00Z',
                    'resumeSignature':signature,'annotationSha256':contract['splits']['train']['annotationSha256'],
                    'validationAnnotationSha256':contract['splits']['valid']['annotationSha256']}
                (folder/'run.json').write_text(json.dumps(metadata));(folder/'parent.pt').write_bytes(b'not-a-model-preflight-must-never-load-it')
                checkpoint_receipt(folder/'parent.pt',signature,{'0':'tile'}if generic else contract['classes'])
                output=root/('output-detector'if generic else'output-classifier')
                argv=[sys.executable,'-m',module,'--dataset',str(root/'data'),'--parent-run',str(folder/'run.json'),
                      '--parent-checkpoint',str(folder/'parent.pt'),'--output',str(output),'--preflight-only']
                argv+=['--prepared',str(root/'yolo')]if generic else['--crops',str(root/'crops'),'--cache',str(root/'cache')]
                result=subprocess.run(argv,capture_output=True,text=True,timeout=60)
                self.assertEqual(result.returncode,0,result.stderr)
                receipt=json.loads(result.stdout);self.assertFalse(receipt['trainingStarted']);self.assertTrue(receipt['derivativeReady'])
                self.assertFalse(output.exists())

    def test_terminal_parent_is_a_new_warm_start_and_wrong_receipt_is_rejected(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);contract=fixture(root/'data');run=root/'parent';run.mkdir()
            signature={'data':contract['splits']};meta={'classes':contract['classes'],'resumeSignature':signature,
                'annotationSha256':contract['splits']['train']['annotationSha256'],
                'validationAnnotationSha256':contract['splits']['valid']['annotationSha256']}
            (run/'run.json').write_text(json.dumps(meta));(run/'last.pt').write_bytes(b'not-loaded-as-a-model')
            checkpoint_receipt(run/'last.pt',signature,contract['classes'])
            with self.assertRaises(ValueError):parent_contract(run/'run.json',run/'last.pt',contract)
            status={'state':'timed_out','child_pid':None,'finished_at':'2026-10-27T03:42:36Z','run_dir':str(run)}
            (run/'status.json').write_text(json.dumps(status))
            _,evidence=parent_contract(run/'run.json',run/'last.pt',contract,run/'status.json',generic=False)
            self.assertEqual(evidence['sourceStatus']['trainingCompletion'],'terminal_checkpoint_warm_start')
            status['run_dir']=str(root/'wrong');(run/'status.json').write_text(json.dumps(status))
            with self.assertRaises(ValueError):parent_contract(run/'run.json',run/'last.pt',contract,run/'status.json')

if __name__=='__main__':unittest.main()
