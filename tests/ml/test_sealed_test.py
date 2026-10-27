"""Prevent test tuning and changed data from passing a frozen winner contract."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from ml.eval.sealed_test import freeze_lock, test_contract
from ml.eval.yolo_evaluate import evaluate
from ml.inference.manifest import FACE_TOKENS
from ml.train.provenance import sha256, dataset_contract


def dataset(root):
    classes={str(i):token for i,token in enumerate(sorted(FACE_TOKENS))}
    for index,split in enumerate(('train','valid','test')):
        directory=root/split;directory.mkdir(parents=True)
        image=directory/'source.png';Image.new('RGB',(40,60),(100+index*40,200,100)).save(image)
        coco={'categories':[{'id':int(i)+1,'name':token} for i,token in classes.items()],
            'images':[{'id':1,'file_name':'source.png','width':40,'height':60,'group_id':split,'sha256':sha256(image)}],
            'annotations':[{'id':1,'image_id':1,'category_id':1,'bbox':[5,10,20,30]}]}
        (directory/'_annotations.coco.json').write_text(json.dumps(coco))
    return classes


class SealedContractTests(unittest.TestCase):
    def test_fixed_operating_point_does_not_call_threshold_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);classes=dataset(root/'data')
            class Predictor:
                def predict(self,image):return [{'classId':classes['0'],'bbox':[5,10,20,30],'score':.95}]
            metadata={'dataContract':dataset_contract(root/'data',True),'winnerLockSha256':'f'*64}
            with patch('ml.eval.coco_metrics.coco_metrics',return_value={}),patch('ml.eval.yolo_evaluate.choose_threshold',side_effect=AssertionError('must not tune test')),contextlib.redirect_stdout(io.StringIO()):
                report=evaluate(Predictor(),root/'data',root/'result',metadata,thresholds=(.5,),split='test',select_threshold=False,
                    expected_annotation_sha256=sha256(root/'data/test/_annotations.coco.json'))
            self.assertEqual(report['split'],'sealed_test_only');self.assertEqual(report['bestThreshold'],.5)
            self.assertFalse(report['testUsedForSelection']);self.assertTrue(report['operatingPointLocked'])

    def test_test_without_lock_or_with_threshold_grid_is_rejected_before_reading(self):
        for thresholds,metadata,select in [((.5,),{},False),((.3,.5),{'winnerLockSha256':'a'},False),((.5,),{'winnerLockSha256':'a'},True)]:
            with tempfile.TemporaryDirectory() as temporary,self.assertRaises(ValueError):
                evaluate(None,Path(temporary)/'absent',Path(temporary)/'out',metadata,thresholds=thresholds,
                    split='test',select_threshold=select,expected_annotation_sha256='a'*64)

    def test_test_pixel_and_annotation_hashes_and_group_separation_are_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);classes=dataset(root)
            annotation=root/'test/_annotations.coco.json';expected=sha256(annotation)
            proof=test_contract(root,expected,classes);self.assertEqual(proof['images'],1)
            Image.new('RGB',(40,60),'red').save(root/'test/source.png')
            with self.assertRaisesRegex(ValueError,'bytes'):test_contract(root,expected,classes)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);classes=dataset(root);annotation=root/'test/_annotations.coco.json'
            data=json.loads(annotation.read_text());data['images'][0]['group_id']='train';annotation.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'group leakage'):test_contract(root,sha256(annotation),classes)

    def test_winner_lock_is_immutable_and_does_not_open_test_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);model=root/'model';model.mkdir()
            classes={str(i):token for i,token in enumerate(sorted(FACE_TOKENS))}
            (model/'weights.pt').write_bytes(b'model')
            report={'evaluationDevice':'cuda','partial':False,'split':'validation_only','bestThreshold':.5,'acceptThreshold':.7}
            (model/'validation-report.json').write_text(json.dumps(report))
            manifest={'architecture':'yolo26s','format':'ultralytics-pt','version':'locked-v1','classes':classes,'resolution':640,
                'threshold':.5,'acceptThreshold':.7,'filesSha256':{'weights.pt':sha256(model/'weights.pt')},
                'validationReportSha256':sha256(model/'validation-report.json')}
            (model/'model.json').write_text(json.dumps(manifest))
            metadata=root/'metadata.json';metadata.write_text(json.dumps({'prepared_splits':{'test':{'annotation_sha256':'f'*64}}}))
            value=freeze_lock(model,metadata,root/'winner.json')
            self.assertFalse(value['testOpened']);self.assertFalse(value['testMayTuneThresholdOrModel'])
            with self.assertRaises(FileExistsError):freeze_lock(model,metadata,root/'winner.json')


if __name__=='__main__':unittest.main()
