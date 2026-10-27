"""COCO's unmatched marker must never collide with a source annotation ID0."""
import copy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ml.eval.coco_metrics import coco_metrics


class CocoIdTests(unittest.TestCase):
    def test_evaluator_gets_nonzero_unique_annotation_ids_without_changing_truth(self):
        captured=[]
        class COCO:
            def __init__(self,data):captured.append(data)
            def loadRes(self,predictions):return predictions
        class Evaluation:
            def __init__(self,*args,**kwargs):self.stats=[1.,1.,1.]
            def evaluate(self):pass
            def accumulate(self):pass
            def summarize(self):pass
        annotations=[{'id':0,'image_id':0,'category_id':1,'bbox':[1,2,3,4]},
                     {'id':5,'image_id':0,'category_id':1,'bbox':[7,2,3,4]}]
        original=copy.deepcopy(annotations)
        with patch.dict(sys.modules,{'faster_coco_eval':SimpleNamespace(COCO=COCO,COCOeval_faster=Evaluation)}):
            report=coco_metrics([{'id':0}],annotations,[{'image_id':0}],{'0':'tile'})
        self.assertEqual(annotations,original)
        self.assertEqual([a['id'] for a in captured[0]['annotations']],[1,2])
        self.assertEqual([a['bbox'] for a in captured[0]['annotations']],[a['bbox'] for a in original])
        self.assertEqual(report['mAP50_95'],1.)


if __name__=='__main__':unittest.main()
