"""Actual HTTP evaluator contracts with inert transport; never cloud calls."""
import json
from pathlib import Path
import tempfile
import time
import unittest

import httpx

from infra.portable import http_evaluate as target
from ml.eval.metrics import Tile


IMAGE = {'width': 100, 'height': 100}
TRUTH = [Tile('1m', (1, 2, 20, 30))]


def response(request_id='fixed', **updates):
    value = {'requestId':request_id, 'modelVersion':'selected', 'runtimeBackend':'onnxruntime-cpu',
             'image':IMAGE, 'tiles':[{'classId':'1m','bbox':[1,2,20,30],'score':.9}],
             'quality':{'status':'ok','reasons':[]},
             'timings':{'inferenceMs':400,'postprocessMs':2,'decodeMs':3,'totalMs':500}}
    value.update(updates)
    return value


class HttpEvaluationTests(unittest.TestCase):
    def test_private_reference_is_optional_and_public_control_shapes_are_supported(self):
        self.assertIsNone(target.baseline_rows(None))
        rows = [{'imageId': i, 'metrics': {}} for i in range(150)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'control.json'
            for per_image in (rows, {'baseline': rows}, {'eff320': rows}):
                path.write_text(json.dumps({'annotationSha256':target.ANNOTATION_SHA,'perImage':per_image}))
                self.assertEqual(target.baseline_rows(path),rows)
            path.write_text(json.dumps({'annotationSha256':'changed','perImage':rows}))
            with self.assertRaises(ValueError):target.baseline_rows(path)

    def test_endpoint_rejects_credentials_redirect_locations_and_non_modal(self):
        self.assertEqual(target.validate_endpoint('https://unit.modal.run/recognize'),
                         'https://unit.modal.run/recognize')
        for endpoint in ('http://unit.modal.run/recognize', 'https://unit.modal.run.evil/recognize',
                         'https://secret@unit.modal.run/recognize',
                         'https://unit.modal.run/recognize?token=secret'):
            with self.assertRaises(ValueError): target.validate_endpoint(endpoint)

    def test_scores_returned_boxes_and_quality_without_rewriting_acceptance(self):
        body = response(quality={'status':'needs_retake','reasons':['uncertain']})
        metric = target.decode_response(body, request_id='fixed',version='selected',image=IMAGE,truth=TRUTH)
        self.assertTrue(metric['model_exact'])
        self.assertFalse(metric['hand_exact_match'])
        # Same class multiset but wrong boxes is not an exact hand.
        body = response(tiles=[{'classId':'1m','bbox':[70,70,20,20],'score':.9}])
        metric = target.decode_response(body,request_id='fixed',version='selected',image=IMAGE,truth=TRUTH)
        self.assertTrue(metric['multiset_exact']); self.assertFalse(metric['model_exact'])

    def test_identity_dimensions_and_invalid_timings_fail_closed(self):
        for change in ({'modelVersion':'wrong'}, {'requestId':'wrong'}, {'runtimeBackend':'torch'},
                       {'image':{'width':99,'height':100}},
                       {'quality':{'status':'ok','reasons':['bad']}},
                       {'timings':{'inferenceMs':float('nan'),'postprocessMs':0,'decodeMs':0,'totalMs':1}}):
            with self.assertRaises(ValueError):
                target.decode_response(response(**change),request_id='fixed',version='selected',image=IMAGE,truth=TRUTH)

    def test_transport_retry_is_bounded_and_secret_never_enters_checkpoints(self):
        seen = []
        def handler(request):
            seen.append(request)
            if len(seen) == 1: raise httpx.ConnectError('do not log this private error',request=request)
            self.assertEqual(request.headers['authorization'],'Bearer secret-value')
            self.assertIn(b'name="image"',request.content)
            return httpx.Response(200,json=response(request.headers['x-request-id']))
        with httpx.Client(transport=httpx.MockTransport(handler),headers={'Authorization':'Bearer secret-value'}) as client:
            row = target.request_image(client,'https://unit.modal.run/recognize',{'id':0,'data':b'private-pixels'},
                                       IMAGE,TRUTH,'selected',time.monotonic()+20,30,True)
        self.assertEqual(len(seen),2); self.assertTrue(row['metrics']['hand_exact_match'])
        self.assertFalse(row['warmEligible'])
        encoded = json.dumps(row)
        for private in ('secret-value','private-pixels','do not log this private error'):
            self.assertNotIn(private,encoded)

    def test_http_failure_stays_in_denominator_is_not_retried_or_timed(self):
        with httpx.Client(transport=httpx.MockTransport(lambda request:httpx.Response(503))) as client:
            row = target.request_image(client,'https://unit.modal.run/recognize',{'id':0,'data':b'pixels'},
                                       IMAGE,TRUTH,'selected',time.monotonic()+20,30,True)
        self.assertEqual(len(row['attempts']),1)
        result = target.summary([row])
        self.assertEqual((result['images'],result['successes'],result['failed_requests']),(1,0,1))
        self.assertEqual(result['warmTimings']['inferenceMs']['samples'],0)

    def test_exhausted_budget_still_scores_failure_and_checkpoint_is_immutable(self):
        with httpx.Client(transport=httpx.MockTransport(lambda request:self.fail('must not send'))) as client:
            row = target.request_image(client,'https://unit.modal.run/recognize',{'id':0,'data':b'pixels'},
                                       IMAGE,TRUTH,'selected',time.monotonic()-1,30,True)
        self.assertTrue(row['metrics']['failed']); self.assertEqual(row['metrics']['truth_count'],1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'checkpoint.json'
            target.write_fresh(path,row)
            with self.assertRaises(FileExistsError):target.write_fresh(path,{'changed':True})
            self.assertEqual(json.loads(path.read_text()),row)


if __name__ == '__main__': unittest.main()
