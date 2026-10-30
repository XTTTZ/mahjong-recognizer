"""Meaningful CPU contracts against pinned production preprocessing/postprocessing."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from ml.inference import onnx_runtime as target
from ml.inference.cascade_model import crop_face,image_tensor
from ml.inference.manifest import FACE_TOKENS,file_sha256


class PortableContracts(unittest.TestCase):
    @unittest.skipUnless(all(importlib.util.find_spec(n)for n in ['torch','cv2','ultralytics']),'Optional pinned export environment required')
    def test_letterbox_pixels_and_axis_specific_inverse_match_pinned_ultralytics(self):
        import torch
        from ultralytics.data.augment import LetterBox
        from ultralytics.utils.ops import scale_boxes
        rng=np.random.default_rng(20261030)
        for w,h in [(1001,657),(777,1231),(4032,3024),(1920,1080),(1000,1000)]:
            pixels=rng.integers(0,256,(h,w,3),dtype=np.uint8);image=Image.fromarray(pixels)
            tensor,transform=target.detector_tensor(image)
            expected=LetterBox((960,960),auto=False)(image=pixels)
            np.testing.assert_array_equal(tensor,np.ascontiguousarray(expected.transpose(2,0,1)[None],dtype=np.float32)/np.float32(255.))
            raw=np.zeros((1,5,18900),dtype=np.float32);raw[0,:,0]=[391.3,419.7,127.1,115.9,.9]
            tiles=target.detector_tiles(raw,image.size,transform)
            expectedbox=torch.tensor([[391.3-127.1/2,419.7-115.9/2,391.3+127.1/2,419.7+115.9/2]],dtype=torch.float32)
            # Use same arithmetic as raw model xywh conversion to avoid unrelated literal rounding.
            source=torch.from_numpy(raw[0,:4,0].copy());expectedbox[0,:2]=source[:2]-source[2:]/2;expectedbox[0,2:]=source[:2]+source[2:]/2
            expectedbox=scale_boxes((960,960),expectedbox,(h,w))[0].numpy();expectedxywh=np.concatenate((expectedbox[:2],expectedbox[2:]-expectedbox[:2]))
            np.testing.assert_array_equal(np.array(tiles[0]['bbox'],dtype=np.float32),expectedxywh)
        # This aspect ratio explicitly distinguishes axis-specific from scalar inversion.
        _,axes=target.detector_tensor(Image.new('RGB',(1001,657)))
        self.assertNotEqual(axes[0],axes[1])

    @unittest.skipUnless(importlib.util.find_spec('torch'),'Optional torch parity environment required')
    def test_crop_bytes_and_normalization_match_original_pil_and_torch_contract(self):
        rng=np.random.default_rng(41);image=Image.fromarray(rng.integers(0,256,(161,289,3),dtype=np.uint8))
        detections=[{'bbox':[-2.7,1.2,79.9,111.2]},{'bbox':[181.31,119.48,98.1,40.9]}]
        batch=target.classifier_tensor(image,detections)
        expected=np.stack([image_tensor(crop_face(image,d['bbox'],320,.08)).numpy() for d in detections])
        np.testing.assert_array_equal(batch,expected);self.assertTrue(batch.flags.c_contiguous);self.assertEqual(batch.dtype,np.float32)

    @unittest.skipUnless(all(importlib.util.find_spec(n)for n in ['torch','torchvision']),'Optional torchvision parity environment required')
    def test_nms_matches_torchvision_and_keeps_201_for_guard(self):
        import torch
        from torchvision.ops import nms
        rng=np.random.default_rng(2026);xy=rng.uniform(0,500,(150,2)).astype(np.float32);size=rng.uniform(20,190,(150,2)).astype(np.float32)
        boxes=np.concatenate((xy,xy+size),1);scores=np.linspace(.99,.06,150,dtype=np.float32)
        expected=nms(torch.from_numpy(boxes),torch.from_numpy(scores),.6).tolist()
        self.assertEqual(target.numpy_nms(boxes,scores),expected)
        raw=np.zeros((1,5,18900),dtype=np.float32)
        for i in range(201):raw[0,:,i]=[10+(i%20)*40,10+(i//20)*40,16,16,.9-i*.001]
        self.assertEqual(len(target.detector_tiles(raw,(960,960),(1,1,0,0))),201)

    def test_threshold_is_strict_and_head_finite_dtype_shape_validated(self):
        raw=np.zeros((1,5,18900),dtype=np.float32);raw[0,:,0]=[100,100,20,20,.05]
        self.assertEqual(target.detector_tiles(raw,(960,960),(1,1,0,0)),[])
        raw[0,4,0]=.05001;self.assertEqual(len(target.detector_tiles(raw,(960,960),(1,1,0,0))),1)
        for bad in [raw.astype(np.float16),raw[:,:,:300],np.full(raw.shape,np.nan,dtype=np.float32)]:
            with self.assertRaises(ValueError):target.detector_tiles(bad,(960,960),(1,1,0,0))

    def test_batches_do_not_drop_tiles_and_guard_precedes_classifier(self):
        model=object.__new__(target.OnnxCascadePredictor);model.classifier_batch_size=32;model.manifest={'classes':{str(i):v for i,v in enumerate(sorted(FACE_TOKENS))}}
        model.detector=types.SimpleNamespace(run=lambda *args:[object()]);calls=[]
        def classify(_,feed):
            count=len(feed['images']);calls.append(count);return [np.zeros((count,42),dtype=np.float32)]
        model.classifier=types.SimpleNamespace(run=classify)
        detections=[{'bbox':[0,0,10,10],'score':.9} for _ in range(65)]
        with patch.object(target,'detector_tensor',return_value=(object(),(1,1,0,0))),patch.object(target,'detector_tiles',return_value=detections):
            raw,timing=model.predict_with_timings(Image.new('RGB',(20,20)))
        self.assertEqual(calls,[32,32,1]);self.assertEqual(len(raw),65);self.assertEqual(timing['classifierBatches'],3)
        calls.clear()
        with patch.object(target,'detector_tensor',return_value=(object(),(1,1,0,0))),patch.object(target,'detector_tiles',return_value=detections*4):
            with self.assertRaisesRegex(ValueError,'safety limit'):model.predict_with_timings(Image.new('RGB',(20,20)))
        self.assertEqual(calls,[])

    def test_softmax_stable_and_invalid_outputs_rejected(self):
        logits=np.arange(84,dtype=np.float32).reshape(2,42)*1000
        probabilities=target.softmax(logits)
        expected=np.zeros((2,42));expected[:,-1]=1
        np.testing.assert_array_equal(probabilities,expected)
        for bad in [np.zeros((2,41)),np.full((2,42),np.inf)]:
            with self.assertRaises(ValueError):target.softmax(bad)

    def test_hash_and_geometry_are_fail_closed(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp)
            for name in ['detector.onnx','classifier.onnx']:(root/name).write_bytes(name.encode())
            manifest={'schemaVersion':1,'format':'onnx-fp32-cascade','version':'fixture','classes':{str(i):x for i,x in enumerate(sorted(FACE_TOKENS))},'resolution':960,'classifierResolution':320,'cropContext':.08,'detectorThreshold':.05,'threshold':.35,'acceptThreshold':.7,'nmsIou':.6,'maxDetections':200,'detectorNmsMaxDetections':300,'maxClassifierBatch':32,'opset':17,'filesSha256':{n:file_sha256(root/n)for n in ['detector.onnx','classifier.onnx']}}
            (root/'manifest.json').write_text(json.dumps(manifest));self.assertEqual(target.verify_portable_directory(root)[1],manifest)
            for field,value in [('maxDetections',300),('cropContext',.02),('detectorNmsMaxDetections',200),('opset',18)]:
                bad=copy.deepcopy(manifest);bad[field]=value;(root/'manifest.json').write_text(json.dumps(bad))
                with self.assertRaises(ValueError):target.verify_portable_directory(root)
            (root/'manifest.json').write_text(json.dumps(manifest));(root/'classifier.onnx').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'SHA mismatch'):target.verify_portable_directory(root)

if __name__=='__main__':unittest.main()
