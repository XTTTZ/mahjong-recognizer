"""Optional CPU tensor arithmetic/architecture checks; no optimizer/backward/GPU."""
import importlib.util
import unittest
from pathlib import Path

@unittest.skipUnless(importlib.util.find_spec('torch')and importlib.util.find_spec('torchvision'),'optional pinned CPU tensor runtime')
class TensorContracts(unittest.TestCase):
    def test_distillation_preserves_ground_truth_for_wrong_or_uncertain_teacher(self):
        import torch
        from ml.train.student_common import distillation_loss
        torch.set_num_threads(2)
        logits=torch.tensor([[2.,0.,-1.],[.2,2.,-.8]])
        labels=torch.tensor([0,1]);weights=torch.ones(3)
        config={'distillationAlpha':.5,'temperature':2.,'teacherMinimumConfidence':.7}
        wrong=torch.tensor([[-2.,4.,-1.],[4.,-2.,-1.]],requires_grad=True)
        base=distillation_loss(logits,labels,None,weights,config)[0]
        result=distillation_loss(logits,labels,wrong,weights,config)
        self.assertTrue(torch.equal(base,result[0]));self.assertEqual(float(result[3]),0)
        uncertain=torch.zeros(2,3)
        self.assertTrue(torch.equal(base,distillation_loss(logits,labels,uncertain,weights,config)[0]))
        correct=torch.tensor([[4.,-2.,-1.],[-2.,4.,-1.]],requires_grad=True)
        result=distillation_loss(logits,labels,correct,weights,config)
        self.assertTrue(torch.isfinite(result[0]));self.assertEqual(float(result[3]),1)
        self.assertFalse(result[0].requires_grad) # Teacher gradients never enter the loss.

    def test_official_student_checkpoint_loads_strictly_and_outputs42_on_cpu(self):
        import torch
        from ml.train.student_common import build_student,WEIGHTS_FILENAME
        path=Path(__file__).resolve().parents[2]/'artifacts/models/pretrained'/WEIGHTS_FILENAME
        if not path.is_file():self.skipTest('Official offline pretrained asset not present')
        model=build_student(path).cpu().eval()
        self.assertEqual(model.classifier[3].out_features,42)
        with torch.inference_mode():output=model(torch.zeros(1,3,224,224))
        self.assertEqual(tuple(output.shape),(1,42));self.assertTrue(torch.isfinite(output).all())
        self.assertLess(sum(p.numel()for p in model.parameters()),5_000_000)

    def test_fx_static_graph_can_convert_and_script_without_a_training_step(self):
        import torch,warnings
        from ml.train.student_common import build_student
        from torch.ao.quantization import get_default_qconfig_mapping
        from torch.ao.quantization.quantize_fx import prepare_fx,convert_fx
        if 'qnnpack'not in torch.backends.quantized.supported_engines:self.skipTest('No CPU INT8 backend')
        # Mac uses qnnpack for local API coverage; target Windows fbgemm is still
        # a mandatory future runtime check, never claimed from this test.
        torch.set_num_threads(2);torch.backends.quantized.engine='qnnpack'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');model=build_student().eval()
            prepared=prepare_fx(model,get_default_qconfig_mapping('qnnpack'),(torch.zeros(1,3,224,224),))
            with torch.inference_mode():prepared(torch.zeros(1,3,224,224))
            quantized=convert_fx(prepared).eval();scripted=torch.jit.script(quantized)
            with torch.inference_mode():output=scripted(torch.zeros(2,3,224,224))
        self.assertEqual(tuple(output.shape),(2,42));self.assertTrue(torch.isfinite(output).all())
        self.assertGreater(sum('quantized'in type(m).__module__ and 'Conv'in type(m).__name__ for m in quantized.modules()),0)

    def test_student_adapter_preserves_actual_detection_boxes_and_fused_scores(self):
        import torch,tempfile
        from PIL import Image
        from ml.inference.student_model import RawStudentCascade
        from ml.train.student_common import build_student,ARCHITECTURE
        from ml.inference.manifest import FACE_TOKENS
        classes={str(i):name for i,name in enumerate(sorted(FACE_TOKENS))}
        config={'resolution':224,'context':.08}
        class Detector:
            def predict(self,image):return [{'classId':'tile','bbox':[10.,20.,30.,40.],'score':.83},{'classId':'tile','bbox':[50.,30.,25.,45.],'score':.79}]
        with tempfile.TemporaryDirectory()as tmp:
            path=Path(tmp)/'student.pt';torch.save({'architecture':ARCHITECTURE,'model':build_student().state_dict(),'classes':classes,'config':config},path)
            adapter=RawStudentCascade(Detector(),path,classes,config,device='cpu')
            result=adapter.predict(Image.new('RGB',(100,100),'white'))
        self.assertEqual(len(result),2);self.assertEqual(result[0]['bbox'],[10.,20.,30.,40.])
        for row,confidence in zip(result,(.83,.79)):
            self.assertIn(row['classId'],classes.values());self.assertAlmostEqual(row['score'],confidence*row['classifierScore'])


if __name__=='__main__':unittest.main()
