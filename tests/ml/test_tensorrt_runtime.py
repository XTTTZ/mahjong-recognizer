"""CPU mocks validate real compatibility contracts; no SDK, GPU or model forward."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from ml.inference import tensorrt_runtime as target


def metadata():
    return {'names':{'0':'tile'},'task':'detect','batch':1,'imgsz':[960,960],
        'channels':3,'end2end':False,'args':{'nms':None,'quantize':32},'description':'保留YOLO输出'}


def manifest():
    return {'version':'fixture','filesSha256':{'detector.pt':'a'*64,'classifier.pt':'b'*64},
            'resolution':960,'classifierResolution':320,'cropContext':.08,'classes':{'0':'fixture'}}


def hardware(name='NVIDIA A10G',sms=80,cc=(8,6)):
    return {'gpu':name,'smCount':sms,'computeCapability':list(cc)}


def fake_trt():
    configs=[]
    class Config:
        hardware_compatibility_level='NONE'
        def set_flag(self,value):self.flag=value
        def set_memory_pool_limit(self,pool,size):self.workspace=(pool,size)
        def add_optimization_profile(self,profile):self.profile=profile;return 0
    class Profile:
        def set_shape(self,name,minimum,optimum,maximum):self.value=(name,minimum,optimum,maximum)
    class Builder:
        def __init__(self,logger):pass
        def create_network(self,flags):return types.SimpleNamespace(kind=None)
        def create_builder_config(self):value=Config();configs.append(value);return value
        def create_optimization_profile(self):return Profile()
        def build_serialized_network(self,network,config):return b'plan-'+network.kind
    class Parser:
        def __init__(self,network,logger):self.network=network;self.num_errors=0
        def parse(self,data):self.network.kind=data;return True
    class Logger:
        WARNING=1
        def __init__(self,*args):pass
    class Engine:
        hardware_compatibility_level='AMPERE_PLUS'
        num_io_tensors=2
        def __init__(self,kind):self.kind=kind
        def get_tensor_name(self,i):return ['images','logits'if self.kind=='classifier'else'output0'][i]
        def get_tensor_dtype(self,name):return 'float32'
        def get_tensor_mode(self,name):return 'input'if name=='images'else'output'
        def get_tensor_location(self,name):return 'device'
        def get_tensor_format(self,name):return 'linear'
        def get_tensor_shape(self,name):
            if name=='images':return (-1,3,320,320)if self.kind=='classifier'else(1,3,960,960)
            return (-1,42)if self.kind=='classifier'else(1,5,18900)
        def get_tensor_profile_shape(self,name,index):return [(1,3,320,320),(14,3,320,320),(32,3,320,320)]
        def create_execution_context(self):return object()
    class Runtime:
        def __init__(self,logger):pass
        def deserialize_cuda_engine(self,plan):return Engine('classifier'if plan==b'plan-classifier'else'detector')
    module=types.SimpleNamespace(__version__='10.13.3.9',Builder=Builder,OnnxParser=Parser,Logger=Logger,Runtime=Runtime,
        HardwareCompatibilityLevel=types.SimpleNamespace(AMPERE_PLUS='AMPERE_PLUS'),
        NetworkDefinitionCreationFlag=types.SimpleNamespace(EXPLICIT_BATCH=0),MemoryPoolType=types.SimpleNamespace(WORKSPACE='workspace'),
        BuilderFlag=types.SimpleNamespace(FP16='fp16'),TensorLocation=types.SimpleNamespace(DEVICE='device'),
        TensorFormat=types.SimpleNamespace(LINEAR='linear'),TensorIOMode=types.SimpleNamespace(INPUT='input'),float32='float32',float16='float16')
    return module,configs,Engine


def receipt():
    trt,_,Engine=fake_trt();m=manifest()
    return {'schemaVersion':2,'hardwareCompatibility':'AMPERE_PLUS',
        'engineHardwareCompatibility':{name:'AMPERE_PLUS'for name in target.ENGINE_FILES},**hardware(),
        'modelVersion':m['version'],'modelFilesSha256':m['filesSha256'],'precision':'fp16','calibrationDataUsed':False,
        'maxBatch':32,'optimalBatch':14,'enginesSha256':{name:'c'*64 for name in target.ENGINE_FILES},
        'torch':'2.8.0','cuda':'12.8','tensorrt':'10.13.3.9',
        'ioContracts':{'detector.engine':target.engine_contract(Engine('detector'),trt,'detector',960,metadata()),
                       'classifier.engine':target.engine_contract(Engine('classifier'),trt,'classifier',320)}}


class TensorRTCompatibilityTests(unittest.TestCase):
    def test_yolo_header_roundtrip_preserves_metadata_and_original_plan(self):
        with tempfile.TemporaryDirectory()as temp:
            path=Path(temp)/'detector.engine';target.write_detector_engine(path,metadata(),b'original-plan')
            raw=path.read_bytes();length=int.from_bytes(raw[:4],'little',signed=True)
            self.assertEqual(json.loads(raw[4:4+length]),metadata());self.assertEqual(raw[4+length:],b'original-plan')
            self.assertEqual(target.read_detector_engine(path),(metadata(),b'original-plan'))
            for invalid in (b'',b'abcd',(-1).to_bytes(4,'little',signed=True)+b'junk',b'\xff\xff\xff\x7fgarbage'):
                path.write_bytes(invalid)
                with self.assertRaises(ValueError):target.read_detector_engine(path)

    def test_both_builders_set_actual_hardware_flag_and_classifier_profile(self):
        trt,configs,_=fake_trt()
        with tempfile.TemporaryDirectory()as temp:
            for kind,size in [('detector',960),('classifier',320)]:
                path=Path(temp)/(kind+'.onnx');path.write_bytes(kind.encode())
                self.assertEqual(target._build_plan(path,trt,object(),kind,size),b'plan-'+kind.encode())
        self.assertEqual(len(configs),2)
        for config in configs:
            self.assertEqual(config.hardware_compatibility_level,'AMPERE_PLUS');self.assertEqual(config.flag,'fp16')
            self.assertEqual(config.workspace,('workspace',2*1024**3))
        self.assertFalse(hasattr(configs[0],'profile'))
        self.assertEqual(configs[1].profile.value,('images',(1,3,320,320),(14,3,320,320),(32,3,320,320)))

    def test_actual_engine_property_and_io_must_match_even_when_receipt_claims_compatibility(self):
        trt,_,Engine=fake_trt();engine=Engine('classifier')
        engine.hardware_compatibility_level='NONE'
        with self.assertRaisesRegex(ValueError,'actually AMPERE_PLUS'):target.engine_contract(engine,trt,'classifier',320)
        engine.hardware_compatibility_level='AMPERE_PLUS';engine.get_tensor_dtype=lambda name:'float16'
        with self.assertRaisesRegex(ValueError,'FP32 NCHW'):target.engine_contract(engine,trt,'classifier',320)
        engine=Engine('classifier');engine.get_tensor_shape=lambda name:(-1,3,320,320)if name=='images'else(-1,41)
        with self.assertRaisesRegex(ValueError,'batch/class'):target.engine_contract(engine,trt,'classifier',320)
        engine=Engine('classifier');engine.get_tensor_format=lambda name:'CHW2'
        with self.assertRaisesRegex(ValueError,'contiguous'):target.engine_contract(engine,trt,'classifier',320)

    def test_80sm_to72sm_is_allowed_only_with_real_explicit_compatibility(self):
        versions={'torch':'2.8.0','cuda':'12.8','tensorrt':'10.13.3.9'}
        target.validate_runtime_receipt(receipt(),manifest(),hardware('NVIDIA A10',72),versions)
        for field,bad in [('hardwareCompatibility','NONE'),('schemaVersion',1),('engineHardwareCompatibility',{}),('smCount',0),('computeCapability',[7,5]),('enginesSha256',{}),('ioContracts',{}),('tensorrt','10.12.0')]:
            value=receipt();value[field]=bad
            with self.subTest(field=field),self.assertRaises(ValueError):target.validate_runtime_receipt(value,manifest(),hardware('NVIDIA A10',72),versions)
        with self.assertRaises(ValueError):target.validate_runtime_receipt(receipt(),manifest(),hardware('NVIDIA T4',40,(7,5)),versions)

    def test_metadata_rejects_changed_nms_branch_labels_or_geometry(self):
        target._validate_detector_metadata(metadata(),manifest())
        for field,value in [('names',{'0':'other'}),('imgsz',[640,640]),('batch',2),('end2end',None),('args',{'nms':False})]:
            row=metadata();row[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):target._validate_detector_metadata(row,manifest())

    def test_build_export_callback_keeps_fp32_graph_nms_none_and_writes_strict_receipt(self):
        trt,configs,_=fake_trt();calls={}
        class YOLO:
            def __init__(self,path,task):self.path=Path(path)
            def add_callback(self,event,function):calls['event']=event;self.callback=function
            def export(self,**options):
                calls['options']=options;self.callback(types.SimpleNamespace(metadata=metadata()))
                path=self.path.with_suffix('.onnx');path.write_bytes(b'detector');return str(path)
        class Classifier:
            def load_state_dict(self,state):pass
            def eval(self):return self
            def cuda(self):return self
        torch=types.SimpleNamespace(__version__='2.8.0',version=types.SimpleNamespace(cuda='12.8'),
            cuda=types.SimpleNamespace(is_available=lambda:True,current_device=lambda:0,get_device_name=lambda d:'NVIDIA A10G',
                get_device_properties=lambda d:types.SimpleNamespace(multi_processor_count=80),get_device_capability=lambda d:(8,6)),
            load=lambda *args,**kwargs:{'model':{},'classes':manifest()['classes'],'config':{'resolution':320,'context':.08}},
            zeros=lambda *args,**kwargs:object(),onnx=types.SimpleNamespace(export=lambda model,inputs,path,**kwargs:Path(path).write_bytes(b'classifier')))
        with tempfile.TemporaryDirectory()as temp:
            directory=Path(temp);(directory/'detector.pt').write_bytes(b'pt');(directory/'classifier.pt').write_bytes(b'pt')
            out=directory/'engines';out.mkdir();(out/'runtime.json').write_text('old unsafe receipt')
            with patch.dict(sys.modules,{'torch':torch,'tensorrt':trt,'ultralytics':types.SimpleNamespace(YOLO=YOLO)}),\
                 patch('ml.inference.cascade_model.verify_model_directory',return_value=(directory,manifest())),\
                 patch('ml.inference.cascade_model.build_classifier',return_value=Classifier()),contextlib.redirect_stdout(io.StringIO()):
                target.build_engines(directory,out)
            value=json.loads((out/'runtime.json').read_text())
            self.assertEqual(calls['event'],'on_export_end');self.assertEqual(calls['options']['format'],'onnx')
            self.assertEqual(calls['options']['quantize'],32);self.assertIsNone(calls['options']['nms'])
            self.assertFalse(calls['options']['dynamic']);self.assertEqual(value['smCount'],80)
            self.assertEqual(value['hardwareCompatibility'],'AMPERE_PLUS');self.assertEqual(len(configs),2)
            self.assertEqual(target.read_detector_engine(out/'detector.engine'),(metadata(),b'plan-detector'))
            self.assertEqual((out/'classifier.engine').read_bytes(),b'plan-classifier')
            target.validate_runtime_receipt(value,manifest(),hardware('NVIDIA A10',72),{'torch':'2.8.0','cuda':'12.8','tensorrt':'10.13.3.9'})

    def test_activation_checks_real_engine_flag_hashes_and_preserves_original_on_failure(self):
        trt,_,Engine=fake_trt();calls=[]
        torch=types.SimpleNamespace(__version__='2.8.0',version=types.SimpleNamespace(cuda='12.8'),
            cuda=types.SimpleNamespace(is_available=lambda:True,current_device=lambda:0,get_device_name=lambda d:'NVIDIA A10',
                get_device_properties=lambda d:types.SimpleNamespace(multi_processor_count=72),get_device_capability=lambda d:(8,6),empty_cache=lambda:None))
        class YOLO:
            def __init__(self,path,task):calls.append(str(path))
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);target.write_detector_engine(root/'detector.engine',metadata(),b'plan-detector')
            (root/'classifier.engine').write_bytes(b'plan-classifier');value=receipt()
            value['detectorMetadataSha256']=hashlib.sha256(target._metadata_bytes(metadata())).hexdigest()
            value['enginesSha256']={name:hashlib.sha256((root/name).read_bytes()).hexdigest()for name in target.ENGINE_FILES}
            def prediction():return types.SimpleNamespace(manifest=manifest(),version='fixture',detector=types.SimpleNamespace(model='original-detector'),classifier='original-classifier',runtime_backend='torch-bf16')
            (root/'runtime.json').write_text(json.dumps(value))
            with patch.dict(sys.modules,{'torch':torch,'tensorrt':trt,'ultralytics':types.SimpleNamespace(YOLO=YOLO)}):
                model=prediction();result=target.activate_tensorrt(model,root)
                self.assertTrue(result['hardwareCompatibilityVerified']);self.assertEqual(result['runtimeHardware']['smCount'],72)
                self.assertEqual(result['smCount'],80);self.assertEqual(model.runtime_backend,'tensorrt-fp16')
                calls.clear();Engine.hardware_compatibility_level='NONE';model=prediction()
                with self.assertRaisesRegex(ValueError,'actually AMPERE_PLUS'):target.activate_tensorrt(model,root)
                self.assertEqual(calls,[]);self.assertEqual(model.classifier,'original-classifier')
                Engine.hardware_compatibility_level='AMPERE_PLUS';model=prediction()
                (root/'classifier.engine').write_bytes(b'tamper')
                with self.assertRaisesRegex(ValueError,'SHA mismatch'):target.activate_tensorrt(model,root)
                self.assertEqual(model.detector.model,'original-detector')

    def test_old_receipt_is_rejected_before_engine_deserialization(self):
        trt,_,_=fake_trt()
        def forbidden(*args,**kwargs):raise AssertionError('Unsafe engine deserialized')
        trt.Runtime=forbidden
        torch=types.SimpleNamespace(__version__='2.8.0',version=types.SimpleNamespace(cuda='12.8'),
            cuda=types.SimpleNamespace(is_available=lambda:True,current_device=lambda:0,get_device_name=lambda d:'NVIDIA A10',
                get_device_properties=lambda d:types.SimpleNamespace(multi_processor_count=72),get_device_capability=lambda d:(8,6)))
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);value=receipt();value['schemaVersion']=1;(root/'runtime.json').write_text(json.dumps(value))
            with patch.dict(sys.modules,{'torch':torch,'tensorrt':trt,'ultralytics':types.SimpleNamespace(YOLO=forbidden)}):
                with self.assertRaisesRegex(ValueError,'AMPERE_PLUS'):target.activate_tensorrt(types.SimpleNamespace(manifest=manifest()),root)

    def test_classifier_refuses_bad_tensor_before_binding_or_execution(self):
        trt,_,_=fake_trt();torch=types.SimpleNamespace(float32='float32')
        classifier=target.TensorRTClassifier.__new__(target.TensorRTClassifier);classifier.resolution=320
        for extra in ({'is_cuda':False},{'dtype':'float16'},{'shape':(33,3,320,320)},{'shape':(1,3,224,224)},{'ndim':3}):
            values={'is_cuda':True,'dtype':'float32','shape':(1,3,320,320),'ndim':4};values.update(extra)
            with patch.dict(sys.modules,{'torch':torch,'tensorrt':trt}),self.assertRaisesRegex(ValueError,'CUDA FP32 NCHW'):
                classifier(types.SimpleNamespace(**values))


if __name__=='__main__':unittest.main()
