"""CPU endpoint imports/configuration tested without Modal network or model sessions."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch, Mock


def load_scope(env=None):
    options={};asgi={};entered={}
    class App:
        def __init__(self,*a,**kw):pass
        def cls(self,**kw):options.update(kw);return lambda cls:cls
    def entry(**kw):entered.update(kw);return lambda f:f
    def expose(**kw):asgi.update(kw);return lambda f:f
    stub=types.SimpleNamespace(App=App,is_local=lambda:False,enter=entry,asgi_app=expose)
    scope={'__file__':'/root/cpu_app.py','__name__':'cpu_app_test'}
    with patch.dict(sys.modules,{'modal':stub}),patch.dict('os.environ',env or {},clear=True):
        exec(compile(Path('infra/modal/cpu_app.py').read_text(),'/root/cpu_app.py','exec'),scope)
    return scope,options,asgi,entered

class ModalCpuApp(unittest.TestCase):
    def test_public_build_path_uploads_only_bootstrap_and_pins_three_files(self):
        with tempfile.TemporaryDirectory()as d:
            directory=Path(d)
            for name in ('manifest.json','detector.onnx','classifier.onnx'):(directory/name).write_bytes(name.encode())
            calls=[];environment={}
            class Image:
                @classmethod
                def debian_slim(cls,**kw):return cls()
                def pip_install(self,*a,**kw):return self
                def add_local_file(self,*a,**kw):calls.append(('file',a,kw));return self
                def add_local_dir(self,*a,**kw):calls.append(('dir',a,kw));return self
                def run_commands(self,*a,**kw):calls.append(('command',a,kw));return self
                def env(self,value):environment.update(value);return self
            class App:
                def __init__(self,*a,**kw):pass
                def cls(self,**kw):return lambda cls:cls
            stub=types.SimpleNamespace(App=App,Image=Image,is_local=lambda:True,
                enter=lambda **kw:lambda f:f,asgi_app=lambda **kw:lambda f:f)
            scope={'__file__':str(Path('infra/modal/cpu_app.py').resolve()),'__name__':'cpu_build_test'}
            env={'MAHJONG_CPU_MODEL_DIR':str(directory),'MAHJONG_CPU_PUBLIC_BASE_URL':'https://github.com/XTTTZ/mahjong-recognizer/releases/download/runtime-compact-fp32-20261109/'}
            with patch.dict(sys.modules,{'modal':stub}),patch.dict('os.environ',env,clear=True), \
                 patch('ml.inference.candidate_onnx_runtime.verify_candidate_directory',return_value=(directory,{'version':'selected'})), \
                 patch('infra.modal.package.prepare_upload',side_effect=AssertionError('No SDK model upload in public path')), \
                 patch('infra.modal.package.prepare_source',return_value=directory):
                exec(compile(Path('infra/modal/cpu_app.py').read_text(),scope['__file__'],'exec'),scope)
            spec=json.loads(environment['MAHJONG_CPU_PUBLIC_MODEL_SPEC'])
            self.assertEqual(set(spec['files']),{'manifest.json','detector.onnx','classifier.onnx'})
            self.assertTrue(all(item['bytes']==len(name)for name,item in spec['files'].items()))
            self.assertTrue(any(c[0]=='file'and str(c[1][0]).endswith('fetch_public_model.py')for c in calls))
            self.assertFalse(any(c[0]=='dir'and c[2].get('remote_path')=='/model-upload'for c in calls))
            self.assertTrue(any(c[0]=='command'and '/build/fetch_public_model.py --out /model'in c[1][0]for c in calls))

    def test_no_gpu_and_proxy_auth_scale_to_zero_snapshot(self):
        scope,options,asgi,entered=load_scope()
        self.assertIsNone(scope['image']);self.assertNotIn('ROOT',scope)
        self.assertNotIn('gpu',options);self.assertNotIn('experimental_options',options)
        self.assertEqual(options['cpu'],(1,1));self.assertEqual(options['memory'],2048)
        self.assertEqual((options['min_containers'],options['max_containers']),(0,1))
        self.assertTrue(options['enable_memory_snapshot']);self.assertEqual(entered,{'snap':True})
        self.assertTrue(asgi['requires_proxy_auth']);self.assertEqual(options['timeout'],90)
    def test_bounded_resources_and_remote_settings(self):
        scope,options,_,entry=load_scope({'MAHJONG_CPU_CORES':'2','MAHJONG_CPU_MEMORY_MB':'4096','MAHJONG_CPU_BATCH':'8','MAHJONG_CPU_SNAPSHOT':'0'})
        self.assertEqual(options['cpu'],(2,2));self.assertEqual(scope['SETTINGS']['batch'],8);self.assertFalse(entry['snap'])
        for key,value in [('MAHJONG_CPU_CORES','3'),('MAHJONG_CPU_MEMORY_MB','1024'),('MAHJONG_CPU_BATCH','32'),('MAHJONG_CPU_SNAPSHOT','maybe')]:
            with self.subTest(key=key),self.assertRaises(ValueError):load_scope({key:value})
    def test_wrong_manifest_cannot_be_warmed(self):
        scope,_,_,_=load_scope();predictor=types.SimpleNamespace(manifest={'version':'wrong'})
        with patch('ml.inference.candidate_onnx_runtime.CandidateOnnxPredictor',return_value=predictor),patch.dict('os.environ',{'MAHJONG_CPU_EXPECTED_MODEL':json.dumps({'version':'selected','manifestSha256':'a'*64})}):
            with self.assertRaisesRegex(ValueError,'differs'):scope['CpuRecognizer']().load()
    def test_shared_http_contract_is_retained(self):
        scope,_,_,_=load_scope();instance=scope['CpuRecognizer']();instance.predictor=types.SimpleNamespace(manifest={'acceptThreshold':.7})
        factory=Mock(return_value='same-contract')
        with patch.dict(sys.modules,{'ml.inference.service':types.SimpleNamespace(create_app=factory)}):
            self.assertEqual(instance.api(),'same-contract');factory.assert_called_once_with(instance.predictor,.7)

if __name__=='__main__':unittest.main()
