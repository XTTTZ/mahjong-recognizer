"""Cloud imports must not assume a local checkout or model directory exists."""
from pathlib import Path
import sys
import types
import json
import unittest
from unittest.mock import patch
from infra.modal.runtime import runtime_environment


class ContainerImportTests(unittest.TestCase):
    def test_gpu_runtime_choices_survive_remote_import(self):
        env=runtime_environment('tensorrt',['A10'],True,'selected','f'*64)
        self.assertEqual(env['MAHJONG_GPU'],'A10')
        self.assertEqual(env['MAHJONG_RUNTIME'],'tensorrt')
        self.assertEqual(env['MAHJONG_SNAPSHOT'],'1')
        self.assertEqual(json.loads(env['MAHJONG_EXPECTED_MODEL'])['manifestSha256'],'f'*64)
        self.assertEqual(runtime_environment('torch',['L4','A10'],False,'v','d')['MAHJONG_GPU'],'L4,A10')
    def test_modal_container_imports_from_shallow_root(self):
        options={}
        class App:
            def __init__(self, *a, **kw): pass
            def cls(self, **kw):
                options.update(kw)
                return lambda cls: cls
        stub=types.SimpleNamespace(App=App,is_local=lambda:False,
                                   enter=lambda **kw:lambda f:f,asgi_app=lambda **kw:lambda f:f)
        source=Path('infra/modal/app.py').read_text()
        scope={'__file__':'/root/app.py','__name__':'mahjong_container_import'}
        with patch.dict(sys.modules,{'modal':stub}), patch.dict('os.environ',{},clear=True):
            exec(compile(source,'/root/app.py','exec'),scope)
        self.assertIn('Recognizer',scope)
        self.assertNotIn('ROOT',scope)
        self.assertIsNone(scope['image'])
        self.assertEqual(options['gpu'],['L4','A10'])
        self.assertEqual((options['min_containers'],options['max_containers']),(0,1))
        self.assertEqual(options['timeout'],90)
        self.assertTrue(options['enable_memory_snapshot'])
        self.assertTrue(options['experimental_options']['enable_gpu_snapshot'])

    def test_reused_image_cannot_serve_different_weights(self):
        class App:
            def __init__(self,*a,**kw): pass
            def cls(self,**kw): return lambda cls:cls
        stub=types.SimpleNamespace(App=App,is_local=lambda:False,
            enter=lambda **kw:lambda f:f,asgi_app=lambda **kw:lambda f:f)
        scope={'__file__':'/root/app.py','__name__':'mahjong_container_import'}
        with patch.dict(sys.modules,{'modal':stub}),patch.dict('os.environ',{},clear=True):
            exec(compile(Path('infra/modal/app.py').read_text(),'/root/app.py','exec'),scope)
        predictor=types.SimpleNamespace(manifest={'version':'other-model','filesSha256':{}})
        expected=json.dumps({'version':'selected-model','manifestSha256':'a'*64})
        with patch.dict(sys.modules,{'torch':types.SimpleNamespace(set_num_threads=lambda _:None)}), \
             patch('ml.inference.model.load_predictor',return_value=predictor),patch.dict('os.environ',{'MAHJONG_EXPECTED_MODEL':expected}):
            with self.assertRaisesRegex(ValueError,'does not contain'):
                scope['Recognizer']().load()


if __name__=='__main__': unittest.main()
