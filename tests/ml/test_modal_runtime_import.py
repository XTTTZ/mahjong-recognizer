"""Cloud imports must not assume a local checkout or model directory exists."""
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


class ContainerImportTests(unittest.TestCase):
    def test_modal_container_imports_from_shallow_root(self):
        options={}
        class App:
            def __init__(self, *a, **kw): pass
            def cls(self, **kw):
                options.update(kw)
                return lambda cls: cls
        stub=types.SimpleNamespace(App=App,is_local=lambda:False,
                                   enter=lambda:lambda f:f,asgi_app=lambda **kw:lambda f:f)
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


if __name__=='__main__': unittest.main()
