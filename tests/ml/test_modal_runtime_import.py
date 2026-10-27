"""Cloud imports must not assume a local checkout or model directory exists."""
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


class ContainerImportTests(unittest.TestCase):
    def test_modal_container_imports_from_shallow_root(self):
        class App:
            def __init__(self, *a, **kw): pass
            def cls(self, **kw): return lambda cls: cls
        stub=types.SimpleNamespace(App=App,is_local=lambda:False,
                                   enter=lambda:lambda f:f,asgi_app=lambda **kw:lambda f:f)
        source=Path('infra/modal/app.py').read_text()
        scope={'__file__':'/root/app.py','__name__':'mahjong_container_import'}
        with patch.dict(sys.modules,{'modal':stub}):
            exec(compile(source,'/root/app.py','exec'),scope)
        self.assertIn('Recognizer',scope)
        self.assertNotIn('ROOT',scope)
        self.assertIsNone(scope['image'])


if __name__=='__main__': unittest.main()
