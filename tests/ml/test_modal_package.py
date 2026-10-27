"""Exercise exact-byte model assembly and corruption checks without cloud calls."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from infra.modal.package import prepare_upload


class UploadPackageTests(unittest.TestCase):
    def test_chunks_reassemble_exact_bytes_and_cache_rejects_damage(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=root/'source'; source.mkdir()
            payload=bytes(range(256))*18
            (source/'checkpoint.pth').write_bytes(payload)
            (source/'model.json').write_text('{"version":"trial"}')
            upload=prepare_upload(source,root/'cache',chunk_bytes=1024)
            self.assertEqual(upload,prepare_upload(source,root/'cache',chunk_bytes=1024))
            target=root/'model'
            assembler=(upload/'assemble.py').read_text().replace("Path('/model-upload')",f'Path({str(upload)!r})').replace("Path('/model')",f'Path({str(target)!r})')
            subprocess.run([sys.executable,'-c',assembler],check=True,capture_output=True)
            self.assertEqual((target/'checkpoint.pth').read_bytes(),payload)
            self.assertEqual((target/'model.json').read_bytes(),(source/'model.json').read_bytes())
            next(upload.glob('*.chunk')).write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError,'chunk changed'):
                prepare_upload(source,root/'cache',chunk_bytes=1024)
            self.assertNotEqual(subprocess.run([sys.executable,'-c',assembler],capture_output=True).returncode,0)


if __name__=='__main__': unittest.main()
