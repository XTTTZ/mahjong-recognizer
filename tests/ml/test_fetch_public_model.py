"""Public model transport uses frozen bytes and atomic directories; no network in tests."""
from io import BytesIO
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.request import Request
from infra.portable import fetch_public_model as target

BASE='https://github.com/XTTTZ/mahjong-recognizer/releases/download/runtime-compact-fp32-20261109/'
DATA={'manifest.json':b'{"version":"fixture"}','detector.onnx':b'detector bytes','classifier.onnx':b'classifier bytes'}
SPEC={name:{'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}for name,data in DATA.items()}

class Response(BytesIO):
    def __init__(self,data,url):super().__init__(data);self.url=url
    def geturl(self):return self.url

class PublicModelFetch(unittest.TestCase):
    def test_exact_download_is_atomic_then_exact_existing_reused(self):
        def open_(req,timeout):
            self.assertEqual(timeout,30)
            self.assertNotIn('Authorization',req.headers)
            return Response(DATA[req.full_url.rsplit('/',1)[1]],'https://release-assets.githubusercontent.com/frozen-file')
        with tempfile.TemporaryDirectory()as d,patch.object(target,'_open_url',side_effect=open_)as network:
            out=Path(d)/'model';receipt=target.fetch_package(BASE,SPEC,out)
            self.assertEqual(receipt['state'],'downloaded_verified');self.assertFalse(receipt['credentialsUsed'])
            self.assertEqual({p.name:p.read_bytes()for p in out.iterdir()},DATA)
            self.assertEqual(target.fetch_package(BASE,SPEC,out)['state'],'verified_existing')
            self.assertEqual(network.call_count,3)
    def test_bad_hash_short_oversize_and_network_failure_leave_no_destination(self):
        for variant in ('hash','short','oversize','network'):
            with self.subTest(variant=variant),tempfile.TemporaryDirectory()as d:
                out=Path(d)/'model'
                def open_(req,timeout):
                    if variant=='network':raise OSError('mock transport down')
                    data=DATA[req.full_url.rsplit('/',1)[1]]
                    data=bytes([data[0]^1])+data[1:] if variant=='hash' else data[:-1] if variant=='short' else data+b'x'
                    return Response(data,req.full_url)
                with patch.object(target,'_open_url',side_effect=open_),self.assertRaises((ValueError,OSError)):
                    target.fetch_package(BASE,SPEC,out)
                self.assertFalse(out.exists());self.assertEqual(list(Path(d).iterdir()),[])
    def test_unapproved_url_spec_and_redirect_rejected_before_fetch(self):
        for url in ('http://github.com/XTTTZ/mahjong-recognizer/releases/download/tag/',BASE+'?token=x',BASE.replace('XTTTZ','other'),BASE+'../',BASE.replace('github.com','github.com.evil')):
            with self.subTest(url=url),patch.object(target,'_open_url')as opener,self.assertRaises(ValueError):
                target.fetch_package(url,SPEC,'unused');opener.assert_not_called()
        with patch.object(target,'_open_url')as opener,self.assertRaises(ValueError):target.fetch_package(BASE,{'../evil':SPEC['manifest.json']},'unused')
        req=Request(BASE+'manifest.json')
        for url in ('http://release-assets.githubusercontent.com/file','https://evil.test/file'):
            with self.assertRaises(ValueError):target.PublicRedirects().redirect_request(req,None,302,'',{},url)
    def test_existing_different_directory_not_overwritten(self):
        with tempfile.TemporaryDirectory()as d:
            out=Path(d)/'model';out.mkdir();(out/'manifest.json').write_bytes(b'keep')
            with patch.object(target,'_open_url')as network,self.assertRaises(FileExistsError):target.fetch_package(BASE,SPEC,out)
            self.assertEqual((out/'manifest.json').read_bytes(),b'keep');network.assert_not_called()

if __name__=='__main__':unittest.main()
