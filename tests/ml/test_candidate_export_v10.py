"""Copied source identity and graph reuse are verified without tensor execution."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from infra.portable.candidate_export import transported_parent, reuse_stage, base_manifest
from ml.inference.manifest import FACE_TOKENS, file_sha256
from ml.train.yolo_train import checkpoint_receipt

CLASSES={str(i):name for i,name in enumerate(sorted(FACE_TOKENS))}

class CandidateSourceTransport(unittest.TestCase):
    def fixture(self, root):
        source=root/'source';source.mkdir();contract={'classes':CLASSES,'splits':{'train':{'annotationSha256':'train'},'valid':{'annotationSha256':'valid'}}}
        run={'classes':CLASSES,'resumeSignature':{'data':contract['splits']},'annotationSha256':'train','validationAnnotationSha256':'valid','finishedAt':None}
        rp=source/'run.json';rp.write_text(json.dumps(run));cp=source/'last.pt';cp.write_bytes(b'signed complete epoch')
        checkpoint_receipt(cp,run['resumeSignature'],CLASSES)
        sp=source/'status.json';sp.write_text(json.dumps({'state':'timed_out','run_dir':'D:\\mahjong\\runs\\source','child_pid':None,'finished_at':'2026-10-27T03:42:36Z'}))
        mapping={'schemaVersion':1,'sources':[{'localDirectory':str(source),'originalDirectory':'D:/mahjong/runs/source','filesSha256':{p.name:file_sha256(p)for p in source.iterdir()}}]}
        mp=root/'transport.json';mp.write_text(json.dumps(mapping));return contract,rp,cp,sp,mp,mapping
    def test_copied_inactive_run_keeps_original_bytes_and_requires_explicit_map(self):
        with tempfile.TemporaryDirectory()as d:
            contract,rp,cp,sp,mp,_=self.fixture(Path(d));before={p:p.read_bytes()for p in (rp,sp)}
            with self.assertRaises(ValueError):transported_parent(rp,cp,contract,sp,False)
            run,evidence=transported_parent(rp,cp,contract,sp,False,mp)
            self.assertIsNone(run['finishedAt']);self.assertTrue(evidence['sourceStatus']['originalBytesPreserved'])
            self.assertEqual(evidence['sourceStatus']['terminalReceiptSha256'],file_sha256(sp))
            self.assertEqual(before,{p:p.read_bytes()for p in (rp,sp)})
    def test_hash_directory_and_live_status_tampering_are_rejected(self):
        for variant in ('hash','directory','active','missing_binding'):
            with self.subTest(variant=variant),tempfile.TemporaryDirectory()as d:
                contract,rp,cp,sp,mp,mapping=self.fixture(Path(d))
                if variant=='hash':cp.write_bytes(b'changed')
                elif variant=='directory':mapping['sources'][0]['originalDirectory']='D:/other/run'
                elif variant=='missing_binding':mapping['sources'][0]['filesSha256'].pop('last.pt.receipt.json')
                else:
                    status=json.loads(sp.read_text());status['child_pid']=123;sp.write_text(json.dumps(status));mapping['sources'][0]['filesSha256']['status.json']=file_sha256(sp)
                mp.write_text(json.dumps(mapping))
                with self.assertRaises(ValueError):transported_parent(rp,cp,contract,sp,False,mp)

class CandidateStageReuse(unittest.TestCase):
    def fixture(self, root):
        directory=root/'profile';directory.mkdir();(directory/'detector.onnx').write_bytes(b'detector');(directory/'classifier.onnx').write_bytes(b'classifier')
        detector={'architecture':'yolov8n-generic','resolution':800};classifier={'architecture':'mobilenet-v3-large','config':{'resolution':224,'context':.08},'classes':CLASSES}
        manifest=base_manifest(detector,classifier,'fixture');evidence={n:{'runSha256':n+'run','checkpointSha256':n+'checkpoint'}for n in ('detector','classifier')}
        manifest.update(sourceEvidence=evidence,filesSha256={n:file_sha256(directory/n)for n in ('detector.onnx','classifier.onnx')},filesBytes={n:(directory/n).stat().st_size for n in ('detector.onnx','classifier.onnx')})
        (directory/'manifest.json').write_text(json.dumps(manifest));out=root/'out';out.mkdir();return directory,manifest,evidence,out
    def test_reuse_preserves_bytes_and_allows_independent_other_stage(self):
        with tempfile.TemporaryDirectory()as d:
            directory,target,evidence,out=self.fixture(Path(d))
            target['resolution']=640;target['detectorArchitecture']='yolo26s-generic'
            result=reuse_stage(directory,'classifier',target,evidence,out)
            self.assertEqual(result['stageSha256'],file_sha256(out/'classifier.onnx'))
            self.assertEqual((out/'classifier.onnx').read_bytes(),b'classifier')
    def test_source_geometry_or_file_tamper_cannot_be_reused(self):
        for variant in ('source','geometry','file'):
            with self.subTest(variant=variant),tempfile.TemporaryDirectory()as d:
                directory,target,evidence,out=self.fixture(Path(d))
                if variant=='source':evidence=copy.deepcopy(evidence);evidence['classifier']['checkpointSha256']='other'
                elif variant=='geometry':target['cropContext']=.02
                else:(directory/'classifier.onnx').write_bytes(b'changed')
                with self.assertRaises(ValueError):reuse_stage(directory,'classifier',target,evidence,out)

if __name__=='__main__':unittest.main()
