"""Offline suite, provenance and calibration tests; no model training or SSH."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from infra.windows.prepare_efficiency_suite_v7 import make_suite,write_plan,bundle
from infra.windows.offline_bundle import verify
from infra.windows.validate_cli import validate
from ml.train.student_common import calibration_indices,validate_recipe,WEIGHTS_FILENAME
from ml.train.yolo_train import checkpoint_receipt
from ml.train.provenance import dataset_contract,sha256
from ml.data.prepare_crops import prepare as prepare_crops
from ml.data.prepare_crop_cache_v6 import prepare as prepare_cache

ROOT=Path(__file__).resolve().parents[2]


def fixture_options():
    return {'expected_environment':{'python':'3.11.17','packages':{'fixture':'1'}},
        'parent_assets':[{'path':'D:/fixture/parent.pt','sha256':'a'*64}],
        'student_assets':[{'path':'D:/fixture/'+WEIGHTS_FILENAME,'sha256':'b'*64}]}


class EfficiencySuiteTests(unittest.TestCase):
    def test_all_remaining_seven_and_controlled_students_fit_estimates_with_reserve(self):
        suite=make_suite(ROOT,**fixture_options());q=suite['queueConfig'];jobs=q['experiments']
        self.assertEqual(len(jobs),11);self.assertEqual(sum(len(j['steps'])for j in jobs),20)
        self.assertEqual(q['deadline_hours'],12);self.assertEqual(q['reserve_minutes'],60)
        self.assertEqual(sum(j['estimated_minutes']for j in jobs),558)
        self.assertLessEqual(suite['budget']['estimatedMinutes']+60,720)
        self.assertFalse(suite['trainingStarted']);self.assertFalse(suite['submitted']);self.assertFalse(suite['newWindowAuthorized'])
        self.assertFalse(any(j['id'].endswith('v8-control')for j in jobs))
        self.assertEqual(sum('student'not in j['id']for j in jobs),7)
        for j in jobs:
            self.assertIn('v7-efficiency-',j['id']);self.assertNotIn('v6-refine',j['run_dir'])
            if 'int8'in j['id']:
                self.assertEqual(len(j['depends_on']),1);self.assertTrue(j['depends_on'][0]in{p['id']for p in jobs[:jobs.index(j)]})
                self.assertEqual(j['steps'][0]['env']['CUDA_VISIBLE_DEVICES'],'-1')
            else:self.assertNotIn('depends_on',j)
        students=[j for j in jobs if j['id'].endswith(('student-ce','student-kd'))]
        for flag in ('--weights','--teacher-run','--teacher-checkpoint','--resolution','--batch','--epochs','--seed','--lr'):
            self.assertEqual(students[0]['steps'][0]['argv'][students[0]['steps'][0]['argv'].index(flag)+1],students[1]['steps'][0]['argv'][students[1]['steps'][0]['argv'].index(flag)+1])

    def test_all_twenty_argument_boundaries_stop_before_io_or_tensor_import(self):
        for job in make_suite(ROOT,**fixture_options())['queueConfig']['experiments']:
            for step in job['steps']:
                with self.subTest(job=job['id'],step=step['name']):self.assertEqual(validate(step['argv'])['state'],'argv_parsed_only')

    def test_cold_bundle_and_command_permissions(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);plan=root/'v7-12h';write_plan(ROOT,plan,**fixture_options())
            for name in ('prepare.cmd','check.cmd','watch.cmd'):self.assertNotIn('--start',(plan/name).read_text())
            self.assertIn('--start',(plan/'submit.cmd').read_text())
            receipt=bundle(ROOT,plan,root/'code.zip');manifest=verify(root/'code.zip',receipt['sha256'])
            for name in ('ml/train/student_train_v7.py','ml/train/student_quantize_v7.py','ml/eval/student_evaluate_v7.py','docs/models/V7_EFFICIENCY.md'):
                self.assertIn(name,manifest['files'])
            self.assertFalse(any(n.endswith(('.pt','.pth'))for n in manifest['files']))
            self.assertFalse(receipt['trainingStarted']);self.assertFalse(receipt['windowsV7Checked'])


class StudentPolicies(unittest.TestCase):
    def test_train_calibration_is_deterministic_balanced_and_prefers_unique_groups(self):
        records=[{'annotationId':i,'label':i%42,'groupId':f'group-{i//42}'}for i in range(42*20)]
        a=calibration_indices(records,672);b=calibration_indices(records,672)
        self.assertEqual(a,b);self.assertEqual(len(a),672);self.assertEqual(len(set(a)),672)
        from collections import Counter
        self.assertEqual(set(Counter(records[i]['label']for i in a).values()),{16})
        self.assertEqual({records[i]['label']for i in a},set(range(42)))
        with self.assertRaises(ValueError):calibration_indices(records[:41])
        with self.assertRaises(ValueError):calibration_indices(records,41)

    def test_invalid_student_recipes_fail_before_tensor_execution(self):
        config={'resolution':224,'context':.08,'distillationAlpha':.5,'temperature':2,'teacherMinimumConfidence':.7,'epochs':20,'batch':16,'gradientAccumulation':2,'workers':2,'lr':.0003,'maximumTrainMinutes':40}
        self.assertEqual(validate_recipe(config),config)
        for key,value in [('resolution',320),('context',.2),('distillationAlpha',1),('temperature',float('nan')),('lr',0),('maximumTrainMinutes',float('inf'))]:
            with self.subTest(key=key),self.assertRaises(ValueError):validate_recipe(dict(config,**{key:value}))

    def test_actual_student_preflight_checks_signed_data_without_importing_torch_or_test(self):
        from tests.ml.test_v6_preparation import fixture
        from ml.train import student_train_v7
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);fixture(root/'data')
            for split in ('train','valid'):
                path=root/'data'/split/'_annotations.coco.json';coco=json.loads(path.read_text())
                coco['annotations']=[{'id':i,'image_id':1,'category_id':i+1,'bbox':[12.3,22.7,32.5,41.2]}for i in range(42)];path.write_text(json.dumps(coco))
            contract=dataset_contract(root/'data',True);prepare_crops(root/'data',root/'crops');prepare_cache(root/'data',root/'crops',root/'cache')
            teacher=root/'teacher';teacher.mkdir();weight=teacher/'last.pt';weight.write_bytes(b'fixture-signed-teacher')
            signature={'data':contract['splits'],'classes':contract['classes']}
            run={'architecture':'efficientnet-v2-s','classes':contract['classes'],'config':{'resolution':320,'context':.08},'resumeSignature':signature,'finishedAt':'fixture',
                'annotationSha256':contract['splits']['train']['annotationSha256'],'validationAnnotationSha256':contract['splits']['valid']['annotationSha256']}
            (teacher/'run.json').write_text(json.dumps(run));checkpoint_receipt(weight,signature,contract['classes'])
            pretrained=root/WEIGHTS_FILENAME;pretrained.write_bytes(b'fixture-pretrained')
            argv=['student','--dataset',str(root/'data'),'--crops',str(root/'crops'),'--cache',str(root/'cache'),'--weights',str(pretrained),
                  '--teacher-run',str(teacher/'run.json'),'--teacher-checkpoint',str(weight),'--output',str(root/'unused'),'--preflight-only']
            import builtins
            original_import=builtins.__import__
            def guard(name,*args,**kwargs):
                if name=='torch'or name.startswith('torch.'):raise AssertionError('Preflight imported tensor runtime')
                return original_import(name,*args,**kwargs)
            stream=io.StringIO()
            with patch.object(sys,'argv',argv),patch.object(student_train_v7,'WEIGHTS_SHA256',sha256(pretrained)),patch('builtins.__import__',side_effect=guard),contextlib.redirect_stdout(stream):student_train_v7.main()
            result=json.loads(stream.getvalue());self.assertTrue(result['derivativeReady']);self.assertFalse(result['trainingStarted'])
            self.assertFalse((root/'unused').exists());self.assertEqual((root/'data/test/_annotations.coco.json').read_text(),'SEALED_DO_NOT_OPEN')
            pretrained.write_bytes(b'tampered')
            with patch.object(sys,'argv',argv),self.assertRaisesRegex(ValueError,'SHA256'):student_train_v7.main()


if __name__=='__main__':unittest.main()
