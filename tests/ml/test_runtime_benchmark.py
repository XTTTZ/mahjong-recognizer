import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from infra.modal.benchmark_runtime import (
    benchmark_gpu, compare_runtime, digest, load_validation, localization_pairs, output_drift,
    percentile, plain_json, register_remote, score_output, summarize_runtime, validate_coco,
    verify_engine_receipt, verify_manifest,
)
from ml.eval.metrics import Tile
from ml.inference.manifest import FACE_TOKENS


def manifest():
    return {'architecture': 'yolo-efficientnet-v2-s-cascade', 'version': 'fixture',
            'classes': {str(i): token for i, token in enumerate(sorted(FACE_TOKENS))},
            'threshold': .35, 'acceptThreshold': .7, 'resolution': 960,
            'filesSha256': {'detector.pt': 'a' * 64, 'classifier.pt': 'b' * 64}}


def prediction(label='1m', score=.8, x=0):
    return {'classId': label, 'score': score, 'bbox': [x, 0, 10, 20]}


def scored(raw, truth=None):
    truth = [Tile('1m', (0, 0, 10, 20))] if truth is None else truth
    result = score_output(raw, truth, 100, 100)
    result.update(imageId=1, inferenceMs=[10., 20.], repeatMetricUnstable=False)
    return result


class RuntimeBenchmarkTests(unittest.TestCase):
    def test_import_does_not_import_gpu_or_modal(self):
        # A separate interpreter avoids contamination from other ML test imports.
        import subprocess
        result = subprocess.run([sys.executable, '-c',
            "import sys;import infra.modal.benchmark_runtime;assert 'torch' not in sys.modules;assert 'modal' not in sys.modules"],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_operating_point_and_ontology_are_locked(self):
        verify_manifest(manifest())
        for key, value in [('threshold', .5), ('acceptThreshold', .6), ('architecture', 'yolov8s')]:
            invalid = manifest(); invalid[key] = value
            with self.assertRaises(ValueError):
                verify_manifest(invalid)

    def test_remote_function_refuses_cpu_without_model_forward(self):
        stub = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
        with patch.dict(sys.modules, {'torch': stub}):
            with self.assertRaisesRegex(RuntimeError, 'GPU required'):
                benchmark_gpu([], b'{}', {}, manifest())

    def test_prebuilt_image_uses_from_id_without_importing_deployment_app(self):
        calls = {}
        class Image:
            def add_local_file(self, path, remote_path, **options):
                calls['source'] = (path, remote_path, options)
                return self
        image = Image()
        class App:
            def __init__(self, name): calls['appName'] = name
            def function(self, **options):
                calls['functionOptions'] = options
                return lambda function: function
        def from_id(image_id):
            calls['imageId'] = image_id
            return image
        modal_stub = types.SimpleNamespace(App=App, Image=types.SimpleNamespace(from_id=from_id))
        # Importing app.py in this branch would fail immediately.
        with patch.dict(sys.modules, {'modal': modal_stub, 'infra.modal.app': None}):
            app, remote = register_remote('A10', 'im-fixture123')
        self.assertEqual(calls['imageId'], 'im-fixture123')
        self.assertEqual(calls['appName'], 'mahjong-runtime-benchmark')
        self.assertEqual(calls['functionOptions']['gpu'], 'A10')
        self.assertEqual(calls['functionOptions']['image'], image)
        self.assertEqual(calls['functionOptions']['max_containers'], 1)
        self.assertFalse(calls['functionOptions'].get('serialized', False))
        self.assertEqual(remote.__module__, 'infra.modal.benchmark_runtime')
        self.assertIs(remote, benchmark_gpu)

    def test_cli_alias_registers_canonical_module_function_without_serialization(self):
        import runpy
        captured = {}
        image = types.SimpleNamespace(add_local_file=lambda *args, **kwargs: image)
        class App:
            def __init__(self, name): pass
            def function(self, **kwargs):
                captured.update(kwargs)
                return lambda function: function
        stub = types.SimpleNamespace(App=App, Image=types.SimpleNamespace(from_id=lambda _: image))
        scope = runpy.run_path('infra/modal/benchmark_runtime.py', run_name='cli_alias_fixture')
        with patch.dict(sys.modules, {'modal': stub, 'infra.modal.app': None}):
            _, remote = scope['register_remote']('A10', 'im-fixture123')
        self.assertEqual(remote.__module__, 'infra.modal.benchmark_runtime')
        self.assertIs(remote, benchmark_gpu)
        self.assertFalse(captured.get('serialized', False))

    def test_invalid_image_id_rejected_before_sdk_import(self):
        for invalid in ('', 'not-an-image', 'im-path/escape'):
            with patch.dict(sys.modules, {'modal': None}):
                with self.assertRaisesRegex(ValueError, 'image ID'):
                    register_remote('L4', invalid)

    def test_engine_receipt_requires_same_gpu_and_model_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ('detector.engine', 'classifier.engine'):
                (root / name).write_bytes(name.encode())
            original = {'modelVersion': 'fixture', 'modelFilesSha256': manifest()['filesSha256'],
                        'gpu': 'NVIDIA A10', 'computeCapability': [8, 6], 'precision': 'fp16',
                        'calibrationDataUsed': False,
                        'enginesSha256': {name: digest((root / name).read_bytes())
                                         for name in ('detector.engine', 'classifier.engine')}}
            path = root / 'runtime.json'; path.write_text(json.dumps(original))
            verify_engine_receipt(root, manifest(), 'NVIDIA A10', (8, 6))
            with self.assertRaisesRegex(ValueError, 'hardwareCompatibility'):
                verify_engine_receipt(root, manifest(), 'NVIDIA A10G', (8, 6))
            original['hardwareCompatibility'] = 'AMPERE_PLUS'; path.write_text(json.dumps(original))
            verify_engine_receipt(root, manifest(), 'NVIDIA A10G', (8, 6))
            alias = dict(original); alias['gpu'] = 'NVIDIA A10G'; path.write_text(json.dumps(alias))
            verify_engine_receipt(root, manifest(), 'NVIDIA A10', (8, 6))
            path.write_text(json.dumps(original))
            for gpu, capability in [('NVIDIA L4', (8, 9)), ('NVIDIA A10', (8, 9))]:
                with self.assertRaises(ValueError):
                    verify_engine_receipt(root, manifest(), gpu, capability)
            with self.assertRaisesRegex(ValueError, 'gpu'):
                verify_engine_receipt(root, manifest(), 'NVIDIA RTX A5000', (8, 6))
            with self.assertRaisesRegex(ValueError, 'computeCapability'):
                verify_engine_receipt(root, manifest(), 'NVIDIA A10G', (8, 9))
            for key, bad in [('modelVersion', 'other'), ('modelFilesSha256', {}), ('precision', 'int8')]:
                modified = dict(original); modified[key] = bad; path.write_text(json.dumps(modified))
                with self.assertRaisesRegex(ValueError, key):
                    verify_engine_receipt(root, manifest(), 'NVIDIA A10', (8, 6))
            path.write_text(json.dumps(original)); (root / 'classifier.engine').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                verify_engine_receipt(root, manifest(), 'NVIDIA A10', (8, 6))

    def test_remote_result_strips_foreign_scalar_subclasses_before_pickle(self):
        import pickle
        ForeignVersion = type('TorchVersion', (str,), {'__module__': 'torch.torch_version'})
        ForeignInt = type('ForeignInt', (int,), {'__module__': 'unavailable_test_module'})
        value = {'torch': ForeignVersion('2.8.0'), 'nested': [{'count': ForeignInt(3)}]}
        clean = plain_json(value)
        self.assertIs(type(clean['torch']), str)
        self.assertIs(type(clean['nested'][0]['count']), int)
        with patch.dict(sys.modules, {'torch': None, 'torch.torch_version': None, 'unavailable_test_module': None}):
            self.assertEqual(pickle.loads(pickle.dumps(clean)), clean)
        with self.assertRaises(ValueError):
            plain_json({'latency': float('nan')})

    def test_strict_threshold_and_shared_acceptance_rounding(self):
        at_floor = scored([prediction(score=.35)])
        self.assertEqual(at_floor['metrics']['prediction_count'], 0)
        rejected = scored([prediction(score=.699999)])
        self.assertEqual(rejected['tiles'][0]['score'], .7)
        self.assertTrue(rejected['tiles'][0]['uncertain'])
        self.assertFalse(rejected['metrics']['hand_exact_match'])
        self.assertTrue(scored([prediction(score=.7)])['metrics']['hand_exact_match'])

    def test_rejected_failed_and_wrong_requests_stay_in_denominator(self):
        good, rejected, wrong = scored([prediction()]), scored([prediction(score=.6)]), scored([prediction('2m')])
        failed = scored([]); failed.update(score_output([], [Tile('1m', (0, 0, 10, 20))], 100, 100, 'fixture'))
        records = {}
        for i, row in enumerate([good, rejected, wrong, failed]):
            row['imageId'] = i
            records[i] = {'oracle_derived': False}
        result = summarize_runtime([good, rejected, wrong, failed], records)['all']
        self.assertEqual(result['images'], 4)
        self.assertEqual(result['hand_exact_match'], .25)
        self.assertEqual(result['rejectedButModelExact'], 1)
        self.assertEqual(result['acceptedWrongImages'], 1)
        self.assertEqual(result['failed_requests'], 1)

    def test_output_matching_recovers_greedy_repeated_tile_case(self):
        left = [prediction(x=2), prediction(x=0)]
        right = [prediction(x=1), prediction(x=5)]
        self.assertEqual(len(localization_pairs(left, right)), 2)

    def test_score_count_and_class_drift_are_distinct(self):
        base = scored([prediction()])
        changed = scored([prediction('2m', .80001)])
        delta = output_drift(base, changed)
        self.assertFalse(delta['countChanged'])
        self.assertTrue(delta['classMultisetChanged'])
        self.assertEqual(delta['localizedClassChanges'], [['1m', '2m']])
        self.assertAlmostEqual(delta['rawScoreAbsoluteMax'], .00001)

    def test_full_validation_and_stable_correctness_required_for_gate(self):
        base = scored([prediction()]); records = {1: {'oracle_derived': False}}
        self.assertTrue(compare_runtime([base], [copy.deepcopy(base)], records)['promotionGate']['passed'])
        self.assertFalse(compare_runtime([base], [base], records, partial=True)['promotionGate']['passed'])
        self.assertFalse(compare_runtime([base], [], records)['promotionGate']['passed'])
        unstable = copy.deepcopy(base); unstable['repeatMetricUnstable'] = True
        self.assertFalse(compare_runtime([base], [unstable], records)['promotionGate']['passed'])
        bad = scored([prediction('2m')])
        report = compare_runtime([base], [bad], records)
        self.assertFalse(report['promotionGate']['passed'])
        self.assertEqual(report['lostAutomaticExactImageIds'], [1])
        self.assertEqual(report['newAcceptedWrongImageIds'], [1])

    def test_gate_does_not_allow_net_equal_swap_of_exact_hands(self):
        good, bad = scored([prediction()]), scored([prediction('2m')])
        b1, b2, c1, c2 = map(copy.deepcopy, [good, bad, bad, good])
        for row in (b2, c2): row['imageId'] = 2
        report = compare_runtime([b1, b2], [c1, c2], {1: {}, 2: {}})
        self.assertEqual(report['primaryAutomaticExactDelta'], 0)
        self.assertFalse(report['promotionGate']['passed'])

    def test_raw_native_group_excludes_oracle_roi(self):
        truth = [Tile('1m', (i * 8, 0, 5, 10)) for i in range(10)]
        raw = [{'classId': t.class_id, 'score': .8, 'bbox': list(t.bbox)} for t in truth]
        a, b = scored(raw, truth), scored(raw, truth); b['imageId'] = 2
        groups = summarize_runtime([a, b], {1: {}, 2: {'oracle_derived': True}})
        self.assertEqual(groups['raw_ten_or_more_tiles']['images'], 1)
        self.assertEqual(groups['oracle_roi_ten_or_more_tiles']['images'], 1)

    def test_percentiles_include_every_measured_repeat(self):
        self.assertEqual(percentile([1, 2, 3], .5), 2)
        self.assertAlmostEqual(percentile([1, 2, 3], .95), 2.9)

    def test_valid_only_loader_checks_annotations_pixels_and_geometry(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); valid = root / 'valid'; valid.mkdir()
            Image.new('RGB', (40, 50)).save(valid / 'one.png')
            m = manifest()
            coco = {'images': [{'id': 1, 'file_name': 'one.png', 'width': 40, 'height': 50,
                               'group_id': 'fixture', 'sha256': digest((valid / 'one.png').read_bytes())}],
                    'annotations': [], 'categories': [{'id': i + 1, 'name': m['classes'][str(i)]} for i in range(42)]}
            annotation = json.dumps(coco).encode(); (valid / '_annotations.coco.json').write_bytes(annotation)
            payload, actual, metadata = load_validation(valid, digest(annotation), m)
            self.assertEqual(len(payload), 1)
            self.assertEqual(actual, annotation)
            self.assertFalse(metadata['partial'])
            with self.assertRaises(ValueError): load_validation(root / 'test', digest(annotation), m)
            with self.assertRaises(ValueError): load_validation(valid, '0' * 64, m)
            (valid / 'one.png').write_bytes(b'changed')
            with self.assertRaises(ValueError): load_validation(valid, digest(annotation), m)

    def test_image_path_escape_rejected_without_opening_target(self):
        with tempfile.TemporaryDirectory() as temp:
            valid = Path(temp) / 'valid'; valid.mkdir()
            m = manifest()
            coco = {'images': [{'id': 1, 'file_name': '../test/never-open.png', 'width': 40, 'height': 50,
                               'group_id': 'fixture', 'sha256': 'a' * 64}], 'annotations': [],
                    'categories': [{'id': i + 1, 'name': m['classes'][str(i)]} for i in range(42)]}
            annotation = json.dumps(coco).encode(); (valid / '_annotations.coco.json').write_bytes(annotation)
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                load_validation(valid, digest(annotation), m)

    def test_duplicate_ids_and_bad_truth_rejected(self):
        m = manifest()
        coco = {'images': [{'id': 1, 'width': 40, 'height': 50, 'group_id': 'a', 'sha256': 'a' * 64}],
                'annotations': [{'id': 0, 'image_id': 1, 'category_id': 1, 'bbox': [39, 0, 2, 2]}],
                'categories': [{'id': i + 1, 'name': m['classes'][str(i)]} for i in range(42)]}
        with self.assertRaisesRegex(ValueError, 'Out-of-bounds'): validate_coco(coco, m)
        coco['annotations'] = []; coco['images'] *= 2
        with self.assertRaisesRegex(ValueError, 'unique'): validate_coco(coco, m)


if __name__ == '__main__':
    unittest.main()
