"""Validation-only, paired GPU runtime benchmark; never runs inference locally.

Default: verify the frozen model and valid/ bytes, without importing Modal or
PyTorch. Add --run to start one ephemeral, bounded L4/A10 function. See the
adjacent benchmark_runtime.md for commands and the promotion gate.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from io import BytesIO
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import time

THRESHOLD = .35
ACCEPT_THRESHOLD = .7
IOU_THRESHOLD = .5


def digest(data):
    return hashlib.sha256(data).hexdigest()


def plain_json(value):
    """Strip SDK/tensor-library scalar subclasses before Modal pickles the result."""
    return json.loads(json.dumps(value, allow_nan=False))


def verify_manifest(manifest):
    from ml.inference.manifest import validate_labels
    validate_labels(manifest)
    if (manifest.get('architecture') != 'yolo-efficientnet-v2-s-cascade'
            or len(manifest['classes']) != 42
            or manifest['threshold'] != THRESHOLD
            or manifest['acceptThreshold'] != ACCEPT_THRESHOLD):
        raise ValueError('Requires the frozen 42-face cascade at threshold .35 / acceptance .7')
    if set(manifest.get('filesSha256', {})) != {'detector.pt', 'classifier.pt'}:
        raise ValueError('Both model stage hashes are required')


def verify_engine_receipt(engine_dir, manifest, gpu_name, compute_capability):
    """Reject another model/GPU or altered engines before any comparison forward."""
    root = Path(engine_dir)
    receipt = json.loads((root / 'runtime.json').read_text())
    # Modal's A10 pool can report A10 or A10G. This is a name-only alias;
    # compute capability, model/engine bytes and the runtime gate stay strict.
    def gpu_family(name):
        return 'NVIDIA A10 family' if name in ('NVIDIA A10', 'NVIDIA A10G') else name
    expected = {'modelVersion': manifest['version'], 'modelFilesSha256': manifest['filesSha256'],
                'gpu': gpu_family(gpu_name), 'computeCapability': list(compute_capability),
                'precision': 'fp16', 'calibrationDataUsed': False,
                'engineFiles': ['classifier.engine', 'detector.engine']}
    actual = {key: receipt.get(key) for key in expected}
    actual['gpu'] = gpu_family(receipt.get('gpu'))
    actual['engineFiles'] = sorted(receipt.get('enginesSha256', {}))
    mismatches = {key: {'expected': value, 'actual': actual[key]}
                  for key, value in expected.items() if actual[key] != value}
    if gpu_family(gpu_name) == 'NVIDIA A10 family' and list(compute_capability) != [8, 6]:
        mismatches['gpuFamilyComputeCapability'] = {'expected': [8, 6], 'actual': list(compute_capability)}
    if (receipt.get('gpu') != gpu_name
            and gpu_family(receipt.get('gpu')) == gpu_family(gpu_name) == 'NVIDIA A10 family'
            and receipt.get('hardwareCompatibility') != 'AMPERE_PLUS'):
        mismatches['hardwareCompatibility'] = {'expected': 'AMPERE_PLUS for A10/A10G cross-GPU reuse',
                                              'actual': receipt.get('hardwareCompatibility')}
    if receipt.get('calibrationDataUsed') is not False:
        mismatches['calibrationDataUsed'] = {'expected': False, 'actual': receipt.get('calibrationDataUsed')}
    if mismatches:
        if 'gpu' in mismatches:
            mismatches['gpu'].update(expectedRaw=gpu_name, actualRaw=receipt.get('gpu'))
        raise ValueError('TensorRT receipt mismatch: ' + json.dumps(mismatches, sort_keys=True))
    for name, sha256 in receipt['enginesSha256'].items():
        actual_digest = digest((root / name).read_bytes())
        if actual_digest != sha256:
            raise ValueError(f'TensorRT engine SHA256 mismatch: {name}; expected={sha256}, actual={actual_digest}')
    return receipt


def validate_coco(coco, manifest):
    """Validate only the supplied valid annotations; no dataset contract opens train/test."""
    from ml.eval.metrics import Tile
    verify_manifest(manifest)
    categories = sorted(coco['categories'], key=lambda c: c['id'])
    if ([c['id'] for c in categories] != list(range(1, 43))
            or {str(i): c['name'] for i, c in enumerate(categories)} != manifest['classes']):
        raise ValueError('Validation classes differ from the frozen model')
    images = {r['id']: r for r in coco['images']}
    if not images or len(images) != len(coco['images']):
        raise ValueError('Validation image IDs must be unique and nonempty')
    truth, annotation_ids = defaultdict(list), set()
    for record in images.values():
        if (type(record['width']) is not int or type(record['height']) is not int
                or min(record['width'], record['height']) < 32 or not record.get('group_id')
                or len(record.get('sha256', '')) != 64):
            raise ValueError('Use frozen, grouped validation images with byte hashes and dimensions')
    for row in coco['annotations']:
        if (row['id'] in annotation_ids or row['image_id'] not in images
                or row['category_id'] not in range(1, 43) or row.get('iscrowd', 0)):
            raise ValueError('Invalid, duplicated or crowd annotation')
        annotation_ids.add(row['id'])
        tile = Tile(manifest['classes'][str(row['category_id'] - 1)], tuple(row['bbox']))
        image = images[row['image_id']]
        x, y, w, h = tile.bbox
        if min(x, y) < 0 or x + w > image['width'] + 1e-5 or y + h > image['height'] + 1e-5:
            raise ValueError('Out-of-bounds truth box')
        truth[row['image_id']].append(tile)
    return images, truth


def load_validation(valid_dir, annotation_sha256, manifest, limit=None):
    """Read explicit valid/ only; reject test paths, symlink escapes and changed pixels."""
    from PIL import Image
    valid_dir = Path(valid_dir).resolve()
    if valid_dir.name != 'valid':
        raise ValueError('Explicit frozen valid/ directory required; test is never supported')
    annotation_bytes = (valid_dir / '_annotations.coco.json').read_bytes()
    if digest(annotation_bytes) != annotation_sha256:
        raise ValueError('Validation annotation SHA256 differs from the explicitly frozen value')
    coco = json.loads(annotation_bytes)
    images, truth = validate_coco(coco, manifest)
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError('limit must be positive')
    selected = list(images.values())[:limit]
    payload, seen_paths = [], set()
    for record in selected:
        path = (valid_dir / record['file_name']).resolve()
        if not path.is_relative_to(valid_dir) or path in seen_paths:
            raise ValueError('Unsafe or duplicate validation image path')
        seen_paths.add(path)
        data = path.read_bytes()
        if digest(data) != record['sha256']:
            raise ValueError('Validation pixels differ from the frozen image hash')
        with Image.open(BytesIO(data)) as image:
            if image.size != (record['width'], record['height']) or image.getexif().get(274, 1) != 1:
                raise ValueError('Normalize dimensions/EXIF in a reviewed derivative before benchmarking')
            image.load()
        payload.append({'imageId': record['id'], 'data': data})
    counts = Counter(len(truth[r['id']]) for r in selected)
    metadata = {'split': 'validation_only', 'testOpened': False,
        'annotationSha256': annotation_sha256, 'fullImages': len(images),
        'selectedImages': len(selected), 'partial': len(selected) != len(images),
        'groups': len({r['group_id'] for r in selected}),
        'tileCountDistribution': dict(sorted(counts.items())),
        'imageTreeSha256': digest(json.dumps(sorted((str(r['id']), r['sha256']) for r in selected)).encode()),
        'payloadBytes': sum(len(p['data']) for p in payload)}
    return payload, annotation_bytes, metadata


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def score_output(raw, truth, width, height, error=None):
    from ml.eval.metrics import Tile, score_image
    from ml.inference.postprocess import MAX_DETECTIONS, normalize_tiles
    if error is not None:
        return {'tiles': [], 'rawPredictions': [], 'reasons': [], 'error': error,
                'metrics': score_image(truth, [], accepted=False, failed=True)}
    retained = [tile for tile in raw if tile['score'] > THRESHOLD]
    if len(retained) > MAX_DETECTIONS:
        return score_output([], truth, width, height, 'crowded_scene')
    tiles, reasons = normalize_tiles(retained, width, height, ACCEPT_THRESHOLD)
    return {'tiles': tiles, 'rawPredictions': retained, 'reasons': reasons, 'error': None,
            'metrics': score_image(truth, [Tile(t['classId'], tuple(t['bbox'])) for t in tiles],
                                   accepted=not reasons, threshold=IOU_THRESHOLD)}


def localization_pairs(left, right):
    """Maximum bipartite IoU matching, independent of class, for output drift."""
    from ml.eval.metrics import Tile, iou
    a = [Tile(t['classId'], tuple(t['bbox'])) for t in left]
    b = [Tile(t['classId'], tuple(t['bbox'])) for t in right]
    edges = [sorted((j for j, y in enumerate(b) if iou(x, y) >= IOU_THRESHOLD),
                    key=lambda j: (-iou(x, b[j]), j)) for x in a]
    assigned = {}
    def augment(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in assigned or augment(assigned[j], seen):
                assigned[j] = i
                return True
        return False
    for i in range(len(a)):
        augment(i, set())
    return sorted((i, j) for j, i in assigned.items())


def output_drift(baseline, candidate):
    left, right = baseline['tiles'], candidate['tiles']
    pairs = localization_pairs(left, right)
    deltas = [abs(left[i]['score'] - right[j]['score']) for i, j in pairs]
    raw_left, raw_right = baseline['rawPredictions'], candidate['rawPredictions']
    raw_pairs = localization_pairs(raw_left, raw_right)
    raw_deltas = [abs(raw_left[i]['score'] - raw_right[j]['score']) for i, j in raw_pairs]
    flips = [[left[i]['classId'], right[j]['classId']] for i, j in pairs
             if left[i]['classId'] != right[j]['classId']]
    return {'countChanged': len(left) != len(right),
        'classMultisetChanged': Counter(t['classId'] for t in left) != Counter(t['classId'] for t in right),
        'localizedClassChanges': flips, 'localizedPairs': len(pairs),
        'unmatchedBaseline': len(left) - len(pairs), 'unmatchedCandidate': len(right) - len(pairs),
        'normalizedScoreAbsoluteMax': max(deltas, default=0.),
        'rawScoreAbsoluteMax': max(raw_deltas, default=0.),
        'bboxCoordinateAbsoluteMax': max((abs(a - b) for i, j in pairs
            for a, b in zip(left[i]['bbox'], right[j]['bbox'])), default=0.),
        'acceptanceChanged': baseline['metrics']['accepted'] != candidate['metrics']['accepted'],
        'handExactChanged': baseline['metrics']['hand_exact_match'] != candidate['metrics']['hand_exact_match'],
        'modelExactChanged': baseline['metrics']['model_exact'] != candidate['metrics']['model_exact']}


def image_groups(record, count):
    groups = ['all']
    if count == 1:
        groups.append('single_tile')
    if count >= 10:
        groups += ['ten_or_more_tiles', 'oracle_roi_ten_or_more_tiles'
                   if record.get('oracle_derived', False) else 'raw_ten_or_more_tiles']
    if count in (13, 14):
        groups.append('thirteen_or_fourteen_tiles')
    return groups


def summarize_runtime(rows, records):
    from ml.eval.metrics import summarize
    groups = defaultdict(list)
    for row in rows:
        for name in image_groups(records[row['imageId']], row['metrics']['truth_count']):
            groups[name].append(row)
    result = {}
    for name, entries in groups.items():
        scores = [r['metrics'] for r in entries]
        times = [elapsed for r in entries for elapsed in r['inferenceMs']]
        truth_count = sum(s['truth_count'] for s in scores)
        prediction_count = sum(s['prediction_count'] for s in scores)
        matched = sum(s['matched'] for s in scores)
        localized = sum(s['localization_matches'] for s in scores)
        result[name] = dict(summarize(scores), rejectedImages=sum(not s['accepted'] for s in scores),
            rejectedButModelExact=sum(s['model_exact'] and not s['accepted'] for s in scores),
            acceptedWrongImages=sum(s['accepted'] and not s['model_exact'] for s in scores),
            uncertainTiles=sum(sum(t['uncertain'] for t in r['tiles']) for r in entries),
            tiles={'truth': truth_count, 'predictions': prediction_count, 'classAndIoUMatched': matched,
                   'localizationMatched': localized,
                   'precisionIoU50': matched / prediction_count if prediction_count else None,
                   'recallIoU50': matched / truth_count if truth_count else None,
                   'localizationRecallIoU50': localized / truth_count if truth_count else None},
            inferenceMs={'p50': percentile(times, .5), 'p95': percentile(times, .95),
                         'mean': statistics.mean(times), 'samples': len(times)})
    return result


def compare_runtime(baseline, candidate, records, *, partial=False):
    base = {r['imageId']: r for r in baseline}
    current = {r['imageId']: r for r in candidate}
    complete = len(base) == len(baseline) == len(candidate) == len(records) and base.keys() == current.keys() == records.keys()
    shared = [key for key in base if key in current]
    details = [{'imageId': key, **output_drift(base[key], current[key])} for key in shared]
    primary = [key for key in shared if 'raw_ten_or_more_tiles' in image_groups(records[key], base[key]['metrics']['truth_count'])]
    if not primary:
        primary = shared
    lost_auto = [key for key in shared if base[key]['metrics']['hand_exact_match'] and not current[key]['metrics']['hand_exact_match']]
    lost_model = [key for key in shared if base[key]['metrics']['model_exact'] and not current[key]['metrics']['model_exact']]
    added_wrong = [key for key in shared if current[key]['metrics']['accepted'] and not current[key]['metrics']['model_exact']
                   and not (base[key]['metrics']['accepted'] and not base[key]['metrics']['model_exact'])]
    instability = [key for key in shared if base[key].get('repeatMetricUnstable') or current[key].get('repeatMetricUnstable')]
    reasons = []
    if partial or not complete:
        reasons.append('requires full frozen validation coverage')
    if lost_auto:
        reasons.append('baseline automatic exact hands regressed')
    if lost_model:
        reasons.append('baseline model-exact hands regressed')
    if added_wrong:
        reasons.append('new accepted wrong hands')
    if any(current[key]['metrics']['failed'] for key in shared):
        reasons.append('candidate inference failures')
    if instability:
        reasons.append('repeated class/count/acceptance or correctness is unstable')
    baseline_times = [t for row in baseline for t in row['inferenceMs']]
    candidate_times = [t for row in candidate for t in row['inferenceMs']]
    speedup = {}
    for name, quantile in (('p50', .5), ('p95', .95)):
        before, after = percentile(baseline_times, quantile), percentile(candidate_times, quantile)
        speedup[name] = {'baselineMs': before, 'candidateMs': after,
                        'speedup': before / after if before is not None and after else None}
    return {'promotionGate': {'passed': not reasons, 'reasons': reasons,
            'policy': 'full validation; preserve every baseline exact hand; no new accepted wrong hands/failures; stable repeats'},
        'primaryGroup': 'raw_ten_or_more_tiles' if primary != shared else 'all',
        'primaryImages': len(primary),
        'primaryAutomaticExactDelta': sum(current[k]['metrics']['hand_exact_match'] - base[k]['metrics']['hand_exact_match'] for k in primary),
        'primaryModelExactDelta': sum(current[k]['metrics']['model_exact'] - base[k]['metrics']['model_exact'] for k in primary),
        'primaryAutomaticExactRateDelta': sum(current[k]['metrics']['hand_exact_match'] - base[k]['metrics']['hand_exact_match'] for k in primary) / len(primary) if primary else None,
        'latencyComparison': speedup,
        'lostAutomaticExactImageIds': lost_auto, 'lostModelExactImageIds': lost_model,
        'newAcceptedWrongImageIds': added_wrong, 'unstableRepeatImageIds': instability,
        'countDriftImages': sum(d['countChanged'] for d in details),
        'classMultisetDriftImages': sum(d['classMultisetChanged'] for d in details),
        'acceptanceDriftImages': sum(d['acceptanceChanged'] for d in details),
        'localizedClassChanges': dict(Counter('->'.join(pair) for d in details for pair in d['localizedClassChanges'])),
        'maxRawScoreAbsoluteDrift': max((d['rawScoreAbsoluteMax'] for d in details), default=0.),
        'maxNormalizedScoreAbsoluteDrift': max((d['normalizedScoreAbsoluteMax'] for d in details), default=0.),
        'perImageDrift': details}


def benchmark_gpu(payload, annotation_bytes, metadata, expected_manifest, with_tensorrt=False,
                  repeats=3, warmups=2, gpu_type='L4'):
    """Remote entry: all model construction/forward is confined to a CUDA GPU."""
    import gc
    import torch
    from PIL import Image
    from ml.inference.cascade_model import CascadePredictor, verify_model_directory
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required; CPU inference is deliberately unsupported')
    if gpu_type not in ('L4', 'A10') or gpu_type not in torch.cuda.get_device_name():
        raise RuntimeError('Benchmark GPU differs from the explicitly requested GPU')
    if not 1 <= repeats <= 5 or not 1 <= warmups <= 3:
        raise ValueError('Bounded repeat/warmup budget required')
    torch.set_num_threads(2)
    directory, manifest = verify_model_directory('/model')
    if (manifest != expected_manifest
            or digest((directory / 'model.json').read_bytes()) != metadata['modelManifestSha256']
            or digest(annotation_bytes) != metadata['annotationSha256']):
        raise ValueError('Remote model or annotation bytes differ from local preflight')
    if with_tensorrt:
        verify_engine_receipt('/engines', manifest, torch.cuda.get_device_name(), torch.cuda.get_device_capability())
    records, truth = validate_coco(json.loads(annotation_bytes), manifest)
    selected = {row['imageId']: records[row['imageId']] for row in payload}
    if len(selected) != len(payload) or len(payload) != metadata['selectedImages']:
        raise ValueError('Duplicate or missing remote images')
    modes = ['torch-original', 'torch-fast-preprocess'] + (['tensorrt-fp16'] if with_tensorrt else [])
    models, load_ms, results, latencies = {}, {}, {name: [] for name in modes}, defaultdict(list)
    runtime_receipt = None
    for mode in modes:
        started = time.perf_counter()
        predictor = CascadePredictor(directory, device='cuda')
        if mode != 'torch-original':
            predictor.enable_fast_preprocess()
        if mode == 'tensorrt-fp16':
            from ml.inference.tensorrt_runtime import activate_tensorrt
            runtime_receipt = activate_tensorrt(predictor, '/engines')
        predictor.warmup()
        torch.cuda.synchronize()
        load_ms[mode] = (time.perf_counter() - started) * 1000
        models[mode] = predictor
    for index, item in enumerate(payload):
        record = selected[item['imageId']]
        if digest(item['data']) != record['sha256']:
            raise ValueError('Remote image bytes differ from signed validation')
        with Image.open(BytesIO(item['data'])) as source:
            if source.size != (record['width'], record['height']) or source.getexif().get(274, 1) != 1:
                raise ValueError('Remote validation geometry changed')
            image = source.convert('RGB')
        if index == 0:
            for mode in modes:
                for _ in range(warmups):
                    models[mode].predict(image)
                torch.cuda.synchronize()
        observed = defaultdict(list)
        for repeat in range(repeats):
            # Rotate mode order on each image/repeat to reduce ordering bias.
            offset = (index + repeat) % len(modes)
            for mode in modes[offset:] + modes[:offset]:
                torch.cuda.synchronize()
                started, raw, error = time.perf_counter(), [], None
                try:
                    raw = models[mode].predict(image)
                    torch.cuda.synchronize()
                except Exception as exception:
                    error = type(exception).__name__ + ': ' + str(exception)[:200]
                    torch.cuda.synchronize()
                elapsed = (time.perf_counter() - started) * 1000
                try:
                    row = score_output(raw, truth[record['id']], image.width, image.height, error)
                except Exception as exception:
                    row = score_output([], truth[record['id']], image.width, image.height, type(exception).__name__)
                observed[mode].append(row)
                latencies[(mode, record['id'])].append(elapsed)
        for mode in modes:
            first = observed[mode][0]
            metric_keys = ('hand_exact_match', 'model_exact', 'accepted', 'failed', 'prediction_count')
            first['repeatMetricUnstable'] = any(
                any(first['metrics'][key] != other['metrics'][key] for key in metric_keys)
                or Counter(t['classId'] for t in first['tiles']) != Counter(t['classId'] for t in other['tiles'])
                for other in observed[mode][1:])
            first.update(imageId=record['id'], inferenceMs=latencies[(mode, record['id'])],
                         repeatMetrics=[r['metrics'].copy() for r in observed[mode]],
                         repeatMaxRawScoreDrift=max((output_drift(first, r)['rawScoreAbsoluteMax']
                                                    for r in observed[mode][1:]), default=0.))
            results[mode].append(first)
        if (index + 1) % 10 == 0:
            print(json.dumps({'event': 'validation_runtime_benchmark', 'images': index + 1,
                              'total': len(payload), 'modes': modes}), flush=True)
    report = {'schemaVersion': 1, 'completedAt': datetime.now(timezone.utc).isoformat(),
        'data': metadata, 'model': manifest, 'operatingPointLocked': True,
        'threshold': THRESHOLD, 'acceptThreshold': ACCEPT_THRESHOLD, 'iouThreshold': IOU_THRESHOLD,
        'modelTraining': False, 'testOpened': False, 'externalPhoneBenchmark': False,
        'webPreprocessingApplied': False,
        'gpu': torch.cuda.get_device_name(), 'computeCapability': list(torch.cuda.get_device_capability()),
        'torch': str(torch.__version__), 'cuda': torch.version.cuda, 'cpuThreads': torch.get_num_threads(),
        'repeats': repeats, 'warmupImageCallsPerMode': warmups,
        'timingScope': 'GPU-synchronized detector + PIL crop + classifier only; excludes decode, normalize, model load, network and cold start',
        'loadAndSyntheticWarmupMs': load_ms, 'tensorRTRuntime': runtime_receipt,
        'runtimeSourceSha256': {name: digest(Path('/app/ml/inference', name).read_bytes())
                               for name in ('cascade_model.py', 'tensorrt_runtime.py', 'postprocess.py')},
        'summaries': {mode: summarize_runtime(rows, selected) for mode, rows in results.items()},
        'comparisons': {mode: compare_runtime(results['torch-original'], results[mode], selected,
                                            partial=metadata['partial']) for mode in modes[1:]},
        'perImage': results,
        'notes': ['Native frozen valid pixels; browser canvas longest-edge2560/JPEG.92 consistency is a separate validation-only check.',
                  'API boxes and scores use shared rounding; raw score drift is reported separately.',
                  'Promotion is validation evidence for this GPU/runtime, not a new sealed-test or production accuracy claim.']}
    models.clear()
    gc.collect()
    torch.cuda.empty_cache()
    return plain_json(report)


def register_remote(gpu_type, image_id=None):
    """Reuse an immutable image directly, or lazily construct the deployment image."""
    if image_id is not None:
        if not isinstance(image_id, str) or not re.fullmatch(r'im-[a-zA-Z0-9]+', image_id):
            raise ValueError('Expected a Modal built image ID im-...')
        import modal
        # Do not import app.py: doing so would include its model/ML build graph
        # and could recompile TensorRT after unrelated training source changes.
        app = modal.App('mahjong-runtime-benchmark')
        image = modal.Image.from_id(image_id)
    else:
        from infra.modal.app import app, image
    # Carry only this small tool, not validation pixels or the whole checkout.
    benchmark_image = image.add_local_file(Path(__file__), '/app/infra/modal/benchmark_runtime.py', copy=True)
    # A CLI may run under Python 3.12 while the image uses Python 3.11. Register
    # the importable module function, never a serialized __main__ function.
    from importlib import import_module
    remote_function = import_module('infra.modal.benchmark_runtime').benchmark_gpu
    remote = app.function(image=benchmark_image, gpu=gpu_type, cpu=2, memory=8192, timeout=3600,
        min_containers=0, max_containers=1, scaledown_window=2)(remote_function)
    return app, remote


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--valid-dir', type=Path, required=True)
    parser.add_argument('--annotation-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, default=os.environ.get('MAHJONG_MODEL_DIR'))
    parser.add_argument('--gpu', choices=('L4', 'A10'), default=os.environ.get('MAHJONG_GPU', 'L4'))
    parser.add_argument('--with-tensorrt', action='store_true')
    parser.add_argument('--image-id', help='Reuse built im-... directly; skips app.py import and engine build graph')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--limit', type=int, help='Partial spot check only; can never pass the promotion gate')
    parser.add_argument('--run', action='store_true', help='Explicitly start bounded remote GPU inference')
    args = parser.parse_args()
    if args.model_dir is None:
        parser.error('--model-dir or MAHJONG_MODEL_DIR is required')
    if args.out.exists():
        raise FileExistsError('Choose a fresh output file; existing benchmark evidence is immutable')
    if args.gpu not in ('L4', 'A10') or not 1 <= args.repeats <= 5 or not 1 <= args.warmups <= 3:
        raise ValueError('Use exactly one GPU and bounded repeats/warmups')
    if args.image_id is not None and not re.fullmatch(r'im-[a-zA-Z0-9]+', args.image_id):
        raise ValueError('Expected a Modal built image ID im-...')
    from ml.inference.model import inspect_model
    manifest = inspect_model(args.model_dir)
    payload, annotations, metadata = load_validation(args.valid_dir, args.annotation_sha256, manifest, args.limit)
    metadata.update(modelManifestSha256=digest((args.model_dir / 'model.json').read_bytes()),
                    prebuiltImageId=args.image_id)
    preflight = {'preflightPassed': True, 'gpuStarted': False, 'localForwardExecuted': False,
                 'modelVersion': manifest['version'], 'modelFilesSha256': manifest['filesSha256'],
                 'gpuRequested': args.gpu, 'withTensorRT': args.with_tensorrt,
                 'prebuiltImageId': args.image_id, 'data': metadata}
    if not args.run:
        print(json.dumps(preflight, indent=2))
        return preflight
    os.environ['MAHJONG_MODEL_DIR'] = str(args.model_dir.resolve())
    os.environ['MAHJONG_GPU'] = args.gpu
    os.environ['MAHJONG_RUNTIME'] = 'tensorrt' if args.with_tensorrt else 'torch'
    os.environ.setdefault('MAHJONG_MODAL_APP', 'mahjong-runtime-benchmark')
    import modal
    app, remote = register_remote(args.gpu, args.image_id)
    with modal.enable_output(), app.run():
        report = remote.remote(payload, annotations, metadata, manifest, args.with_tensorrt,
                               args.repeats, args.warmups, args.gpu)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'report': str(args.out), 'comparisons': {
        mode: {k: v for k, v in result.items() if k != 'perImageDrift'}
        for mode, result in report['comparisons'].items()}}, indent=2))
    return report


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    main()
