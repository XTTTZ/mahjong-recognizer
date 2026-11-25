"""Compare the selected 42-face detector with the signed compact cascade on the same CPU."""
import argparse
from collections import defaultdict
from io import BytesIO
import json
import os
from pathlib import Path
import platform
import time

from infra.portable.candidate_benchmark import ANNOTATION_SHA, validation_payload
from infra.portable.quality_gate import paired_quality_gate, percentile
from ml.eval.metrics import Tile, score_image, summarize
from ml.eval.recover_terminal import checked_run
from ml.inference.postprocess import MAX_DETECTIONS, normalize_tiles
from ml.train.provenance import atomic_json, dataset_contract, sha256
from ml.train.yolo_train import verify_receipt


def checked_control(component, contract):
    checkpoint = Path(component['checkpoint']['path'])
    if Path(component['receipt']['path']) != Path(str(checkpoint) + '.receipt.json'):
        raise ValueError('Compact control receipt must accompany its checkpoint')
    for key in ('checkpoint', 'run', 'receipt'):
        asset = component[key]
        if sha256(Path(asset['path'])) != asset['sha256']:
            raise ValueError(f'Compact control {key} hash changed')
    run, _ = checked_run(Path(component['run']['path']), contract)
    verify_receipt(checkpoint, run)
    return checkpoint, run


def paired_measure(models, payload, coco):
    from PIL import Image
    tokens = {c['id']: c['name'] for c in coco['categories']}
    truth = defaultdict(list)
    for a in coco['annotations']:
        truth[a['image_id']].append(Tile(tokens[a['category_id']], tuple(a['bbox'])))
    rows = {name: [] for name in models}
    for index, record in enumerate(payload):
        image = Image.open(BytesIO(record['data'])).convert('RGB')
        targets = truth[record['id']]
        for name in (list(models) if index % 2 == 0 else reversed(models)):
            start = time.perf_counter()
            raw = models[name].predict(image)
            retained = [tile for tile in raw if tile['score'] > .35]
            crowded = len(retained) > MAX_DETECTIONS
            tiles, reasons = ([], ['crowded']) if crowded else normalize_tiles(retained, *image.size, .7)
            ms = (time.perf_counter() - start) * 1000
            metrics = score_image(targets, [Tile(t['classId'], tuple(t['bbox'])) for t in tiles],
                                  accepted=not reasons, failed=crowded)
            rows[name].append({'imageId': record['id'], 'pipelineMs': ms, 'metrics': metrics})
    summaries = {}
    for name, results in rows.items():
        summaries[name] = {}
        for key, group in [('all150', results), ('primary82', [r for r in results if r['metrics']['truth_count'] >= 10]),
                           ('thirteenFourteen45', [r for r in results if r['metrics']['truth_count'] in (13, 14)])]:
            scores = [r['metrics'] for r in group]
            timings = [r['pipelineMs'] for r in group]
            summaries[name][key] = {**summarize(scores), 'wrongAccepts': sum(s['accepted'] and not s['model_exact'] for s in scores),
                                    'p50Ms': percentile(timings, .5) if timings else None,
                                    'p95Ms': percentile(timings, .95) if timings else None, 'timedImages': len(group)}
    return {'perImage': rows, 'summaries': summaries, 'strictGate': paired_quality_gate(rows['baseline'], rows['candidate'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'run', 'selection', 'control-record', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a fresh immutable benchmark report')
    os.environ.update(CUDA_VISIBLE_DEVICES='-1', YOLO_OFFLINE='true', YOLO_AUTOINSTALL='false',
                      OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', YOLO_CONFIG_DIR=str(args.run.parent/'yolo-config'))
    import torch
    from PIL import Image
    from ml.inference.yolo_model import RawYOLO
    from ml.inference.student_model import RawStudentCascade
    if torch.cuda.is_available():
        raise RuntimeError('CPU benchmark cannot see CUDA')
    torch.set_num_interop_threads(1)
    torch.set_num_threads(1)
    contract = dataset_contract(args.dataset, verify_images=True)
    payload, coco = validation_payload(args.dataset/'valid')
    direct, _ = checked_run(args.run, contract)
    selection = json.loads(args.selection.read_text(encoding='utf-8'))
    checkpoint = Path(selection['selectedCheckpoint']).resolve()
    if (direct.get('generic') or direct['architecture'] != 'yolov8n'
            or checkpoint not in {(args.run.parent/'train/weights'/n).resolve() for n in ('best.pt', 'last.pt')}
            or selection['checkpointSha256'] != sha256(checkpoint) or not selection['checkpointSelectionComplete']
            or selection['partial'] or not selection['operatingPointLocked'] or selection['inferenceFailedImages']
            or selection['validationAnnotationSha256'] != ANNOTATION_SHA):
        raise ValueError('A terminal, failure-free, locked direct detector selection is required')
    verify_receipt(checkpoint, direct)
    baseline = json.loads(args.control_record.read_text(encoding='utf-8'))
    det, det_run = checked_control(baseline['detector'], contract)
    cls, cls_run = checked_control(baseline['classifier'], contract)
    locator = RawYOLO(det, direct['classes'], det_run['resolution'], device='cpu', generic=True)
    models = {'baseline': RawStudentCascade(locator, cls, direct['classes'], cls_run['config']),
              'candidate': RawYOLO(checkpoint, direct['classes'], direct['resolution'], device='cpu')}
    warm = Image.open(BytesIO(payload[0]['data'])).convert('RGB')
    for model in models.values():
        model.predict(warm)
    # Ultralytics selects its own thread count on first prediction.
    torch.set_num_threads(1)
    import cv2
    cv2.setNumThreads(1)
    measured = paired_measure(models, payload, coco)
    if torch.get_num_threads() != 1 or any(s['timedImages'] != n for m in measured['summaries'].values()
                                            for k, n in [('all150', 150), ('primary82', 82), ('thirteenFourteen45', 45)]
                                            for s in [m[k]]):
        raise RuntimeError('Benchmark thread count or frozen cohort coverage changed')
    report = {**measured, 'execution': 'local-cpu', 'runtime': 'pytorch-fp32', 'torchVersion': torch.__version__,
              'hardware': platform.platform(), 'processor': platform.processor(), 'intraOpThreads': 1, 'interOpThreads': 1,
              'pairedOnSameHardware': True, 'annotationSha256': ANNOTATION_SHA, 'testOpened': False,
              'modelHashes': {'candidate': sha256(checkpoint), 'baselineDetector': sha256(det), 'baselineClassifier': sha256(cls)},
              'thresholds': {'detector': .05, 'fused': .35, 'accept': .7}, 'productionPromotion': False,
              'timingScope': 'Alternating order, one warm PyTorch CPU pass per model per image, includes normalization, excludes decode/load/network/ground truth scoring.'}
    atomic_json(args.out, report)


if __name__ == '__main__':
    main()
