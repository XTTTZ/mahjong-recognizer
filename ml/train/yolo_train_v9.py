"""Fresh COCO-initialized nano detector with a complete-epoch wall-time budget.

Preflight verifies bytes and train/valid provenance without loading a model.
This is a new one-class detector, never a small-model warm start from YOLO-s.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from ml.data.prepare_yolo import verify_prepared
from ml.train.provenance import dataset_contract, sha256
from ml.train.yolo_train import training_options, shared_signature
from ml.train.yolo_train_v6 import train_bounded_detector

WEIGHTS_FILENAME = 'yolov8n.pt'
WEIGHTS_SHA256 = 'f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36'
WEIGHTS_URL = 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt'


def options(args):
    config = training_options(args)
    # Fresh COCO initialization keeps three warmup epochs, unlike the mature
    # V6 warm start. Disable patience: the explicit complete-epoch budget wins.
    config.update(patience=0)
    if args.geometry == 'mild':
        config.update(degrees=8., translate=.04, scale=.1, perspective=.0004)
    return config


def validate_recipe(args):
    if (args.resolution not in (640, 800, 960) or args.batch != 8 or
        not 1 <= args.epochs <= 60 or args.workers not in (0, 1, 2) or
        not math.isfinite(args.lr) or not 0 < args.lr < .1 or
        not math.isfinite(args.max_train_minutes) or args.max_train_minutes <= 0):
        raise ValueError('Reviewed nano recipe needs 640/800/960, batch8 and a finite positive budget')
    return args


def initialization_contract(weights):
    weights = Path(weights)
    if weights.name != WEIGHTS_FILENAME or not weights.is_file():
        raise ValueError('Cache the official pinned yolov8n.pt before preflight')
    digest = sha256(weights)
    if digest != WEIGHTS_SHA256:
        raise ValueError('Official nano initialization hash mismatch')
    return {'kind': 'official_coco_pretrained', 'architecture': 'yolov8n',
            'sha256': digest, 'source': WEIGHTS_URL, 'resume': False,
            'semantics': 'new one-class detector; fresh optimizer and schedule'}


def reviewed_derivative(prepared, review_path, contract):
    review = json.loads(Path(review_path).read_text(encoding='utf-8'))
    if (not review.get('contactSheetsVisuallyReviewed') or not review.get('accepted') or
        review.get('trainAnnotationSha256') != contract['splits']['train']['annotationSha256'] or
        review.get('trainImageTreeSha256') != contract['splits']['train']['imageTreeSha256']):
        raise ValueError('Nano training requires the signed manually reviewed train backgrounds')
    if not (prepared/'prepared.json').is_file():
        return None
    derivative = verify_prepared(prepared, contract, True)
    if (derivative.get('backgroundReviewSha256') != sha256(review_path) or
        derivative.get('backgrounds') != review['accepted']):
        raise ValueError('Prepared backgrounds differ from the frozen manual review')
    return derivative


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'prepared', 'weights', 'background-review', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--resolution', type=int, choices=(640, 800, 960), default=800)
    p.add_argument('--epochs', type=int, default=45)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--lr', type=float, default=.0003)
    p.add_argument('--augmentation', choices=('none', 'photometric'), default='photometric')
    p.add_argument('--geometry', choices=('none', 'mild'), default='mild')
    p.add_argument('--max-train-minutes', type=float, default=85)
    p.add_argument('--preflight-only', action='store_true')
    args = validate_recipe(p.parse_args())
    initial = initialization_contract(args.weights)
    contract = dataset_contract(args.dataset, verify_images=True)
    if contract['ontology'] != 'faces42':
        raise ValueError('Nano training requires the reviewed 42-face source')
    derivative = reviewed_derivative(args.prepared, args.background_review, contract)
    ready = derivative is not None
    train_options = options(args)
    config = {k: v for k, v in train_options.items() if k not in {'data', 'project', 'name'}}
    config.update(geometry=args.geometry, maximumTrainMinutes=args.max_train_minutes,
                  initialization='official_coco_pretrained', oomPolicy='fail_without_batch_changes')
    signature = shared_signature(contract, config, args.weights)
    signature.update(initialization=initial, backgroundReviewSha256=sha256(args.background_review),
                     derivativeSha256=sha256(args.prepared/'prepared.json') if ready else None)
    source_root = Path(__file__).resolve().parents[1]
    for name in ('train/yolo_train_v9.py', 'train/yolo_train_v6.py',
                 'train/v6_common.py', 'data/prepare_background_v6.py'):
        signature['code'][name] = sha256(source_root/name)
    if args.preflight_only:
        print(json.dumps({'trainingStarted': False, 'derivativeReady': ready,
                          'config': config, 'initialization': initial,
                          'backgrounds': len(derivative.get('backgrounds', [])) if ready else 0,
                          'dataContract': contract, 'testOpened': False}))
        return
    if not ready:
        raise FileNotFoundError('Prepare the reviewed train-only derivative before training')
    run = {'architecture': 'yolov8n-generic', 'generic': True,
           'resolution': args.resolution, 'classes': contract['classes'], 'ontology': 'faces42',
           'resumeSignature': signature, 'config': config, 'initialization': initial,
           'initializationNote': 'Independent COCO-initialized nano detector',
           'startedAt': datetime.now(timezone.utc).isoformat(), 'plannedEpochs': args.epochs,
           'annotationSha256': contract['splits']['train']['annotationSha256'],
           'validationAnnotationSha256': contract['splits']['valid']['annotationSha256'],
           'testOpened': False}
    train_bounded_detector(args, train_options, run, args.weights)


if __name__ == '__main__':
    main()
