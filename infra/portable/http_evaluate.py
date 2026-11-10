"""Bounded frozen-validation evaluation through an existing protected HTTP endpoint.

Default mode is offline preflight. Credentials are read only by the HTTP client,
never logged or embedded in artifacts. Every image gets an immutable checkpoint.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit
import uuid

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from infra.portable.candidate_benchmark import ANNOTATION_SHA, validation_payload
from infra.portable.quality_gate import paired_quality_gate, percentile
from ml.eval.metrics import Tile, score_image, summarize


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_fresh(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def validate_endpoint(endpoint):
    parts = urlsplit(endpoint)
    if (parts.scheme != 'https' or not parts.hostname or
            not parts.hostname.endswith('.modal.run') or parts.username or
            parts.password or parts.query or parts.fragment or
            parts.path != '/recognize'):
        raise ValueError('Require a direct HTTPS Modal /recognize endpoint without credentials')
    return endpoint


def baseline_rows(path):
    """An independently generated control is optional; no private historical file is required."""
    if path is None:
        return None
    report = json.loads(Path(path).read_text())
    if report.get('annotationSha256') != ANNOTATION_SHA:
        raise ValueError('Reference annotation differs')
    rows = report['perImage']
    if isinstance(rows, dict):
        rows = rows.get('baseline', rows.get('eff320'))
    if not isinstance(rows, list) or len(rows) != 150:
        raise ValueError('Reference must contain all150 per-image metrics')
    return rows


def decode_response(body, *, request_id, version, image, truth):
    if body.get('requestId') != request_id or body.get('modelVersion') != version:
        raise ValueError('Response identity differs from request or selected model')
    if body.get('runtimeBackend') != 'onnxruntime-cpu':
        raise ValueError('Require the selected CPU runtime')
    if body.get('image') != {'width': image['width'], 'height': image['height']}:
        raise ValueError('Response dimensions differ from frozen native pixels')
    quality = body['quality']
    if quality['status'] not in ('ok', 'needs_retake') or not isinstance(quality['reasons'], list):
        raise ValueError('Invalid quality contract')
    if (quality['status'] == 'ok') != (not quality['reasons']):
        raise ValueError('Quality status and reasons disagree')
    timings = body['timings']
    for field in ('inferenceMs', 'postprocessMs', 'decodeMs', 'totalMs'):
        value = timings[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid timing contract')
    # The endpoint already applies the frozen >.35 fused floor and .7 acceptance.
    # Re-normalizing rounded scores would change the actual HTTP contract.
    predicted = [Tile(t['classId'], tuple(t['bbox'])) for t in body['tiles']]
    return score_image(truth, predicted, accepted=quality['status'] == 'ok')


def summary(rows):
    result = summarize([r['metrics'] for r in rows])
    warm = [r for r in rows if r.get('timings') and r.get('warmEligible') and not r['metrics']['failed']]
    timing = {}
    for name in ('inferenceMs', 'postprocessMs', 'decodeMs', 'totalMs', 'inferenceAndNormalizationMs', 'httpWallMs'):
        values = [(r['httpWallMs'] if name == 'httpWallMs' else
                   r['timings']['inferenceMs'] + r['timings']['postprocessMs']
                   if name == 'inferenceAndNormalizationMs' else r['timings'][name]) for r in warm]
        timing[name] = {'samples': len(values), 'p50': percentile(values, .5) if values else None,
                        'p95': percentile(values, .95) if values else None}
    result.update(modelExactCount=sum(r['metrics']['model_exact'] for r in rows),
                  wrongAccepts=sum(r['metrics']['accepted'] and not r['metrics']['model_exact'] for r in rows),
                  failedIds=[r['imageId'] for r in rows if r['metrics']['failed']],
                  wrongAcceptIds=[r['imageId'] for r in rows if r['metrics']['accepted'] and not r['metrics']['model_exact']],
                  retryCount=sum(max(0, len(r.get('attempts', []))-1) for r in rows), warmTimings=timing)
    return result


def request_image(client, endpoint, payload, image, truth, version, deadline, timeout, warm):
    import httpx
    row = {'imageId': payload['id'], 'imageSha256': hashlib.sha256(payload['data']).hexdigest(),
           'startedAt': utc_now(), 'attempts': [], 'timings': None, 'warmEligible': warm,
           'metrics': score_image(truth, [], accepted=False, failed=True)}
    for attempt in range(2):
        request_id = str(uuid.uuid4())
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            row['errorType'] = 'OverallBudgetExceeded'
            break
        start = time.perf_counter()
        receipt = {'attempt': attempt + 1, 'requestId': request_id, 'startedAt': utc_now()}
        try:
            response = client.post(endpoint, files={'image': ('validation.jpg', payload['data'], 'image/jpeg')},
                                   headers={'x-request-id': request_id}, timeout=min(timeout, remaining))
            receipt['httpWallMs'] = round((time.perf_counter()-start)*1000, 3)
            receipt['httpStatus'] = response.status_code
            row['httpWallMs'] = receipt['httpWallMs']
            if response.status_code != 200:
                row['errorType'] = 'HTTPStatusError'
                break
            body = response.json()
            metric = decode_response(body, request_id=request_id, version=version, image=image, truth=truth)
            row.update(metrics=metric, timings=body['timings'], quality=body['quality'], tiles=body['tiles'],
                       modelVersion=body['modelVersion'], runtimeBackend=body['runtimeBackend'])
            row.pop('errorType', None)
            break
        except httpx.TransportError as error:
            # Only transport failures get one retry. Do not print exception text:
            # transport implementations can include headers or other private data.
            receipt['errorType'] = type(error).__name__
            receipt['httpWallMs'] = round((time.perf_counter()-start)*1000, 3)
            row['errorType'] = type(error).__name__
            row['warmEligible'] = False
            if attempt == 1:
                break
        except (ValueError, KeyError, TypeError) as error:
            receipt['errorType'] = type(error).__name__
            row['errorType'] = 'ResponseContractError'
            break
        finally:
            row['attempts'].append(receipt)
    row['completedAt'] = utc_now()
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--valid-dir', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--token-file', type=Path, default=Path('.secrets/modal-proxy-token'))
    parser.add_argument('--baseline', type=Path, help='Optional frozen control report for historical per-image diagnostics')
    parser.add_argument('--timeout', type=int, choices=(30,45), default=45)
    parser.add_argument('--budget-seconds', type=int, default=1200)
    parser.add_argument('--run-http', action='store_true')
    args = parser.parse_args()
    if not 60 <= args.budget_seconds <= 1200:
        raise ValueError('Bound the complete run to at most20minutes')
    endpoint = validate_endpoint(args.endpoint)
    payload, annotation = validation_payload(args.valid_dir)
    manifest_data = (args.model_dir/'manifest.json').read_bytes()
    manifest = json.loads(manifest_data)
    categories = {c['id']: c['name'] for c in annotation['categories']}
    if set(categories.values()) != set(manifest['classes'].values()):
        raise ValueError('Frozen COCO and model class sets differ')
    if manifest['threshold'] != .35 or manifest['acceptThreshold'] != .7:
        raise ValueError('Require the selected frozen production operating point')
    if args.out.exists() or args.out.with_suffix('').exists():
        raise FileExistsError('Preserve immutable report and checkpoints; choose a fresh path')
    if not args.run_http:
        print(json.dumps({'ready':True,'images':len(payload),'httpStarted':False,'trainingStarted':False}))
        return
    import httpx
    token = args.token_file.read_text().strip()
    if not token or '\n' in token or '\r' in token:
        raise ValueError('Invalid token file')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    checkpoint_dir = args.out.with_suffix('')
    checkpoint_dir.mkdir()
    truth = {image['id']: [] for image in annotation['images']}
    images = {image['id']: image for image in annotation['images']}
    for tile in annotation['annotations']:
        truth[tile['image_id']].append(Tile(categories[tile['category_id']], tuple(tile['bbox'])))
    baseline = baseline_rows(args.baseline)
    started_at = utc_now(); start = time.monotonic(); deadline = start+args.budget_seconds
    rows = []
    with httpx.Client(headers={'Authorization':'Bearer '+token}, follow_redirects=False) as client:
        # One unscored warmup, saved separately. It cannot improve quality denominators.
        warmup = request_image(client, endpoint, payload[0], images[payload[0]['id']], truth[payload[0]['id']],
                               manifest['version'], deadline, args.timeout, False)
        write_fresh(checkpoint_dir/'warmup.json', warmup)
        for index, item in enumerate(payload):
            row = request_image(client, endpoint, item, images[item['id']], truth[item['id']], manifest['version'],
                                deadline, args.timeout, not warmup['metrics']['failed'] or index > 0)
            write_fresh(checkpoint_dir/f'{index+1:03d}-image-{item["id"]}.json', row)
            rows.append(row)
            print(json.dumps({'completed':len(rows),'total':len(payload),'imageId':item['id'],
                              'exact':row['metrics']['hand_exact_match'], 'failed':row['metrics']['failed'],
                              'inferenceMs':row['timings']['inferenceMs'] if row['timings'] else None,
                              'elapsedSeconds':round(time.monotonic()-start,1)}),flush=True)
    groups = {'all150':rows,'primary82':[r for r in rows if r['metrics']['truth_count']>=10],
              'thirteenFourteen45':[r for r in rows if r['metrics']['truth_count'] in (13,14)]}
    summaries = {name:summary(selected) for name,selected in groups.items()}
    speed = summaries['thirteenFourteen45']['warmTimings']['inferenceAndNormalizationMs']
    result = {'createdAt':utc_now(),'startedAt':started_at,'execution':'actual-protected-modal-http',
              'endpoint':endpoint,'concurrency':1,'allocatedPhysicalCores':1,'memoryMiB':2048,'classifierBatch':4,
              'minContainers':0,'maxContainers':1,'cpuSnapshot':True,'precision':'float32',
              'partial':False,'complete':len(rows)==150,'trainingStarted':False,'testOpened':False,
              'phoneHardwareMeasured':False,'browserCanvasTransformApplied':False,
              'annotationSha256':ANNOTATION_SHA,'manifestSha256':hashlib.sha256(manifest_data).hexdigest(),
              'modelVersion':manifest['version'],'filesSha256':manifest['filesSha256'],
              'thresholds':{'detector':.05,'fused':.35,'accept':.7,'iou':.5},
              'elapsedSeconds':round(time.monotonic()-start,3),'budgetSeconds':args.budget_seconds,
              'warmup':warmup,'summaries':summaries,'perImage':rows,
              'historicalReference':str(args.baseline) if args.baseline else None,
              'historicalQualityGate':paired_quality_gate(baseline,rows) if baseline is not None else None,
              'warmLatencyTargetPassed':speed['samples']==45 and speed['p95']<=2000 and not any(r['metrics']['failed'] for r in rows),
              'timingScope':'Native frozen JPEG pixels, one sequential actual HTTP request per image after separately recorded warmup. Server inferenceMs includes threadpool/lock wait; inferenceAndNormalizationMs adds server postprocessMs. totalMs also includes multipart upload/decode; HTTP wall includes network/proxy. No own concurrent queue; unrelated traffic cannot be excluded. Retries excluded from warm timing, retained in quality denominator.',
              'checkpointDirectory':str(checkpoint_dir),'privateArtifact':True}
    write_fresh(args.out, result)
    print(json.dumps({'report':str(args.out),'complete':result['complete'],'summaries':summaries,
                      'historicalQualityGate':result['historicalQualityGate'],'warmLatencyTargetPassed':result['warmLatencyTargetPassed']}),flush=True)


if __name__ == '__main__':
    main()
