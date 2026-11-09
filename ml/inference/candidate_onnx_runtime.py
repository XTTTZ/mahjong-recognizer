"""Research-only FP32 candidate profiles; never used by the production factory."""
import json
import math
from pathlib import Path
import time

from ml.inference.manifest import file_sha256, model_file, validate_labels
from ml.inference.onnx_runtime import detector_tensor, classifier_tensor, numpy_nms, softmax
from ml.inference.cascade_model import combine_tiles


DETECTORS = {'yolov8n-generic', 'yolov8s-generic', 'yolo26s-generic'}
RESOLUTIONS = (640, 800, 960)


def head_points(resolution):
    if type(resolution) is not int or resolution not in RESOLUTIONS:
        raise ValueError('Candidate detector resolution must be640/800/960')
    return sum((resolution // stride) ** 2 for stride in (8, 16, 32))


def io_contracts(resolution, classifier_resolution):
    return {
        'detector.onnx': {'inputs': {'images': [1, 3, resolution, resolution]},
                          'outputs': {'output0': [1, 5, head_points(resolution)]}, 'float32': True},
        'classifier.onnx': {'inputs': {'images': ['batch', 3, classifier_resolution, classifier_resolution]},
                            'outputs': {'logits': ['batch', 42]}, 'float32': True},
    }


def validate_candidate_manifest(manifest):
    validate_labels(manifest)
    required = {'schemaVersion': 1, 'format': 'research-onnx-fp32-cascade', 'researchOnly': True,
                'ontology': 'faces42', 'detectorThreshold': .05, 'threshold': .35, 'acceptThreshold': .7,
                'nmsIou': .6, 'detectorNmsMaxDetections': 300, 'maxDetections': 200,
                'maxClassifierBatch': 32, 'opset': 17, 'precision': 'float32'}
    if any(manifest.get(k) != v for k, v in required.items()):
        raise ValueError('Candidate must use explicit research profile and fixed production thresholds')
    head_points(manifest['resolution'])
    if manifest.get('detectorArchitecture') not in DETECTORS:
        raise ValueError('Unsupported candidate detector')
    sizes = {'mobilenet-v3-large': (192, 224), 'efficientnet-v2-s': (320,)}
    if manifest.get('classifierResolution') not in sizes.get(manifest.get('classifierArchitecture'), ()):
        raise ValueError('Unsupported candidate classifier geometry')
    context = manifest.get('cropContext')
    if isinstance(context, bool) or not isinstance(context, (int, float)) or not math.isfinite(context) or not 0 <= context <= .16:
        raise ValueError('Invalid candidate crop context')
    if manifest.get('ioContracts') != io_contracts(manifest['resolution'], manifest['classifierResolution']):
        raise ValueError('Candidate manifest IO contract mismatch')
    if set(manifest.get('filesSha256', {})) != {'detector.onnx', 'classifier.onnx'}:
        raise ValueError('Both candidate stage hashes are required')


def verify_candidate_directory(model_dir):
    directory = Path(model_dir)
    manifest = json.loads((directory / 'manifest.json').read_text())
    validate_candidate_manifest(manifest)
    for name, expected in manifest['filesSha256'].items():
        path = model_file(directory, name)
        if file_sha256(path) != expected or path.stat().st_size != manifest.get('filesBytes', {}).get(name):
            raise ValueError('Candidate stage hash or size mismatch: ' + name)
    return directory, manifest


def candidate_detector_tiles(output, image_size, transform, resolution):
    """The same raw single-class NMS/axis geometry as the frozen adapter, at explicit size."""
    import numpy as np
    values = np.asarray(output)
    if values.shape != (1, 5, head_points(resolution)) or values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError('Candidate detector raw head shape/type/values differ from manifest')
    values = values[0].T
    values = values[values[:, 4] > .05]
    if not len(values):
        return []
    if np.any(values[:, 2:4] <= 0) or np.any(values[:, 4] > 1):
        raise ValueError('Invalid candidate detector box or probability')
    boxes = np.empty((len(values), 4), dtype=np.float32)
    boxes[:, :2] = values[:, :2] - values[:, 2:4] / 2
    boxes[:, 2:] = values[:, :2] + values[:, 2:4] / 2
    keep = numpy_nms(boxes, values[:, 4], .6, 300)
    w, h = image_size
    gx, gy, px, py = transform
    boxes = boxes[keep]
    boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - px) / gx).clip(0, w)
    boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - py) / gy).clip(0, h)
    # Invalid crops fail loudly in classifier_tensor; never silently drop a detection.
    return [{'classId': 'tile', 'label': 'tile',
             'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)], 'score': float(score)}
            for (x1, y1, x2, y2), score in zip(boxes, values[keep, 4])]


class CandidateOnnxPredictor:
    def __init__(self, model_dir, threads=2, classifier_batch_size=4):
        import onnxruntime as ort
        if type(threads) is not int or not 1 <= threads <= 16 or type(classifier_batch_size) is not int or not 1 <= classifier_batch_size <= 32:
            raise ValueError('Bound CPU threads1..16 and classifier batch1..32')
        directory, self.manifest = verify_candidate_directory(model_dir)
        self.version = self.manifest['version']
        self.classifier_batch_size = classifier_batch_size
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry('session.intra_op.allow_spinning', '0')
        self.detector = ort.InferenceSession(str(directory / 'detector.onnx'), options, providers=['CPUExecutionProvider'])
        self.classifier = ort.InferenceSession(str(directory / 'classifier.onnx'), options, providers=['CPUExecutionProvider'])
        for name, session in [('detector.onnx', self.detector), ('classifier.onnx', self.classifier)]:
            inputs, outputs = session.get_inputs(), session.get_outputs()
            actual = {'inputs': {x.name: x.shape for x in inputs}, 'outputs': {x.name: x.shape for x in outputs},
                      'float32': all(x.type == 'tensor(float)' for x in inputs + outputs)}
            if actual != self.manifest['ioContracts'][name] or session.get_providers() != ['CPUExecutionProvider']:
                raise ValueError('Actual candidate ONNX IO/provider differs from manifest')

    def predict_with_timings(self, image):
        started = time.perf_counter()
        image = image.convert('RGB')
        tensor, transform = detector_tensor(image, self.manifest['resolution'])
        ready = time.perf_counter()
        output = self.detector.run(['output0'], {'images': tensor})[0]
        detected = time.perf_counter()
        detections = candidate_detector_tiles(output, image.size, transform, self.manifest['resolution'])
        if len(detections) > 200:
            raise ValueError('Detector exceeded the API safety limit')
        post = time.perf_counter()
        values, prep_seconds, infer_seconds = [], 0., 0.
        for first in range(0, len(detections), self.classifier_batch_size):
            start = time.perf_counter()
            part = detections[first:first+self.classifier_batch_size]
            batch = classifier_tensor(image, part, self.manifest['classifierResolution'], self.manifest['cropContext'])
            begin = time.perf_counter()
            logits = self.classifier.run(['logits'], {'images': batch})[0]
            if logits.shape != (len(part), 42):
                raise ValueError('Candidate classifier output dropped detections')
            values.extend(softmax(logits).tolist())
            done = time.perf_counter()
            prep_seconds += begin-start
            infer_seconds += done-begin
        raw = combine_tiles(detections, values, self.manifest['classes'])
        done = time.perf_counter()
        return raw, {'detectorPreprocessMs': (ready-started)*1000, 'detectorForwardMs': (detected-ready)*1000,
                     'detectorPostprocessMs': (post-detected)*1000, 'cropPreprocessMs': prep_seconds*1000,
                     'classifierForwardAndSoftmaxMs': infer_seconds*1000, 'totalMs': (done-started)*1000,
                     'detectedFaces': len(detections), 'classifierBatches': math.ceil(len(detections)/self.classifier_batch_size)}

    def predict_raw(self, image):
        return self.predict_with_timings(image)[0]

    def predict(self, image):
        return [tile for tile in self.predict_raw(image) if tile['score'] > .35]
