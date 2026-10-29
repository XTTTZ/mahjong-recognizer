"""Real AMPERE_PLUS FP16 engines, preserving original PT and YOLO output semantics.

A10G80SM and A1072SM need hardware-compatible tactics, not a name alias or a
suppressed warning. APIs verified against NVIDIA/TensorRT release/10.13 Python
bindings and Ultralytics8.4.189 exporter/engine source. No calibration is used.
"""
from pathlib import Path
import hashlib
import json

HARDWARE_COMPATIBILITY = 'AMPERE_PLUS'
ENGINE_FILES = {'detector.engine', 'classifier.engine'}


def _validate_hardware(value):
    cc = value.get('computeCapability')
    if (not isinstance(value.get('gpu'), str) or not value['gpu'] or
        type(value.get('smCount')) is not int or value['smCount'] <= 0 or
        not isinstance(cc, list) or len(cc) != 2 or any(type(v) is not int or v < 0 for v in cc) or cc[0] < 8):
        raise ValueError('AMPERE_PLUS requires GPU, positive SM count and compute capability>=8.0')


def _hardware(torch):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for TensorRT engines')
    device = torch.cuda.current_device()
    value = {'gpu': torch.cuda.get_device_name(device),
             'smCount': int(torch.cuda.get_device_properties(device).multi_processor_count),
             'computeCapability': list(torch.cuda.get_device_capability(device))}
    _validate_hardware(value)
    return value


def _metadata_bytes(metadata):
    return json.dumps(metadata, ensure_ascii=True, sort_keys=True, separators=(',', ':')).encode('utf-8')


def write_detector_engine(path, metadata, plan):
    """Ultralytics header: int32 little-endian JSON byte length + JSON + plan."""
    encoded = _metadata_bytes(metadata)
    if not 0 < len(encoded) <= 1024**2 or not plan:
        raise ValueError('Invalid detector metadata/plan')
    Path(path).write_bytes(len(encoded).to_bytes(4, 'little', signed=True) + encoded + bytes(plan))


def read_detector_engine(path):
    raw = Path(path).read_bytes()
    if len(raw) < 5:
        raise ValueError('Missing Ultralytics engine header')
    length = int.from_bytes(raw[:4], 'little', signed=True)
    if not 0 < length <= 1024**2 or 4 + length >= len(raw):
        raise ValueError('Invalid Ultralytics engine header length')
    metadata = json.loads(raw[4:4 + length])
    if not isinstance(metadata, dict):
        raise ValueError('Detector metadata must be an object')
    return metadata, raw[4 + length:]


def _validate_detector_metadata(metadata, manifest):
    names = metadata.get('names')
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names, key=lambda k: int(k))]
    if (names != ['tile'] or metadata.get('task') != 'detect' or metadata.get('batch') != 1 or
        metadata.get('imgsz') != [manifest['resolution'], manifest['resolution']] or
        metadata.get('channels', 3) != 3 or type(metadata.get('end2end')) is not bool):
        raise ValueError('Detector metadata differs from frozen generic YOLO geometry/head')
    if metadata.get('args', {}).get('nms') is not None:
        raise ValueError('Preserve original nms=None; do not switch the YOLO26 output branch')


def _dtype_name(dtype, trt):
    if dtype == trt.float32:
        return 'float32'
    if dtype == trt.float16:
        return 'float16'
    raise ValueError('Only floating32/16 TensorRT IO is supported')


def engine_contract(engine, trt, kind, resolution, metadata=None):
    if engine.hardware_compatibility_level != trt.HardwareCompatibilityLevel.AMPERE_PLUS:
        raise ValueError('Serialized engine is not actually AMPERE_PLUS compatible')
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    expected = {'images', 'logits' if kind == 'classifier' else 'output0'}
    if set(names) != expected or len(names) != 2:
        raise ValueError('Unexpected TensorRT IO tensors')
    result = {}
    for name in names:
        if (engine.get_tensor_location(name) != trt.TensorLocation.DEVICE or
            engine.get_tensor_format(name) != trt.TensorFormat.LINEAR):
            raise ValueError('TensorRT IO must be contiguous device tensors')
        result[name] = {'shape': list(engine.get_tensor_shape(name)),
            'dtype': _dtype_name(engine.get_tensor_dtype(name), trt),
            'mode': 'input' if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else 'output'}
    if result['images'] != {'shape': [-1 if kind == 'classifier' else 1, 3, resolution, resolution],
                            'dtype': 'float32', 'mode': 'input'}:
        raise ValueError('FP32 NCHW input differs from original ONNX geometry')
    output_name = 'logits' if kind == 'classifier' else 'output0'
    shape = result[output_name]['shape']
    if result[output_name]['mode'] != 'output':
        raise ValueError('TensorRT output mode mismatch')
    if kind == 'classifier':
        profile = [list(s) for s in engine.get_tensor_profile_shape('images', 0)]
        if shape != [-1, 42] or profile != [[1, 3, resolution, resolution], [14, 3, resolution, resolution], [32, 3, resolution, resolution]]:
            raise ValueError('Classifier batch/class dimension differs from reviewed profile')
        result['profile'] = profile
    else:
        # Preserve the head chosen by Exporter; do not reshape/NMS it ourselves.
        end2end = metadata['end2end']
        if (len(shape) != 3 or shape[0] != 1 or any(v <= 0 for v in shape) or
            (shape[-1] != 6 if end2end else shape[1] != 5)):
            raise ValueError('Detector output differs from original generic YOLO head')
    return result


def _build_plan(onnx_path, trt, logger, kind, resolution):
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(Path(onnx_path).read_bytes()):
        raise RuntimeError('ONNX parse failed: ' + str([str(parser.get_error(i)) for i in range(parser.num_errors)]))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * 1024**3)
    config.set_flag(trt.BuilderFlag.FP16)
    config.hardware_compatibility_level = trt.HardwareCompatibilityLevel.AMPERE_PLUS
    if config.hardware_compatibility_level != trt.HardwareCompatibilityLevel.AMPERE_PLUS:
        raise RuntimeError('Builder rejected AMPERE_PLUS compatibility')
    if kind == 'classifier':
        profile = builder.create_optimization_profile()
        profile.set_shape('images', (1, 3, resolution, resolution), (14, 3, resolution, resolution), (32, 3, resolution, resolution))
        if config.add_optimization_profile(profile) < 0:
            raise RuntimeError('Classifier optimization profile rejected')
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(kind + ' compatible TensorRT build failed')
    return bytes(plan)


def build_engines(model_dir='/model', output_dir='/engines'):
    import torch
    import tensorrt as trt
    from ultralytics import YOLO
    from ml.inference.cascade_model import build_classifier, verify_model_directory
    from ml.inference.manifest import file_sha256
    hardware = _hardware(torch)
    if not str(trt.__version__).startswith('10.13.'):
        raise RuntimeError('Use pinned TensorRT10.13')
    directory, manifest = verify_model_directory(model_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # An interrupted rebuild must not leave an old receipt blessing new files.
    (output / 'runtime.json').unlink(missing_ok=True)
    detector = YOLO(str(directory / 'detector.pt'), task='detect')
    captured = {}
    def exported(exporter):
        captured.update(json.loads(json.dumps(exporter.metadata)))
    detector.add_callback('on_export_end', exported)
    # Old format=engine used FP32 ONNX then FP16 TRT kernels. ONNX quantize=16
    # would half the graph+IO. nms=False would switch YOLO26's end2end branch.
    detector_onnx = Path(detector.export(format='onnx', imgsz=manifest['resolution'], batch=1,
        quantize=32, dynamic=False, device=0, simplify=False, nms=None))
    _validate_detector_metadata(captured, manifest)
    classifier = build_classifier()
    saved = torch.load(directory / 'classifier.pt', map_location='cpu', weights_only=True)
    if (saved['classes'] != manifest['classes'] or saved['config']['resolution'] != manifest['classifierResolution'] or
        saved['config']['context'] != manifest['cropContext']):
        raise ValueError('Classifier classes/geometry differ from manifest')
    classifier.load_state_dict(saved['model'])
    classifier.eval().cuda()
    resolution = manifest['classifierResolution']
    classifier_onnx = output / 'classifier.onnx'
    torch.onnx.export(classifier, (torch.zeros(1, 3, resolution, resolution, device='cuda'),), str(classifier_onnx),
        input_names=['images'], output_names=['logits'], opset_version=17,
        dynamic_axes={'images': {0: 'batch'}, 'logits': {0: 'batch'}}, dynamo=False)
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    contracts = {}
    for kind, path, size in [('detector', detector_onnx, manifest['resolution']), ('classifier', classifier_onnx, resolution)]:
        plan = _build_plan(path, trt, logger, kind, size)
        engine = runtime.deserialize_cuda_engine(plan)
        if engine is None:
            raise RuntimeError('Compatible engine verification could not deserialize ' + kind)
        contracts[kind + '.engine'] = engine_contract(engine, trt, kind, size, captured if kind == 'detector' else None)
        if kind == 'detector':
            write_detector_engine(output / 'detector.engine', captured, plan)
        else:
            (output / 'classifier.engine').write_bytes(plan)
        del engine
    receipt = {'schemaVersion': 2, 'precision': 'fp16', 'hardwareCompatibility': HARDWARE_COMPATIBILITY,
        'engineHardwareCompatibility': {name: HARDWARE_COMPATIBILITY for name in sorted(ENGINE_FILES)},
        'modelVersion': manifest['version'], 'modelFilesSha256': manifest['filesSha256'], **hardware,
        'torch': torch.__version__, 'cuda': torch.version.cuda, 'tensorrt': trt.__version__, 'maxBatch': 32, 'optimalBatch': 14,
        'calibrationDataUsed': False, 'ioContracts': contracts,
        'detectorMetadataSha256': hashlib.sha256(_metadata_bytes(captured)).hexdigest(),
        'onnxSha256': {'detector': file_sha256(detector_onnx), 'classifier': file_sha256(classifier_onnx)},
        'buildCodeSha256': file_sha256(__file__),
        'enginesSha256': {name: file_sha256(output / name) for name in sorted(ENGINE_FILES)}}
    temporary = output / 'runtime.json.tmp'
    temporary.write_text(json.dumps(receipt, indent=2) + '\n')
    temporary.replace(output / 'runtime.json')
    classifier_onnx.unlink()
    print(json.dumps(receipt))


def validate_runtime_receipt(receipt, manifest, hardware, versions):
    _validate_hardware(receipt)
    _validate_hardware(hardware)
    if (receipt.get('schemaVersion') != 2 or receipt.get('hardwareCompatibility') != HARDWARE_COMPATIBILITY or
        receipt.get('engineHardwareCompatibility') != {name: HARDWARE_COMPATIBILITY for name in ENGINE_FILES}):
        raise ValueError('Both engines must have explicit AMPERE_PLUS build provenance')
    if (receipt.get('modelVersion') != manifest['version'] or receipt.get('modelFilesSha256') != manifest['filesSha256'] or
        receipt.get('precision') != 'fp16' or receipt.get('calibrationDataUsed') is not False or
        receipt.get('maxBatch') != 32 or receipt.get('optimalBatch') != 14):
        raise ValueError('Engine model/precision/profile receipt mismatch')
    if set(receipt.get('enginesSha256', {})) != ENGINE_FILES or set(receipt.get('ioContracts', {})) != ENGINE_FILES:
        raise ValueError('Both engine hashes and IO contracts are mandatory')
    for field in ('torch', 'cuda', 'tensorrt'):
        if receipt.get(field) != versions[field]:
            raise ValueError('Engine runtime version mismatch: ' + field)
    if not str(versions['tensorrt']).startswith('10.13.'):
        raise ValueError('Use pinned TensorRT10.13')
    # Name/SM equality is not an alias rule: actual engine AMPERE_PLUS property
    # is checked before either context runs. NVIDIA compatibility allows SM/CC changes.


class TensorRTClassifier:
    def __init__(self, path, resolution=320, expected_contract=None):
        import tensorrt as trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:
            raise RuntimeError('TensorRT engine deserialization failed')
        self.contract = engine_contract(self.engine, trt, 'classifier', resolution)
        if expected_contract is not None and self.contract != expected_contract:
            raise ValueError('Classifier engine IO differs from receipt')
        self.context = self.engine.create_execution_context()
        self.resolution = resolution
        if self.context is None:
            raise RuntimeError('TensorRT execution context failed')

    def __call__(self, batch):
        import torch
        import tensorrt as trt
        if (not batch.is_cuda or batch.dtype != torch.float32 or batch.ndim != 4 or
            not 1 <= batch.shape[0] <= 32 or tuple(batch.shape[1:]) != (3, self.resolution, self.resolution)):
            raise ValueError('TensorRT classifier requires CUDA FP32 NCHW within batch1..32')
        batch = batch.contiguous()
        if not self.context.set_input_shape('images', tuple(batch.shape)):
            raise ValueError('TensorRT input shape rejected')
        shape = tuple(self.context.get_tensor_shape('logits'))
        if shape != (batch.shape[0], 42):
            raise ValueError('TensorRT output must cover every tile with42 logits')
        dtype = self.engine.get_tensor_dtype('logits')
        _dtype_name(dtype, trt)
        output = torch.empty(shape, device=batch.device, dtype=torch.float16 if dtype == trt.float16 else torch.float32)
        if not self.context.set_tensor_address('images', batch.data_ptr()) or not self.context.set_tensor_address('logits', output.data_ptr()):
            raise RuntimeError('TensorRT IO address rejected')
        if not self.context.execute_async_v3(stream_handle=torch.cuda.current_stream(batch.device).cuda_stream):
            raise RuntimeError('TensorRT execution failed')
        return output


def activate_tensorrt(predictor, engine_dir='/engines'):
    import torch
    import tensorrt as trt
    from ultralytics import YOLO
    from ml.inference.manifest import file_sha256
    root = Path(engine_dir)
    receipt = json.loads((root / 'runtime.json').read_text())
    hardware = _hardware(torch)
    validate_runtime_receipt(receipt, predictor.manifest, hardware,
        {'torch': torch.__version__, 'cuda': torch.version.cuda, 'tensorrt': trt.__version__})
    for name in sorted(ENGINE_FILES):
        if file_sha256(root / name) != receipt['enginesSha256'][name]:
            raise ValueError('TensorRT engine SHA mismatch: ' + name)
    metadata, plan = read_detector_engine(root / 'detector.engine')
    _validate_detector_metadata(metadata, predictor.manifest)
    if hashlib.sha256(_metadata_bytes(metadata)).hexdigest() != receipt['detectorMetadataSha256']:
        raise ValueError('Detector metadata differs from receipt')
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    if engine is None:
        raise RuntimeError('Detector engine deserialization failed')
    if engine_contract(engine, trt, 'detector', predictor.manifest['resolution'], metadata) != receipt['ioContracts']['detector.engine']:
        raise ValueError('Detector engine IO differs from receipt')
    del engine
    classifier = TensorRTClassifier(root / 'classifier.engine', predictor.manifest['classifierResolution'], receipt['ioContracts']['classifier.engine'])
    model = YOLO(str(root / 'detector.engine'), task='detect')
    predictor.detector.model = model
    predictor.classifier = classifier
    predictor.runtime_backend = 'tensorrt-fp16'
    torch.cuda.empty_cache()
    return dict(receipt, runtimeHardware=hardware, hardwareCompatibilityVerified=True)
