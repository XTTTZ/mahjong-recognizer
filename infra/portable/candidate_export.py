"""CPU-only signed candidate ONNX export; no training, cloud calls or deployment."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PureWindowsPath
import shutil
import sys

from ml.inference.manifest import file_sha256
from ml.inference.candidate_onnx_runtime import DETECTORS, head_points, io_contracts, validate_candidate_manifest
from ml.train.provenance import dataset_contract, check_evaluation_data
from ml.train.v6_common import parent_contract


def transported_parent(run_path, checkpoint, contract, terminal_receipt, generic, transport=None):
    """Verify copied original bytes; an explicit transport map preserves Windows terminal identity."""
    run_path, checkpoint = Path(run_path), Path(checkpoint)
    if transport is None:
        return parent_contract(run_path, checkpoint, contract, terminal_receipt, generic=generic)
    run = json.loads(run_path.read_text())
    if run.get('finishedAt'):
        return parent_contract(run_path, checkpoint, contract, terminal_receipt, generic=generic)
    from ml.train.yolo_train import verify_receipt
    mapping = json.loads(Path(transport).read_text())
    records = [r for r in mapping.get('sources', []) if Path(r['localDirectory']).resolve() == run_path.parent.resolve()]
    if mapping.get('schemaVersion') != 1 or len(records) != 1 or terminal_receipt is None:
        raise ValueError('One explicit source transport binding is required for an unfinished copied run')
    source = records[0]; root = run_path.parent.resolve()
    required = {run_path.resolve(), checkpoint.resolve(), Path(str(checkpoint)+'.receipt.json').resolve(), Path(terminal_receipt).resolve()}
    bound = set()
    for relative, expected in source['filesSha256'].items():
        original = root/relative
        path = original.resolve()
        if original.is_symlink() or not path.is_relative_to(root) or file_sha256(path) != expected:
            raise ValueError('Copied source bytes differ from transport binding')
        bound.add(path)
    if not required <= bound:
        raise ValueError('Transport must bind run, checkpoint, checkpoint receipt and terminal receipt')
    check_evaluation_data(run, contract); verify_receipt(checkpoint, run)
    receipt = json.loads(Path(terminal_receipt).read_text())
    normalize = lambda p: PureWindowsPath(p).as_posix().rstrip('/').casefold()
    if (bool(run.get('generic')) != generic or receipt.get('state') not in {'failed','timed_out','interrupted'} or
        receipt.get('child_pid') is not None or not receipt.get('finished_at') or
        normalize(receipt.get('run_dir','')) != normalize(source['originalDirectory'])):
        raise ValueError('Copied run terminal receipt does not bind the explicit original Windows directory')
    evidence = {'runSha256':file_sha256(run_path), 'checkpointSha256':file_sha256(checkpoint),
        'signatureSha256':hashlib.sha256(json.dumps(run['resumeSignature'],sort_keys=True).encode()).hexdigest(),
        'sourceStatus':{'trainingCompletion':'terminal_checkpoint_export', 'terminalState':receipt['state'],
            'terminalReceiptSha256':file_sha256(terminal_receipt), 'transportMapSha256':file_sha256(transport),
            'originalDirectorySha256':hashlib.sha256(normalize(source['originalDirectory']).encode()).hexdigest(),
            'originalBytesPreserved':True}}
    return run, evidence


def source_contract(dataset, detector_run, detector_checkpoint, classifier_run, classifier_checkpoint,
                    detector_terminal_receipt=None, classifier_terminal_receipt=None, source_transport=None):
    """Require identical verified train/valid pixels and signed inactive source runs."""
    contract = dataset_contract(dataset, verify_images=True)
    if contract['ontology'] != 'faces42':
        raise ValueError('Candidate requires the reviewed42-class dataset')
    detector, de = transported_parent(detector_run, detector_checkpoint, contract, detector_terminal_receipt, True, source_transport)
    classifier, ce = transported_parent(classifier_run, classifier_checkpoint, contract, classifier_terminal_receipt, False, source_transport)
    if detector.get('architecture') not in DETECTORS:
        raise ValueError('Unsupported signed candidate detector')
    head_points(detector['resolution'])
    sizes = {'mobilenet-v3-large': (192, 224), 'efficientnet-v2-s': (320,)}
    if classifier.get('config', {}).get('resolution') not in sizes.get(classifier.get('architecture'), ()):
        raise ValueError('Unsupported signed candidate classifier')
    # Reuse the runtime's exact geometry/threshold validation before loading torch.
    preview = base_manifest(detector, classifier, 'candidate-preflight')
    preview['filesSha256'] = {n: '0'*64 for n in ('detector.onnx', 'classifier.onnx')}
    validate_candidate_manifest(preview)
    evidence = {'detector': {k: de[k] for k in ('runSha256', 'checkpointSha256', 'signatureSha256', 'sourceStatus')},
                'classifier': {k: ce[k] for k in ('runSha256', 'checkpointSha256', 'signatureSha256', 'sourceStatus')},
                'data': contract, 'testOpened': False}
    return detector, classifier, evidence


def base_manifest(detector, classifier, version):
    resolution, size = detector['resolution'], classifier['config']['resolution']
    return {'schemaVersion': 1, 'format': 'research-onnx-fp32-cascade', 'researchOnly': True,
            'version': version, 'ontology': 'faces42', 'classes': classifier['classes'],
            'detectorArchitecture': detector['architecture'], 'classifierArchitecture': classifier['architecture'],
            'resolution': resolution, 'classifierResolution': size, 'cropContext': classifier['config']['context'],
            'detectorThreshold': .05, 'threshold': .35, 'acceptThreshold': .7, 'nmsIou': .6,
            'detectorNmsMaxDetections': 300, 'maxDetections': 200, 'maxClassifierBatch': 32, 'opset': 17,
            'precision': 'float32', 'ioContracts': io_contracts(resolution, size)}


def reuse_stage(profile, stage, target, evidence, out):
    """Copy a previously checked research graph only for identical signed source and geometry."""
    if profile is None:
        return None
    from ml.inference.candidate_onnx_runtime import verify_candidate_directory
    directory, source = verify_candidate_directory(profile)
    name = stage+'.onnx'
    keys = ('resolution','detectorArchitecture') if stage=='detector' else ('classifierResolution','classifierArchitecture','cropContext')
    if (source.get('sourceEvidence',{}).get(stage) != evidence[stage] or source['classes'] != target['classes'] or
        any(source[k] != target[k] for k in keys) or source['ioContracts'][name] != target['ioContracts'][name]):
        raise ValueError('Reused stage must bind identical signed source, classes and geometry')
    shutil.copyfile(directory/name, Path(out)/name)
    return {'sourceManifestSha256':file_sha256(directory/'manifest.json'), 'stageSha256':file_sha256(directory/name)}


def export(dataset, detector_run, detector_checkpoint, classifier_run, classifier_checkpoint, out, *,
           detector_terminal_receipt=None, classifier_terminal_receipt=None, source_transport=None,
           reuse_detector=None, reuse_classifier=None, threads=2):
    if type(threads) is not int or not 1 <= threads <= 16:
        raise ValueError('Export CPU thread limit must be1..16')
    out = Path(out)
    if out.exists():
        raise FileExistsError('Use a fresh candidate export directory')
    detector_run, classifier_run, evidence = source_contract(dataset, detector_run, detector_checkpoint,
        classifier_run, classifier_checkpoint, detector_terminal_receipt, classifier_terminal_receipt, source_transport)
    os.environ.setdefault('YOLO_AUTOINSTALL', 'False')
    os.environ.setdefault('YOLO_OFFLINE', 'True')
    import onnx
    import torch
    from ultralytics import YOLO
    from ml.inference.cascade_model import build_classifier
    from ml.train.student_common import build_student
    from infra.portable.audit_public_export import PRIVATE, string_fields, string_fields_json
    torch.set_num_threads(threads)
    identity = hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()[:16]
    manifest = base_manifest(detector_run, classifier_run, 'research-fp32-' + identity)
    out.mkdir(parents=True)
    reused = {stage: reuse_stage(profile,stage,manifest,evidence,out) for stage,profile in
              [('detector',reuse_detector),('classifier',reuse_classifier)]}
    if not reused['classifier']:
        export_classifier(classifier_run, classifier_checkpoint, evidence, out, torch, build_student, build_classifier)
    if not reused['detector']:
        # Export sidecars must never modify the signed source checkpoint directory.
        copied = out / 'detector.pt'
        shutil.copyfile(detector_checkpoint, copied)
        detector = YOLO(str(copied), task='detect')
        if {str(k): v for k, v in detector.names.items()} != {'0': 'tile'}:
            raise ValueError('Actual candidate checkpoint does not contain the single tile head')
        exported = Path(detector.export(format='onnx', imgsz=detector_run['resolution'], batch=1, quantize=32,
                                      dynamic=False, device='cpu', simplify=False, nms=None, opset=17))
        if exported.resolve() != (out / 'detector.onnx').resolve():
            shutil.copyfile(exported, out / 'detector.onnx')
        copied.unlink()
    manifest['reusedStages'] = {k:v for k,v in reused.items() if v}
    scanned = {}
    for name, expected in manifest['ioContracts'].items():
        graph = onnx.load(out / name)
        # Preserve graph/weights; exporter strings can contain private Windows paths.
        del graph.metadata_props[:]
        graph.doc_string = ''
        graph.producer_name = 'mahjong-candidate-export'
        onnx.helper.set_model_props(graph, {'license': 'AGPL-3.0-or-later', 'purpose': 'research candidate; not production validated'})
        if any(x.data_location == onnx.TensorProto.EXTERNAL for x in graph.graph.initializer):
            raise ValueError('Candidate ONNX must be self-contained')
        shape = lambda v: [d.dim_param or d.dim_value for d in v.type.tensor_type.shape.dim]
        actual = {'inputs': {v.name: shape(v) for v in graph.graph.input}, 'outputs': {v.name: shape(v) for v in graph.graph.output},
                  'float32': all(v.type.tensor_type.elem_type == onnx.TensorProto.FLOAT for v in list(graph.graph.input)+list(graph.graph.output))}
        if actual != expected:
            raise ValueError('Export changed raw branch or IO: ' + name)
        findings = [key for key, value in string_fields(graph) if PRIVATE.search(value)]
        if findings:
            raise ValueError('Private text marker in candidate protobuf fields: ' + str(findings))
        scanned[name] = sum(1 for _ in string_fields(graph))
        onnx.checker.check_model(graph)
        onnx.save(graph, out / name)
    files = ('detector.onnx', 'classifier.onnx')
    manifest.update(sourceEvidence=evidence, filesSha256={n: file_sha256(out/n) for n in files},
        filesBytes={n: (out/n).stat().st_size for n in files},
        preprocessing={'detector': 'OpenCV bilinear centered114 letterbox; RGB NCHW float32/255; rounded per-axis inverse gains',
                       'classifier': 'PIL full crop floor/ceil with manifest context; bilinear letterbox114 at manifest resolution; ImageNet normalization',
                       'postprocess': 'single-class raw head; NMS0.6; detector score times42-way softmax; strict fused>0.35; accept0.7'},
        runtime={n: importlib.metadata.version(n) for n in ('torch', 'torchvision', 'ultralytics', 'onnx', 'numpy', 'Pillow')},
        license='AGPL-3.0-or-later', trainingExecuted=False, gpuUsed=False, qualityValidated=False,
        productionPromotionAllowed=False, quantizationUsed=False, exportScriptSha256=file_sha256(__file__),
        privacyAudit={'protobufStringFieldsScanned': scanned, 'privateMarkersFound': 0})
    validate_candidate_manifest(manifest)
    if any(PRIVATE.search(value) for _, value in string_fields_json(manifest)):
        raise ValueError('Private text marker in candidate manifest')
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    return manifest


def export_classifier(classifier_run, classifier_checkpoint, evidence, out, torch, build_student, build_classifier):
    saved = torch.load(classifier_checkpoint, map_location='cpu', weights_only=True)
    if saved.get('signatureSha256') != evidence['classifier']['signatureSha256'] or saved.get('classes') != classifier_run['classes'] or saved.get('config') != classifier_run['config']:
        raise ValueError('Classifier tensors differ from signed source config/classes')
    if classifier_run['architecture'] == 'mobilenet-v3-large' and saved.get('architecture') != 'mobilenet-v3-large':
        raise ValueError('Student checkpoint architecture mismatch')
    classifier = build_student() if classifier_run['architecture'] == 'mobilenet-v3-large' else build_classifier()
    classifier.load_state_dict(saved['model'], strict=True)
    classifier.eval()
    size = classifier_run['config']['resolution']
    torch.onnx.export(classifier, (torch.zeros(1, 3, size, size),), str(out / 'classifier.onnx'),
        input_names=['images'], output_names=['logits'], opset_version=17,
        dynamic_axes={'images': {0: 'batch'}, 'logits': {0: 'batch'}}, dynamo=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'detector-run', 'detector-checkpoint', 'classifier-run', 'classifier-checkpoint', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--detector-terminal-receipt', type=Path)
    parser.add_argument('--classifier-terminal-receipt', type=Path)
    parser.add_argument('--source-transport', type=Path,
                        help='Hash-bound original directory mapping for copied inactive Windows source runs')
    parser.add_argument('--reuse-detector', type=Path)
    parser.add_argument('--reuse-classifier', type=Path)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--preflight-only', action='store_true')
    args = vars(parser.parse_args())
    if args.pop('preflight_only'):
        args.pop('out'); args.pop('threads'); args.pop('reuse_detector'); args.pop('reuse_classifier')
        _, _, evidence = source_contract(**args)
        print(json.dumps({'trainingStarted': False, 'exportStarted': False, 'sourceEvidence': evidence}, indent=2))
    else:
        print(json.dumps(export(**args), indent=2))


if __name__ == '__main__':
    main()
