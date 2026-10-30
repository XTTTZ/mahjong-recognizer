"""Export frozen production weights, CPU only, no training or data input."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys


def export(model_dir,out,threads=2):
    import onnx
    import torch
    from ultralytics import YOLO
    from ml.inference.cascade_model import build_classifier,verify_model_directory
    from ml.inference.manifest import file_sha256
    torch.set_num_threads(threads)
    directory,source=verify_model_directory(model_dir)
    if out.exists():raise FileExistsError('Preserve previous export; choose a fresh directory')
    out.mkdir(parents=True)
    # Export only copies so exporter sidecars never alter the frozen production package.
    shutil.copyfile(directory/'detector.pt',out/'detector.pt')
    detector=YOLO(str(out/'detector.pt'),task='detect');captured={}
    detector.add_callback('on_export_end',lambda exporter:captured.update(json.loads(json.dumps(exporter.metadata))))
    path=Path(detector.export(format='onnx',imgsz=960,batch=1,quantize=32,dynamic=False,
        device='cpu',simplify=False,nms=None,opset=17))
    if path!=out/'detector.onnx':shutil.copyfile(path,out/'detector.onnx')
    (out/'detector.pt').unlink()
    if captured.get('end2end')is not False or captured.get('args',{}).get('nms')is not None:
        raise ValueError('Exporter changed raw YOLO26 output branch')
    saved=torch.load(directory/'classifier.pt',map_location='cpu',weights_only=True)
    if saved['classes']!=source['classes'] or saved['config']['resolution']!=320 or saved['config']['context']!=.08:
        raise ValueError('Classifier checkpoint contract changed')
    classifier=build_classifier();classifier.load_state_dict(saved['model']);classifier.eval()
    torch.onnx.export(classifier,(torch.zeros(1,3,320,320),),str(out/'classifier.onnx'),input_names=['images'],
        output_names=['logits'],opset_version=17,dynamic_axes={'images':{0:'batch'},'logits':{0:'batch'}},dynamo=False)
    contracts={}
    for name in ['detector.onnx','classifier.onnx']:
        model=onnx.load(out/name)
        # Exporter descriptions contain private training-machine dataset paths.
        # Preserve attribution and portable contracts, sanitize before hashing.
        for prop in model.metadata_props:
            if prop.key=='description':prop.value='Mahjong single-class tile detector; frozen V4 production weights'
        onnx.save(model,out/name)
        onnx.checker.check_model(model)
        if any(v.data_location==onnx.TensorProto.EXTERNAL for v in model.graph.initializer):raise ValueError('Portable model must be self contained')
        def shape(value):return [d.dim_param if d.dim_param else d.dim_value for d in value.type.tensor_type.shape.dim]
        contracts[name]={'inputs':{v.name:shape(v) for v in model.graph.input},'outputs':{v.name:shape(v) for v in model.graph.output},'float32':all(v.type.tensor_type.elem_type==onnx.TensorProto.FLOAT for v in list(model.graph.input)+list(model.graph.output))}
    if contracts!={'detector.onnx':{'inputs':{'images':[1,3,960,960]},'outputs':{'output0':[1,5,18900]},'float32':True},
                  'classifier.onnx':{'inputs':{'images':['batch',3,320,320]},'outputs':{'logits':['batch',42]},'float32':True}}:
        raise ValueError('Unexpected portable shapes: '+str(contracts))
    manifest={'schemaVersion':1,'format':'onnx-fp32-cascade','version':source['version']+'-onnx-fp32-op17',
        'sourceModelVersion':source['version'],'sourceModelManifestSha256':file_sha256(directory/'model.json'),
        'sourceModelFilesSha256':source['filesSha256'],'classes':source['classes'],'ontology':'faces42','resolution':960,
        'classifierResolution':320,'cropContext':.08,'detectorThreshold':.05,'threshold':.35,'acceptThreshold':.7,
        'nmsIou':.6,'detectorNmsMaxDetections':300,'maxDetections':200,'maxClassifierBatch':32,'opset':17,
        'precision':'float32','ioContracts':contracts,'detectorMetadata':captured,
        'preprocessing':{'detector':'RGB uint8->OpenCV INTER_LINEAR aspect preserving resize/center 114 pad to960; NCHW float32/255',
            'classifier':'PIL floor/ceil crop with8% context; complete aspect-preserving bilinear320 letterbox114; RGB NCHW float32; mean .485,.456,.406 std .229,.224,.225',
            'detectorOutput':'raw cx,cy,w,h in960 input pixels +one tile probability; no objectness/sigmoid; external singleclass NMS.6',
            'classificationOutput':'42 logits ->float32 stable softmax; fused=detector*top1Probability; strict fused>.35; every retainedtile>=.7 for acceptance'},
        'filesSha256':{n:file_sha256(out/n) for n in contracts},'filesBytes':{n:(out/n).stat().st_size for n in contracts},
        'license':source['license'],'runtime':{n:importlib.metadata.version(n) for n in ['torch','torchvision','ultralytics','onnx','onnxruntime','numpy','opencv-python','Pillow']},
        'trainingExecuted':False,'gpuUsed':False,'calibrationUsed':False,'qualityValidated':False,
        'exportScriptSha256':file_sha256(__file__)}
    # Upstream metadata can carry source checkpoint paths; public artifact keeps only portable inference fields.
    manifest['detectorMetadata']={k:v for k,v in captured.items() if k in ['names','task','batch','imgsz','channels','end2end','stride','args']}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return manifest

if __name__=='__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]));os.environ.setdefault('YOLO_AUTOINSTALL','False');os.environ.setdefault('YOLO_OFFLINE','True')
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--threads',type=int,default=2);a=p.parse_args()
    print(json.dumps(export(a.model_dir,a.out,a.threads),ensure_ascii=False,indent=2))
