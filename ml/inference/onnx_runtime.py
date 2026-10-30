"""Portable FP32 cascade. CPU ORT only; no torch dependency at inference time."""
import json
import math
from pathlib import Path
import time
from ml.inference.manifest import file_sha256,model_file,validate_labels
from ml.inference.cascade_model import crop_face,combine_tiles


def verify_portable_directory(model_dir):
    directory=Path(model_dir)
    manifest=json.loads((directory/'manifest.json').read_text())
    validate_labels(manifest)
    if (manifest.get('schemaVersion')!=1 or manifest.get('format')!='onnx-fp32-cascade' or
        manifest.get('resolution')!=960 or manifest.get('classifierResolution')!=320 or
        manifest.get('cropContext')!=.08 or manifest.get('detectorThreshold')!=.05 or
        manifest.get('threshold')!=.35 or manifest.get('acceptThreshold')!=.7 or
        manifest.get('nmsIou')!=.6 or manifest.get('maxDetections')!=200 or
        manifest.get('detectorNmsMaxDetections')!=300 or len(manifest['classes'])!=42 or
        manifest.get('maxClassifierBatch')!=32 or manifest.get('opset')!=17):
        raise ValueError('Portable model contract differs from reviewed production cascade')
    if set(manifest.get('filesSha256',{}))!={'detector.onnx','classifier.onnx'}:
        raise ValueError('Both portable stage hashes required')
    for name,digest in manifest['filesSha256'].items():
        if file_sha256(model_file(directory,name))!=digest:
            raise ValueError('Portable stage SHA mismatch: '+name)
    return directory,manifest


def detector_tensor(image,resolution=960):
    import cv2
    import numpy as np
    # Ultralytics rect=False LetterBox: OpenCV bilinear, centered 114 padding.
    source=np.asarray(image.convert('RGB'))
    h,w=source.shape[:2]
    gain=min(resolution/h,resolution/w)
    nw,nh=round(w*gain),round(h*gain)
    dw,dh=(resolution-nw)/2,(resolution-nh)/2
    left,top=round(dw-.1),round(dh-.1)
    right,bottom=round(dw+.1),round(dh+.1)
    if (nw,nh)!=(w,h):source=cv2.resize(source,(nw,nh),interpolation=cv2.INTER_LINEAR)
    source=cv2.copyMakeBorder(source,top,bottom,left,right,cv2.BORDER_CONSTANT,value=(114,114,114))
    # Pinned Ultralytics8.4.189 scale_boxes reverses each rounded resize axis,
    # not the pre-rounding scalar gain; this also controls exact PIL crop boundaries.
    return np.ascontiguousarray(source.transpose(2,0,1)[None],dtype=np.float32)/np.float32(255.),(nw/w,nh/h,left,top)


def numpy_nms(boxes,scores,iou_threshold=.6,max_detections=300):
    """Single-class Ultralytics-style confidence sorting, IoU suppression."""
    import numpy as np
    order=np.argsort(-scores,kind='stable');keep=[]
    area=(boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    while len(order) and len(keep)<max_detections:
        i=int(order[0]);keep.append(i);rest=order[1:]
        if not len(rest):break
        x1=np.maximum(boxes[i,0],boxes[rest,0]);y1=np.maximum(boxes[i,1],boxes[rest,1])
        x2=np.minimum(boxes[i,2],boxes[rest,2]);y2=np.minimum(boxes[i,3],boxes[rest,3])
        inter=np.maximum(x2-x1,0)*np.maximum(y2-y1,0)
        overlap=inter/(area[i]+area[rest]-inter)
        order=rest[overlap<=iou_threshold]
    return keep


def detector_tiles(output,image_size,transform,threshold=.05,iou_threshold=.6):
    import numpy as np
    values=np.asarray(output)
    if values.shape!=(1,5,18900) or values.dtype!=np.float32 or not np.isfinite(values).all():
        raise ValueError('Detector must return finite FP32 [1,5,18900] raw head')
    values=values[0].T
    # One generic class probability, no extra objectness product or sigmoid.
    values=values[values[:,4]>threshold]
    if not len(values):return []
    boxes=np.empty((len(values),4),dtype=np.float32)
    boxes[:,:2]=values[:,:2]-values[:,2:4]/2
    boxes[:,2:]=values[:,:2]+values[:,2:4]/2
    keep=numpy_nms(boxes,values[:,4],iou_threshold,300)
    w,h=image_size;gain_x,gain_y,padx,pady=transform
    boxes=boxes[keep];scores=values[keep,4]
    boxes[:,[0,2]]=(boxes[:,[0,2]]-padx)/gain_x
    boxes[:,[1,3]]=(boxes[:,[1,3]]-pady)/gain_y
    boxes[:,[0,2]]=boxes[:,[0,2]].clip(0,w)
    boxes[:,[1,3]]=boxes[:,[1,3]].clip(0,h)
    return [{'classId':'tile','label':'tile','bbox':[float(x1),float(y1),float(x2-x1),float(y2-y1)],'score':float(score)}
            for (x1,y1,x2,y2),score in zip(boxes,scores)]


def classifier_tensor(image,detections,resolution=320,context=.08):
    import numpy as np
    values=np.stack([np.asarray(crop_face(image,d['bbox'],resolution,context),dtype=np.float32) for d in detections])
    values=values.transpose(0,3,1,2)/np.float32(255.)
    return np.ascontiguousarray((values-np.asarray([.485,.456,.406],dtype=np.float32)[None,:,None,None])/
        np.asarray([.229,.224,.225],dtype=np.float32)[None,:,None,None])


def softmax(values):
    import numpy as np
    values=np.asarray(values)
    if values.ndim!=2 or values.shape[1]!=42 or not np.isfinite(values).all():
        raise ValueError('Classifier must return finite [batch,42] logits')
    values=values-values.max(axis=1,keepdims=True);values=np.exp(values)
    return values/values.sum(axis=1,keepdims=True)


class OnnxCascadePredictor:
    def __init__(self,model_dir,threads=2,classifier_batch_size=32):
        import onnxruntime as ort
        if type(threads)is not int or not 1<=threads<=16 or type(classifier_batch_size)is not int or not 1<=classifier_batch_size<=32:
            raise ValueError('Bound CPU threads1..16 and classifier batch1..32')
        directory,self.manifest=verify_portable_directory(model_dir)
        self.version=self.manifest['version'];self.classifier_batch_size=classifier_batch_size
        options=ort.SessionOptions();options.intra_op_num_threads=threads;options.inter_op_num_threads=1
        options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry('session.intra_op.allow_spinning','0')
        self.detector=ort.InferenceSession(str(directory/'detector.onnx'),options,providers=['CPUExecutionProvider'])
        self.classifier=ort.InferenceSession(str(directory/'classifier.onnx'),options,providers=['CPUExecutionProvider'])
        for session,output,shape in [(self.detector,'output0',[1,5,18900]),(self.classifier,'logits',['batch',42])]:
            ins,outs=session.get_inputs(),session.get_outputs()
            size=960 if output=='output0' else 320
            if (len(ins)!=1 or ins[0].name!='images' or ins[0].type!='tensor(float)' or
                ins[0].shape!=([1,3,size,size] if output=='output0' else ['batch',3,size,size]) or
                len(outs)!=1 or outs[0].name!=output or outs[0].type!='tensor(float)' or outs[0].shape!=shape or
                session.get_providers()!=['CPUExecutionProvider']):
                raise ValueError('Actual ONNX IO/provider differs from manifest')
    def predict_with_timings(self,image):
        started=time.perf_counter();image=image.convert('RGB');tensor,transform=detector_tensor(image)
        prepared=time.perf_counter();output=self.detector.run(['output0'],{'images':tensor})[0]
        detected=time.perf_counter();detections=detector_tiles(output,image.size,transform)
        if len(detections)>200:raise ValueError('Detector exceeded the API safety limit')
        postprocessed=time.perf_counter();probabilities=[];prep_seconds=0;classifier_seconds=0
        for first in range(0,len(detections),self.classifier_batch_size):
            t=time.perf_counter();part=detections[first:first+self.classifier_batch_size];batch=classifier_tensor(image,part)
            ready=time.perf_counter();logits=self.classifier.run(['logits'],{'images':batch})[0]
            if logits.shape!=(len(part),42):raise ValueError('Classifier batch output dropped detections')
            probabilities.extend(softmax(logits).tolist());after=time.perf_counter();prep_seconds+=ready-t;classifier_seconds+=after-ready
        raw=combine_tiles(detections,probabilities,self.manifest['classes']);finished=time.perf_counter()
        return raw,{'detectorPreprocessMs':(prepared-started)*1000,'detectorForwardMs':(detected-prepared)*1000,
            'detectorPostprocessMs':(postprocessed-detected)*1000,'cropPreprocessMs':prep_seconds*1000,
            'classifierForwardAndSoftmaxMs':classifier_seconds*1000,'totalMs':(finished-started)*1000,
            'detectedFaces':len(detections),'classifierBatches':math.ceil(len(detections)/self.classifier_batch_size)}
    def predict_raw(self,image):return self.predict_with_timings(image)[0]
    def predict(self,image):return [tile for tile in self.predict_raw(image) if tile['score']>self.manifest['threshold']]
