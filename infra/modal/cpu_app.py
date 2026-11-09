"""Separate scale-to-zero CPU endpoint for a frozen research ONNX profile.

MAHJONG_CPU_MODEL_DIR=/path/to/profile python -m modal deploy infra/modal/cpu_app.py
Never imports or changes the production GPU app. No credentials enter the image.
"""
import json
import os
from pathlib import Path
import sys
import modal


def cpu_settings(env):
    cores=int(env.get('MAHJONG_CPU_CORES','1'))
    memory=int(env.get('MAHJONG_CPU_MEMORY_MB','2048'))
    batch=int(env.get('MAHJONG_CPU_BATCH','4'))
    if cores not in (1,2) or memory not in (2048,4096) or batch not in (1,4,8):
        raise ValueError('CPU profile requires1/2 physical cores,2048/4096MiB and batch1/4/8')
    snapshot=env.get('MAHJONG_CPU_SNAPSHOT','1')
    if snapshot not in ('0','1'):raise ValueError('Snapshot flag must be0/1')
    return {'cores':cores,'memory':memory,'batch':batch,'snapshot':snapshot=='1'}


SETTINGS=cpu_settings(os.environ)
app=modal.App(os.environ.get('MAHJONG_CPU_MODAL_APP','mahjong-cpu-inference'))
image=None
if modal.is_local():
    ROOT=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(ROOT))
    from ml.inference.candidate_onnx_runtime import verify_candidate_directory
    from ml.inference.manifest import file_sha256
    from infra.modal.package import prepare_upload,prepare_source
    model_dir=os.environ.get('MAHJONG_CPU_MODEL_DIR')
    if not model_dir:raise RuntimeError('Set MAHJONG_CPU_MODEL_DIR to a frozen ONNX candidate directory')
    directory,manifest=verify_candidate_directory(model_dir)
    if any(p.name not in {'manifest.json','detector.onnx','classifier.onnx','NOTICE','LICENSE','SHA256SUMS'} for p in directory.iterdir()):
        raise ValueError('CPU upload directory may contain only the selected portable model and attribution files')
    source=prepare_source(ROOT/'ml/inference',ROOT/'artifacts/models/modal-cpu-inference-source')
    expected={'version':manifest['version'],'manifestSha256':file_sha256(directory/'manifest.json')}
    image=(modal.Image.debian_slim(python_version='3.11')
        .pip_install('onnxruntime==1.23.2','numpy==1.26.4','opencv-python-headless==4.11.0.86','Pillow==11.3.0','fastapi==0.118.0'))
    public_base=os.environ.get('MAHJONG_CPU_PUBLIC_BASE_URL')
    if public_base:
        from infra.portable.fetch_public_model import validate_spec
        public_files={name:{'sha256':file_sha256(directory/name),'bytes':(directory/name).stat().st_size}
                      for name in ('manifest.json','detector.onnx','classifier.onnx')}
        public_base,public_files=validate_spec(public_base,public_files)
        image=(image.add_local_file(ROOT/'infra/portable/fetch_public_model.py',remote_path='/build/fetch_public_model.py',copy=True)
            .env({'MAHJONG_CPU_PUBLIC_MODEL_SPEC':json.dumps({'baseUrl':public_base,'files':public_files},sort_keys=True)})
            .run_commands('python /build/fetch_public_model.py --out /model'))
    else:
        upload=prepare_upload(directory,ROOT/'artifacts/models/modal-cpu-upload')
        image=(image.add_local_dir(upload,remote_path='/model-upload',copy=True)
            .run_commands('python /model-upload/assemble.py && rm -rf /model-upload'))
    image=(image.add_local_dir(source,remote_path='/app/ml/inference',copy=True)
        .env({'PYTHONPATH':'/app','MAHJONG_CPU_CORES':str(SETTINGS['cores']),
              'MAHJONG_CPU_MEMORY_MB':str(SETTINGS['memory']),'MAHJONG_CPU_BATCH':str(SETTINGS['batch']),
              'MAHJONG_CPU_SNAPSHOT':'1'if SETTINGS['snapshot']else'0',
              'OMP_NUM_THREADS':str(SETTINGS['cores']),'OPENBLAS_NUM_THREADS':str(SETTINGS['cores']),
              'MAHJONG_CPU_EXPECTED_MODEL':json.dumps(expected,sort_keys=True)}))


@app.cls(image=image,cpu=(SETTINGS['cores'],SETTINGS['cores']),memory=SETTINGS['memory'],
         min_containers=0,max_containers=1,scaledown_window=10,timeout=90,
         enable_memory_snapshot=SETTINGS['snapshot'])
class CpuRecognizer:
    @modal.enter(snap=SETTINGS['snapshot'])
    def load(self):
        sys.path.insert(0,'/app')
        from ml.inference.candidate_onnx_runtime import CandidateOnnxPredictor
        from ml.inference.manifest import file_sha256
        expected=json.loads(os.environ['MAHJONG_CPU_EXPECTED_MODEL'])
        self.predictor=CandidateOnnxPredictor('/model',SETTINGS['cores'],SETTINGS['batch'])
        if self.predictor.manifest['version']!=expected['version'] or file_sha256('/model/manifest.json')!=expected['manifestSha256']:
            raise ValueError('CPU image differs from the selected model manifest')
        self.predictor.runtime_backend='onnxruntime-cpu'
        # Capture both detector and dynamic classifier kernel initialization in the CPU snapshot.
        from PIL import Image
        import numpy as np
        self.predictor.predict(Image.new('RGB',(960,640),(114,114,114)))
        size=self.predictor.manifest['classifierResolution']
        for count in sorted({1,SETTINGS['batch']}):
            self.predictor.classifier.run(['logits'],{'images':np.zeros((count,3,size,size),dtype=np.float32)})

    @modal.asgi_app(requires_proxy_auth=True)
    def api(self):
        from ml.inference.service import create_app
        return create_app(self.predictor,self.predictor.manifest['acceptThreshold'])
