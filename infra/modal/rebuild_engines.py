"""Rebuild engines from an existing frozen image, preserving weights and packages.

Only an explicit invocation of this tool starts GPU compilation. This does not
publish a service or run training/validation data through any model.
"""
import argparse
import json
import os
from pathlib import Path
import sys


def runtime_receipt():
    return json.loads(Path('/engines/runtime.json').read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-image-id',required=True)
    parser.add_argument('--gpu',choices=['L4','A10'],required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists(): raise FileExistsError('Choose a fresh build receipt')
    import modal
    from ml.inference.tensorrt_runtime import build_engines
    root=Path(__file__).resolve().parents[2]
    app=modal.App('mahjong-compatible-engines-build')
    image=(modal.Image.from_id(args.base_image_id)
        .add_local_file(root/'ml/inference/tensorrt_runtime.py', '/app/ml/inference/tensorrt_runtime.py',copy=True)
        .run_function(build_engines,gpu=args.gpu,cpu=2,memory=8192,timeout=1800))
    from infra.modal.rebuild_engines import runtime_receipt as remote_receipt
    image=image.add_local_file(Path(__file__),'/app/infra/modal/rebuild_engines.py',copy=True)
    reader=app.function(image=image,cpu=.125,memory=512,timeout=60)(remote_receipt)
    with modal.enable_output(),app.run():
        receipt=reader.remote()
        result={'baseImageId':args.base_image_id,'imageId':image.object_id,'gpu':args.gpu,
                'runtime':receipt,'serviceDeployed':False,'trainingStarted':False}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    main()
