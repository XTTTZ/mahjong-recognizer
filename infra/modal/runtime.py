"""Pinned inference-only requirements for each reviewed adapter; no Modal import."""


def packages_for(architecture):
    common = ['fastapi==0.118.0', 'pillow==11.3.0', 'numpy==2.2.6']
    if architecture in {'rfdetr-small', 'rfdetr-medium', 'rfdetr-large'}:
        return common + ['rfdetr==1.11.2', 'transformers==5.20.2', 'vernier==0.5.3',
                         'scipy==1.16.2', 'safetensors==0.8.0']
    if architecture == 'rtdetr-v2-r18vd':
        return common + ['transformers==5.20.2', 'scipy==1.16.2', 'safetensors==0.8.0']
    if architecture in {'yolov8s', 'yolo26s', 'yolo26m', 'yolo-efficientnet-v2-s-cascade'}:
        return common + ['ultralytics==8.4.189', 'opencv-python==4.12.0.88']
    raise ValueError('Unsupported inference architecture')
