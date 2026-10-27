"""Architecture selection for the shared HTTP service; only reviewed adapters."""
import json
from pathlib import Path


def inspect_model(model_dir):
    path = Path(model_dir)
    architecture = json.loads((path / "model.json").read_text(encoding="utf-8"))["architecture"]
    if architecture.startswith("rfdetr-"):
        from ml.inference.rfdetr_model import verify_model_directory
    elif architecture == "rtdetr-v2-r18vd":
        from ml.inference.rtdetr_model import verify_model_directory
    elif architecture in {"yolov8s", "yolo26s", "yolo26m"}:
        from ml.inference.yolo_model import verify_model_directory
    elif architecture == "yolo-efficientnet-v2-s-cascade":
        from ml.inference.cascade_model import verify_model_directory
    else:
        raise ValueError("Unsupported model architecture")
    return verify_model_directory(path)[1]


def load_predictor(model_dir, device="cuda"):
    manifest = inspect_model(model_dir)
    if manifest["architecture"].startswith("rfdetr-"):
        from ml.inference.rfdetr_model import RFDETRPredictor
        return RFDETRPredictor(model_dir, device=device)
    if manifest["architecture"] in {"yolov8s", "yolo26s", "yolo26m"}:
        from ml.inference.yolo_model import YOLOPredictor
        return YOLOPredictor(model_dir, device=device)
    if manifest["architecture"] == "yolo-efficientnet-v2-s-cascade":
        from ml.inference.cascade_model import CascadePredictor
        return CascadePredictor(model_dir, device=device)
    from ml.inference.rtdetr_model import RTDETRPredictor
    return RTDETRPredictor(model_dir, device=device)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Verify packaged files and labels without GPU or network")
    parser.add_argument("model_dir", type=Path)
    arguments = parser.parse_args()
    manifest = inspect_model(arguments.model_dir)
    print(json.dumps({"architecture": manifest["architecture"], "version": manifest["version"],
                      "resolution": manifest["resolution"], "filesVerified": True,
                      "productionValidated": manifest.get("productionValidated", False),
                      "gpuProbed": False}, ensure_ascii=False, indent=2))
