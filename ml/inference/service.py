"""HTTP contract and image handling shared by local tests and Modal deployment."""
from email import policy
from email.parser import BytesParser
from io import BytesIO
import time
import warnings
from typing import Protocol

from fastapi import FastAPI, HTTPException, Request
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool
from ml.inference.postprocess import MAX_DETECTIONS, normalize_tiles

MAX_BYTES = 12 * 1024 * 1024
MAX_UPLOAD = MAX_BYTES + 64 * 1024
MAX_PIXELS = 40_000_000


class Predictor(Protocol):
    version: str
    def predict(self, image: Image.Image) -> list[dict]: ...


def decode_image(data: bytes) -> Image.Image:
    if not data:
        raise HTTPException(400, "empty image")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "image too large")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = Image.open(BytesIO(data))
            if image.width * image.height > MAX_PIXELS:
                raise HTTPException(413, "image pixel limit")
            if image.format not in {"JPEG", "PNG", "WEBP"}:
                raise HTTPException(415, "unsupported image format")
            if min(image.size) < 32:
                raise HTTPException(422, "image too small")
            image = ImageOps.exif_transpose(image)
            image.load()
            return image.convert("RGB")
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise HTTPException(413, "image pixel limit")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(415, "invalid image")



def create_app(predictor: Predictor, accept_threshold: float = 0.7) -> FastAPI:
    app = FastAPI(title="Mahjong inference", docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/health")
    def health():
        return {"modelVersion": predictor.version, "loaded": True}

    @app.post("/recognize")
    async def recognize(request: Request):
        started = time.perf_counter()
        content_type = request.headers.get("content-type", "")
        if not content_type.startswith("multipart/form-data;") or "boundary=" not in content_type:
            raise HTTPException(415, "multipart image required")
        chunks = []
        total = 0
        async for chunk in request.stream():
            total += len(chunk)
            if total > MAX_UPLOAD:
                raise HTTPException(413, "upload too large")
            chunks.append(chunk)
        message = BytesParser(policy=policy.default).parsebytes(
            b"Content-Type: " + content_type.encode("ascii", "strict") + b"\r\nMIME-Version: 1.0\r\n\r\n" + b"".join(chunks))
        parts = list(message.iter_parts())
        images = [p for p in parts if p.get_param("name", header="content-disposition") == "image"]
        file_parts = [p for p in parts if p.get_filename() is not None]
        if len(images) != 1 or len(file_parts) != 1 or file_parts[0] is not images[0]:
            raise HTTPException(400, "exactly one image required")
        image = decode_image(images[0].get_payload(decode=True) or b"")
        raw = await run_in_threadpool(predictor.predict, image)
        if len(raw) > MAX_DETECTIONS:
            raise HTTPException(422, "scene too crowded; photograph one group of tiles")
        tiles, reasons = normalize_tiles(raw, image.width, image.height, accept_threshold)
        return {
            "requestId": request.headers.get("x-request-id", ""),
            "modelVersion": predictor.version,
            "image": {"width": image.width, "height": image.height}, "tiles": tiles,
            "quality": {"status": "needs_retake" if reasons else "ok", "reasons": reasons},
            "timings": {"totalMs": round((time.perf_counter() - started) * 1000, 1)},
        }

    return app
