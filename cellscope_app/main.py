"""FastAPI entrypoint. Only the dedicated static directory is publicly served."""
from contextlib import asynccontextmanager
import logging
from pathlib import Path
import threading
from typing import Literal

import torch

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .inference import Detector, decode_image
from .classification import SubtypeClassifier
from .pipeline import IntegratedPipeline

MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_REQUEST_BYTES = MAX_UPLOAD_BYTES + 64 * 1024
STATIC = Path(__file__).resolve().parent / "static"
detector = Detector()
classifier = SubtypeClassifier()
pipeline = IntegratedPipeline(detector, classifier)
analysis_slot = threading.BoundedSemaphore(1)


class UploadLimit:
    """Bound the complete multipart body before the parser can spool it to disk."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        if scope["method"] == "POST" and scope["path"] == "/api/analyze":
            chunks, size = [], 0
            while True:
                event = await receive()
                if event["type"] == "http.disconnect":
                    return
                chunk = event.get("body", b"")
                size += len(chunk)
                if size > MAX_REQUEST_BYTES:
                    return await JSONResponse({"detail": "Upload too large. Maximum image size is 12 MB."}, 413)(scope, receive, send)
                chunks.append(chunk)
                if not event.get("more_body", False):
                    break
            body = b"".join(chunks)
            original_receive = receive

            async def replay():
                nonlocal body
                if body is not None:
                    value, body = body, None
                    return {"type": "http.request", "body": value, "more_body": False}
                return await original_receive()

            receive = replay
        await self.app(scope, receive, send)


@asynccontextmanager
async def lifespan(app):
    torch.set_num_threads(2)
    await run_in_threadpool(detector.load)
    await run_in_threadpool(classifier.load)
    yield


app = FastAPI(title="CellScope — Integrated Blood Cell Analysis", lifespan=lifespan)
app.add_middleware(UploadLimit)


@app.middleware("http")
async def private_responses(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


@app.get("/api/health")
def health():
    return {**detector.status(), "classifier": classifier.status(),
            "integrated_ready": detector.model is not None and classifier.model is not None}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), confidence: float = Query(0.25, ge=0.05, le=0.95),
                  mode: Literal["detection", "integrated"] = "detection",
                  subtype_confidence: float = Query(.70, ge=0., le=1.),
                  padding: float = Query(.12, ge=0., le=.25)):
    try:
        if Path(file.filename or "").suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            raise HTTPException(415, "Choose a JPG, JPEG, or PNG file.")
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Upload too large. Maximum image size is 12 MB.")
        if not data:
            raise HTTPException(400, "The uploaded file is empty.")
        try:
            image = await run_in_threadpool(decode_image, data)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if detector.model is None:
            raise HTTPException(503, detector.error or "The model is not ready. Restart the server.")
        if mode == "integrated" and classifier.model is None:
            raise HTTPException(503, classifier.error or "Classifier unavailable. Use detection-only mode or restart.")
        if not analysis_slot.acquire(blocking=False):
            raise HTTPException(429, "Another image is being analyzed. Please try again when it finishes.")
        try:
            report = await run_in_threadpool(pipeline.analyze, image, confidence, mode, subtype_confidence, padding)
            # Basename only: never expose a client's local path in exports.
            report["filename"] = (file.filename or "image").replace("\\", "/").split("/")[-1]
            return report
        except Exception as exc:
            logging.getLogger(__name__).exception("Inference failed")
            raise HTTPException(500, "Analysis failed. Try a smaller microscopy image or restart the local server.") from exc
        finally:
            analysis_slot.release()
    finally:
        await file.close()


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
