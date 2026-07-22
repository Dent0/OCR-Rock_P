import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from celery.result import AsyncResult
from prometheus_fastapi_instrumentator import Instrumentator
from app.tasks import process_document
from app.config import UPLOAD_DIR, SUPPORTED_EXTENSIONS, DEFAULT_ENGINE

app = FastAPI(
    title="OCR Service",
    description="Асинхронный сервис для распознавания текста",
    version="1.0.0"
)

# Добавляем Prometheus метрики
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app, endpoint="/metrics")


@app.get("/")
async def root():
    return {"message": "OCR Service", "docs": "/docs"}


@app.post("/ocr/submit")
async def submit_ocr(
        file: UploadFile = File(...),
        engine: str = Form(DEFAULT_ENGINE)
):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    if engine not in ["tesseract", "easy", "paddle"]:
        raise HTTPException(400, f"Unsupported engine: {engine}")

    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    task = process_document.apply_async(args=[file_path, engine])

    return {"task_id": task.id, "file_name": file.filename, "engine": engine, "status": "pending"}


@app.get("/ocr/status/{task_id}")
async def get_status(task_id: str):
    task = AsyncResult(task_id, app=process_document)

    if task.state == 'PENDING':
        return {"task_id": task_id, "status": "pending", "progress": 0}
    elif task.state == 'PROCESSING':
        meta = task.info or {}
        return {"task_id": task_id, "status": "processing", "progress": meta.get('progress', 0)}
    elif task.state == 'SUCCESS':
        return {"task_id": task_id, "status": "done", "result": task.result}
    elif task.state == 'FAILURE':
        return {"task_id": task_id, "status": "failed", "error": str(task.info)}
    else:
        return {"task_id": task_id, "status": task.state}


@app.get("/ocr/result/{task_id}")
async def get_result(task_id: str):
    task = AsyncResult(task_id, app=process_document)

    if task.state == 'SUCCESS':
        return task.result
    elif task.state == 'FAILURE':
        raise HTTPException(500, f"Task failed: {str(task.info)}")
    elif task.state == 'PENDING':
        raise HTTPException(404, "Task is still pending")
    else:
        raise HTTPException(404, f"Task is in state: {task.state}")


@app.get("/health")
async def health():
    return {"status": "ok"}