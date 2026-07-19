import os
from celery import Celery
from app.config import REDIS_URL
from app.ocr_engine import process_single_image

app = Celery('tasks', broker=REDIS_URL, backend=REDIS_URL)

app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Europe/Moscow',
    enable_utc=True,
    task_track_started=True,
    result_expires=3600,
)

@app.task(bind=True)
def process_document(self, file_path: str, engine: str = "tesseract"):
    print(f"🔍 ENGINE RECEIVED: {engine}")  # ← ДОБАВЛЕНО
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        result = process_single_image(file_path, engine)
        result['task_id'] = self.request.id
        print(f"RESULT ENGINE: {result.get('engine')}")  # ← ДОБАВЛЕНО
        return result

    except Exception as e:
        return {
            'task_id': self.request.id,
            'status': 'failed',
            'error': str(e)
        }
