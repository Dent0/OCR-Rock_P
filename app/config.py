import os
from pathlib import Path

# Базовая директория
BASE_DIR = Path(__file__).resolve().parent.parent

# Настройки Redis
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

# Директории для файлов
UPLOAD_DIR = os.getenv("UPLOAD_DIR", str(BASE_DIR / "uploads"))
RESULT_DIR = os.getenv("RESULT_DIR", str(BASE_DIR / "results"))

# Создаём директории
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Настройки OCR
DEFAULT_ENGINE = os.getenv("DEFAULT_ENGINE", "tesseract")

# Поддерживаемые форматы
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".pdf"}