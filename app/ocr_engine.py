import os
import sys
import time
import tempfile
from typing import Dict, Any

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr_batch import preprocess_image, get_ocr_engine


def process_single_image(image_path: str, engine_name: str = "tesseract") -> Dict[str, Any]:
    start_time = time.perf_counter()

    try:
        preprocess_start = time.perf_counter()
        processed = preprocess_image(image_path)
        preprocessing_time = time.perf_counter() - preprocess_start

        temp_path = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
        cv2.imwrite(temp_path, processed)

        # Для Tesseract явно указываем русский язык
        if engine_name == 'tesseract':
            from classes.ocr_tesseract import TesseractOCRM
            engine = TesseractOCRM(lang='eng+rus', debug=False)
        else:
            engine = get_ocr_engine(engine_name, debug=False)

        text, ocr_time = engine.recognize(temp_path)

        os.unlink(temp_path)

        total_time = time.perf_counter() - start_time

        return {
            "status": "success",
            "text": text.strip(),
            "char_count": len(text.strip()),
            "word_count": len(text.strip().split()),
            "total_time": total_time,
            "preprocessing_time": preprocessing_time,
            "ocr_time": ocr_time,
            "engine": engine_name
        }

    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "engine": engine_name,
            "total_time": time.perf_counter() - start_time
        }