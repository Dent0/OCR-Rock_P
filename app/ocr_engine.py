import os
import sys
import time
import tempfile
from typing import Dict, Any
from pdf2image import convert_from_path

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr_batch import preprocess_image, get_ocr_engine


def process_single_image(image_path: str, engine_name: str = "tesseract") -> Dict[str, Any]:
    start_time = time.perf_counter()

    try:
        # Проверяем, PDF ли это
        ext = os.path.splitext(image_path)[1].lower()

        if ext == '.pdf':
            # Конвертируем PDF в изображения
            images = convert_from_path(image_path, dpi=200)
            if not images:
                raise ValueError("PDF не содержит страниц")

            # Обрабатываем все страницы
            all_text = []
            total_preprocessing = 0
            total_ocr = 0

            for i, img in enumerate(images):
                # Сохраняем страницу как PNG
                temp_path = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
                img.save(temp_path, 'PNG')

                # Предобработка страницы с замером
                preprocess_start = time.perf_counter()
                processed = preprocess_image(temp_path)
                total_preprocessing += time.perf_counter() - preprocess_start

                cv2.imwrite(temp_path, processed)

                # OCR с замером
                if engine_name == 'tesseract':
                    from classes.ocr_tesseract import TesseractOCRM
                    engine = TesseractOCRM(lang='eng+rus', debug=False)
                else:
                    engine = get_ocr_engine(engine_name, debug=False)

                ocr_start = time.perf_counter()
                text, ocr_time = engine.recognize(temp_path)
                total_ocr += time.perf_counter() - ocr_start

                all_text.append(text.strip())

                os.unlink(temp_path)

            # Объединяем текст со всех страниц
            full_text = '\n\n'.join(all_text)

            total_time = time.perf_counter() - start_time

            return {
                "status": "success",
                "text": full_text,
                "char_count": len(full_text),
                "word_count": len(full_text.split()),
                "total_time": total_time,
                "preprocessing_time": total_preprocessing,
                "ocr_time": total_ocr,
                "engine": engine_name,
                "pages": len(images)
            }

        # Обработка одного изображения (старый код)
        preprocess_start = time.perf_counter()
        processed = preprocess_image(image_path)
        preprocessing_time = time.perf_counter() - preprocess_start

        temp_path = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
        cv2.imwrite(temp_path, processed)

        if engine_name == 'tesseract':
            from classes.ocr_tesseract import TesseractOCRM
            engine = TesseractOCRM(lang='eng+rus', debug=False)
        else:
            engine = get_ocr_engine(engine_name, debug=False)

        ocr_start = time.perf_counter()
        text, ocr_time = engine.recognize(temp_path)
        ocr_time = time.perf_counter() - ocr_start

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