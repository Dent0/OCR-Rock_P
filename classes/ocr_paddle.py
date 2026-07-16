from paddleocr import PaddleOCR
import os
import time

class PaddleOCRM:
    def __init__(self, lang='ru', debug=False):
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
        self.debug = debug
        self.lang = lang
        self.ocr = None

    def load(self):
        if self.ocr is None:
            if self.debug:
                print(f"Загрузка PaddleOCR (lang={self.lang})")
            start = time.time()
            self.ocr = PaddleOCR(lang=self.lang)
            if self.debug:
                print(f"Загружено за {time.time() - start:.2f} сек")

    def recognize(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Файл не найден: {image_path}")

        self.load()
        start = time.time()

        results = self.ocr.predict(image_path)

        elapsed = time.time() - start

        text_parts = []
        if results and isinstance(results, list):
            for page in results:
                if isinstance(page, dict):
                    if 'rec_texts' in page:
                        text_parts.extend(page['rec_texts'])
                    elif 'text' in page:
                        text_parts.append(page['text'])

        text = '\n'.join(text_parts) if text_parts else "(текст не найден)"

        if self.debug:
            print(f"{elapsed:.3f} сек | {len(text)} символов")

        return text, elapsed