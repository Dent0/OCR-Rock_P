import os
import time
from paddleocr import PaddleOCR


class PaddleOCRModel:
    def __init__(self, lang='ru', debug=False):
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
        self.debug = debug
        self.lang = lang
        self.ocr = None

    def load(self):
        if self.ocr is None:
            if self.debug: print("Загрузка PaddleOCR")
            start = time.time()
            self.ocr = PaddleOCR(lang=self.lang)
            if self.debug: print(f"Загружено за {time.time() - start:.2f} сек")

    def recognize(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Файл не найден: {image_path}")

        self.load()
        start = time.time()

        results = self.ocr.predict(image_path)

        elapsed = time.time() - start

        # Извлечение текста из правильного поля
        text_parts = []

        if results and isinstance(results, list):
            for page in results:
                if isinstance(page, dict):
                    # Основное поле с текстом
                    if 'rec_texts' in page:
                        text_parts.extend(page['rec_texts'])
                    # Запасной вариант
                    elif 'text' in page:
                        text_parts.append(page['text'])

        # Если ничего не нашли
        if not text_parts:
            text_parts = ["(текст не найден)"]

        return ' '.join(text_parts), elapsed


# === ТЕСТ ===
if __name__ == "__main__":
    model = PaddleOCRModel(lang='ru')

    image_path = "test_image.png"

    text, elapsed = model.recognize(image_path)

    print("\n" + "=" * 40)
    print("📝 РЕЗУЛЬТАТ:")
    print("=" * 40)
    print(text)
    print("=" * 40)
    print(f"⏱️ {elapsed:.3f} сек")