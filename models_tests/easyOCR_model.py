import os
import time
import easyocr


class EasyOCRModel:
    def __init__(self, lang_list=None, use_gpu=False):
        """
        Инициализация EasyOCR

        Args:
            lang_list: Список языков (по умолчанию ['ru', 'en'])
            use_gpu: Использовать GPU (на macOS False)
        """
        if lang_list is None:
            lang_list = ['ru', 'en']

        self.lang_list = lang_list
        self.use_gpu = use_gpu
        self.reader = None

    def load(self):
        """Загрузка модели (ленивая инициализация)"""
        if self.reader is None:
            print("⏳ Загрузка EasyOCR...")
            start = time.time()
            self.reader = easyocr.Reader(self.lang_list, gpu=self.use_gpu)
            print(f"✅ Загружено за {time.time() - start:.2f} сек")

    def recognize(self, image_path):
        """
        Распознавание текста на изображении

        Args:
            image_path: Путь к изображению

        Returns:
            tuple: (распознанный текст, время выполнения)
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Файл не найден: {image_path}")

        self.load()

        start = time.time()
        results = self.reader.readtext(image_path, detail=0)
        elapsed = time.time() - start

        # Объединение текста
        text = ' '.join(results) if results else ""

        return text.strip(), elapsed


# === ТЕСТ ===
if __name__ == "__main__":
    model = EasyOCRModel(lang_list=['ru', 'en'], use_gpu=False)

    image_path = "test_image.png"

    text, elapsed = model.recognize(image_path)

    print("\n" + "=" * 40)
    print("📝 РЕЗУЛЬТАТ EASYOCR:")
    print("=" * 40)
    print(text)
    print("=" * 40)
    print(f"⏱️ {elapsed:.3f} сек")