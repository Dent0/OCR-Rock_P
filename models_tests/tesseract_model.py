import os
import time
import pytesseract
import cv2


class TesseractModel:
    def __init__(self, lang='eng+rus', tesseract_path=None, debug = False):
        """
        Инициализация Tesseract

        Args:
            lang: Язык ('rus+eng', 'eng', 'rus')
            tesseract_path: Путь к tesseract (для macOS обычно не нужен)
        """
        self.lang = lang
        self.debug = debug

        # Для macOS Tesseract автоматически находится в PATH
        # Если нужно указать вручную (для Windows или специфических установок)
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

        # Проверка установки
        try:
            version = pytesseract.get_tesseract_version()
            if self.debug: print(f"Tesseract версия: {version}")
        except Exception as e:
            if self.debug:
                print(f"Tesseract не найден!")
                print("Установите: brew install tesseract tesseract-lang")
            raise e

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

        # Загрузка изображения
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")

        # Конвертация в RGB (Tesseract ожидает RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Распознавание
        start = time.time()
        text = pytesseract.image_to_string(img_rgb, lang=self.lang)
        elapsed = time.time() - start

        return text.strip(), elapsed


# === ТЕСТ ===
if __name__ == "__main__":
    model = TesseractModel(lang='rus+eng')

    image_path = "test_image.png"

    text, elapsed = model.recognize(image_path)

    print("\n" + "=" * 40)
    print("РЕЗУЛЬТАТ TESSERACT:")
    print("=" * 40)
    print(text)
    print("=" * 40)
    print(f"{elapsed:.3f} сек")