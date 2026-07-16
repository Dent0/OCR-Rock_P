import pytesseract
import time
import os
import platform


class TesseractOCRM:
    def __init__(self, lang='rus+eng', config='', debug=False, tesseract_path=None):
        self.lang = lang
        self.config = config
        self.debug = debug

        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        elif platform.system() == 'Windows':
            paths = [r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                     r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe']
            for p in paths:
                if os.path.exists(p):
                    pytesseract.pytesseract.tesseract_cmd = p
                    break

    def recognize(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Файл не найден: {image_path}")

        start = time.time()
        text = pytesseract.image_to_string(image_path, lang=self.lang, config=self.config).strip()
        elapsed = time.time() - start

        if self.debug:
            print(f"{elapsed:.3f} сек | {len(text)} символов")

        return text, elapsed