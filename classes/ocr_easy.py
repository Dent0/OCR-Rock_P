import easyocr
import os
import time
import warnings
import platform

warnings.filterwarnings("ignore")


class EasyOCRM:
    def __init__(self, lang_list=['ru', 'en'], gpu=False, debug=False):
        self.debug = debug
        self.gpu = gpu
        self.lang_list = lang_list
        self.reader = None

        # Автоматическое отключение GPU на macOS
        if platform.system() == 'Darwin':
            if self.gpu:
                if self.debug:
                    print("GPU отключен: на macOS EasyOCR с GPU работает нестабильно")
                self.gpu = False

        self._load()

    def _load(self):
        if self.debug:
            print(f"Загрузка EasyOCR: {self.lang_list}, GPU={self.gpu}")
        start = time.time()
        self.reader = easyocr.Reader(self.lang_list, gpu=self.gpu, verbose=False)
        if self.debug:
            print(f"Загружено за {time.time() - start:.2f} сек")

    def recognize(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Файл не найден: {image_path}")

        start = time.time()
        result = self.reader.readtext(image_path, detail=0)
        elapsed = time.time() - start

        text = '\n'.join(result) if result else "(текст не найден)"

        if self.debug:
            print(f"{elapsed:.3f} сек | {len(text)} символов")

        return text, elapsed