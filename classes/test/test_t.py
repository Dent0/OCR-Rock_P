import platform
from classes.ocr_tesseract import TesseractOCRM

print(f"Операционная система: {platform.system()}")

ocr = TesseractOCRM(lang='rus+eng', debug=True)

try:
    text, elapsed = ocr.recognize("image/test2.png")

    print("\n" + "=" * 40)
    print("РЕЗУЛЬТАТ TESSERACT:")
    print("=" * 40)
    print(text)
    print("=" * 40)
    print(f"{elapsed:.3f} сек")
    print("\nРезультат сохранён в result.txt")

except Exception as e:
    print(f"Ошибка: {e}")