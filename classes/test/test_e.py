from classes.ocr_easy import EasyOCRM

# Windows (с GPU)
ocr = EasyOCRM(lang_list=['ru', 'en'], gpu=True, debug=True)

# macOS (GPU автоматически отключится)
ocr = EasyOCRM(lang_list=['ru', 'en'], gpu=True, debug=True)

text, elapsed = ocr.recognize("image/test_image.png")

print("\n" + "=" * 40)
print("РЕЗУЛЬТАТ EASYOCR:")
print("=" * 40)
print(text)
print("=" * 40)
print(f"{elapsed:.3f} сек")