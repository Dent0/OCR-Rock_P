from classes.ocr_paddle import PaddleOCRM

ocr = PaddleOCRM(lang='ru', debug=True)

text, elapsed = ocr.recognize("image/test_image.png")

print("\n" + "=" * 40)
print("РЕЗУЛЬТАТ PADDLEOCR:")
print("=" * 40)
print(text)
print("=" * 40)
print(f"{elapsed:.3f} сек")