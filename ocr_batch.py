#!/usr/bin/env python3
"""
ocr_batch.py - CLI tool for batch OCR processing with preprocessing pipeline.
Cross-platform: Windows, macOS, Linux
"""

import os
import sys
import json
import time
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import platform
import re

import cv2
import numpy as np

# Импорт OCR классов из папки classes
from classes.ocr_tesseract import TesseractOCRM
from classes.ocr_paddle import PaddleOCRM
from classes.ocr_easy import EasyOCRM

# Попытка импорта tqdm для прогресс-бара
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    tqdm = None
    TQDM_AVAILABLE = False
    print("Для прогресс-бара установите: pip install tqdm")

# Попытка импорта pdf2image для pdf
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    convert_from_path = None
    PDF2IMAGE_AVAILABLE = False
    print("Для поддержки PDF установите: "
          "\npip install pdf2imag;"
          "\n(macOS): brew install poppler"
          "\n(windows):"
          "\n1. Скачать poppler: https://github.com/oschwartz10612/poppler-windows/releases/"
          "\n2. Распаковать в C: poppler"
          "\n3. Добавить в PATH: C: poppler bin")


def get_platform_info():
    """Возвращает информацию о платформе"""
    system = platform.system()
    is_windows = system == 'Windows'
    is_macos = system == 'Darwin'
    is_linux = system == 'Linux'
    return {
        'system': system,
        'is_windows': is_windows,
        'is_macos': is_macos,
        'is_linux': is_linux,
        'cpu_count': mp.cpu_count()
    }


def natural_sort_key(filename: str) -> List:
    """
    Естественная сортировка файлов (числа в имени сортируются правильно).
    Например: image1.png, image2.png, image10.png
    """

    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]

    return alphanum_key(filename)


def get_images_from_list(list_arg: str, input_dir: str, output_dir: str = None) -> List[str]:
    """
    Получает список изображений из файла или из переданных имен.
    Поддерживает PDF файлы (конвертирует в изображения).
    """
    images = []

    # Если передан путь к файлу
    if os.path.isfile(list_arg):
        with open(list_arg, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    images.extend(resolve_file_path(line, input_dir, output_dir))
    else:
        # Переданы имена файлов через пробел
        for name in list_arg.split():
            images.extend(resolve_file_path(name, input_dir, output_dir))

    return images


def resolve_file_path(file_path: str, input_dir: str, output_dir: str = None) -> List[str]:
    """
    Определяет тип файла и возвращает список путей к изображениям.
    """
    result = []

    # Проверяем, есть ли полный путь или только имя
    if os.path.exists(file_path):
        full_path = file_path
    else:
        full_path = os.path.join(input_dir, file_path)
        if not os.path.exists(full_path):
            print(f"Warning: File not found: {file_path}")
            return result

    # Проверяем расширение
    ext = Path(full_path).suffix.lower()

    if ext in {'.pdf'}:
        # Конвертируем PDF
        if output_dir is None:
            output_dir = input_dir
        result.extend(convert_pdf_to_images(full_path, output_dir))
    elif ext in {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}:
        result.append(full_path)
    else:
        print(f"Warning: Unsupported file type: {full_path}")

    return result


def convert_pdf_to_images(pdf_path: str, output_dir: str, dpi: int = 200, debug: bool = False) -> List[str]:
    """
    Конвертирует PDF в изображения.
    Возвращает список путей к сохранённым изображениям.
    """
    if not PDF2IMAGE_AVAILABLE:
        raise ImportError("pdf2image не установлен. Установите: pip install pdf2image")

    # Создаём папку для страниц PDF
    pdf_name = Path(pdf_path).stem
    pdf_output_dir = Path(output_dir) / f"{pdf_name}_pages"
    pdf_output_dir.mkdir(parents=True, exist_ok=True)

    # Конвертируем PDF в изображения
    if debug: print(f"  Конвертация PDF: {pdf_path} (DPI={dpi})")
    images = convert_from_path(pdf_path, dpi=dpi)

    image_paths = []
    for i, img in enumerate(images):
        # Сохраняем каждую страницу как PNG
        page_path = pdf_output_dir / f"{pdf_name}_page_{i + 1:03d}.png"
        img.save(str(page_path), 'PNG')
        image_paths.append(str(page_path))
        if debug: print(f"    Страница {i + 1}: {page_path.name}")

    if debug: print(f"  Всего страниц: {len(image_paths)}")
    return image_paths


def filter_images_by_range(images: List[str], start: Optional[int], end: Optional[int]) -> List[str]:
    """
    Фильтрует список изображений по диапазону.
    start - начальный индекс (1-based, включительно)
    end - конечный индекс (1-based, включительно)
    """
    if not images:
        return images

    # Сортируем изображения естественным образом
    sorted_images = sorted(images, key=lambda x: natural_sort_key(os.path.basename(x)))

    total = len(sorted_images)

    # Определяем начальный индекс
    start_idx = 0
    if start is not None:
        start_idx = max(0, start - 1)  # Преобразуем в 0-based
        if start_idx >= total:
            print(f"Warning: start={start} > total images ({total}), starting from beginning")
            start_idx = 0

    # Определяем конечный индекс
    end_idx = total
    if end is not None:
        end_idx = min(total, end)  # Преобразуем в 0-based (end - включительно)
        if end_idx <= 0:
            print(f"Warning: end={end} <= 0, processing all images")
            end_idx = total

    # Проверяем, что диапазон корректен
    if start_idx >= end_idx:
        print(f"Warning: start={start} > end={end}, processing all images")
        return sorted_images

    return sorted_images[start_idx:end_idx]


def deskew_image_pca(image):
    """
    Коррекция перекоса через PCA (Principal Component Analysis)
    Более точный метод для текста
    """
    # 1. Определяем яркость изображения
    mean_brightness = np.mean(image)
    is_light = mean_brightness > 127

    # 2. Бинаризация для выделения текста
    if is_light:
        binary = cv2.adaptiveThreshold(image, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
    else:
        binary = cv2.adaptiveThreshold(image, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

    # 3. Находим координаты всех белых пикселей (текст)
    points = np.column_stack(np.where(binary > 0))

    if len(points) < 10:
        return image, 0

    # 4. Вычисляем PCA
    mean = np.mean(points, axis=0)
    centered = points - mean
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # Сортировка по убыванию собственных значений
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Главный компонент (направление текста)
    main_axis = eigenvectors[:, 0]

    # Вычисляем угол (в градусах)
    angle = np.arctan2(main_axis[0], main_axis[1]) * 180 / np.pi

    # Приводим к диапазону [-45°, 45°]
    if angle > 45:
        angle -= 90
    elif angle < -45:
        angle += 90

    # 5. Применяем поворот (в OpenCV поворачиваем на тот же угол)
    if abs(angle) > 2.8:
        h, w = image.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed = cv2.warpAffine(image, M, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=255)

        # Обрезаем серые края после поворота
        coords = np.column_stack(np.where(deskewed < 250))
        if len(coords) > 10:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            pad = 15
            y_min = max(0, y_min - pad)
            y_max = min(h, y_max + pad)
            x_min = max(0, x_min - pad)
            x_max = min(w, x_max + pad)
            deskewed = deskewed[y_min:y_max, x_min:x_max]

        return deskewed, angle

    return image, 0


def preprocess_image(image_path: str, target_dpi: int = 300) -> np.ndarray:
    """
    Full preprocessing pipeline using OpenCV with improved PCA-based deskew.
    """
    img_path = Path(image_path)
    if not img_path.exists():
        raise ValueError(f"Cannot read image: {image_path}")

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Denoise (Non-local Means)
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10,
                                        templateWindowSize=7,
                                        searchWindowSize=21)

    # 3. Deskew (улучшенный через PCA)
    deskewed, _ = deskew_image_pca(denoised)

    # 4. Otsu threshold (binarization)
    _, binary = cv2.threshold(deskewed, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 5. DPI normalization
    current_dpi = 300
    if current_dpi != target_dpi:
        scale = target_dpi / current_dpi
        new_w = int(binary.shape[1] * scale)
        new_h = int(binary.shape[0] * scale)
        binary = cv2.resize(binary, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    return binary


def get_ocr_engine(engine_name: str, debug: bool = False):
    """
    Return initialized OCR engine instance.
    """
    if engine_name == "tesseract":
        return TesseractOCRM(lang='rus+eng', debug=debug)
    elif engine_name == "paddle":
        return PaddleOCRM(lang='ru', debug=debug)
    elif engine_name == "easy":
        return EasyOCRM(lang_list=['ru', 'en'], gpu=False, debug=debug)
    else:
        raise ValueError(f"Unknown engine: {engine_name}. Choose from: tesseract, paddle, easy")


def save_preprocessed_image(img: np.ndarray, output_path: str):
    """Save preprocessed image for debugging."""
    cv2.imwrite(output_path, img)


def worker_func(args: Tuple[str, str, str, bool]) -> Dict[str, Any]:
    """
    Worker function for multiprocessing.
    Args: (image_path, output_dir, engine_name, save_preprocessed)
    Returns: dict with image_path, text, status, error, processing_time
    """
    image_path, output_dir, engine_name, save_preproc = args
    result: Dict[str, Any] = {
        "image": os.path.basename(image_path),
        "image_path": image_path,
        "status": "success",
        "text": "",
        "error": None,
        "processing_time": 0.0,
        "preprocessing_time": 0.0,
        "ocr_time": 0.0
    }

    try:
        # Preprocess
        preprocess_start = time.perf_counter()
        processed = preprocess_image(image_path)
        result["preprocessing_time"] = time.perf_counter() - preprocess_start

        # Save preprocessed image if requested
        if save_preproc:
            stem = Path(image_path).stem
            preproc_path = Path(output_dir) / f"{stem}_preprocessed.png"
            save_preprocessed_image(processed, str(preproc_path))

        # Save processed image temporarily for OCR
        temp_path = Path(output_dir) / f"_temp_{os.getpid()}.png"
        cv2.imwrite(str(temp_path), processed)

        # OCR
        engine = get_ocr_engine(engine_name, debug=False)
        text, ocr_time = engine.recognize(str(temp_path))
        result["ocr_time"] = ocr_time
        result["processing_time"] = result["preprocessing_time"] + ocr_time
        result["text"] = text.strip()

        # Cleanup temp file
        if temp_path.exists():
            temp_path.unlink()

        # Save individual result
        stem = Path(image_path).stem
        out_file = Path(output_dir) / f"{stem}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({
                "image": os.path.basename(image_path),
                "text": text.strip(),
                "processing_time": result["processing_time"],
                "preprocessing_time": result["preprocessing_time"],
                "ocr_time": result["ocr_time"]
            }, f, ensure_ascii=False, indent=2)

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)

    return result


def run_batch(input_dir: str, output_dir: str, engine: str, workers: int,
              save_preprocessed: bool = False, start: Optional[int] = None,
              end: Optional[int] = None, list_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Run batch OCR processing with multiprocessing pool.
    Returns statistics and collected results.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    # Получаем список изображений
    if list_file:
        # Используем список из файла или переданных имен
        images = get_images_from_list(list_file, input_dir, str(output_path))
        if not images:
            raise ValueError(f"No images found in list: {list_file}")
        total_all = len(images)
    else:
        # Используем все изображения из папки
        supported_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".pdf"}
        files = [str(p) for p in input_path.iterdir() if p.suffix.lower() in supported_extensions]

        images = []
        for f in files:
            images.extend(resolve_file_path(f, input_dir, str(output_path)))

        if not images:
            raise ValueError(f"No images found in {input_dir}")
        total_all = len(images)

    # Фильтруем изображения по диапазону
    filtered_images = filter_images_by_range(images, start, end)
    total_filtered = len(filtered_images)

    platform_info = get_platform_info()
    print(f"Platform: {platform_info['system']}")
    if list_file:
        print(f"Using image list: {list_file}")
    print(f"Total images: {total_all}")
    if start is not None or end is not None:
        range_info = []
        if start is not None:
            range_info.append(f"from #{start}")
        if end is not None:
            range_info.append(f"to #{end}")
        print(f"Filtered images: {total_filtered} ({' '.join(range_info)})")
    print(f"Processing {total_filtered} images with {workers} workers using {engine} engine...")

    # Настройка multiprocessing для Windows
    if platform_info['is_windows']:
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # уже установлен

    # Prepare args for workers
    worker_args = [(img, str(output_path), engine, save_preprocessed) for img in filtered_images]

    # Run multiprocessing with progress bar
    start_total = time.perf_counter()

    if TQDM_AVAILABLE:
        # С прогресс-баром
        with mp.Pool(processes=workers) as pool:
            results = list(tqdm(
                pool.imap(worker_func, worker_args),
                total=len(worker_args),
                desc="Processing images",
                unit="img",
                colour="green"
            ))
    else:
        # Без прогресс-бара
        with mp.Pool(processes=workers) as pool:
            results = pool.map(worker_func, worker_args)

    total_time = time.perf_counter() - start_total

    # Aggregate results
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]

    stats = {
        "engine": engine,
        "workers": workers,
        "total_images": total_all,
        "total_images_processed": total_filtered,
        "start": start,
        "end": end,
        "list_file": list_file,
        "successful": len(successful),
        "failed": len(failed),
        "total_time": total_time,
        "avg_time_per_image": sum(r["processing_time"] for r in successful) / len(successful) if successful else 0,
        "avg_preprocessing_time": sum(r["preprocessing_time"] for r in successful) / len(
            successful) if successful else 0,
        "avg_ocr_time": sum(r["ocr_time"] for r in successful) / len(successful) if successful else 0,
        "platform": platform_info['system'],
        "results": results
    }

    # Save combined results
    with open(output_path / "all_results.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    return stats


def run_benchmark(input_dir: str, output_dir: str, engine: str,
                  start: Optional[int] = None, end: Optional[int] = None,
                  list_file: Optional[str] = None, workers: Optional[int] = None):
    """
    Run speedup benchmark: 1 worker vs N workers.
    """
    platform_info = get_platform_info()
    cpu_count = platform_info['cpu_count']

    # Определяем число воркеров
    if workers is not None:
        # Используем переданное значение
        n_workers = min(workers, cpu_count)
        print(f"Using {n_workers} workers (specified by --workers)")
    else:
        # Автоопределение в зависимости от платформы
        if platform_info['is_windows']:
            n_workers = min(cpu_count, 4)
        elif platform_info['is_macos']:
            n_workers = min(cpu_count, 6)
        else:
            n_workers = min(cpu_count, 8)
        print(f"Using {n_workers} workers (auto-detected)")

    print("=" * 60)
    print("SPEEDUP BENCHMARK")
    print("=" * 60)
    print(f"Platform: {platform_info['system']}")
    print(f"CPU cores available: {cpu_count}")
    if list_file:
        print(f"Using image list: {list_file}")
    if start is not None or end is not None:
        range_info = []
        if start is not None:
            range_info.append(f"from #{start}")
        if end is not None:
            range_info.append(f"to #{end}")
        print(f"Range: {' '.join(range_info)}")
    print(f"Using {n_workers} workers for parallel run\n")

    # Run with 1 worker
    print("Running with 1 worker...")
    bench1_dir = f"{output_dir}_bench1"
    stats1 = run_batch(input_dir, bench1_dir, engine, 1, save_preprocessed=False,
                       start=start, end=end, list_file=list_file)
    t1 = stats1["total_time"]
    print(f"T(1 worker) = {t1:.2f}s\n")

    # Run with N workers
    print(f"Running with {n_workers} workers...")
    benchN_dir = f"{output_dir}_benchN"
    statsN = run_batch(input_dir, benchN_dir, engine, n_workers, save_preprocessed=False,
                       start=start, end=end, list_file=list_file)
    tN = statsN["total_time"]
    print(f"T({n_workers} workers) = {tN:.2f}s\n")

    speedup = t1 / tN
    efficiency = (speedup / n_workers) * 100

    # Print table
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\n--- SPEEDUP TABLE ---")
    print("+-------------+-----------+-----------+")
    print("| Workers     | Time (s)  | Speedup   |")
    print("+-------------+-----------+-----------+")
    print(f"| 1           | {t1:9.2f} | 1.00x     |")
    print(f"| {n_workers:<11} | {tN:9.2f} | {speedup:6.2f}x    |")
    print("+-------------+-----------+-----------+")
    print(f"| Efficiency  |           | {efficiency:5.1f}%    |")
    print("+-------------+-----------+-----------+")

    # Additional stats
    print("\n--- DETAILED STATS ---")
    print(f"Total images processed: {stats1['total_images_processed']}")
    print(f"T(1 worker):  {t1:.2f}s (avg per image: {stats1['avg_time_per_image']:.3f}s)")
    print(f"T({n_workers} workers): {tN:.2f}s (avg per image: {statsN['avg_time_per_image']:.3f}s)")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Efficiency: {efficiency:.1f}%")

    # Save benchmark results
    benchmark_results = {
        "platform": platform_info['system'],
        "engine": engine,
        "workers_1": 1,
        "workers_n": n_workers,
        "time_1_worker": t1,
        "time_n_workers": tN,
        "speedup": speedup,
        "efficiency": efficiency,
        "start": start,
        "end": end,
        "list_file": list_file
    }
    with open(f"{output_dir}_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Batch OCR with preprocessing pipeline (cross-platform)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-i", "--input", required=True, help="Input directory with images")
    parser.add_argument("-o", "--output", required=True, help="Output directory for results")
    parser.add_argument("-w", "--workers", type=int, default=1,
                        help="Number of worker processes (default: 1)")
    parser.add_argument("--engine", choices=["tesseract", "paddle", "easy"], default="tesseract",
                        help="OCR engine to use (default: tesseract)")
    parser.add_argument("-sp", "--save-preprocessed", action="store_true",
                        help="Save preprocessed images for debugging")
    parser.add_argument("-bm", "--benchmark", action="store_true",
                        help="Run speedup benchmark (1 vs N workers)")
    parser.add_argument("--no-progress", action="store_true",
                        help="Disable progress bar")
    parser.add_argument("-s", "--start", type=int, default=None,
                        help="Start processing from image number N (1-based, inclusive)")
    parser.add_argument("-e", "--end", type=int, default=None,
                        help="End processing at image number M (1-based, inclusive)")
    parser.add_argument("-l", "--list", type=str, default=None,
                        help="TXT file with image names OR space-separated image names (e.g., --list img1.png img2.png)")

    args = parser.parse_args()

    # Отключаем прогресс-бар если запрошено
    global TQDM_AVAILABLE
    if args.no_progress:
        TQDM_AVAILABLE = False

    # Проверка валидности диапазона
    if args.start is not None and args.start < 1:
        print(f"Warning: start={args.start} < 1, using 1")
        args.start = 1

    if args.end is not None and args.end < 1:
        print(f"Warning: end={args.end} < 1, ignoring")
        args.end = None

    try:
        if args.benchmark:
            run_benchmark(args.input, args.output, args.engine,
                          args.start, args.end, args.list, args.workers)
        else:
            stats = run_batch(args.input, args.output, args.engine, args.workers,
                              args.save_preprocessed, args.start, args.end, args.list)

            print("\n" + "=" * 60)
            print("PROCESSING COMPLETE")
            print("=" * 60)
            print(f"Platform: {stats['platform']}")
            print(f"Processed: {stats['total_images_processed']}/{stats['total_images']}")
            if args.list:
                print(f"Using list: {args.list}")
            if args.start is not None or args.end is not None:
                range_info = []
                if args.start is not None:
                    range_info.append(f"from #{args.start}")
                if args.end is not None:
                    range_info.append(f"to #{args.end}")
                print(f"Range: {' '.join(range_info)}")
            print(f"Successful: {stats['successful']}/{stats['total_images_processed']}")
            print(f"Failed: {stats['failed']}/{stats['total_images_processed']}")
            print(f"Total time: {stats['total_time']:.2f}s")
            print(f"Average time per image: {stats['avg_time_per_image']:.3f}s")
            print(f"Results saved to: {args.output}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()