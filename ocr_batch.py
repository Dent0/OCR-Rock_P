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
from typing import Dict, Any, Tuple
import platform

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
    TQDM_AVAILABLE = False
    print("Для прогресс-бара установите: pip install tqdm")


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


def preprocess_image(image_path: str, target_dpi: int = 300) -> np.ndarray:
    """
    Full preprocessing pipeline using OpenCV.
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

    # 3. Deskew
    coords = np.column_stack(np.where(denoised > 0))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) > 0.5:
            h, w = denoised.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            denoised = cv2.warpAffine(denoised, M, (w, h),
                                      flags=cv2.INTER_CUBIC,
                                      borderMode=cv2.BORDER_REPLICATE)

    # 4. Otsu threshold (binarization)
    _, binary = cv2.threshold(denoised, 0, 255,
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
    result = {
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
        ocr_start = time.perf_counter()
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
              save_preprocessed: bool = False) -> Dict[str, Any]:
    """
    Run batch OCR processing with multiprocessing pool.
    Returns statistics and collected results.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    image_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    images = [str(p) for p in input_path.iterdir() if p.suffix.lower() in image_extensions]

    if not images:
        raise ValueError(f"No images found in {input_dir}")

    platform_info = get_platform_info()
    print(f"Platform: {platform_info['system']}")
    print(f"Processing {len(images)} images with {workers} workers using {engine} engine...")

    # Настройка multiprocessing для Windows
    if platform_info['is_windows']:
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # уже установлен

    # Prepare args for workers
    worker_args = [(img, str(output_path), engine, save_preprocessed) for img in images]

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
        # Без прогресс-бара (или для 1 воркера)
        with mp.Pool(processes=workers) as pool:
            results = pool.map(worker_func, worker_args)

    total_time = time.perf_counter() - start_total

    # Aggregate results
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]

    stats = {
        "engine": engine,
        "workers": workers,
        "total_images": len(images),
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


def run_benchmark(input_dir: str, output_dir: str, engine: str):
    """
    Run speedup benchmark: 1 worker vs N workers.
    """
    platform_info = get_platform_info()
    cpu_count = platform_info['cpu_count']

    # Ограничение числа воркеров в зависимости от платформы
    if platform_info['is_windows']:
        n_workers = min(cpu_count, 4)
    elif platform_info['is_macos']:
        n_workers = min(cpu_count, 6)
    else:
        n_workers = min(cpu_count, 8)

    print("=" * 60)
    print("SPEEDUP BENCHMARK")
    print("=" * 60)
    print(f"Platform: {platform_info['system']}")
    print(f"CPU cores available: {cpu_count}")
    print(f"Using {n_workers} workers for parallel run\n")

    # Run with 1 worker
    print("Running with 1 worker...")
    bench1_dir = f"{output_dir}_bench1"
    stats1 = run_batch(input_dir, bench1_dir, engine, 1, save_preprocessed=False)
    t1 = stats1["total_time"]
    print(f"T(1 worker) = {t1:.2f}s\n")

    # Run with N workers
    print(f"Running with {n_workers} workers...")
    benchN_dir = f"{output_dir}_benchN"
    statsN = run_batch(input_dir, benchN_dir, engine, n_workers, save_preprocessed=False)
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
    print(f"Total images processed: {stats1['total_images']}")
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
        "efficiency": efficiency
    }
    with open(f"{output_dir}_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Batch OCR with preprocessing pipeline (cross-platform)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", required=True, help="Input directory with images")
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of worker processes (default: 1)")
    parser.add_argument("--engine", choices=["tesseract", "paddle", "easy"], default="tesseract",
                        help="OCR engine to use (default: tesseract)")
    parser.add_argument("--save-preprocessed", action="store_true",
                        help="Save preprocessed images for debugging")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run speedup benchmark (1 vs N workers)")
    parser.add_argument("--no-progress", action="store_true",
                        help="Disable progress bar")

    args = parser.parse_args()

    # Отключаем прогресс-бар если запрошено
    global TQDM_AVAILABLE
    if args.no_progress:
        TQDM_AVAILABLE = False

    try:
        if args.benchmark:
            run_benchmark(args.input, args.output, args.engine)
        else:
            stats = run_batch(args.input, args.output, args.engine, args.workers, args.save_preprocessed)

            print("\n" + "=" * 60)
            print("PROCESSING COMPLETE")
            print("=" * 60)
            print(f"Platform: {stats['platform']}")
            print(f"Successful: {stats['successful']}/{stats['total_images']}")
            print(f"Failed: {stats['failed']}/{stats['total_images']}")
            print(f"Total time: {stats['total_time']:.2f}s")
            print(f"Average time per image: {stats['avg_time_per_image']:.3f}s")
            print(f"Results saved to: {args.output}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()