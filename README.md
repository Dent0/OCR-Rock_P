# OCR Service

---
Асинхронный микросервис для распознавания текста на изображениях и PDF-документах с использованием OCR-движков Tesseract, EasyOCR и PaddleOCR.

---

## Основные возможности

- **Асинхронная обработка** — через Celery + Redis
- **Поддержка трёх OCR-движков**:
  - Tesseract (лёгкий, стабильный)
  - EasyOCR (высокая точность)
  - PaddleOCR (высокая точность, смешанные языки)
- **Поддержка форматов**: PNG, JPG, JPEG, TIFF, BMP, PDF
- **Мониторинг**: Prometheus + Grafana + Flower
- **Нагрузочное тестирование**: Locust
- **Кроссплатформенность**: Windows, macOS, Linux

---

## Архитектура
```
Пользователь → POST /ocr/submit → FastAPI → Celery → Redis → Worker → OCR
↓
Получает task_id
↓
GET /ocr/status/{task_id} → проверка статуса
↓
GET /ocr/result/{task_id} → получение результата
```
---

## Компоненты

| Компонент | Описание | Порт |
|-----------|----------|------|
| **FastAPI** | REST API сервис | 8000 |
| **Celery Worker** | Выполнение OCR задач | - |
| **Redis** | Брокер сообщений и бекенд | 6379 |
| **Flower** | Мониторинг очереди задач | 5555 |
| **Prometheus** | Сбор метрик | 9090 |
| **Grafana** | Визуализация метрик | 3000 |
| **Node Exporter** | Метрики системы (CPU/RAM) | 9100 |

---

## Установка и запуск

---
### Требования

- Docker и Docker Compose
- Git
- Python 3.11+ (для локального запуска)
---
### Запуск через Docker (рекомендуется)

```bash

# 1. Клонировать репозиторий
git clone https://github.com/Dent0/OCR-Rock_P.git
cd OCR-Rock_P

# 2. Запустить все сервисы
docker-compose up -d --build

# 3. Проверить статус контейнеров
docker-compose ps
```

---
### Запуск без Docker
```bash

# 1. Установить зависимости
pip install -r requirements.txt

# 2. Установить poppler (для PDF)
# macOS:
brew install poppler

# Windows:
pip install poppler-utils

# Linux:
sudo apt-get install poppler-utils

# 3. Запустить Redis (локально)
redis-server

# 4. Запустить воркер
celery -A app.tasks worker --loglevel=info --concurrency=2

# 5. Запустить API
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
---
## API Эндпоинты

---
### POST /ocr/submit

---
Отправить файл на распознавание.

Параметры (form-data):

file — изображение или PDF

engine — tesseract | easy | paddle (по умолчанию tesseract)

**Ответ:**
```json
{
  "task_id": "a1b2c3d4-1234-5678-90ab-cdef12345678",
  "file_name": "document.png",
  "engine": "tesseract",
  "status": "pending"
}
```
---
### GET /ocr/status/{task_id}

---
Проверить статус задачи.

**Ответ:**
```json
{
  "task_id": "a1b2c3d4-1234-5678-90ab-cdef12345678",
  "status": "done",
  "result": "text"
}
```
---
### GET /ocr/result/{task_id}

---
Получить результат распознавания.

**Ответ:**

```json
{
  "status": "success",
  "text": "Распознанный текст...",
  "char_count": 3415,
  "word_count": 470,
  "total_time": 7.058,
  "preprocessing_time": 2.460,
  "ocr_time": 4.597,
  "engine": "tesseract",
  "task_id": "a1b2c3d4-1234-5678-90ab-cdef12345678"
}
```
---
### GET /health

---
Проверка состояния сервиса.
```json
{"status": "ok"}
```
---
## Примеры использования

---
### **Через curl**

```bash

# 1. Отправить изображение
curl -X POST http://localhost:8000/ocr/submit \
  -F "file=@images/test_img/image_ru_mech_1.png" \
  -F "engine=tesseract"

# 2. Получить task_id (например: a1b2c3d4-1234-5678-90ab-cdef12345678)

# 3. Проверить статус
curl http://localhost:8000/ocr/status/a1b2c3d4-1234-5678-90ab-cdef12345678

# 4. Получить результат
curl http://localhost:8000/ocr/result/a1b2c3d4-1234-5678-90ab-cdef12345678
```
---
### Через Swagger

Открыть в браузере: http://localhost:8000/docs

---
## Инструкция по использованию `ocr_batch.py`

---

`ocr_batch.py` — CLI-инструмент для массовой обработки изображений с использованием всех ядер CPU.

---
### Установка зависимостей

```bash

# macOS
brew install poppler
pip install opencv-python numpy pytesseract easyocr paddleocr pdf2image tqdm

# Windows
pip install poppler-utils opencv-python numpy pytesseract easyocr paddleocr pdf2image tqdm

# Linux
sudo apt-get install poppler-utils
pip install opencv-python numpy pytesseract easyocr paddleocr pdf2image tqdm
```

---
### Основные аргументы

| Аргумент | Описание | По умолчанию |
|----------|----------|--------------|
| `-i, --input` | Путь к папке с изображениями | Обязательный |
| `-o, --output` | Путь для сохранения результатов | Обязательный |
| `-w, --workers` | Количество воркеров | 1 |
| `--engine` | OCR движок: `tesseract`, `easy`, `paddle` | `tesseract` |
| `-sp, --save-preprocessed` | Сохранять препроцессированные изображения | `False` |
| `-bm, --benchmark` | Запустить тест производительности | `False` |
| `--no-progress` | Отключить прогресс-бар | `False` |
| `-s, --start` | Начать с N-го изображения | `None` |
| `-e, --end` | Закончить на M-м изображении | `None` |
| `-l, --list` | Файл со списком изображений | `None` |

---
### **Примеры**
```bash

# Базовый запуск
python ocr_batch.py -i ./images -o ./results

# С выбором движка и воркеров
python ocr_batch.py -i ./images -o ./results -e easy -w 4

# Обработка диапазона
python ocr_batch.py -i ./images -o ./results -s 2 -e 5

# Запуск теста производительности
python ocr_batch.py -i ./images -o ./results -bm

# Полная команда
python ocr_batch.py -i ./images -o ./results/full -w 4 -e tesseract -s 2 -e 5 -l ./images_list.txt -sp -bm --no-progress
```

---
### Выходные данные
**Результат для одного изображения:**
```json
{
  "image": "image_001.png",
  "text": "Пример распознанного текста",
  "processing_time": 0.453,
  "preprocessing_time": 0.120,
  "ocr_time": 0.333
}
```

**Сводная статистика:**
```json
{
  "engine": "tesseract",
  "workers": 4,
  "total_images": 64,
  "successful": 64,
  "failed": 0,
  "total_time": 12.34,
  "avg_time_per_image": 0.193,
  "platform": "Darwin"
}
```

---
## Мониторинг

| Сервис | URL | Логин / Пароль |
|--------|-----|----------------|
| Swagger (API документация) | http://localhost:8000/docs | - |
| Flower (Celery мониторинг) | http://localhost:5555 | - |
| Prometheus | http://localhost:9090 | - |
| Grafana | http://localhost:3000 | admin / admin |

---
## Нагрузочное тестирование

```bash

# Установить locust
pip install locust

# Запустить тест
locust -f locustfile.py --host=http://localhost:8000 --web-host=127.0.0.1 --web-port=8089
```

Открыть: http://127.0.0.1:8089

**Профили тестирования:**

| Профиль | Пользователи | Spawn rate | Длительность |
|---------|--------------|------------|--------------|
| 1 | 10 | 2 | 2 минуты     |
| 2 | 50 | 5 | 2 минуты     |
| 3 | 100 | 10 | 2 минуты     |

---
## Тестирование

---
### Запуск интеграционных тестов
```bash

docker-compose exec api pytest tests/test_integration.py -v
```

Ожидаемый результат: 7 тестов пройдено успешно.

---
## Структура проекта (без тестовых частей)
```
OCR-Rock_P/
├── app/                    # FastAPI приложение
│   ├── main.py             # Эндпоинты API
│   ├── tasks.py            # Celery задачи
│   ├── config.py           # Настройки
│   └── ocr_engine.py       # Обёртка над OCR
├── classes/                # OCR движки
│   ├── ocr_tesseract.py
│   ├── ocr_paddle.py
│   └── ocr_easy.py
├── tests/                   # Интеграционные тесты
│   └── test_integration.py
├── images/test_img/         # Тестовые изображения
├── uploads/                 # Загруженные файлы
├── results/                 # Результаты распознавания
├── docker-compose.yml       # Оркестрация контейнеров
├── Dockerfile               # Сборка образа
├── requirements.txt         # Зависимости
├── prometheus.yml           # Конфигурация Prometheus
├── locustfile.py            # Нагрузочный тест
├── ocr_batch.py             # CLI инструмент для батчевой обработки
└── README.md                # Этот файл
```
---
## Управление сервисом

```bash

# Запустить все сервисы
docker-compose up -d

# Остановить все сервисы
docker-compose down

# Остановить и удалить все данные (volumes)
docker-compose down -v

# Посмотреть логи API
docker-compose logs api -f

# Посмотреть логи воркера
docker-compose logs worker -f

# Перезапустить воркер
docker-compose restart worker

# Очистить очередь задач
docker-compose exec redis redis-cli FLUSHALL
```
---
## Очистка

```bash

# Очистка папки uploads
rm -rf uploads/* && touch uploads/.gitkeep

# Очистка папки results
rm -rf results/* && touch results/.gitkeep

# Полная очистка системы (включая образы)
docker-compose down -v && docker system prune -f
```
---
## Результаты тестирования

### Интеграционные тесты

| Тест | Результат |
|------|-----------|
| test_health | ✅ PASSED |
| test_root | ✅ PASSED |
| test_submit_invalid_file | ✅ PASSED |
| test_submit_invalid_engine | ✅ PASSED |
| test_full_flow | ✅ PASSED |
| test_status_not_found | ✅ PASSED |
| test_result_not_found | ✅ PASSED |

---
### Нагрузочное тестирование (Locust)

| Профиль | Users | RPS | P50 (мс) | P95 (мс) | Queue Size | CPU (%) | RAM (%) |
|---------|-------|-----|----------|----------|------------|---------|---------|
| 1 | 10 | 4.7 | 8 | 74 | 22 | 55 | +5-6% |
| 2 | 50 | 24.7 | 11 | 130 | 186 | 55 | +5-6% |
| 3 | 100 | 49.8 | 20 | 230 | 386 | 55 | +5-6% |

**Точка насыщения:** 50-80 пользователей  
**Узкое место:** Скорость обработки задач воркером (I/O, OCR)
---
## Лицензия

---
MIT

---

## Контакты

GitHub: [https://github.com/Dent0/OCR-Rock_P](https://github.com/Dent0/OCR-Rock_P)