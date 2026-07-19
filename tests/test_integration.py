import pytest
import os
import time
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    """Тест health-эндпоинта."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root():
    """Тест корневого эндпоинта."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "docs" in response.json()


def test_submit_invalid_file():
    """Тест с неподдерживаемым форматом."""
    response = client.post(
        "/ocr/submit",
        files={"file": ("test.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_submit_invalid_engine():
    """Тест с неподдерживаемым движком."""
    # Создаём временное изображение
    test_image_path = "test_image.png"
    try:
        import cv2
        import numpy as np
        img = np.ones((50, 50, 3), dtype=np.uint8) * 255
        cv2.imwrite(test_image_path, img)

        with open(test_image_path, 'rb') as f:
            response = client.post(
                "/ocr/submit",
                files={"file": ("test_image.png", f, "image/png")},
                data={"engine": "invalid"}  # ← как form-data
            )
        assert response.status_code == 400
        assert "Unsupported engine" in response.json()["detail"]
    finally:
        if os.path.exists(test_image_path):
            os.remove(test_image_path)


@pytest.mark.asyncio
async def test_full_flow():
    """Интеграционный тест: submit -> status -> result."""

    test_image_path = "test_image.png"

    try:
        # Создаём тестовое изображение
        import cv2
        import numpy as np
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        cv2.putText(img, 'Test', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imwrite(test_image_path, img)

        # 1. Отправляем изображение
        with open(test_image_path, 'rb') as f:
            response = client.post(
                "/ocr/submit",
                files={"file": ("test_image.png", f, "image/png")}
            )

        assert response.status_code == 200
        data = response.json()
        task_id = data["task_id"]
        assert task_id is not None
        assert data["file_name"] == "test_image.png"
        assert data["status"] == "pending"

        # 2. Ждём завершения
        status = "pending"
        max_wait = 60
        waited = 0

        while status not in ["done", "failed"] and waited < max_wait:
            time.sleep(2)
            waited += 2
            response = client.get(f"/ocr/status/{task_id}")
            assert response.status_code == 200
            status = response.json()["status"]

        assert status == "done", f"Task failed with status: {status}"

        # 3. Получаем результат
        response = client.get(f"/ocr/result/{task_id}")
        assert response.status_code == 200
        result = response.json()

        assert "text" in result
        assert "char_count" in result
        assert result["status"] == "success"
        assert result["engine"] == "tesseract"

    finally:
        # Очистка
        if os.path.exists(test_image_path):
            os.remove(test_image_path)


def test_status_not_found():
    """Тест статуса для несуществующей задачи."""
    response = client.get("/ocr/status/invalid_task_id")
    # Может вернуть 404 или статус PENDING
    assert response.status_code in [200, 404]


def test_result_not_found():
    """Тест результата для несуществующей задачи."""
    response = client.get("/ocr/result/invalid_task_id")
    # Должен вернуть 404 или 500
    assert response.status_code in [404, 500]