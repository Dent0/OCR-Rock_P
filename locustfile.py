import time
import os
from locust import HttpUser, task, between


class OCRUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        # Используем PDF из папки images
        self.pdf_path = "images/locust_doc.pdf"

    @task
    def submit_document(self):
        if not os.path.exists(self.pdf_path):
            print(f"PDF не найден: {self.pdf_path}")
            return

        with open(self.pdf_path, "rb") as f:
            response = self.client.post(
                "/ocr/submit",
                files={"file": ("test.pdf", f, "application/pdf")},
                data={"engine": "tesseract"}
            )

        if response.status_code == 200:
            task_id = response.json()["task_id"]
            status = "pending"
            waited = 0

            while status not in ["done", "failed"] and waited < 30:
                time.sleep(2)
                waited += 2
                status_response = self.client.get(f"/ocr/status/{task_id}")
                if status_response.status_code == 200:
                    status = status_response.json().get("status", "pending")

            if status == "done":
                self.client.get(f"/ocr/result/{task_id}")