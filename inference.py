import json
import os
from typing import Any, Dict

from baseline_inference import run_task

# Required environment variables (checked by validator)
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-5.2")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")


def _logger(event: str, payload: Dict[str, Any]) -> None:
    print(f"{event} {json.dumps(payload, ensure_ascii=False)}")


def main() -> None:
    tasks = ["triage_packaging", "late_delivery_refund", "defective_replacement_pickup"]
    scores: Dict[str, float] = {}
    for task_id in tasks:
        scores[task_id] = run_task(task_id, logger=_logger)
    print(f"END {json.dumps({'scores': scores}, ensure_ascii=False)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"END {json.dumps({'error': str(exc)})}")
