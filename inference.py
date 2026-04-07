import os
from typing import Any, Dict

from baseline_inference import run_task

# Required environment variables (checked by validator)
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-5.2")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")


def _format_kv(payload: Dict[str, Any]) -> str:
    parts = []
    for key, value in payload.items():
        if isinstance(value, float):
            value = round(value, 3)
        parts.append(f"{key}={value}")
    return " ".join(parts)


def _emit(event: str, payload: Dict[str, Any]) -> None:
    print(f"[{event}] {_format_kv(payload)}", flush=True)


def _step_logger(event: str, payload: Dict[str, Any]) -> None:
    if event != "STEP":
        return
    _emit("STEP", payload)


def main() -> None:
    tasks = ["triage_packaging", "late_delivery_refund", "defective_replacement_pickup"]
    for task_id in tasks:
        _emit("START", {"task": task_id})
        try:
            score, steps = run_task(task_id, logger=_step_logger)
            _emit("END", {"task": task_id, "score": score, "steps": steps})
        except Exception as exc:
            _emit("END", {"task": task_id, "score": 0.0, "steps": 0, "error": str(exc)})


if __name__ == "__main__":
    main()
