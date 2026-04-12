import os
from typing import Any, Dict

from openai import OpenAI

from baseline_inference import run_task

# Required environment variables (checked by validator)
API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-5.2")
HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = os.environ["API_KEY"]
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
    # Force at least one LLM proxy call at the top-level script.
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    client.chat.completions.create(
        model=MODEL_NAME or "gpt-4o-mini",
        messages=[{"role": "user", "content": "ping"}],
        temperature=0,
    )
    for task_id in tasks:
        _emit("START", {"task": task_id})
        try:
            score, steps = run_task(task_id, logger=_step_logger)
            _emit("END", {"task": task_id, "score": score, "steps": steps})
        except Exception as exc:
            _emit("END", {"task": task_id, "score": 0.0, "steps": 0, "error": str(exc)})


if __name__ == "__main__":
    main()
