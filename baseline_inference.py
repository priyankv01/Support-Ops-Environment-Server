from __future__ import annotations

import json
import os
from typing import Any, Dict

from openai import OpenAI

from client import SupportOpsEnv
from models import SupportOpsAction


MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")
BASE_URL = os.getenv("SUPPORT_OPS_BASE_URL", "http://localhost:8000")


SYSTEM_PROMPT = (
    "You are a retail support operations agent. "
    "Choose the next single action to progress the task. "
    "Respond with only a JSON object compatible with SupportOpsAction."
)


def build_prompt(observation: Dict[str, Any]) -> str:
    return (
        f"Task: {observation['task_brief']}\n"
        f"Tickets: {json.dumps(observation['tickets'], ensure_ascii=False)}\n"
        f"Orders: {json.dumps(observation['orders'], ensure_ascii=False)}\n"
        f"Inventory: {json.dumps(observation['inventory'], ensure_ascii=False)}\n"
        f"Available actions: {observation['available_actions']}\n"
        "Return next action JSON."
    )


def parse_action(text: str) -> SupportOpsAction:
    data = json.loads(text)
    return SupportOpsAction(**data)


def run_task(task_id: str) -> float:
    client = OpenAI()
    with SupportOpsEnv(base_url=BASE_URL) as env:
        result = env.reset(task_id=task_id, seed=42)
        total_reward = 0.0
        done = result.done
        observation = result.observation

        while not done:
            prompt = build_prompt(observation.model_dump())
            response = client.responses.create(
                model=MODEL,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            action = parse_action(response.output_text)
            step_result = env.step(action)
            observation = step_result.observation
            total_reward += step_result.reward
            done = step_result.done

        return round(total_reward, 3)


def main() -> None:
    tasks = ["triage_packaging", "late_delivery_refund", "defective_replacement_pickup"]
    scores = {}
    for task_id in tasks:
        scores[task_id] = run_task(task_id)
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
