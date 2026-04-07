---
title: Support Ops Environment Server
emoji: 📦
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
---

# Support Ops Environment

This environment simulates real-world retail customer support operations: ticket triage, refunds, and replacement logistics. It is designed for training and evaluating agentic workflows that resemble actual human support tasks.

## Motivation

Customer support teams regularly triage incoming tickets, apply policies, and coordinate refunds or replacements. This environment models those workflows with structured actions, deterministic graders, and partial-reward shaping to encourage correct intermediate steps.

## Action Space

`SupportOpsAction` fields (all typed, Pydantic):

- `action_type` (str): `view_ticket`, `update_ticket`, `add_note`, `issue_refund`, `check_inventory`, `create_replacement`, `schedule_pickup`, `close_ticket`
- `ticket_id` (str, optional)
- `order_id` (str, optional)
- `priority` (str, optional): `low`, `medium`, `high`
- `status` (str, optional): `new`, `triaged`, `in_progress`, `waiting_on_customer`, `closed`
- `tags` (list[str], optional)
- `note` (str, optional)
- `refund_amount` (float, optional)
- `pickup_date` (str, optional, ISO date)
- `replacement_sku` (str, optional)

## Observation Space

`SupportOpsObservation` fields:

- `message` (str): outcome summary
- `task_brief` (str): task instructions for the episode
- `tickets` (list[Ticket])
- `orders` (list[Order])
- `inventory` (list[InventoryItem])
- `progress` (float 0.0–1.0): graded task progress
- `available_actions` (list[str])
- `reward_details` (SupportOpsReward): `progress_delta`, `penalty`, `total_reward`
- `reward` (float), `done` (bool), `metadata` (dict)

## Tasks and Graders

Each task has a deterministic grader that returns a score in `[0.0, 1.0]`.

1. Easy — `triage_packaging`
   - Objective: set priority to medium, add tag `packaging-damage`, mark status `triaged`.

2. Medium — `late_delivery_refund`
   - Objective: refund 20% of item price (cap $25), add apology + refund note, close ticket.

3. Hard — `defective_replacement_pickup`
   - Objective: check inventory, create replacement, schedule pickup within 7 days, add summary note, close ticket.

## Reward Function

- Reward is the **progress delta** between steps.
- Invalid actions incur a small penalty (reward reduction).
- Zero-progress actions incur a smaller penalty to discourage infinite loops.
- Episode ends when progress reaches `1.0` or the max step limit is reached.

## Setup

### Local install

```bash
pip install -e .
```

### Run server locally

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

### Docker

```bash
docker build -t support-ops-env -f server/Dockerfile .
docker run -p 8000:8000 support-ops-env
```

## Baseline (OpenAI API)

The baseline script uses the OpenAI Python client and reads `OPENAI_API_KEY` from your environment:

```bash
pip install -e ".[baseline]"
export OPENAI_API_KEY=...
python baseline_inference.py
```

The script runs all three tasks with a deterministic seed and prints a JSON dict of scores.

## Baseline Scores

Run the baseline script to reproduce scores on your machine and model configuration.

## Project Structure

```
support_ops_env/
├── __init__.py
├── baseline_inference.py
├── client.py
├── models.py
├── openenv.yaml
├── pyproject.toml
└── server/
    ├── __init__.py
    ├── app.py
    ├── support_ops_environment.py
    ├── requirements.txt
    └── Dockerfile
```
