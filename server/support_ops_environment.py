# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Support Ops Environment implementation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

import random

from openenv.core.env_server.interfaces import Environment

from models import (
    InventoryItem,
    Order,
    SupportOpsAction,
    SupportOpsObservation,
    SupportOpsReward,
    SupportOpsState,
    Ticket,
)


@dataclass
class TaskSpec:
    task_id: str
    difficulty: str
    brief: str
    ticket_id: str
    order_id: Optional[str]
    grader: Callable[["SupportOpsEnvironment"], float]


class SupportOpsEnvironment(Environment[SupportOpsAction, SupportOpsObservation, SupportOpsState]):
    """Retail Support Ops environment with triage, refunds, and replacements."""

    def __init__(self) -> None:
        self._state = SupportOpsState(episode_id=str(uuid4()), step_count=0)
        self._rng = random.Random()
        self._task: Optional[TaskSpec] = None
        self._tickets: Dict[str, Ticket] = {}
        self._orders: Dict[str, Order] = {}
        self._inventory: Dict[str, InventoryItem] = {}
        self._progress = 0.0
        self._max_steps = 20
        self._flags: Dict[str, Any] = {}

        self._tasks = self._build_tasks()

    def _build_tasks(self) -> List[TaskSpec]:
        return [
            TaskSpec(
                task_id="triage_packaging",
                difficulty="easy",
                brief=(
                    "Ticket T-1001 reports damaged packaging on delivery. "
                    "Triage the ticket: set priority to medium, add tag "
                    "'packaging-damage', and mark status as triaged."
                ),
                ticket_id="T-1001",
                order_id="O-5001",
                grader=self._grade_task_easy,
            ),
            TaskSpec(
                task_id="late_delivery_refund",
                difficulty="medium",
                brief=(
                    "Ticket T-2001 reports a delivery 5 days late. "
                    "Policy: refund 20% of item price (cap $25) and close the ticket. "
                    "Add a customer note that includes an apology and refund confirmation."
                ),
                ticket_id="T-2001",
                order_id="O-6001",
                grader=self._grade_task_medium,
            ),
            TaskSpec(
                task_id="defective_replacement_pickup",
                difficulty="hard",
                brief=(
                    "Ticket T-3001 reports a defective blender. "
                    "Check inventory, create a replacement order, schedule a pickup "
                    "within 7 days, leave a summary note, and close the ticket."
                ),
                ticket_id="T-3001",
                order_id="O-7001",
                grader=self._grade_task_hard,
            ),
        ]

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        difficulty: Optional[str] = None,
        **kwargs: Any,
    ) -> SupportOpsObservation:
        if seed is not None:
            self._rng.seed(seed)

        self._state = SupportOpsState(episode_id=episode_id or str(uuid4()), step_count=0)
        self._progress = 0.0
        self._flags = {
            "inventory_checked": False,
            "pickup_scheduled": False,
            "replacement_created": False,
        }

        candidates = self._tasks
        if difficulty:
            candidates = [t for t in candidates if t.difficulty == difficulty]
        if task_id:
            candidates = [t for t in candidates if t.task_id == task_id]
        self._task = self._rng.choice(candidates) if candidates else self._tasks[0]

        self._load_task_data(self._task)
        self._state.task_id = self._task.task_id
        self._state.difficulty = self._task.difficulty
        self._state.progress = self._progress
        self._state.max_steps = self._max_steps

        return self._build_observation(
            message="Environment reset. Review the task brief and take actions.",
            reward=0.0,
            done=False,
        )

    def _load_task_data(self, task: TaskSpec) -> None:
        self._tickets = {}
        self._orders = {}
        self._inventory = {}

        if task.task_id == "triage_packaging":
            self._tickets[task.ticket_id] = Ticket(
                ticket_id=task.ticket_id,
                customer_name="Lena Ortiz",
                channel="email",
                issue="Box arrived crushed; items intact but packaging damaged.",
                order_id=task.order_id,
                priority="low",
                status="new",
                tags=[],
            )
            self._orders[task.order_id] = Order(
                order_id=task.order_id,
                item_sku="PACK-12",
                item_name="Ceramic Dinnerware Set",
                item_price=72.0,
                quantity=1,
                status="delivered",
                shipped_at="2026-03-10",
                delivery_eta="2026-03-14",
                delivered_at="2026-03-14",
            )
        elif task.task_id == "late_delivery_refund":
            self._tickets[task.ticket_id] = Ticket(
                ticket_id=task.ticket_id,
                customer_name="Avery Shah",
                channel="chat",
                issue="Shipment arrived late; requesting partial refund.",
                order_id=task.order_id,
                priority="medium",
                status="in_progress",
                tags=["late-delivery"],
            )
            self._orders[task.order_id] = Order(
                order_id=task.order_id,
                item_sku="CHAIR-44",
                item_name="Ergo Office Chair",
                item_price=80.0,
                quantity=1,
                status="delivered",
                shipped_at="2026-03-11",
                delivery_eta="2026-03-15",
                delivered_at="2026-03-20",
            )
        elif task.task_id == "defective_replacement_pickup":
            self._tickets[task.ticket_id] = Ticket(
                ticket_id=task.ticket_id,
                customer_name="Nia Campbell",
                channel="email",
                issue="Motor failed after 2 uses; wants replacement and pickup.",
                order_id=task.order_id,
                priority="high",
                status="in_progress",
                tags=["defective"],
            )
            self._orders[task.order_id] = Order(
                order_id=task.order_id,
                item_sku="BLND-200",
                item_name="PulsePro Blender",
                item_price=140.0,
                quantity=1,
                status="delivered",
                shipped_at="2026-03-08",
                delivery_eta="2026-03-12",
                delivered_at="2026-03-12",
            )
            self._inventory["BLND-200"] = InventoryItem(
                sku="BLND-200",
                name="PulsePro Blender",
                available_qty=4,
            )

    def step(self, action: SupportOpsAction, **kwargs: Any) -> SupportOpsObservation:
        self._state.step_count += 1

        message, valid_action = self._apply_action(action)
        previous_progress = self._progress
        self._progress = self._task.grader(self) if self._task else 0.0
        progress_delta = max(0.0, self._progress - previous_progress)
        penalty = 0.05 if not valid_action else (0.01 if progress_delta == 0.0 else 0.0)
        reward = max(0.0, progress_delta - penalty)
        done = self._progress >= 1.0 or self._state.step_count >= self._max_steps

        if done and self._progress < 1.0:
            message += " Episode ended: max steps reached."

        self._state.progress = self._progress

        return self._build_observation(
            message=message,
            reward=reward,
            done=done,
            progress_delta=progress_delta,
            penalty=penalty,
        )

    def _apply_action(self, action: SupportOpsAction) -> tuple[str, bool]:
        if self._task is None:
            return "No active task. Call reset() first.", False

        action_type = action.action_type
        if action_type == "view_ticket":
            ticket = self._tickets.get(action.ticket_id or self._task.ticket_id)
            if not ticket:
                return "Ticket not found.", False
            return f"Viewed ticket {ticket.ticket_id}.", True

        if action_type == "update_ticket":
            ticket = self._tickets.get(action.ticket_id or self._task.ticket_id)
            if not ticket:
                return "Ticket not found.", False
            if action.priority:
                ticket.priority = action.priority
            if action.status:
                ticket.status = action.status
            if action.tags:
                for tag in action.tags:
                    if tag not in ticket.tags:
                        ticket.tags.append(tag)
            return f"Updated ticket {ticket.ticket_id}.", True

        if action_type == "add_note":
            ticket = self._tickets.get(action.ticket_id or self._task.ticket_id)
            if not ticket:
                return "Ticket not found.", False
            if not action.note:
                return "Note text required.", False
            ticket.notes.append(action.note)
            return f"Added note to ticket {ticket.ticket_id}.", True

        if action_type == "issue_refund":
            if not action.order_id:
                return "order_id required to issue refund.", False
            order = self._orders.get(action.order_id)
            if not order:
                return "Order not found.", False
            if action.refund_amount is None:
                return "refund_amount required.", False
            order.refund_total += float(action.refund_amount)
            return (
                f"Issued refund of ${action.refund_amount:.2f} for order {order.order_id}.",
                True,
            )

        if action_type == "check_inventory":
            sku = action.replacement_sku or (self._task.order_id and self._orders[self._task.order_id].item_sku)
            item = self._inventory.get(sku) if sku else None
            self._flags["inventory_checked"] = True
            if not item:
                return "Inventory item not found.", False
            return f"Inventory for {item.sku}: {item.available_qty} available.", True

        if action_type == "create_replacement":
            if not action.order_id:
                return "order_id required to create replacement.", False
            order = self._orders.get(action.order_id)
            if not order:
                return "Order not found.", False
            sku = action.replacement_sku or order.item_sku
            item = self._inventory.get(sku)
            if not item or item.available_qty <= 0:
                return f"Replacement SKU {sku} unavailable.", False
            item.available_qty -= 1
            order.status = "replacement_created"
            self._flags["replacement_created"] = True
            return f"Replacement created for order {order.order_id}.", True

        if action_type == "schedule_pickup":
            if not action.pickup_date:
                return "pickup_date required.", False
            self._flags["pickup_scheduled"] = True
            self._flags["pickup_date"] = action.pickup_date
            return f"Pickup scheduled for {action.pickup_date}.", True

        if action_type == "close_ticket":
            ticket = self._tickets.get(action.ticket_id or self._task.ticket_id)
            if not ticket:
                return "Ticket not found.", False
            ticket.status = "closed"
            return f"Closed ticket {ticket.ticket_id}.", True

        return f"Unknown action type: {action_type}", False

    def _build_observation(
        self,
        message: str,
        reward: float,
        done: bool,
        progress_delta: float = 0.0,
        penalty: float = 0.0,
    ) -> SupportOpsObservation:
        reward_details = SupportOpsReward(
            progress_delta=progress_delta,
            penalty=penalty,
            total_reward=reward,
        )
        return SupportOpsObservation(
            message=message,
            task_brief=self._task.brief if self._task else "",
            tickets=list(self._tickets.values()),
            orders=list(self._orders.values()),
            inventory=list(self._inventory.values()),
            progress=self._progress,
            available_actions=[
                "view_ticket",
                "update_ticket",
                "add_note",
                "issue_refund",
                "check_inventory",
                "create_replacement",
                "schedule_pickup",
                "close_ticket",
            ],
            reward_details=reward_details,
            reward=reward,
            done=done,
            metadata={
                "task_id": self._task.task_id if self._task else None,
                "difficulty": self._task.difficulty if self._task else None,
                "step_count": self._state.step_count,
                "max_steps": self._max_steps,
            },
        )

    def _grade_task_easy(self) -> float:
        ticket = self._tickets.get("T-1001")
        if not ticket:
            return 0.0
        score = 0.0
        score += 1.0 / 3.0 if ticket.priority == "medium" else 0.0
        score += 1.0 / 3.0 if "packaging-damage" in ticket.tags else 0.0
        score += 1.0 / 3.0 if ticket.status == "triaged" else 0.0
        return round(score, 3)

    def _grade_task_medium(self) -> float:
        ticket = self._tickets.get("T-2001")
        order = self._orders.get("O-6001")
        if not ticket or not order:
            return 0.0
        expected_refund = min(order.item_price * 0.20, 25.0)
        refund_correct = abs(order.refund_total - expected_refund) <= 0.5
        note_ok = any(
            "sorry" in note.lower() or "apolog" in note.lower()
            for note in ticket.notes
        ) and any("refund" in note.lower() for note in ticket.notes)
        closed = ticket.status == "closed"

        score = 0.0
        score += 0.4 if refund_correct else 0.0
        score += 0.3 if note_ok else 0.0
        score += 0.3 if closed else 0.0
        return round(score, 3)

    def _grade_task_hard(self) -> float:
        ticket = self._tickets.get("T-3001")
        order = self._orders.get("O-7001")
        if not ticket or not order:
            return 0.0
        score = 0.0
        score += 0.2 if self._flags.get("inventory_checked") else 0.0
        score += 0.2 if self._flags.get("replacement_created") else 0.0
        score += 0.2 if self._flags.get("pickup_scheduled") else 0.0
        score += 0.2 if any(ticket.notes) else 0.0
        score += 0.2 if ticket.status == "closed" else 0.0
        return round(score, 3)

    @property
    def state(self) -> SupportOpsState:
        return self._state


if __name__ == "__main__":
    env = SupportOpsEnvironment()
    obs = env.reset(seed=42)
    print(obs.task_brief)
