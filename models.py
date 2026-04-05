# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
Data models for the Support Ops OpenEnv environment.

This environment simulates real-world retail support operations: ticket triage,
refund processing, and replacement logistics.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from openenv.core.env_server.types import Action, Observation, State


class Ticket(BaseModel):
    ticket_id: str
    customer_name: str
    channel: Literal["email", "chat"]
    issue: str
    order_id: Optional[str] = None
    priority: Literal["low", "medium", "high"] = "low"
    status: Literal["new", "triaged", "in_progress", "waiting_on_customer", "closed"] = "new"
    tags: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class Order(BaseModel):
    order_id: str
    item_sku: str
    item_name: str
    item_price: float
    quantity: int
    status: Literal["processing", "shipped", "delivered", "replacement_created"] = "processing"
    shipped_at: Optional[str] = None
    delivery_eta: Optional[str] = None
    delivered_at: Optional[str] = None
    refund_total: float = 0.0


class InventoryItem(BaseModel):
    sku: str
    name: str
    available_qty: int


class SupportOpsAction(Action):
    action_type: Literal[
        "view_ticket",
        "update_ticket",
        "add_note",
        "issue_refund",
        "check_inventory",
        "create_replacement",
        "schedule_pickup",
        "close_ticket",
    ] = Field(..., description="Action to perform in the environment")
    ticket_id: Optional[str] = None
    order_id: Optional[str] = None
    priority: Optional[Literal["low", "medium", "high"]] = None
    status: Optional[Literal["new", "triaged", "in_progress", "waiting_on_customer", "closed"]] = None
    tags: Optional[List[str]] = None
    note: Optional[str] = None
    refund_amount: Optional[float] = None
    pickup_date: Optional[str] = None
    replacement_sku: Optional[str] = None


class SupportOpsReward(BaseModel):
    progress_delta: float = Field(0.0, ge=0.0, le=1.0)
    penalty: float = Field(0.0, ge=0.0, le=1.0)
    total_reward: float = Field(0.0, ge=0.0, le=1.0)


class SupportOpsObservation(Observation):
    message: str = Field(..., description="Human-readable summary of the result")
    task_brief: str = Field(..., description="Task instructions for the current episode")
    tickets: List[Ticket] = Field(default_factory=list)
    orders: List[Order] = Field(default_factory=list)
    inventory: List[InventoryItem] = Field(default_factory=list)
    progress: float = Field(0.0, ge=0.0, le=1.0, description="Progress score for the task")
    available_actions: List[str] = Field(default_factory=list)
    reward_details: SupportOpsReward = Field(default_factory=SupportOpsReward)


class SupportOpsState(State):
    task_id: Optional[str] = None
    difficulty: Optional[str] = None
    progress: float = 0.0
    max_steps: int = 20
