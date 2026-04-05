# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Client for the Support Ops environment."""
from __future__ import annotations

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from support_ops_env.models import SupportOpsAction, SupportOpsObservation, SupportOpsState


class SupportOpsEnv(EnvClient[SupportOpsAction, SupportOpsObservation, SupportOpsState]):
    """WebSocket client for Support Ops environment."""

    def _step_payload(self, action: SupportOpsAction) -> dict:
        return action.model_dump()

    def _parse_result(self, payload: dict) -> StepResult[SupportOpsObservation]:
        obs_payload = payload.get("observation", payload)
        observation = SupportOpsObservation(**obs_payload)
        reward = payload.get("reward", observation.reward)
        done = payload.get("done", observation.done)
        return StepResult(observation=observation, reward=reward, done=done)

    def _parse_state(self, payload: dict) -> SupportOpsState:
        return SupportOpsState(**payload)
