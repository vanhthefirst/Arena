"""Pedestrian group ids read from the engine's InteractionFrame."""

from __future__ import annotations

import pytest

pytest.importorskip("rclpy")

from arena_humansim.core.interaction_kinds import InteractionType, MembershipRole
from arena_humansim.utils.types import InteractionOutcome
from arena_humansim_msgs.msg import AgentFrame, InteractionFrame
from task_generator.manager.realizer import Realizer
from task_generator.simulators.human.arena_humansim.arena_humansim import ArenaHumanSimulator, group_ids

ACTIVE, FORMING = InteractionOutcome.ACTIVE, InteractionOutcome.FORMING
PARTICIPANT, QUEUED = MembershipRole.PARTICIPANT, MembershipRole.QUEUED


def _frame(*interactions: tuple[int, InteractionType, InteractionOutcome, list[tuple[int, MembershipRole]]]) -> InteractionFrame:
    offsets, ids, roles = [0], [], []
    for _, _, _, members in interactions:
        ids += [member for member, _ in members]
        roles += [role for _, role in members]
        offsets.append(len(ids))
    return InteractionFrame(
        interaction_id=[i for i, _, _, _ in interactions],
        interaction_type=[t for _, t, _, _ in interactions],
        outcome=[o for _, _, o, _ in interactions],
        member_offset=offsets,
        member_id=ids,
        member_role=roles,
    )


FRAME = _frame(
    (3, InteractionType.GROUP_WALK, ACTIVE, [(1, PARTICIPANT), (2, PARTICIPANT), (3, PARTICIPANT)]),
    (5, InteractionType.GROUP_CONVERSATION, ACTIVE, [(2, PARTICIPANT), (4, PARTICIPANT)]),
    (7, InteractionType.GROUP_WALK, FORMING, [(6, PARTICIPANT)]),
    (8, InteractionType.SERVICE, ACTIVE, [(9, PARTICIPANT)]),
    (9, InteractionType.TALK_TO, ACTIVE, [(10, PARTICIPANT), (11, PARTICIPANT), (12, QUEUED)]),
)


def test_active_group_participants_get_their_lowest_group_id() -> None:
    assert group_ids(FRAME) == {1: 3, 2: 3, 3: 3, 4: 5, 10: 9, 11: 9}


def test_forming_queued_and_non_group_interactions_give_no_group() -> None:
    groups = group_ids(FRAME)
    assert 6 not in groups
    assert 9 not in groups
    assert 12 not in groups
    assert group_ids(InteractionFrame()) == {}


def test_published_pedestrians_carry_the_group_of_their_engine_agent() -> None:
    adapter = object.__new__(ArenaHumanSimulator)
    adapter._agent_gestures = {}
    adapter._group_ids = {}
    adapter._agent_names = {}
    adapter._agent_types = {}
    adapter._agent_type_paths = {}
    adapter._realizer = Realizer(Realizer._Configuration(x=0.0, y=0.0, prefix="env_0"))
    adapter._interactions_callback(FRAME)
    agents = AgentFrame(agent_id=[1, 4, 6], x=[0.0, 1.0, 2.0], y=[0.0, 0.0, 0.0], theta=[0.0] * 3, vx=[0.0] * 3, vy=[0.0] * 3, animation_state=[0] * 3)

    peds = adapter._agent_states_to_pedestrians(agents)

    assert [(p.id, p.group_id) for p in peds.pedestrians] == [(1, 3), (4, 5), (6, -1)]
