"""
第四阶段：任务序列版 MCHA 重分配测试。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from config.params import WEIGHTS, MCHA
from src.core.models import AssignmentPlan, Battlefield, Target, TaskNode, Threat, UAV, UavTaskSequence
from src.core.sequence_eval import evaluate_uav_task_sequence
from src.re_allocation.events import Event, EventType, analyze_plan_event_impact, apply_event_to_battlefield
from src.re_allocation.mcha import run_mcha_for_plan


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def build_plan_battlefield() -> Battlefield:
    uavs = [
        UAV(id=0, x=0.0, y=0.0, speed=100.0, ammo=2, range_left=300.0),
        UAV(id=1, x=0.0, y=30.0, speed=100.0, ammo=2, range_left=300.0),
        UAV(id=2, x=0.0, y=70.0, speed=100.0, ammo=2, range_left=300.0),
    ]
    targets = [
        Target(id=0, x=60.0, y=0.0, value=8.0, required_uavs=1),
        Target(id=1, x=70.0, y=30.0, value=10.0, required_uavs=1),
        Target(id=2, x=60.0, y=70.0, value=7.0, required_uavs=1),
    ]
    threats: list[Threat] = []
    return Battlefield(uavs=uavs, targets=targets, threats=threats, map_size=(100.0, 100.0))


def build_initial_plan() -> AssignmentPlan:
    return AssignmentPlan(
        uav_task_sequences={
            0: UavTaskSequence(
                uav_id=0,
                tasks=[
                    TaskNode(target_id=0, order=0),
                    TaskNode(target_id=1, order=1),
                ],
            ),
            1: UavTaskSequence(
                uav_id=1,
                tasks=[
                    TaskNode(target_id=2, order=0),
                ],
            ),
            2: UavTaskSequence(uav_id=2, tasks=[]),
        },
        target_assignees={
            0: [0],
            1: [0],
            2: [1],
        },
        total_cost=0.0,
    )


def assert_plan_feasible(battlefield: Battlefield, plan: AssignmentPlan) -> None:
    for uav in battlefield.uavs:
        sequence = plan.uav_task_sequences[uav.id]
        evaluated = evaluate_uav_task_sequence(battlefield, sequence)
        assert_true(evaluated.is_feasible, f'UAV-{uav.id} 任务序列应满足 ammo 和 range 约束')


def test_target_demand_increase_appends_missing_task_to_sequence_plan():
    battlefield = build_plan_battlefield()
    plan = build_initial_plan()
    event = Event(
        type=EventType.TARGET_DEMAND_INCREASED,
        data={
            'target_id': 1,
            'new_required_uavs': 2,
        },
    )

    apply_event_to_battlefield(event, battlefield)
    state = analyze_plan_event_impact(event, battlefield, plan)
    result = run_mcha_for_plan(battlefield, WEIGHTS, state, MCHA)

    updated_plan = result.assignment_plan
    assert_true(state.open_targets == [1], '需求增加后开放目标应为目标1')
    assert_true(state.remaining_demand == {1: 1}, '目标1应只需要补充1架 UAV')
    assert_true(result.remaining_demand[1] == 0, 'MCHA 应补齐目标1的新增需求')

    target_1_assignees = updated_plan.target_assignees[1]
    assert_true(len(target_1_assignees) == 2, '目标1应由两架 UAV 执行')
    assert_true(0 in target_1_assignees, '目标1原有执行 UAV 应保留')
    assert_true(len(set(target_1_assignees)) == 2, '同一目标不应重复分给同一 UAV')

    assert_true(updated_plan.uav_task_sequences[0].target_ids() == [0, 1], '原 UAV-0 任务链应保持不变')
    new_uav_ids = [uav_id for uav_id in target_1_assignees if uav_id != 0]
    assert_true(len(new_uav_ids) == 1, '应找到一个新的 UAV 补充目标1')
    new_sequence = updated_plan.uav_task_sequences[new_uav_ids[0]]
    assert_true(new_sequence.target_ids()[-1] == 1, '新增任务应追加到补位 UAV 的任务链尾部')

    assert_plan_feasible(battlefield, updated_plan)
    assignment = updated_plan.to_assignment_matrix(num_uavs=3, num_targets=3)
    assert_true(int(np.sum(assignment[:, 1])) == 2, '兼容矩阵中目标1也应满足新增需求')


TEST_CASES = [
    test_target_demand_increase_appends_missing_task_to_sequence_plan,
]


if __name__ == '__main__':
    for test_case in TEST_CASES:
        test_case()
    print(f'通过 {len(TEST_CASES)} 项任务序列版 MCHA 测试')
