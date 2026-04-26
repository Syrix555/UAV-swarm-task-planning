"""
第四阶段：任务序列版 MCHA 重分配测试。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from config.params import WEIGHTS, MCHA, MCHA_TEST
from data.scenario_reallocation import create_reallocation_scenario
from src.core.models import AssignmentPlan, Battlefield, Target, TaskNode, Threat, UAV, UavTaskSequence
from src.core.sequence_eval import evaluate_uav_task_sequence
from src.pre_allocation.pso import run_pso
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


def assert_all_targets_satisfied(battlefield: Battlefield, plan: AssignmentPlan) -> None:
    for target in battlefield.targets:
        assigned_count = len(plan.target_assignees.get(target.id, []))
        assert_true(
            assigned_count >= target.required_uavs,
            f'目标{target.id} 应满足需求: assigned={assigned_count}, required={target.required_uavs}',
        )


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


def test_uav_lost_releases_whole_sequence_and_repairs_open_tasks():
    battlefield = build_plan_battlefield()
    plan = build_initial_plan()
    event = Event(
        type=EventType.UAV_LOST,
        data={
            'uav_id': 0,
        },
    )

    state = analyze_plan_event_impact(event, battlefield, plan)
    result = run_mcha_for_plan(battlefield, WEIGHTS, state, MCHA)

    updated_plan = result.assignment_plan
    assert_true(state.open_targets == [0, 1], 'UAV 损失后应释放其整条任务链')
    assert_true(state.remaining_demand == {0: 1, 1: 1}, '释放目标应重新产生需求缺口')
    assert_true(0 not in state.available_uavs, '损失 UAV 不应再参与重分配')
    assert_true(updated_plan.uav_task_sequences[0].task_count() == 0, '损失 UAV 的任务链应被清空')
    assert_true(result.remaining_demand[0] == 0 and result.remaining_demand[1] == 0, 'MCHA 应补齐释放任务')

    for target_id in [0, 1, 2]:
        required = battlefield.get_target(target_id).required_uavs
        assigned_count = len(updated_plan.target_assignees.get(target_id, []))
        assert_true(assigned_count == required, f'目标{target_id} 应满足需求')

    for target_id in [0, 1]:
        assert_true(0 not in updated_plan.target_assignees[target_id], '损失 UAV 不应出现在目标执行者中')

    assert_plan_feasible(battlefield, updated_plan)
    assignment = updated_plan.to_assignment_matrix(num_uavs=3, num_targets=3)
    assert_true(int(np.sum(assignment[0])) == 0, '兼容矩阵中损失 UAV 也应无任务')


def test_target_added_appends_new_target_to_available_uav_sequence():
    battlefield = build_plan_battlefield()
    plan = build_initial_plan()
    new_target = Target(id=3, x=80.0, y=72.0, value=9.5, required_uavs=1)
    event = Event(
        type=EventType.TARGET_ADDED,
        data={
            'target': new_target,
        },
    )

    apply_event_to_battlefield(event, battlefield)
    state = analyze_plan_event_impact(event, battlefield, plan)
    result = run_mcha_for_plan(battlefield, WEIGHTS, state, MCHA)

    updated_plan = result.assignment_plan
    assert_true(state.open_targets == [3], '新增目标应进入开放任务集合')
    assert_true(state.remaining_demand == {3: 1}, '新增目标应产生对应需求')
    assert_true(result.remaining_demand[3] == 0, 'MCHA 应补齐新增目标需求')
    assert_true(len(updated_plan.target_assignees[3]) == 1, '新增目标应分配给1架 UAV')

    new_uav_id = updated_plan.target_assignees[3][0]
    assert_true(updated_plan.uav_task_sequences[new_uav_id].target_ids()[-1] == 3, '新增目标应追加到任务链尾部')
    assert_true(updated_plan.uav_task_sequences[0].target_ids() == [0, 1], '原有任务链应保持不变')

    assert_plan_feasible(battlefield, updated_plan)
    assignment = updated_plan.to_assignment_matrix(num_uavs=3, num_targets=4)
    assert_true(assignment.shape == (3, 4), '兼容矩阵应扩展到新增目标列')
    assert_true(int(np.sum(assignment[:, 3])) == 1, '兼容矩阵中新增目标应满足需求')


def test_target_added_supports_required_uavs_greater_than_one():
    battlefield = build_plan_battlefield()
    plan = build_initial_plan()
    new_target = Target(id=3, x=80.0, y=72.0, value=9.5, required_uavs=2)
    event = Event(
        type=EventType.TARGET_ADDED,
        data={
            'target': new_target,
        },
    )

    apply_event_to_battlefield(event, battlefield)
    state = analyze_plan_event_impact(event, battlefield, plan)
    result = run_mcha_for_plan(battlefield, WEIGHTS, state, MCHA)

    updated_plan = result.assignment_plan
    target_assignees = updated_plan.target_assignees[3]
    assert_true(state.remaining_demand == {3: 2}, '新增目标 required_uavs=2 时应产生两个需求槽位')
    assert_true(result.remaining_demand[3] == 0, 'MCHA 应补齐新增目标多 UAV 需求')
    assert_true(len(target_assignees) == 2, '新增目标应分配给两架 UAV')
    assert_true(len(set(target_assignees)) == 2, '新增目标不应重复分给同一 UAV')

    for uav_id in target_assignees:
        assert_true(updated_plan.uav_task_sequences[uav_id].target_ids()[-1] == 3, '新增目标应追加到每个补位 UAV 的任务链尾部')

    assert_plan_feasible(battlefield, updated_plan)
    assignment = updated_plan.to_assignment_matrix(num_uavs=3, num_targets=4)
    assert_true(int(np.sum(assignment[:, 3])) == 2, '兼容矩阵中新增目标应满足 required_uavs=2')


def test_threat_added_releases_sequence_suffix_and_repairs_open_tasks():
    battlefield = build_plan_battlefield()
    plan = build_initial_plan()
    event = Event(
        type=EventType.THREAT_ADDED,
        data={
            'threat': Threat(id=0, x=65.0, y=15.0, radius=15.0),
            'threat_threshold': 1.0,
        },
    )

    apply_event_to_battlefield(event, battlefield)
    state = analyze_plan_event_impact(event, battlefield, plan)
    result = run_mcha_for_plan(battlefield, WEIGHTS, state, MCHA)

    updated_plan = result.assignment_plan
    assert_true(state.locked_plan.uav_task_sequences[0].target_ids() == [0], '中间航段受威胁时应保留安全前缀')
    assert_true(state.open_targets == [1], '受威胁后缀任务应进入开放任务集合')
    assert_true(state.remaining_demand == {1: 1}, '释放目标1后应产生1个需求缺口')
    assert_true((0, 1) in (state.forbidden_pairs or []), '受威胁释放的原 UAV-目标组合应加入禁忌集合')
    assert_true(result.remaining_demand[1] == 0, 'MCHA 应补齐受威胁释放任务')
    assert_true(0 not in updated_plan.target_assignees[1], '受威胁释放的任务不应直接分回原 UAV')
    assert_true(len(updated_plan.target_assignees[1]) == 1, '目标1应重新分配给1架 UAV')

    assert_plan_feasible(battlefield, updated_plan)
    assignment = updated_plan.to_assignment_matrix(num_uavs=3, num_targets=3)
    assert_true(int(np.sum(assignment[:, 1])) == 1, '兼容矩阵中目标1应满足需求')


def test_threat_added_releases_whole_sequence_when_first_segment_affected():
    battlefield = build_plan_battlefield()
    plan = build_initial_plan()
    event = Event(
        type=EventType.THREAT_ADDED,
        data={
            'threat': Threat(id=0, x=30.0, y=0.0, radius=15.0),
            'threat_threshold': 1.0,
        },
    )

    apply_event_to_battlefield(event, battlefield)
    state = analyze_plan_event_impact(event, battlefield, plan)
    result = run_mcha_for_plan(battlefield, WEIGHTS, state, MCHA)

    updated_plan = result.assignment_plan
    assert_true(state.locked_plan.uav_task_sequences[0].target_ids() == [], '首段受威胁时应释放整条任务链')
    assert_true(state.open_targets == [0, 1], '整链释放后应开放原链上的全部目标')
    assert_true(state.remaining_demand == {0: 1, 1: 1}, '整链释放后两个目标都应产生需求缺口')
    assert_true((0, 0) in (state.forbidden_pairs or []), '目标0的原受威胁分配应加入禁忌集合')
    assert_true((0, 1) in (state.forbidden_pairs or []), '目标1的原受威胁分配应加入禁忌集合')
    assert_true(result.remaining_demand[0] == 0 and result.remaining_demand[1] == 0, 'MCHA 应补齐整链释放任务')

    for target_id in [0, 1]:
        assert_true(0 not in updated_plan.target_assignees[target_id], '受威胁释放目标不应分回原 UAV')
        assert_true(len(updated_plan.target_assignees[target_id]) == 1, f'目标{target_id} 应重新分配给1架 UAV')

    assert_plan_feasible(battlefield, updated_plan)


def test_threat_added_noop_when_no_segment_exceeds_threshold():
    battlefield = build_plan_battlefield()
    plan = build_initial_plan()
    event = Event(
        type=EventType.THREAT_ADDED,
        data={
            'threat': Threat(id=0, x=95.0, y=95.0, radius=5.0),
            'threat_threshold': 1.0,
        },
    )

    apply_event_to_battlefield(event, battlefield)
    state = analyze_plan_event_impact(event, battlefield, plan)

    assert_true(state.locked_plan.uav_task_sequences[0].target_ids() == [0, 1], '无显著威胁时原 UAV-0 任务链应保持不变')
    assert_true(state.locked_plan.uav_task_sequences[1].target_ids() == [2], '无显著威胁时原 UAV-1 任务链应保持不变')
    assert_true(state.open_targets == [], '无显著威胁时不应开放任务')
    assert_true(state.remaining_demand == {}, '无显著威胁时不应产生需求缺口')
    assert_true(state.forbidden_pairs == [], '无释放任务时禁忌集合应为空')


def test_reallocation_scenario_threat_added_keeps_all_targets_satisfied():
    battlefield = create_reallocation_scenario()
    np.random.seed(7)
    _, _, _, plan = run_pso(
        battlefield,
        WEIGHTS,
        return_assignment_plan=True,
    )
    event = Event(
        type=EventType.THREAT_ADDED,
        data={
            'threat': Threat(
                id=MCHA_TEST['new_threat_id'],
                x=MCHA_TEST['new_threat_x'],
                y=MCHA_TEST['new_threat_y'],
                radius=MCHA_TEST['new_threat_radius'],
            ),
            'threat_threshold': MCHA_TEST['threat_threshold'],
        },
    )

    apply_event_to_battlefield(event, battlefield)
    state = analyze_plan_event_impact(event, battlefield, plan)
    result = run_mcha_for_plan(battlefield, WEIGHTS, state, MCHA)

    assert_true(len(state.open_targets) > 0, '重分配场景中的新增威胁应触发任务释放')
    assert_true(all(value == 0 for value in result.remaining_demand.values()), '新增威胁释放任务应全部被补齐')
    assert_all_targets_satisfied(battlefield, result.assignment_plan)
    assert_plan_feasible(battlefield, result.assignment_plan)


TEST_CASES = [
    test_target_demand_increase_appends_missing_task_to_sequence_plan,
    test_uav_lost_releases_whole_sequence_and_repairs_open_tasks,
    test_target_added_appends_new_target_to_available_uav_sequence,
    test_target_added_supports_required_uavs_greater_than_one,
    test_threat_added_releases_sequence_suffix_and_repairs_open_tasks,
    test_threat_added_releases_whole_sequence_when_first_segment_affected,
    test_threat_added_noop_when_no_segment_exceeds_threshold,
    test_reallocation_scenario_threat_added_keeps_all_targets_satisfied,
]


if __name__ == '__main__':
    for test_case in TEST_CASES:
        test_case()
    print(f'通过 {len(TEST_CASES)} 项任务序列版 MCHA 测试')
