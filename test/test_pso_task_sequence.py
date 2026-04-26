"""
第三阶段：PSO 预分配任务序列测试。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from config.params import WEIGHTS
from src.core.models import AssignmentPlan, Battlefield, Target, Threat, UAV
from src.core.sequence_eval import evaluate_uav_task_sequence
from src.pre_allocation.pso import logistic_init, run_pso


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def build_multitask_battlefield() -> Battlefield:
    uavs = [
        UAV(id=0, x=0.0, y=0.0, speed=100.0, ammo=2, range_left=300.0),
        UAV(id=1, x=0.0, y=20.0, speed=100.0, ammo=2, range_left=300.0),
    ]
    targets = [
        Target(id=0, x=40.0, y=0.0, value=10.0, required_uavs=1),
        Target(id=1, x=45.0, y=20.0, value=9.0, required_uavs=1),
        Target(id=2, x=55.0, y=10.0, value=8.0, required_uavs=1),
    ]
    threats: list[Threat] = []
    return Battlefield(uavs=uavs, targets=targets, threats=threats, map_size=(100.0, 100.0))


def test_logistic_initialization_supports_more_slots_than_uavs_when_ammo_allows():
    np.random.seed(7)

    population = logistic_init(
        num_particles=6,
        dim=3,
        num_uavs=2,
        uav_capacities=[2, 2],
    )

    assert_true(population.shape == (6, 3), '初始化种群维度应匹配任务槽位数量')
    for particle in population:
        counts = np.bincount(particle, minlength=2)
        assert_true(np.all(counts <= np.array([2, 2])), '初始化粒子不应超过 UAV ammo 容量')


def test_run_pso_can_return_assignment_plan_with_multi_task_sequences():
    battlefield = build_multitask_battlefield()
    pso_params = {
        'num_particles': 16,
        'max_iter': 30,
        'w_start': 0.9,
        'w_end': 0.4,
        'c1': 1.2,
        'c2': 1.8,
    }

    np.random.seed(11)
    assignment, etas, curve, plan = run_pso(
        battlefield,
        WEIGHTS,
        pso_params,
        return_assignment_plan=True,
    )

    assert_true(isinstance(plan, AssignmentPlan), 'PSO 应能输出 AssignmentPlan')
    assert_true(assignment.shape == (2, 3), '兼容旧流程的分配矩阵形状应正确')
    assert_true(etas.shape == (2, 3), 'ETA 矩阵形状应正确')
    assert_true(len(curve) == pso_params['max_iter'] + 1, '收敛曲线长度应包含初始代')

    for target in battlefield.targets:
        assigned_count = int(np.sum(assignment[:, target.id]))
        assert_true(assigned_count == target.required_uavs, '每个目标应满足 required_uavs')
        assert_true(len(plan.target_assignees[target.id]) == target.required_uavs, '目标反向索引应满足需求数量')

    task_counts = [
        sequence.task_count()
        for sequence in plan.uav_task_sequences.values()
    ]
    assert_true(sum(task_counts) == 3, '任务序列总任务数应等于目标需求总量')
    assert_true(max(task_counts) >= 2, '任务数多于 UAV 数时，至少一架 UAV 应承担任务序列')

    for uav in battlefield.uavs:
        sequence = plan.uav_task_sequences[uav.id]
        assert_true(sequence.task_count() <= uav.ammo, '任务序列长度不应超过 ammo')
        evaluated = evaluate_uav_task_sequence(battlefield, sequence)
        assert_true(evaluated.is_feasible, 'PSO 输出的任务序列应满足 ammo 和 range 约束')

    roundtrip_assignment = plan.to_assignment_matrix(num_uavs=2, num_targets=3)
    assert_true(np.array_equal(roundtrip_assignment, assignment), 'AssignmentPlan 应能投影回兼容矩阵')


TEST_CASES = [
    test_logistic_initialization_supports_more_slots_than_uavs_when_ammo_allows,
    test_run_pso_can_return_assignment_plan_with_multi_task_sequences,
]


if __name__ == '__main__':
    for test_case in TEST_CASES:
        test_case()
    print(f'通过 {len(TEST_CASES)} 项第三阶段 PSO 任务序列测试')
