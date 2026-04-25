"""
第一阶段数据结构兼容性测试。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.core.models import AssignmentPlan, TaskNode, UavTaskSequence


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def test_assignment_plan_from_matrix_preserves_pairs():
    assignment = np.array([
        [1, 0, 1],
        [0, 1, 0],
    ], dtype=int)

    plan = AssignmentPlan.from_assignment_matrix(assignment)

    assert_true(plan.uav_task_sequences[0].target_ids() == [0, 2], 'UAV0 应保留两个目标节点')
    assert_true(plan.uav_task_sequences[1].target_ids() == [1], 'UAV1 应保留单个目标节点')
    assert_true(plan.target_assignees == {0: [0], 1: [1], 2: [0]}, '目标反向索引应正确生成')


def test_assignment_plan_to_matrix_roundtrip():
    plan = AssignmentPlan(
        uav_task_sequences={
            0: UavTaskSequence(uav_id=0, tasks=[TaskNode(target_id=1, order=0), TaskNode(target_id=2, order=1)]),
            1: UavTaskSequence(uav_id=1, tasks=[TaskNode(target_id=0, order=0)]),
        },
        target_assignees={0: [1], 1: [0], 2: [0]},
    )

    assignment = plan.to_assignment_matrix(num_uavs=2, num_targets=3)
    expected = np.array([
        [0, 1, 1],
        [1, 0, 0],
    ], dtype=int)

    assert_true(np.array_equal(assignment, expected), '任务序列应能投影回旧分配矩阵')


def test_empty_assignment_plan_initializes_all_uavs():
    plan = AssignmentPlan.empty([0, 2, 5])

    assert_true(sorted(plan.uav_task_sequences.keys()) == [0, 2, 5], '应为所有 UAV 初始化空任务序列')
    assert_true(plan.target_assignees == {}, '空计划不应包含目标反向索引')
    assert_true(all(sequence.task_count() == 0 for sequence in plan.uav_task_sequences.values()), '所有任务序列初始应为空')


TEST_CASES = [
    test_assignment_plan_from_matrix_preserves_pairs,
    test_assignment_plan_to_matrix_roundtrip,
    test_empty_assignment_plan_initializes_all_uavs,
]


if __name__ == '__main__':
    for test_case in TEST_CASES:
        test_case()
    print(f'通过 {len(TEST_CASES)} 项第一阶段数据结构测试')
