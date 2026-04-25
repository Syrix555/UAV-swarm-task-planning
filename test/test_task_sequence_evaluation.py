"""
第二阶段：任务链评估测试。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.core.models import Battlefield, Target, TaskNode, Threat, UAV, UavTaskSequence
from src.core.sequence_eval import evaluate_uav_task_sequence


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def build_sequence_battlefield() -> Battlefield:
    uavs = [
        UAV(id=0, x=0.0, y=0.0, speed=10.0, ammo=3, range_left=100.0),
    ]
    targets = [
        Target(id=0, x=30.0, y=0.0, value=10.0, service_time=1.0),
        Target(id=1, x=30.0, y=40.0, value=8.0, service_time=2.0),
        Target(id=2, x=60.0, y=40.0, value=6.0, time_window_start=6.0, time_window_end=8.0),
    ]
    return Battlefield(uavs=uavs, targets=targets, threats=[], map_size=(100.0, 100.0))


def test_sequence_evaluation_accumulates_distance_and_arrival_times():
    battlefield = build_sequence_battlefield()
    sequence = UavTaskSequence(
        uav_id=0,
        tasks=[TaskNode(target_id=0, order=0), TaskNode(target_id=1, order=1)],
    )

    result = evaluate_uav_task_sequence(battlefield, sequence)

    assert_true(abs(result.total_distance - 70.0) < 1e-6, '累计距离应为 30 + 40 = 70')
    assert_true(abs(result.arrival_times[0] - 3.0) < 1e-6, '首个目标到达时刻应为 30/10 = 3')
    assert_true(abs(result.arrival_times[1] - 8.0) < 1e-6, '第二个目标到达时刻应累计前序服务时间')
    assert_true(abs(result.total_travel_time - 7.0) < 1e-6, '纯飞行时间应为总距离/速度 = 7')
    assert_true(abs(result.completion_time - 10.0) < 1e-6, '完成时刻应包含服务时间 1 + 2')
    assert_true(result.is_ammo_feasible, '该序列不应违反 ammo 约束')
    assert_true(result.is_range_feasible, '该序列不应违反 range 约束')


def test_sequence_evaluation_flags_ammo_violation():
    battlefield = build_sequence_battlefield()
    battlefield.get_uav(0).ammo = 1
    sequence = UavTaskSequence(
        uav_id=0,
        tasks=[TaskNode(target_id=0, order=0), TaskNode(target_id=1, order=1)],
    )

    result = evaluate_uav_task_sequence(battlefield, sequence)

    assert_true(not result.is_ammo_feasible, '任务数量超过 ammo 时应判定为不可行')
    assert_true(not result.is_feasible, 'ammo 不可行时整体应不可行')


def test_sequence_evaluation_flags_range_violation():
    battlefield = build_sequence_battlefield()
    battlefield.get_uav(0).range_left = 60.0
    sequence = UavTaskSequence(
        uav_id=0,
        tasks=[TaskNode(target_id=0, order=0), TaskNode(target_id=1, order=1)],
    )

    result = evaluate_uav_task_sequence(battlefield, sequence)

    assert_true(not result.is_range_feasible, '累计距离超过剩余航程时应判定为不可行')
    assert_true(not result.is_feasible, 'range 不可行时整体应不可行')


def test_sequence_evaluation_applies_time_window_penalty_from_cumulative_arrival():
    battlefield = build_sequence_battlefield()
    sequence = UavTaskSequence(
        uav_id=0,
        tasks=[TaskNode(target_id=0, order=0), TaskNode(target_id=1, order=1), TaskNode(target_id=2, order=2)],
    )

    result = evaluate_uav_task_sequence(battlefield, sequence, alpha=2.0)

    assert_true(abs(result.arrival_times[2] - 13.0) < 1e-6, '第三个目标到达时刻应由完整任务链累计得到')
    assert_true(abs(result.time_window_penalty - 10.0) < 1e-6, '超出时间窗 5 个单位时应产生 alpha*5 的惩罚')


def test_sequence_evaluation_returns_updated_sequence_with_planned_fields():
    battlefield = build_sequence_battlefield()
    sequence = UavTaskSequence(
        uav_id=0,
        tasks=[TaskNode(target_id=0, order=0), TaskNode(target_id=1, order=1)],
    )

    result = evaluate_uav_task_sequence(battlefield, sequence)

    assert_true(abs(result.evaluated_sequence.tasks[0].planned_arrival_time - 3.0) < 1e-6, '应回填第一个任务节点的到达时刻')
    assert_true(abs(result.evaluated_sequence.tasks[1].estimated_path_length - 40.0) < 1e-6, '应记录该节点对应航段长度')
    assert_true(result.evaluated_sequence.tasks[1].planned_service_time == 2.0, '应回填目标服务时间')


TEST_CASES = [
    test_sequence_evaluation_accumulates_distance_and_arrival_times,
    test_sequence_evaluation_flags_ammo_violation,
    test_sequence_evaluation_flags_range_violation,
    test_sequence_evaluation_applies_time_window_penalty_from_cumulative_arrival,
    test_sequence_evaluation_returns_updated_sequence_with_planned_fields,
]


if __name__ == '__main__':
    for test_case in TEST_CASES:
        test_case()
    print(f'通过 {len(TEST_CASES)} 项任务链评估测试')
