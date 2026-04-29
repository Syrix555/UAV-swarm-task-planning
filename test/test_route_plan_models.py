"""
任务序列航迹规划结果数据结构测试。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.route_planning.planner import (
    AssignmentRoutePlan,
    PathPlanningResult,
    RouteSegment,
    UavRoutePlan,
    plan_routes_for_assignment_plan,
)
from src.core.models import AssignmentPlan, Battlefield, Target, TaskNode, UAV, UavTaskSequence


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def build_result(success: bool, final_path, path_length: float, failure_reason=None) -> PathPlanningResult:
    return PathPlanningResult(
        success=success,
        original_path=list(final_path),
        los_path=list(final_path),
        kinematic_path=list(final_path),
        smoothed_path=list(final_path),
        final_path=list(final_path) if success else [],
        used_kinematic_constraint=False,
        used_smoothing=False,
        kinematic_mode='disabled',
        fallback_reason=None,
        failure_reason=failure_reason,
        path_length=path_length,
        estimated_min_turn_radius=float('inf'),
    )


def build_segment(
    uav_id: int,
    order: int,
    start_kind: str,
    start_id: int,
    end_target_id: int,
    final_path,
    path_length: float,
    success: bool = True,
) -> RouteSegment:
    return RouteSegment(
        uav_id=uav_id,
        segment_order=order,
        start_kind=start_kind,
        start_id=start_id,
        end_target_id=end_target_id,
        start_xy=final_path[0],
        end_xy=final_path[-1],
        result=build_result(
            success=success,
            final_path=final_path,
            path_length=path_length,
            failure_reason=None if success else 'no_path_found',
        ),
    )


def build_route_battlefield(targets=None) -> Battlefield:
    uavs = [
        UAV(id=0, x=0.0, y=0.0, speed=100.0, ammo=3, range_left=300.0),
        UAV(id=1, x=0.0, y=10.0, speed=100.0, ammo=2, range_left=300.0),
    ]
    if targets is None:
        targets = [
            Target(id=0, x=10.0, y=0.0, value=10.0),
            Target(id=1, x=20.0, y=0.0, value=8.0),
            Target(id=2, x=10.0, y=10.0, value=6.0),
        ]
    return Battlefield(uavs=uavs, targets=targets, threats=[], map_size=(40.0, 40.0))


def build_route_plan() -> AssignmentPlan:
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
        },
        target_assignees={0: [0], 1: [0], 2: [1]},
    )


def test_uav_route_plan_merges_segment_paths_without_duplicate_joint():
    first = build_segment(0, 0, 'uav', 0, 1, [(0.0, 0.0), (1.0, 1.0)], 1.4)
    second = build_segment(0, 1, 'target', 1, 2, [(1.0, 1.0), (2.0, 1.0)], 1.0)
    route = UavRoutePlan(uav_id=0, target_ids=[1, 2], segments=[first, second])

    assert_true(route.success, '全部航段成功时 UAV 航迹应成功')
    assert_true(route.active, '存在目标序列时 UAV 应为 active')
    assert_true(route.total_path_length == 2.4, '总航迹长度应累加成功航段长度')
    assert_true(
        route.full_path == [(0.0, 0.0), (1.0, 1.0), (2.0, 1.0)],
        '拼接完整航迹时不应重复相邻航段连接点',
    )


def test_failed_segments_are_reported_and_excluded_from_path_length():
    success_segment = build_segment(1, 0, 'uav', 1, 3, [(0.0, 0.0), (3.0, 0.0)], 3.0)
    failed_segment = build_segment(1, 1, 'target', 3, 4, [(3.0, 0.0), (4.0, 0.0)], 1.0, success=False)
    route = UavRoutePlan(uav_id=1, target_ids=[3, 4], segments=[success_segment, failed_segment])

    assert_true(not route.success, '存在失败航段时 UAV 航迹应失败')
    assert_true(route.failed_segments == [failed_segment], '应能定位失败航段')
    assert_true(route.total_path_length == 3.0, '失败航段长度不应计入总航迹长度')
    assert_true(route.full_path == [(0.0, 0.0), (3.0, 0.0)], '失败航段不应拼入完整航迹')


def test_assignment_route_plan_summarizes_all_uavs():
    route_ok = UavRoutePlan(
        uav_id=0,
        target_ids=[1],
        segments=[build_segment(0, 0, 'uav', 0, 1, [(0.0, 0.0), (1.0, 0.0)], 1.0)],
    )
    route_failed = UavRoutePlan(
        uav_id=1,
        target_ids=[2],
        segments=[build_segment(1, 0, 'uav', 1, 2, [(0.0, 1.0), (1.0, 1.0)], 1.0, success=False)],
    )
    inactive_route = UavRoutePlan(uav_id=2, target_ids=[], segments=[])
    plan = AssignmentRoutePlan(
        uav_route_plans={
            0: route_ok,
            1: route_failed,
            2: inactive_route,
        },
        source='test_plan',
    )

    assert_true(not plan.success, '存在失败 UAV 航迹时整体航迹规划应失败')
    assert_true(plan.active_uav_count == 2, '只统计有任务的 UAV')
    assert_true(plan.segment_count == 2, '整体航段数应累加所有 UAV 航段')
    assert_true(plan.failed_uavs == [1], '应能汇总失败 UAV')
    assert_true(len(plan.failed_segments) == 1, '应能汇总失败航段')
    assert_true(plan.total_path_length == 1.0, '整体长度只累加成功航段')


def test_plan_routes_for_assignment_plan_builds_segments_for_each_task():
    battlefield = build_route_battlefield()
    route_plan = plan_routes_for_assignment_plan(
        battlefield,
        build_route_plan(),
        params={'enable_bspline_after_kinematic': False, 'safety_margin': 0.0},
        source='unit_test',
    )

    assert_true(route_plan.source == 'unit_test', '应保留航迹规划来源标记')
    assert_true(route_plan.success, '简单无威胁任务链应全部规划成功')
    assert_true(route_plan.active_uav_count == 2, '两架有任务 UAV 都应计为 active')
    assert_true(route_plan.segment_count == 3, '每个任务节点应生成一段航迹')

    uav0_route = route_plan.uav_route_plans[0]
    assert_true(uav0_route.target_ids == [0, 1], 'UAV 航迹应保留任务序列')
    assert_true(len(uav0_route.segments) == 2, 'UAV0 应生成两段任务链航迹')
    assert_true(uav0_route.segments[0].start_kind == 'uav', '首段应从 UAV 起点出发')
    assert_true(uav0_route.segments[1].start_kind == 'target', '后续航段应从上一目标出发')
    assert_true(uav0_route.segments[1].start_id == 0, '第二段起点目标应为上一任务目标')
    assert_true(len(uav0_route.full_path) >= 2, '成功航迹应能拼接完整路径')


def test_plan_routes_for_assignment_plan_stops_current_uav_after_failed_segment():
    targets = [
        Target(id=0, x=10.0, y=0.0, value=10.0),
        Target(id=1, x=1000.0, y=0.0, value=8.0),
        Target(id=2, x=20.0, y=0.0, value=6.0),
    ]
    battlefield = build_route_battlefield(targets=targets)
    assignment_plan = AssignmentPlan(
        uav_task_sequences={
            0: UavTaskSequence(
                uav_id=0,
                tasks=[
                    TaskNode(target_id=0, order=0),
                    TaskNode(target_id=1, order=1),
                    TaskNode(target_id=2, order=2),
                ],
            ),
            1: UavTaskSequence(uav_id=1, tasks=[]),
        },
        target_assignees={0: [0], 1: [0], 2: [0]},
    )

    route_plan = plan_routes_for_assignment_plan(
        battlefield,
        assignment_plan,
        params={'enable_bspline_after_kinematic': False, 'safety_margin': 0.0},
    )

    uav0_route = route_plan.uav_route_plans[0]
    assert_true(not route_plan.success, '存在不可达航段时整体规划应失败')
    assert_true(not uav0_route.success, '存在不可达航段时该 UAV 航迹应失败')
    assert_true(len(uav0_route.segments) == 2, '当前 UAV 遇到失败航段后应停止继续规划')
    assert_true(uav0_route.failed_segments[0].end_target_id == 1, '失败航段应定位到不可达目标')
    assert_true(uav0_route.failed_segments[0].failure_reason == 'no_path_found', '越界目标应返回 no_path_found')


TEST_CASES = [
    ('UAV 航段路径拼接', test_uav_route_plan_merges_segment_paths_without_duplicate_joint),
    ('失败航段统计', test_failed_segments_are_reported_and_excluded_from_path_length),
    ('整体航迹结果汇总', test_assignment_route_plan_summarizes_all_uavs),
    ('AssignmentPlan 转多 UAV 任务链航迹', test_plan_routes_for_assignment_plan_builds_segments_for_each_task),
    ('任务链遇到失败航段停止当前 UAV', test_plan_routes_for_assignment_plan_stops_current_uav_after_failed_segment),
]


def main():
    print('开始运行任务序列航迹规划结果数据结构测试...')
    passed = 0
    for name, test_func in TEST_CASES:
        try:
            test_func()
            passed += 1
            print(f'[PASS] {name}')
        except Exception as exc:
            print(f'[FAIL] {name}: {exc}')
            raise

    print(f'测试完成：{passed}/{len(TEST_CASES)} 通过')


if __name__ == '__main__':
    main()
