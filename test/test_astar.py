"""
路径规划基础测试脚本
验证 A* 避障、LOS 简化、运动学约束路径、局部 B 样条平滑与回退处理。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from config.params import ASTAR
from data.scenario_small import create_small_scenario
from src.core.models import Battlefield, Threat
from src.route_planning.geometry import estimate_min_turn_radius, path_intersects_any_threat
from src.route_planning.planner import plan_path_between_points, plan_path_for_uav


DEFAULT_PARAMS = dict(ASTAR)


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def build_blocked_battlefield() -> Battlefield:
    battlefield = create_small_scenario()
    battlefield.threats = [
        Threat(id=100, x=50.0, y=50.0, radius=80.0),
    ]
    return battlefield


def test_plan_path_avoids_inflated_threats():
    battlefield = create_small_scenario()
    result = plan_path_for_uav(battlefield, uav_id=0, target_id=0, params=DEFAULT_PARAMS)

    assert_true(result.success, 'A* 应为基础场景找到可行路径')
    assert_true(len(result.original_path) >= 2, '原始路径至少应包含起点和终点')
    assert_true(
        not path_intersects_any_threat(result.final_path, battlefield.threats, DEFAULT_PARAMS['safety_margin']),
        '最终路径不得进入膨胀威胁区',
    )


def test_plan_path_between_points_matches_uav_to_target_wrapper():
    battlefield = create_small_scenario()
    uav = battlefield.get_uav(0)
    target = battlefield.get_target(0)

    direct_result = plan_path_between_points(
        battlefield,
        uav_id=0,
        start_xy=(uav.x, uav.y),
        goal_xy=(target.x, target.y),
        params=DEFAULT_PARAMS,
    )
    wrapper_result = plan_path_for_uav(battlefield, uav_id=0, target_id=0, params=DEFAULT_PARAMS)

    assert_true(direct_result.success == wrapper_result.success, '通用点到点接口应保持原 UAV->目标接口语义')
    assert_true(direct_result.failure_reason == wrapper_result.failure_reason, '失败原因应与原接口一致')
    assert_true(len(direct_result.final_path) == len(wrapper_result.final_path), '最终路径采样点数量应与原接口一致')


def test_plan_path_between_points_supports_target_to_target_segment():
    battlefield = create_small_scenario()
    battlefield.threats = []
    start_target = battlefield.get_target(0)
    end_target = battlefield.get_target(1)

    result = plan_path_between_points(
        battlefield,
        uav_id=0,
        start_xy=(start_target.x, start_target.y),
        goal_xy=(end_target.x, end_target.y),
        params=DEFAULT_PARAMS,
    )

    assert_true(result.success, '通用点到点接口应支持任务链中的目标到目标航段')
    assert_true(len(result.final_path) >= 2, '目标到目标航段应返回有效路径')


def test_los_simplification_reduces_waypoints():
    battlefield = create_small_scenario()
    result = plan_path_for_uav(battlefield, uav_id=1, target_id=1, params=DEFAULT_PARAMS)

    assert_true(result.success, '应找到可行路径')
    assert_true(len(result.los_path) <= len(result.original_path), 'LOS 简化后节点数不应增加')
    assert_true(
        not path_intersects_any_threat(result.los_path, battlefield.threats, DEFAULT_PARAMS['safety_margin']),
        'LOS 路径不得进入膨胀威胁区',
    )


def test_kinematic_path_respects_min_turn_radius():
    battlefield = create_small_scenario()
    params = dict(DEFAULT_PARAMS)
    params['enable_bspline_after_kinematic'] = False
    params['min_turn_radius'] = 2.0

    result = plan_path_for_uav(battlefield, uav_id=0, target_id=1, params=params)

    assert_true(result.success, '启用最小转弯半径后仍应找到可行路径')
    assert_true(len(result.los_path) >= 3, '该测试场景应包含至少一个 LOS 拐点')
    assert_true(len(result.kinematic_path) >= 2, '运动学约束路径不应为空')
    assert_true(result.used_kinematic_constraint, '存在拐点时应优先使用运动学约束路径')
    assert_true(result.kinematic_mode in ('adaptive', 'windowed'), '应输出有效的运动学模式')
    assert_true(
        estimate_min_turn_radius(result.final_path) >= params['min_turn_radius'] - 0.1,
        '最终路径应基本满足最小转弯半径约束',
    )


def test_kinematic_metadata_is_reported():
    battlefield = create_small_scenario()
    params = dict(DEFAULT_PARAMS)
    params['enable_bspline_after_kinematic'] = False
    params['min_turn_radius'] = 2.0

    result = plan_path_for_uav(battlefield, uav_id=1, target_id=1, params=params)

    assert_true(result.success, '元数据场景应找到可行路径')
    assert_true(result.kinematic_mode in ('adaptive', 'windowed', 'fallback', 'disabled'), '应返回合法 kinematic_mode')
    assert_true(result.estimated_min_turn_radius > 0.0, '应返回估计最小转弯半径')


def test_valid_bspline_path_is_used_when_available():
    battlefield = create_small_scenario()
    params = dict(DEFAULT_PARAMS)
    params['smoothing_factor'] = 1.0
    params['sample_step'] = 0.25
    params['corner_angle_threshold_deg'] = 10.0

    result = plan_path_for_uav(battlefield, uav_id=0, target_id=1, params=params)

    assert_true(result.success, '应找到可行路径')
    assert_true(len(result.smoothed_path) >= 2, '局部 B 样条输出不应为空')
    if result.used_smoothing:
        assert_true(result.final_path == result.smoothed_path, '使用平滑时最终路径应来自局部 B 样条结果')
    else:
        assert_true(
            result.fallback_reason is not None,
            '未使用平滑时应记录回退原因',
        )


def test_smoothing_collision_triggers_fallback():
    battlefield = create_small_scenario()
    params = dict(DEFAULT_PARAMS)
    params['smoothing_factor'] = 3.0
    params['sample_step'] = 0.25

    result = plan_path_for_uav(battlefield, uav_id=2, target_id=2, params=params)

    assert_true(result.success, '即使平滑失败，也应通过回退保留可行路径')
    assert_true(
        result.final_path in (result.smoothed_path, result.kinematic_path, result.los_path, result.original_path),
        '最终路径应来自候选路径之一',
    )
    assert_true(
        not path_intersects_any_threat(result.final_path, battlefield.threats, params['safety_margin']),
        '回退后的最终路径仍不得进入膨胀威胁区',
    )


def test_no_solution_returns_failure():
    battlefield = build_blocked_battlefield()
    result = plan_path_for_uav(battlefield, uav_id=0, target_id=0, params=DEFAULT_PARAMS)

    assert_true(not result.success, '完全阻断时应返回失败状态')
    assert_true(result.failure_reason == 'no_path_found', '失败原因应为 no_path_found')


def test_range_limit_is_checked():
    battlefield = create_small_scenario()
    battlefield.get_uav(0).range_left = 20.0
    result = plan_path_for_uav(battlefield, uav_id=0, target_id=0, params=DEFAULT_PARAMS)

    assert_true(not result.success, '航程不足时应返回失败状态')
    assert_true(result.failure_reason == 'out_of_range', '失败原因应为 out_of_range')


TEST_CASES = [
    ('基础避障', test_plan_path_avoids_inflated_threats),
    ('通用点到点接口兼容原接口', test_plan_path_between_points_matches_uav_to_target_wrapper),
    ('通用点到点接口支持目标到目标航段', test_plan_path_between_points_supports_target_to_target_segment),
    ('LOS 简化', test_los_simplification_reduces_waypoints),
    ('最小转弯半径约束', test_kinematic_path_respects_min_turn_radius),
    ('运动学元数据输出', test_kinematic_metadata_is_reported),
    ('严格 B 样条可用时优先使用', test_valid_bspline_path_is_used_when_available),
    ('平滑失败回退', test_smoothing_collision_triggers_fallback),
    ('无解返回失败', test_no_solution_returns_failure),
    ('航程约束', test_range_limit_is_checked),
]


def main():
    print('开始运行路径规划测试...')
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
