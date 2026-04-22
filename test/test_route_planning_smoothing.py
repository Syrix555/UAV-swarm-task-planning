"""
局部 B 样条平滑与运动学圆弧测试脚本
验证直线保持、拐点局部平滑、圆弧约束与异常处理。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.route_planning.geometry import (
    estimate_min_turn_radius,
    generate_kinematic_path,
    generate_kinematic_path_details,
)
from src.route_planning.smoothing import BSplineSmoothingError, smooth_bspline


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def test_collinear_path_stays_straight():
    path_points = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0)]
    smoothed = smooth_bspline(path_points, degree=3, smoothing_factor=1.0, sample_step=1.0)

    assert_true(all(abs(point[1]) < 1e-6 for point in smoothed), '共线路径局部平滑后仍应保持直线')


def test_single_corner_is_smoothed_locally():
    path_points = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (20.0, 10.0)]
    smoothed = smooth_bspline(path_points, degree=3, smoothing_factor=1.0, sample_step=0.5)

    assert_true(len(smoothed) > len(path_points), '拐点附近应生成额外采样点')
    assert_true(smoothed[0] == path_points[0], '局部平滑应保持起点不变')
    assert_true(smoothed[-1] == path_points[-1], '局部平滑应保持终点不变')


def test_kinematic_path_respects_turn_radius():
    path_points = [(0.0, 0.0), (12.0, 0.0), (12.0, 12.0)]
    kinematic = generate_kinematic_path(path_points, min_turn_radius=3.0, sample_step=0.25)

    assert_true(kinematic[0] == path_points[0], '运动学路径应保持起点不变')
    assert_true(kinematic[-1] == path_points[-1], '运动学路径应保持终点不变')
    assert_true(len(kinematic) > len(path_points), '圆弧替换后应产生更稠密的采样点')
    assert_true(estimate_min_turn_radius(kinematic) >= 2.95, '运动学路径应基本满足给定最小转弯半径')


def test_adaptive_kinematic_mode_reports_adaptive():
    path_points = [(0.0, 0.0), (12.0, 0.0), (12.0, 12.0)]
    result = generate_kinematic_path_details(path_points, min_turn_radius=3.0, sample_step=0.25)

    assert_true(result.mode == 'adaptive', '单拐点充足场景应进入 adaptive 模式')
    assert_true(result.reason is None, 'adaptive 成功时不应记录失败原因')
    assert_true(len(result.applied_radii) == 1, '单拐点应仅应用一个圆弧半径')


def test_windowed_kinematic_mode_reports_windowed():
    path_points = [(0.0, 0.0), (8.0, 0.0), (8.0, 8.0), (16.0, 8.0), (16.0, 16.0)]
    result = generate_kinematic_path_details(path_points, min_turn_radius=3.0, sample_step=0.25)

    assert_true(result.mode == 'windowed', '相邻拐点冲突场景应进入 windowed 模式')
    assert_true(len(result.applied_radii) >= 2, 'windowed 模式应至少保留两个有效圆弧')
    assert_true(estimate_min_turn_radius(result.path_points) >= 2.95, 'windowed 路径仍应满足最小转弯半径')


def test_medium_like_adjacent_corners_stay_near_windowed_threshold():
    path_points = [(10.0, 10.0), (60.0, 59.0), (64.0, 59.0), (70.0, 64.0), (80.0, 80.0)]
    result = generate_kinematic_path_details(path_points, min_turn_radius=3.0, sample_step=0.25)

    assert_true(result.mode == 'adaptive', '该 medium-like 场景应仍由 adaptive 模式处理')
    assert_true(result.reason is None, 'adaptive 成功时不应返回失败原因')
    assert_true(len(result.applied_radii) == 3, '该场景应保留三个连续圆弧')
    assert_true(estimate_min_turn_radius(result.path_points) >= 2.95, '路径应满足最小转弯半径')


def test_insufficient_corner_space_falls_back():
    path_points = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0)]
    result = generate_kinematic_path_details(path_points, min_turn_radius=3.0, sample_step=0.25)

    assert_true(result.mode == 'fallback', '空间不足时应回退到折线路径')
    assert_true(result.reason == 'turn_radius_segment_too_short', '应返回线段过短原因')


def test_short_path_stays_valid():
    path_points = [(0.0, 0.0), (5.0, 3.0), (10.0, 0.0)]
    smoothed = smooth_bspline(path_points, degree=5, smoothing_factor=0.5, sample_step=0.5)

    assert_true(len(smoothed) >= len(path_points), '短路径局部平滑后仍应返回有效结果')


def test_bspline_deduplicates_repeated_points():
    path_points = [(0.0, 0.0), (5.0, 5.0), (5.0, 5.0), (10.0, 0.0), (15.0, 5.0)]
    smoothed = smooth_bspline(path_points, degree=3, smoothing_factor=1.0, sample_step=0.5)

    assert_true(len(smoothed) >= 2, '包含重复点的路径仍应能完成局部平滑')


def test_bspline_requires_positive_sample_step():
    path_points = [(0.0, 0.0), (10.0, 5.0), (20.0, 0.0)]
    try:
        smooth_bspline(path_points, degree=2, smoothing_factor=1.0, sample_step=0.0)
    except BSplineSmoothingError:
        return
    raise AssertionError('sample_step 非法时应抛出 BSplineSmoothingError')


def test_adjacent_corners_do_not_create_overlapping_loops():
    path_points = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (20.0, 10.0), (20.0, 20.0)]
    smoothed = smooth_bspline(path_points, degree=3, smoothing_factor=1.0, sample_step=0.5)

    assert_true(smoothed[0] == path_points[0], '连续拐点场景下起点应保持不变')
    assert_true(smoothed[-1] == path_points[-1], '连续拐点场景下终点应保持不变')
    assert_true(len(smoothed) < 120, '连续拐点局部平滑不应产生异常膨胀的重复路径点')


TEST_CASES = [
    ('共线路径保持直线', test_collinear_path_stays_straight),
    ('单拐点局部平滑', test_single_corner_is_smoothed_locally),
    ('运动学路径满足转弯半径', test_kinematic_path_respects_turn_radius),
    ('自适应模式输出', test_adaptive_kinematic_mode_reports_adaptive),
    ('窗口模式输出', test_windowed_kinematic_mode_reports_windowed),
    ('medium近邻拐点接近阈值', test_medium_like_adjacent_corners_stay_near_windowed_threshold),
    ('空间不足时回退', test_insufficient_corner_space_falls_back),
    ('短路径稳定处理', test_short_path_stays_valid),
    ('重复点去重', test_bspline_deduplicates_repeated_points),
    ('连续拐点不重叠回绕', test_adjacent_corners_do_not_create_overlapping_loops),
    ('非法采样步长报错', test_bspline_requires_positive_sample_step),
]


def main():
    print('开始运行局部 B 样条平滑测试...')
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
