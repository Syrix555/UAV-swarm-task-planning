"""
最终任务链航迹规划可视化测试。
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.core.models import AssignmentPlan, Battlefield, Target, TaskNode, UAV, UavTaskSequence
from src.route_planning.planner import plan_routes_for_assignment_plan
from src.visualization.route_planning import (
    plot_assignment_route_plan,
    write_route_plan_summary_csv,
)


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def build_battlefield() -> Battlefield:
    return Battlefield(
        uavs=[
            UAV(id=0, x=0.0, y=0.0, speed=100.0, ammo=2, range_left=200.0),
            UAV(id=1, x=0.0, y=20.0, speed=100.0, ammo=1, range_left=200.0),
        ],
        targets=[
            Target(id=0, x=20.0, y=0.0, value=10.0),
            Target(id=1, x=30.0, y=10.0, value=8.0),
            Target(id=2, x=20.0, y=20.0, value=6.0),
        ],
        threats=[],
        map_size=(50.0, 50.0),
    )


def build_assignment_plan() -> AssignmentPlan:
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


def test_final_route_plot_and_csv_are_saved():
    tmp_dir = tempfile.mkdtemp()
    try:
        battlefield = build_battlefield()
        route_plan = plan_routes_for_assignment_plan(
            battlefield,
            build_assignment_plan(),
            params={'safety_margin': 0.0, 'enable_bspline_after_kinematic': False},
        )
        image_output_path = os.path.join(tmp_dir, 'final_routes.png')
        csv_output_path = os.path.join(tmp_dir, 'final_routes.csv')

        fig, ax = plot_assignment_route_plan(
            battlefield,
            route_plan,
            title='最终航迹规划测试图',
            output_path=image_output_path,
        )
        write_route_plan_summary_csv(route_plan, csv_output_path)

        assert_true(fig is not None and ax is not None, '绘图函数应返回 fig 和 ax')
        assert_true(os.path.exists(image_output_path), '最终航迹图应保存到指定路径')
        assert_true(os.path.getsize(image_output_path) > 0, '输出图片文件不应为空')
        assert_true(os.path.exists(csv_output_path), '最终航迹 CSV 应保存到指定路径')
        assert_true(os.path.getsize(csv_output_path) > 0, '输出 CSV 文件不应为空')
    finally:
        shutil.rmtree(tmp_dir)


TEST_CASES = [
    ('最终航迹图和 CSV 保存', test_final_route_plot_and_csv_are_saved),
]


def main():
    print('开始运行最终任务链航迹规划可视化测试...')
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
