"""
预分配任务序列可视化测试。
"""
import os
import sys
import tempfile

import matplotlib

matplotlib.use('Agg')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.core.models import AssignmentPlan, Battlefield, Target, Threat, UAV, UavTaskSequence
from src.visualization.preallocation import (
    collect_preallocation_metrics,
    plot_cooperative_arrival_windows,
    plot_preallocation_metrics_table,
    plot_target_loads,
    plot_task_sequence_assignment_map,
    plot_uav_task_loads,
)


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def build_visualization_battlefield() -> Battlefield:
    uavs = [
        UAV(id=0, x=0.0, y=0.0, speed=100.0, ammo=2, range_left=300.0),
        UAV(id=1, x=0.0, y=40.0, speed=100.0, ammo=2, range_left=300.0),
    ]
    targets = [
        Target(id=0, x=40.0, y=0.0, value=10.0, required_uavs=2),
        Target(id=1, x=60.0, y=20.0, value=9.0, required_uavs=1),
        Target(id=2, x=45.0, y=45.0, value=8.0, required_uavs=1),
    ]
    threats = [Threat(id=0, x=30.0, y=25.0, radius=8.0)]
    return Battlefield(uavs=uavs, targets=targets, threats=threats, map_size=(80.0, 80.0))


def build_visualization_plan() -> AssignmentPlan:
    plan = AssignmentPlan.empty([0, 1])
    plan.uav_task_sequences[0].append_target(0)
    plan.uav_task_sequences[0].append_target(1)
    plan.uav_task_sequences[1].append_target(0)
    plan.uav_task_sequences[1].append_target(2)
    plan.target_assignees = {
        0: [0, 1],
        1: [0],
        2: [1],
    }
    return plan


def test_plot_task_sequence_assignment_map_saves_file():
    battlefield = build_visualization_battlefield()
    plan = build_visualization_plan()

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = os.path.join(tmp_dir, 'task_sequence_assignment.png')
        fig, ax = plot_task_sequence_assignment_map(
            battlefield,
            plan,
            title='任务序列预分配结果',
            output_path=output_path,
        )

        assert_true(os.path.exists(output_path), '任务序列预分配图应保存到指定路径')
        assert_true(os.path.getsize(output_path) > 0, '输出图片文件不应为空')
        assert_true(len(ax.lines) >= 2, '任务序列图应至少绘制有任务的 UAV 任务链')
        assert_true(len(ax.patches) >= 1, '任务序列图应绘制威胁区和任务链箭头等图形元素')
        fig.clf()


def test_plot_target_loads_saves_file():
    battlefield = build_visualization_battlefield()
    plan = build_visualization_plan()

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = os.path.join(tmp_dir, 'target_loads.png')
        fig, ax = plot_target_loads(
            battlefield,
            plan,
            title='目标需求满足情况',
            output_path=output_path,
        )

        assert_true(os.path.exists(output_path), '目标需求满足情况图应保存到指定路径')
        assert_true(os.path.getsize(output_path) > 0, '输出图片文件不应为空')
        assert_true(len(ax.patches) >= len(battlefield.targets), '目标负载图应为每个目标绘制分配数量柱')
        fig.clf()


def test_plot_cooperative_arrival_windows_saves_file():
    battlefield = build_visualization_battlefield()
    plan = build_visualization_plan()

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = os.path.join(tmp_dir, 'cooperative_arrival_windows.png')
        fig, ax = plot_cooperative_arrival_windows(
            battlefield,
            plan,
            title='协同打击到达时间窗',
            output_path=output_path,
            sync_window=0.2,
        )

        assert_true(os.path.exists(output_path), '协同到达时间窗图应保存到指定路径')
        assert_true(os.path.getsize(output_path) > 0, '输出图片文件不应为空')
        assert_true(len(ax.collections) >= 1, '协同到达时间窗图应绘制 UAV 到达时间点')
        assert_true(len(ax.collections) >= 2, '协同到达时间窗图应绘制同步时间窗区域和 UAV 到达时间点')
        fig.clf()


def test_collect_preallocation_metrics_returns_expected_values():
    battlefield = build_visualization_battlefield()
    plan = build_visualization_plan()

    metrics = collect_preallocation_metrics(
        battlefield,
        plan,
        final_fitness=123.45,
        sync_window=0.2,
    )

    assert_true(metrics['uav_count'] == 2, '综合指标应统计 UAV 总数')
    assert_true(metrics['target_count'] == 3, '综合指标应统计目标总数')
    assert_true(metrics['assigned_task_count'] == 4, '综合指标应统计任务槽分配数量')
    assert_true(metrics['required_task_count'] == 4, '综合指标应统计目标需求总数')
    assert_true(metrics['target_satisfaction_rate'] == 1.0, '当前测试计划应满足全部目标需求')
    assert_true(metrics['max_task_chain_length'] == 2, '当前测试计划最大任务链长度应为 2')
    assert_true(metrics['final_fitness'] == 123.45, '综合指标应保留最终适应度')


def test_plot_uav_task_loads_saves_file():
    battlefield = build_visualization_battlefield()
    plan = build_visualization_plan()

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = os.path.join(tmp_dir, 'uav_task_loads.png')
        fig, ax = plot_uav_task_loads(
            battlefield,
            plan,
            title='UAV 任务负载情况',
            output_path=output_path,
        )

        assert_true(os.path.exists(output_path), 'UAV 任务负载图应保存到指定路径')
        assert_true(os.path.getsize(output_path) > 0, '输出图片文件不应为空')
        assert_true(len(ax.patches) >= len(battlefield.uavs), '任务负载图应为每架 UAV 绘制任务数量柱')
        fig.clf()


def test_plot_preallocation_metrics_table_saves_file_and_csv():
    battlefield = build_visualization_battlefield()
    plan = build_visualization_plan()
    metrics = collect_preallocation_metrics(
        battlefield,
        plan,
        final_fitness=123.45,
        sync_window=0.2,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = os.path.join(tmp_dir, 'preallocation_metrics.png')
        csv_output_path = os.path.join(tmp_dir, 'preallocation_metrics.csv')
        fig, ax = plot_preallocation_metrics_table(
            metrics,
            title='预分配综合指标',
            output_path=output_path,
            csv_output_path=csv_output_path,
        )

        assert_true(os.path.exists(output_path), '预分配综合指标表图应保存到指定路径')
        assert_true(os.path.getsize(output_path) > 0, '输出图片文件不应为空')
        assert_true(os.path.exists(csv_output_path), '预分配综合指标 CSV 应保存到指定路径')
        assert_true(os.path.getsize(csv_output_path) > 0, '输出 CSV 文件不应为空')
        assert_true(not ax.axison, '综合指标表图应隐藏普通坐标轴')
        fig.clf()


if __name__ == '__main__':
    test_plot_task_sequence_assignment_map_saves_file()
    test_plot_target_loads_saves_file()
    test_plot_cooperative_arrival_windows_saves_file()
    test_collect_preallocation_metrics_returns_expected_values()
    test_plot_uav_task_loads_saves_file()
    test_plot_preallocation_metrics_table_saves_file_and_csv()
    print('通过 6 项预分配任务序列可视化测试')
