"""
任务序列版重分配可视化测试。
"""
import os
import sys
import tempfile

import matplotlib

matplotlib.use('Agg')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.core.models import AssignmentPlan, Battlefield, Target, Threat, UAV
from src.re_allocation.events import Event, EventType
from src.re_allocation.mcha import BidResult, BidRoundLog
from src.visualization.reallocation import (
    plot_mcha_candidate_bid_scores,
    plot_mcha_open_demand_repair,
    plot_mcha_winning_bids,
    plot_plan_reallocation_before_after,
    plot_plan_reallocation_diff,
    plot_plan_reallocation_target_loads,
    plot_plan_reallocation_uav_loads,
    write_reallocation_cost_change_csv,
)


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def build_reallocation_visualization_battlefields():
    uavs = [
        UAV(id=0, x=0.0, y=0.0, speed=100.0, ammo=3, range_left=300.0),
        UAV(id=1, x=0.0, y=50.0, speed=100.0, ammo=3, range_left=300.0),
        UAV(id=2, x=0.0, y=90.0, speed=100.0, ammo=3, range_left=300.0),
    ]
    targets = [
        Target(id=0, x=42.0, y=12.0, value=8.0, required_uavs=1),
        Target(id=1, x=64.0, y=26.0, value=9.0, required_uavs=1),
        Target(id=2, x=52.0, y=58.0, value=7.5, required_uavs=1),
        Target(id=3, x=70.0, y=82.0, value=8.5, required_uavs=1),
    ]
    battlefield_before = Battlefield(
        uavs=uavs,
        targets=targets,
        threats=[Threat(id=0, x=32.0, y=42.0, radius=9.0)],
        map_size=(100.0, 100.0),
    )
    battlefield_after = Battlefield(
        uavs=uavs,
        targets=targets,
        threats=[
            Threat(id=0, x=32.0, y=42.0, radius=9.0),
            Threat(id=99, x=48.0, y=48.0, radius=12.0),
        ],
        map_size=(100.0, 100.0),
    )
    return battlefield_before, battlefield_after


def build_before_plan() -> AssignmentPlan:
    plan = AssignmentPlan.empty([0, 1, 2])
    plan.uav_task_sequences[0].append_target(0)
    plan.uav_task_sequences[0].append_target(1)
    plan.uav_task_sequences[1].append_target(2)
    plan.uav_task_sequences[2].append_target(3)
    plan.target_assignees = {
        0: [0],
        1: [0],
        2: [1],
        3: [2],
    }
    return plan


def build_after_plan() -> AssignmentPlan:
    plan = AssignmentPlan.empty([0, 1, 2])
    plan.uav_task_sequences[0].append_target(0)
    plan.uav_task_sequences[1].append_target(2)
    plan.uav_task_sequences[1].append_target(1)
    plan.uav_task_sequences[2].append_target(3)
    plan.target_assignees = {
        0: [0],
        1: [1],
        2: [1],
        3: [2],
    }
    return plan


def test_plot_plan_reallocation_before_after_saves_file():
    battlefield_before, battlefield_after = build_reallocation_visualization_battlefields()
    event = Event(
        type=EventType.THREAT_ADDED,
        data={'threat': battlefield_after.threats[-1]},
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = os.path.join(tmp_dir, 'plan_reallocation_before_after.png')
        fig, axes = plot_plan_reallocation_before_after(
            battlefield_before,
            build_before_plan(),
            battlefield_after,
            build_after_plan(),
            title_before='事件前任务序列',
            title_after='重分配后任务序列',
            event=event,
            output_path=output_path,
        )

        assert_true(os.path.exists(output_path), '任务序列版重分配对比图应保存到指定路径')
        assert_true(os.path.getsize(output_path) > 0, '输出图片文件不应为空')
        assert_true(len(axes) == 2, '重分配对比图应包含事件前和重分配后两个子图')
        assert_true(len(axes[0].lines) >= 2, '事件前子图应绘制任务链')
        assert_true(len(axes[1].lines) >= 2, '重分配后子图应绘制任务链')
        assert_true(len(axes[1].patches) >= len(battlefield_after.threats), '重分配后子图应绘制威胁区及事件高亮')
        fig.clf()


def test_plot_plan_reallocation_diff_saves_file():
    _, battlefield_after = build_reallocation_visualization_battlefields()
    event = Event(
        type=EventType.THREAT_ADDED,
        data={'threat': battlefield_after.threats[-1]},
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = os.path.join(tmp_dir, 'plan_reallocation_diff.png')
        fig, ax = plot_plan_reallocation_diff(
            battlefield_after,
            build_before_plan(),
            build_after_plan(),
            title='任务链变化示意图',
            event=event,
            output_path=output_path,
        )

        assert_true(os.path.exists(output_path), '任务链变化图应保存到指定路径')
        assert_true(os.path.getsize(output_path) > 0, '输出图片文件不应为空')
        assert_true(len(ax.lines) >= 3, '任务链变化图应绘制保持、释放和新增航段')
        assert_true(len(ax.patches) >= len(battlefield_after.threats), '任务链变化图应绘制威胁区和事件高亮')
        fig.clf()


def test_plot_plan_reallocation_target_loads_saves_file():
    battlefield_before, battlefield_after = build_reallocation_visualization_battlefields()
    battlefield_after.get_target(1).required_uavs = 2
    after_plan = build_after_plan()
    after_plan.uav_task_sequences[2].append_target(1)
    after_plan.target_assignees[1] = [1, 2]

    event = Event(
        type=EventType.TARGET_DEMAND_INCREASED,
        data={
            'target_id': 1,
            'old_required_uavs': 1,
            'new_required_uavs': 2,
        },
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = os.path.join(tmp_dir, 'plan_reallocation_target_loads.png')
        fig, ax = plot_plan_reallocation_target_loads(
            battlefield_before,
            build_before_plan(),
            battlefield_after,
            after_plan,
            title='目标需求满足变化图',
            event=event,
            output_path=output_path,
        )

        assert_true(os.path.exists(output_path), '目标需求满足变化图应保存到指定路径')
        assert_true(os.path.getsize(output_path) > 0, '输出图片文件不应为空')
        assert_true(len(ax.patches) >= 2, '目标需求满足变化图应绘制事件前后分配数量柱')
        assert_true(len(ax.collections) >= 1, '目标需求满足变化图应绘制需求数量标记')
        fig.clf()


def test_plot_plan_reallocation_uav_loads_saves_file():
    battlefield_before, battlefield_after = build_reallocation_visualization_battlefields()

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = os.path.join(tmp_dir, 'plan_reallocation_uav_loads.png')
        fig, ax = plot_plan_reallocation_uav_loads(
            battlefield_before,
            build_before_plan(),
            battlefield_after,
            build_after_plan(),
            title='UAV任务负载变化图',
            output_path=output_path,
        )

        assert_true(os.path.exists(output_path), 'UAV任务负载变化图应保存到指定路径')
        assert_true(os.path.getsize(output_path) > 0, '输出图片文件不应为空')
        assert_true(len(ax.lines) >= len(battlefield_after.uavs), 'UAV任务负载变化图应绘制前后负载连线')
        assert_true(len(ax.collections) >= 2, 'UAV任务负载变化图应绘制事件前后负载点')
        fig.clf()


def test_plot_mcha_winning_bids_saves_file():
    selected_bids = [
        BidResult(uav_id=0, target_id=1, score=-8.2),
        BidResult(uav_id=2, target_id=3, score=-4.6),
        BidResult(uav_id=1, target_id=2, score=-6.1),
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = os.path.join(tmp_dir, 'mcha_winning_bids.png')
        fig, ax = plot_mcha_winning_bids(
            selected_bids,
            title='MCHA中标结果图',
            output_path=output_path,
        )

        assert_true(os.path.exists(output_path), 'MCHA中标结果图应保存到指定路径')
        assert_true(os.path.getsize(output_path) > 0, '输出图片文件不应为空')
        assert_true(len(ax.patches) >= len(selected_bids), 'MCHA中标结果图应绘制每个中标结果')
        fig.clf()


def test_plot_mcha_candidate_bid_scores_saves_file():
    bid_round_logs = [
        BidRoundLog(
            iteration=1,
            active_targets=[1],
            available_uavs=[0, 1, 2],
            candidate_bids=[
                BidResult(uav_id=0, target_id=1, score=-8.2),
                BidResult(uav_id=1, target_id=1, score=-5.4),
                BidResult(uav_id=2, target_id=1, score=-6.1),
            ],
            accepted_bids=[
                BidResult(uav_id=1, target_id=1, score=-5.4),
            ],
        ),
        BidRoundLog(
            iteration=2,
            active_targets=[3],
            available_uavs=[0, 2],
            candidate_bids=[
                BidResult(uav_id=0, target_id=3, score=-7.5),
                BidResult(uav_id=2, target_id=3, score=-4.6),
            ],
            accepted_bids=[
                BidResult(uav_id=2, target_id=3, score=-4.6),
            ],
        ),
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = os.path.join(tmp_dir, 'mcha_candidate_bid_scores.png')
        fig, ax = plot_mcha_candidate_bid_scores(
            bid_round_logs,
            title='MCHA候选投标得分图',
            output_path=output_path,
        )

        assert_true(os.path.exists(output_path), 'MCHA候选投标得分图应保存到指定路径')
        assert_true(os.path.getsize(output_path) > 0, '输出图片文件不应为空')
        assert_true(len(ax.collections) >= 2, 'MCHA候选投标得分图应绘制候选投标和中标投标散点')
        fig.clf()


def test_plot_mcha_open_demand_repair_saves_file():
    initial_remaining_demand = {
        1: 2,
        3: 1,
    }
    bid_round_logs = [
        BidRoundLog(
            iteration=1,
            active_targets=[1, 3],
            available_uavs=[0, 1, 2],
            candidate_bids=[
                BidResult(uav_id=0, target_id=1, score=-5.2),
                BidResult(uav_id=1, target_id=3, score=-4.6),
            ],
            accepted_bids=[
                BidResult(uav_id=0, target_id=1, score=-5.2),
                BidResult(uav_id=1, target_id=3, score=-4.6),
            ],
        ),
        BidRoundLog(
            iteration=2,
            active_targets=[1],
            available_uavs=[2],
            candidate_bids=[
                BidResult(uav_id=2, target_id=1, score=-3.8),
            ],
            accepted_bids=[
                BidResult(uav_id=2, target_id=1, score=-3.8),
            ],
        ),
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = os.path.join(tmp_dir, 'mcha_open_demand_repair.png')
        fig, axes = plot_mcha_open_demand_repair(
            initial_remaining_demand,
            bid_round_logs,
            title='开放任务需求修复过程图',
            output_path=output_path,
        )

        assert_true(os.path.exists(output_path), '开放任务需求修复过程图应保存到指定路径')
        assert_true(os.path.getsize(output_path) > 0, '输出图片文件不应为空')
        assert_true(len(axes) == 2, '开放任务需求修复过程图应包含热力图和总需求曲线')
        fig.clf()


def test_write_reallocation_cost_change_csv_saves_file():
    battlefield_before, battlefield_after = build_reallocation_visualization_battlefields()
    weights = {
        'w1': 1.0,
        'w2': 0.1,
        'w3': 1.0,
        'w4': 1.0,
        'alpha': 1.0,
        'sync_window': 0.05,
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = os.path.join(tmp_dir, 'reallocation_cost_change.csv')
        rows = write_reallocation_cost_change_csv(
            battlefield_before,
            build_before_plan(),
            battlefield_after,
            build_before_plan(),
            build_after_plan(),
            weights,
            output_path,
        )

        assert_true(os.path.exists(output_path), '重分配代价变化 CSV 应保存到指定路径')
        assert_true(os.path.getsize(output_path) > 0, '输出 CSV 文件不应为空')
        assert_true(len(rows) >= 6, '重分配代价变化 CSV 应包含主要代价指标')


if __name__ == '__main__':
    test_plot_plan_reallocation_before_after_saves_file()
    test_plot_plan_reallocation_diff_saves_file()
    test_plot_plan_reallocation_target_loads_saves_file()
    test_plot_plan_reallocation_uav_loads_saves_file()
    test_plot_mcha_winning_bids_saves_file()
    test_plot_mcha_candidate_bid_scores_saves_file()
    test_plot_mcha_open_demand_repair_saves_file()
    test_write_reallocation_cost_change_csv_saves_file()
    print('通过 8 项任务序列版重分配可视化测试')
