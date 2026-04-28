"""
论文支撑类可视化测试。
"""
import os
import sys
import tempfile

import matplotlib

matplotlib.use('Agg')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.core.models import Battlefield, Target, Threat, UAV
from src.visualization.paper_support import (
    plot_scenario_elements,
    plot_system_workflow,
)


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def build_paper_support_battlefield() -> Battlefield:
    return Battlefield(
        uavs=[
            UAV(id=0, x=0.0, y=0.0, speed=100.0, ammo=3, range_left=300.0),
            UAV(id=1, x=0.0, y=50.0, speed=100.0, ammo=3, range_left=300.0),
            UAV(id=2, x=0.0, y=90.0, speed=100.0, ammo=3, range_left=300.0),
        ],
        targets=[
            Target(id=0, x=42.0, y=12.0, value=8.0, required_uavs=1),
            Target(id=1, x=64.0, y=26.0, value=9.0, required_uavs=2),
            Target(id=2, x=52.0, y=58.0, value=7.5, required_uavs=1),
            Target(id=3, x=70.0, y=82.0, value=8.5, required_uavs=1),
        ],
        threats=[Threat(id=0, x=32.0, y=42.0, radius=9.0)],
        map_size=(100.0, 100.0),
    )


def test_plot_scenario_elements_saves_file():
    battlefield = build_paper_support_battlefield()

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = os.path.join(tmp_dir, 'scenario_elements.png')
        fig, ax = plot_scenario_elements(
            battlefield,
            title='战场场景要素建模图',
            output_path=output_path,
        )

        assert_true(os.path.exists(output_path), '场景要素建模图应保存到指定路径')
        assert_true(os.path.getsize(output_path) > 0, '输出图片文件不应为空')
        assert_true(len(ax.collections) >= 2, '场景要素建模图应绘制 UAV 和目标散点')
        assert_true(len(ax.patches) >= len(battlefield.threats), '场景要素建模图应绘制威胁区')
        fig.clf()


def test_plot_system_workflow_saves_file():
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = os.path.join(tmp_dir, 'system_workflow.png')
        fig, ax = plot_system_workflow(
            title='无人集群协同打击任务规划完整流程',
            output_path=output_path,
        )

        assert_true(os.path.exists(output_path), '完整流程示意图应保存到指定路径')
        assert_true(os.path.getsize(output_path) > 0, '输出图片文件不应为空')
        assert_true(len(ax.patches) >= 7, '完整流程示意图应绘制主要流程节点')
        fig.clf()


if __name__ == '__main__':
    test_plot_scenario_elements_saves_file()
    test_plot_system_workflow_saves_file()
    print('通过 2 项论文支撑类可视化测试')
