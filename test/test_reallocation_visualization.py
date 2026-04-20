"""
重分配可视化脚本
支持以下事件的论文图生成：
1. UAV_LOST
2. THREAT_ADDED
3. TARGET_ADDED

输出图表：
- 事件前后对比图
- 分配变化图
- 目标满足情况前后对比图
"""
import os
import sys
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from config.params import WEIGHTS, MCHA, MCHA_TEST
from data.scenario_reallocation import create_reallocation_scenario
from src.core.models import Threat, Target
from src.pre_allocation.pso import run_pso
from src.re_allocation.events import Event, EventType, analyze_event_impact, apply_event_to_battlefield
from src.re_allocation.mcha import run_mcha
from src.visualization.reallocation import (
    plot_assignment_diff,
    plot_reallocation_before_after,
    plot_reallocation_target_loads,
)


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

RESULT_DIR = 'results/reallocation'


def expand_assignment_for_display(assignment, new_target_count):
    if new_target_count <= 0:
        return assignment
    extra = np.zeros((assignment.shape[0], new_target_count), dtype=assignment.dtype)
    return np.hstack((assignment, extra))


def build_uav_lost_event():
    return Event(
        type=EventType.UAV_LOST,
        data={'uav_id': MCHA_TEST['lost_uav_id']},
    )


def build_threat_added_event():
    new_threat = Threat(
        id=MCHA_TEST['new_threat_id'],
        x=MCHA_TEST['new_threat_x'],
        y=MCHA_TEST['new_threat_y'],
        radius=MCHA_TEST['new_threat_radius'],
    )
    return Event(
        type=EventType.THREAT_ADDED,
        data={
            'threat': new_threat,
            'threat_threshold': MCHA_TEST['threat_threshold'],
        },
    )


def build_target_added_event():
    new_target = Target(
        id=MCHA_TEST['target_added_id'],
        x=MCHA_TEST['target_added_x'],
        y=MCHA_TEST['target_added_y'],
        value=MCHA_TEST['target_added_value'],
        required_uavs=MCHA_TEST['target_added_required_uavs'],
    )
    return Event(
        type=EventType.TARGET_ADDED,
        data={'target': new_target},
    )


def get_event_and_assignments(event_mode, assignment):
    if event_mode == 'threat_added':
        return build_threat_added_event(), assignment
    if event_mode == 'target_added':
        return build_target_added_event(), expand_assignment_for_display(assignment, 1)
    return build_uav_lost_event(), assignment


def event_title(event):
    if event.type == EventType.UAV_LOST:
        return f"UAV损失事件（UAV-{event.data['uav_id']}）"
    if event.type == EventType.THREAT_ADDED:
        threat = event.data['threat']
        return f"新增威胁事件（Threat-{threat.id}）"
    if event.type == EventType.TARGET_ADDED:
        target = event.data['target']
        return f"新增目标事件（Target-{target.id}）"
    return event.type.value


def main():
    battlefield_before = create_reallocation_scenario()

    print('正在运行PSO预分配...')
    assignment_before, etas_before, curve = run_pso(battlefield_before, WEIGHTS)
    print(f'PSO完成，最终适应度: {curve[-1]:.4f}')

    event_mode = os.environ.get('MCHA_EVENT', 'uav_lost').strip().lower()
    event, assignment_before_for_plot = get_event_and_assignments(event_mode, assignment_before)

    battlefield_after = deepcopy(battlefield_before)
    apply_event_to_battlefield(event, battlefield_after)
    state = analyze_event_impact(event, battlefield_after, assignment_before, etas_before)
    result = run_mcha(battlefield_after, WEIGHTS, state, MCHA)

    print(f'当前事件: {event_title(event)}')
    print(f'open_targets: {state.open_targets}')
    print(f'remaining_demand: {state.remaining_demand}')
    print(f'iterations: {result.iterations}')
    print(f'selected_bids: {len(result.selected_bids)}')

    event_name = event.type.value
    plot_reallocation_before_after(
        battlefield_before,
        assignment_before_for_plot,
        battlefield_after,
        result.assignment,
        title_before='事件前任务分配',
        title_after='重分配后任务分配',
        output_path=os.path.join(RESULT_DIR, f'{event_name}_before_after.png'),
    )

    plot_assignment_diff(
        battlefield_after,
        assignment_before_for_plot,
        result.assignment,
        title='任务分配变化边示意图',
        output_path=os.path.join(RESULT_DIR, f'{event_name}_diff.png'),
    )

    plot_reallocation_target_loads(
        battlefield_before,
        assignment_before_for_plot,
        battlefield_after,
        result.assignment,
        title='重分配前后目标需求满足情况对比图',
        output_path=os.path.join(RESULT_DIR, f'{event_name}_target_loads.png'),
    )

    print('\n图表已保存到 results/reallocation/ 目录：')
    print(f'- {event_name}_before_after.png')
    print(f'- {event_name}_diff.png')
    print(f'- {event_name}_target_loads.png')

    plt.show()


if __name__ == '__main__':
    main()
