"""
需求变化事件可视化脚本
支持：
1. TARGET_DEMAND_INCREASED
2. TARGET_DEMAND_DECREASED

输出图表：
- 事件前后对比图
- 分配变化图
- 目标满足情况前后对比图
- 需求减少释放评分图（仅 decrease）
"""
import os
import sys
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from config.params import WEIGHTS, MCHA, MCHA_TEST
from data.scenario_reallocation import create_reallocation_scenario
from src.re_allocation.events import (
    Event,
    EventType,
    analyze_event_impact,
    apply_event_to_battlefield,
    retention_score,
)
from src.re_allocation.mcha import run_mcha
from src.pre_allocation.pso import run_pso
from src.visualization.common import ensure_output_dir
from src.visualization.reallocation import (
    plot_assignment_diff,
    plot_reallocation_before_after,
    plot_reallocation_target_loads,
)


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

RESULT_DIR = 'results/reallocation'


def build_target_demand_increased_event():
    return Event(
        type=EventType.TARGET_DEMAND_INCREASED,
        data={
            'target_id': MCHA_TEST['target_demand_increase_target_id'],
            'new_required_uavs': MCHA_TEST['target_demand_increase_new_required_uavs'],
        },
    )


def build_target_demand_decreased_event():
    return Event(
        type=EventType.TARGET_DEMAND_DECREASED,
        data={
            'target_id': MCHA_TEST['target_demand_decrease_target_id'],
            'new_required_uavs': MCHA_TEST['target_demand_decrease_new_required_uavs'],
        },
    )


def plot_retention_scores(battlefield, assignment, target_id, output_path=None):
    target = battlefield.get_target(target_id)
    assigned_uavs = np.where(assignment[:, target_id] == 1)[0].tolist()
    scored = []
    for uav_id in assigned_uavs:
        uav = battlefield.get_uav(uav_id)
        score = retention_score(uav, target, assignment, battlefield)
        scored.append((uav_id, score))

    scored.sort(key=lambda item: item[1])

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    labels = [f'UAV-{uav_id}' for uav_id, _ in scored]
    values = [score for _, score in scored]
    colors = ['#d62728'] + ['#1f77b4'] * (len(values) - 1)

    ax.bar(labels, values, color=colors)
    ax.set_xlabel('候选无人机')
    ax.set_ylabel('保留得分')
    ax.set_title(f'目标{target_id}需求减少时的保留得分对比图')
    ax.grid(True, axis='y', alpha=0.3)

    if output_path is not None:
        ensure_output_dir(output_path)
        fig.savefig(output_path, dpi=180)

    return fig, ax


def main():
    battlefield_before = create_reallocation_scenario()

    print('正在运行PSO预分配...')
    assignment_before, etas_before, curve = run_pso(battlefield_before, WEIGHTS)
    print(f'PSO完成，最终适应度: {curve[-1]:.4f}')

    event_mode = os.environ.get('MCHA_EVENT', 'target_demand_increased').strip().lower()
    if event_mode == 'target_demand_decreased':
        event = build_target_demand_decreased_event()
    else:
        event = build_target_demand_increased_event()

    battlefield_after = deepcopy(battlefield_before)
    apply_event_to_battlefield(event, battlefield_after)
    state = analyze_event_impact(event, battlefield_after, assignment_before, etas_before)
    result = run_mcha(battlefield_after, WEIGHTS, state, MCHA)

    print(f'当前事件: {event.type.value}')
    print(f'target_id: {event.data["target_id"]}')
    print(f'new_required_uavs: {event.data["new_required_uavs"]}')
    print(f'open_targets: {state.open_targets}')
    print(f'remaining_demand: {state.remaining_demand}')

    event_name = event.type.value
    plot_reallocation_before_after(
        battlefield_before,
        assignment_before,
        battlefield_after,
        result.assignment,
        title_before='事件前任务分配',
        title_after='重分配后任务分配',
        output_path=os.path.join(RESULT_DIR, f'{event_name}_before_after.png'),
    )

    plot_assignment_diff(
        battlefield_after,
        assignment_before,
        result.assignment,
        title='任务分配变化边示意图',
        output_path=os.path.join(RESULT_DIR, f'{event_name}_diff.png'),
    )

    plot_reallocation_target_loads(
        battlefield_before,
        assignment_before,
        battlefield_after,
        result.assignment,
        title='重分配前后目标需求满足情况对比图',
        output_path=os.path.join(RESULT_DIR, f'{event_name}_target_loads.png'),
    )

    if event.type == EventType.TARGET_DEMAND_DECREASED:
        plot_retention_scores(
            battlefield_before,
            assignment_before,
            event.data['target_id'],
            output_path=os.path.join(RESULT_DIR, f'{event_name}_retention_scores.png'),
        )
        print(f'- {event_name}_retention_scores.png')

    print('\n图表已保存到 results/reallocation/ 目录：')
    print(f'- {event_name}_before_after.png')
    print(f'- {event_name}_diff.png')
    print(f'- {event_name}_target_loads.png')

    plt.show()


if __name__ == '__main__':
    main()
