"""
生成任务序列版重分配论文图。
"""
import os
import sys
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config.params import MCHA, MCHA_TEST, WEIGHTS
from data.scenario_reallocation import create_reallocation_scenario
from src.core.models import Target, Threat
from src.pre_allocation.pso import run_pso
from src.re_allocation.events import (
    Event,
    EventType,
    analyze_plan_event_impact,
    apply_event_to_battlefield,
)
from src.re_allocation.mcha import run_mcha_for_plan
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


RESULT_DIR = 'results/reallocation'


def build_uav_lost_event() -> Event:
    return Event(
        type=EventType.UAV_LOST,
        data={'uav_id': MCHA_TEST['lost_uav_id']},
    )


def build_target_added_event() -> Event:
    return Event(
        type=EventType.TARGET_ADDED,
        data={
            'target': Target(
                id=MCHA_TEST['target_added_id'],
                x=MCHA_TEST['target_added_x'],
                y=MCHA_TEST['target_added_y'],
                value=MCHA_TEST['target_added_value'],
                required_uavs=MCHA_TEST['target_added_required_uavs'],
            )
        },
    )


def build_threat_added_event() -> Event:
    return Event(
        type=EventType.THREAT_ADDED,
        data={
            'threat': Threat(
                id=MCHA_TEST['new_threat_id'],
                x=MCHA_TEST['new_threat_x'],
                y=MCHA_TEST['new_threat_y'],
                radius=MCHA_TEST['new_threat_radius'],
            ),
            'threat_threshold': MCHA_TEST['threat_threshold'],
        },
    )


def build_event(event_mode: str) -> Event:
    if event_mode == 'target_added':
        return build_target_added_event()
    if event_mode == 'target_demand_increased':
        return Event(
            type=EventType.TARGET_DEMAND_INCREASED,
            data={
                'target_id': MCHA_TEST['target_demand_increase_target_id'],
                'new_required_uavs': MCHA_TEST['target_demand_increase_new_required_uavs'],
            },
        )
    if event_mode == 'threat_added':
        return build_threat_added_event()
    return build_uav_lost_event()


def event_title(event: Event) -> str:
    if event.type == EventType.UAV_LOST:
        return f"UAV 损失事件（U{event.data['uav_id']}）"
    if event.type == EventType.TARGET_ADDED:
        target = event.data['target']
        return f"新增目标事件（T{target.id}）"
    if event.type == EventType.TARGET_DEMAND_INCREASED:
        return (
            f"目标需求增加事件"
            f"（T{event.data['target_id']} -> {event.data['new_required_uavs']}架）"
        )
    if event.type == EventType.THREAT_ADDED:
        threat = event.data['threat']
        return f"新增威胁事件（Threat-{threat.id}）"
    return event.type.value


def main():
    event_mode = os.environ.get('MCHA_EVENT', 'uav_lost').strip().lower()
    seed = int(os.environ.get('MCHA_SEED', '42'))
    np.random.seed(seed)

    battlefield_before = create_reallocation_scenario()
    print('正在运行 PSO 预分配...')
    _, _, curve, plan_before = run_pso(
        battlefield_before,
        WEIGHTS,
        return_assignment_plan=True,
    )
    print(f'PSO 完成，最终适应度: {curve[-1]:.4f}')

    event = build_event(event_mode)
    if event.type == EventType.TARGET_DEMAND_INCREASED:
        target_id = event.data['target_id']
        event.data['old_required_uavs'] = battlefield_before.get_target(target_id).required_uavs

    battlefield_after = deepcopy(battlefield_before)
    apply_event_to_battlefield(event, battlefield_after)

    state = analyze_plan_event_impact(event, battlefield_after, plan_before)
    result = run_mcha_for_plan(battlefield_after, WEIGHTS, state, MCHA)

    event_name = event.type.value
    event_result_dir = os.path.join(RESULT_DIR, event_name)
    before_after_output_path = os.path.join(event_result_dir, 'task_sequence_before_after.png')
    diff_output_path = os.path.join(event_result_dir, 'task_sequence_diff.png')
    target_loads_output_path = os.path.join(event_result_dir, 'target_loads.png')
    uav_loads_output_path = os.path.join(event_result_dir, 'uav_loads.png')
    winning_bids_output_path = os.path.join(event_result_dir, 'winning_bids.png')
    candidate_bids_output_path = os.path.join(event_result_dir, 'candidate_bids.png')
    demand_repair_output_path = os.path.join(event_result_dir, 'demand_repair.png')
    cost_change_output_path = os.path.join(event_result_dir, 'cost_change.csv')
    plot_plan_reallocation_before_after(
        battlefield_before,
        plan_before,
        battlefield_after,
        result.assignment_plan,
        title_before=f'事件前任务序列（seed={seed}）',
        title_after=f'重分配后任务序列：{event_title(event)}',
        event=event,
        output_path=before_after_output_path,
    )
    plot_plan_reallocation_diff(
        battlefield_after,
        plan_before,
        result.assignment_plan,
        title=f'任务链变化示意图：{event_title(event)}',
        event=event,
        output_path=diff_output_path,
    )
    plot_plan_reallocation_target_loads(
        battlefield_before,
        plan_before,
        battlefield_after,
        result.assignment_plan,
        title=f'目标需求满足变化图：{event_title(event)}',
        event=event,
        output_path=target_loads_output_path,
    )
    plot_plan_reallocation_uav_loads(
        battlefield_before,
        plan_before,
        battlefield_after,
        result.assignment_plan,
        title=f'UAV任务负载变化图：{event_title(event)}',
        output_path=uav_loads_output_path,
    )
    plot_mcha_winning_bids(
        result.selected_bids,
        title=f'MCHA中标结果图：{event_title(event)}',
        output_path=winning_bids_output_path,
    )
    plot_mcha_candidate_bid_scores(
        result.bid_round_logs,
        title=f'MCHA每轮最优投标与并行中标结果图：{event_title(event)}',            # 每架 UAV 每轮提交一个最优投标；每个目标按剩余需求接收多个中标 UAV
        output_path=candidate_bids_output_path,
    )
    plot_mcha_open_demand_repair(
        state.remaining_demand,
        result.bid_round_logs,
        title=f'开放任务需求修复过程图：{event_title(event)}',
        output_path=demand_repair_output_path,
    )
    write_reallocation_cost_change_csv(
        battlefield_before,
        plan_before,
        battlefield_after,
        state.locked_plan,
        result.assignment_plan,
        WEIGHTS,
        cost_change_output_path,
    )

    print(f'当前事件: {event_title(event)}')
    print(f'open_targets: {state.open_targets}')
    print(f'remaining_demand_before: {state.remaining_demand}')
    print(f'remaining_demand_after: {result.remaining_demand}')
    print(f'iterations: {result.iterations}')
    print(f'selected_bids: {len(result.selected_bids)}')
    print('图表已保存到:')
    print(f'- {before_after_output_path}')
    print(f'- {diff_output_path}')
    print(f'- {target_loads_output_path}')
    print(f'- {uav_loads_output_path}')
    print(f'- {winning_bids_output_path}')
    print(f'- {candidate_bids_output_path}')
    print(f'- {demand_repair_output_path}')
    print(f'- {cost_change_output_path}')

    plt.show()


if __name__ == '__main__':
    main()
