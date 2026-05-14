"""
生成预分配与重分配结果对应的最终任务链航迹图。

运行示例：
    python scripts/visualize_final_routes.py

可选环境变量：
    ROUTE_MODE=all|preallocation|reallocation
    ROUTE_SCENARIO=small|medium|hard
    ROUTE_EVENTS=uav_lost,target_added,threat_added
    ROUTE_SEED=42
"""
import os
import sys
from copy import deepcopy
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config.params import ASTAR, MCHA, MCHA_TEST, WEIGHTS
from data.scenario_hard import create_hard_scenario
from data.scenario_medium import create_medium_scenario
from data.scenario_reallocation import create_reallocation_scenario
from data.scenario_small import create_small_scenario
from src.core.models import Target, Threat
from src.pre_allocation.pso import run_pso
from src.re_allocation.events import (
    Event,
    EventType,
    analyze_plan_event_impact,
    apply_event_to_battlefield,
)
from src.re_allocation.mcha import run_mcha_for_plan
from src.route_planning.planner import plan_routes_for_assignment_plan
from src.visualization.route_planning import (
    plot_assignment_route_plan,
    write_route_plan_summary_csv,
)


RESULT_DIR = 'results/route_planning'
DEFAULT_ROUTE_EVENTS = ('uav_lost', 'target_added', 'threat_added')


def load_preallocation_scenario(name: str):
    if name == 'small':
        return create_small_scenario()
    if name == 'hard':
        return create_hard_scenario()
    return create_medium_scenario()


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
    if event_mode == 'threat_added':
        return build_threat_added_event()
    if event_mode == 'uav_lost':
        return build_uav_lost_event()
    raise ValueError(f'不支持的重分配事件: {event_mode}')


def event_title(event: Event) -> str:
    if event.type == EventType.UAV_LOST:
        return f"UAV 损失事件（U{event.data['uav_id']}）"
    if event.type == EventType.TARGET_ADDED:
        target = event.data['target']
        return f"新增目标事件（T{target.id}）"
    if event.type == EventType.THREAT_ADDED:
        threat = event.data['threat']
        return f"新增威胁事件（Threat-{threat.id}）"
    return event.type.value


def save_route_outputs(battlefield, route_plan, title: str, image_path: str, csv_path: str) -> None:
    fig, _ = plot_assignment_route_plan(
        battlefield,
        route_plan,
        title=title,
        output_path=image_path,
        safety_margin=float(ASTAR.get('safety_margin', 0.0)),
    )
    write_route_plan_summary_csv(route_plan, csv_path)
    plt.close(fig)


def print_route_status(route_plan) -> None:
    print(f'航迹规划成功: {route_plan.success}, 活跃 UAV: {route_plan.active_uav_count}, 航段数: {route_plan.segment_count}')
    if not route_plan.failed_segments:
        return

    print('失败航段:')
    for segment in route_plan.failed_segments:
        print(
            f'- U{segment.uav_id}: '
            f'{segment.start_kind}:{segment.start_id} -> T{segment.end_target_id} '
            f'({segment.failure_reason})'
        )


def generate_preallocation_routes(scenario_name: str, seed: int) -> None:
    np.random.seed(seed)
    battlefield = load_preallocation_scenario(scenario_name)
    print(f'正在生成预分配最终航迹: scenario={scenario_name}, seed={seed}')

    _, _, curve, assignment_plan = run_pso(
        battlefield,
        WEIGHTS,
        return_assignment_plan=True,
    )
    route_plan = plan_routes_for_assignment_plan(
        battlefield,
        assignment_plan,
        params=ASTAR,
        source=f'preallocation_{scenario_name}',
    )

    output_dir = os.path.join(RESULT_DIR, 'preallocation')
    image_path = os.path.join(output_dir, f'{scenario_name}_final_routes.png')
    csv_path = os.path.join(output_dir, f'{scenario_name}_final_routes.csv')
    save_route_outputs(
        battlefield,
        route_plan,
        title=f'预分配最终任务链航迹图（{scenario_name}, seed={seed}）',
        image_path=image_path,
        csv_path=csv_path,
    )

    print(f'预分配适应度: {curve[-1]:.4f}')
    print_route_status(route_plan)
    print(f'- {image_path}')
    print(f'- {csv_path}')


def generate_reallocation_routes(event_mode: str, seed: int) -> None:
    np.random.seed(seed)
    battlefield_before = create_reallocation_scenario()
    event = build_event(event_mode)
    print(f'正在生成重分配最终航迹: event={event_mode}, seed={seed}')

    _, _, curve, plan_before = run_pso(
        battlefield_before,
        WEIGHTS,
        return_assignment_plan=True,
    )

    battlefield_after = deepcopy(battlefield_before)
    apply_event_to_battlefield(event, battlefield_after)
    state = analyze_plan_event_impact(event, battlefield_after, plan_before)
    result = run_mcha_for_plan(battlefield_after, WEIGHTS, state, MCHA)

    route_plan = plan_routes_for_assignment_plan(
        battlefield_after,
        result.assignment_plan,
        params=ASTAR,
        source=f'reallocation_{event_mode}',
    )

    output_dir = os.path.join(RESULT_DIR, 'reallocation', event_mode)
    image_path = os.path.join(output_dir, 'final_routes.png')
    csv_path = os.path.join(output_dir, 'final_routes.csv')
    save_route_outputs(
        battlefield_after,
        route_plan,
        title=f'重分配后最终任务链航迹图：{event_title(event)}',
        image_path=image_path,
        csv_path=csv_path,
    )

    print(f'预分配适应度: {curve[-1]:.4f}')
    print(f'重分配轮次: {result.iterations}, 中标数: {len(result.selected_bids)}')
    print_route_status(route_plan)
    print(f'- {image_path}')
    print(f'- {csv_path}')


def parse_events(raw_events: str | None) -> Iterable[str]:
    if not raw_events:
        return DEFAULT_ROUTE_EVENTS
    return tuple(event.strip().lower() for event in raw_events.split(',') if event.strip())


def main() -> None:
    mode = os.environ.get('ROUTE_MODE', 'all').strip().lower()
    scenario_name = os.environ.get('ROUTE_SCENARIO', 'medium').strip().lower()
    seed = int(os.environ.get('ROUTE_SEED', '42'))
    events = parse_events(os.environ.get('ROUTE_EVENTS'))

    if mode not in {'all', 'preallocation', 'reallocation'}:
        raise ValueError('ROUTE_MODE 只能为 all、preallocation 或 reallocation')

    if mode in {'all', 'preallocation'}:
        generate_preallocation_routes(scenario_name, seed)

    if mode in {'all', 'reallocation'}:
        for event_mode in events:
            generate_reallocation_routes(event_mode, seed)

    print('最终航迹图生成完成。')


if __name__ == '__main__':
    main()
