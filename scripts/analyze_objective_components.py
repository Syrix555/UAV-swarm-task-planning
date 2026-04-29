"""
统计预分配目标函数各分项的数量级。

阶段 1 只做数据采集，不修改目标函数权重。输出结果用于后续归一化、
AHP 初始权重确定和灵敏度实验。
"""
import csv
import os
import sys
from collections import defaultdict
from statistics import mean, pstdev

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config.params import PSO, WEIGHTS
from data.scenario_hard import create_hard_scenario
from data.scenario_medium import create_medium_scenario
from data.scenario_small import create_small_scenario
from src.core.models import AssignmentPlan, Battlefield
from src.core.sequence_eval import evaluate_uav_task_sequence
from src.pre_allocation.pso import run_pso
from src.visualization.common import ensure_output_dir


RESULT_DIR = 'results/weight_analysis'
DEFAULT_SEEDS = list(range(10))
QUICK_SEEDS = [0, 1]
PENALTY = 1e6


COMPONENT_FIELDS = [
    'scenario',
    'seed',
    'final_fitness',
    'distance_cost',
    'threat_cost',
    'explicit_time_window_penalty',
    'cooperative_time_window_penalty',
    'time_window_penalty',
    'task_reward',
    'constraint_penalty',
    'weighted_distance',
    'weighted_threat',
    'weighted_time_window',
    'weighted_reward',
    'objective_without_constraint',
    'target_satisfaction_rate',
    'sync_violation_count',
    'max_sync_gap',
    'assigned_task_count',
    'active_uav_count',
]


SUMMARY_FIELDS = [
    'scenario',
    'metric',
    'mean',
    'std',
    'min',
    'p50',
    'p90',
    'max',
]


def load_scenario(name: str) -> Battlefield:
    if name == 'small':
        return create_small_scenario()
    if name == 'medium':
        return create_medium_scenario()
    if name == 'hard':
        return create_hard_scenario()
    raise ValueError(f'Unsupported scenario: {name}')


def parse_scenarios() -> list[str]:
    raw = os.environ.get('OBJECTIVE_SCENARIO', os.environ.get('PSO_SCENARIO', 'medium'))
    scenarios = [item.strip().lower() for item in raw.split(',') if item.strip()]
    if scenarios == ['all']:
        return ['small', 'medium', 'hard']
    return scenarios


def parse_seeds(quick_mode: bool) -> list[int]:
    raw = os.environ.get('OBJECTIVE_SEEDS')
    if raw:
        return [int(item.strip()) for item in raw.split(',') if item.strip()]
    return QUICK_SEEDS if quick_mode else DEFAULT_SEEDS


def build_pso_params(quick_mode: bool) -> dict:
    params = dict(PSO)
    if quick_mode:
        params['num_particles'] = 30
        params['max_iter'] = 80
    return params


def collect_objective_components(
    battlefield: Battlefield,
    plan: AssignmentPlan,
    weights: dict,
) -> dict[str, float | int]:
    distance_cost = 0.0
    threat_cost = 0.0
    explicit_time_window_penalty = 0.0
    cooperative_arrivals: dict[int, list[float]] = defaultdict(list)
    assigned_target_ids: set[int] = set()
    constraint_penalty = 0.0
    task_counts = []

    for sequence in plan.uav_task_sequences.values():
        uav = battlefield.get_uav(sequence.uav_id)
        task_counts.append(sequence.task_count())
        evaluated = evaluate_uav_task_sequence(
            battlefield,
            sequence,
            alpha=weights['alpha'],
        )

        distance_cost += evaluated.total_distance
        explicit_time_window_penalty += evaluated.time_window_penalty

        if not evaluated.is_ammo_feasible:
            constraint_penalty += PENALTY * (sequence.task_count() - uav.ammo)
        if not evaluated.is_range_feasible:
            constraint_penalty += PENALTY * (evaluated.total_distance - uav.range_left)

        for task in evaluated.evaluated_sequence.tasks:
            cooperative_arrivals[task.target_id].append(task.planned_arrival_time)
            assigned_target_ids.add(task.target_id)

        current_x, current_y = uav.x, uav.y
        for task in sequence.tasks:
            target = battlefield.get_target(task.target_id)
            threat_cost += battlefield.threat_cost_on_line(
                current_x,
                current_y,
                target.x,
                target.y,
            )
            current_x, current_y = target.x, target.y

    target_satisfied_count = 0
    for target in battlefield.targets:
        assigned_count = len(plan.target_assignees.get(target.id, []))
        if assigned_count >= target.required_uavs:
            target_satisfied_count += 1
        else:
            constraint_penalty += PENALTY * (target.required_uavs - assigned_count)

    cooperative_time_window_penalty = 0.0
    sync_violation_count = 0
    max_sync_gap = 0.0
    sync_window = weights.get('sync_window', 0.05)
    for arrivals in cooperative_arrivals.values():
        if len(arrivals) < 2:
            continue
        gap = max(arrivals) - min(arrivals)
        max_sync_gap = max(max_sync_gap, gap)
        if gap <= sync_window + 1e-12:
            continue
        sync_violation_count += 1
        t_syn = float(np.mean(arrivals))
        cooperative_time_window_penalty += float(
            sum(weights['alpha'] * (arrival - t_syn) ** 2 for arrival in arrivals)
        )

    task_reward = float(
        sum(battlefield.get_target(target_id).value for target_id in assigned_target_ids)
    )
    time_window_penalty = explicit_time_window_penalty + cooperative_time_window_penalty

    weighted_distance = weights['w1'] * distance_cost
    weighted_threat = weights['w2'] * threat_cost
    weighted_time_window = weights['w3'] * time_window_penalty
    weighted_reward = -weights['w4'] * task_reward
    objective_without_constraint = (
        weighted_distance
        + weighted_threat
        + weighted_time_window
        + weighted_reward
    )

    assigned_task_count = int(sum(task_counts))
    active_uav_count = int(sum(1 for count in task_counts if count > 0))

    return {
        'distance_cost': float(distance_cost),
        'threat_cost': float(threat_cost),
        'explicit_time_window_penalty': float(explicit_time_window_penalty),
        'cooperative_time_window_penalty': float(cooperative_time_window_penalty),
        'time_window_penalty': float(time_window_penalty),
        'task_reward': float(task_reward),
        'constraint_penalty': float(constraint_penalty),
        'weighted_distance': float(weighted_distance),
        'weighted_threat': float(weighted_threat),
        'weighted_time_window': float(weighted_time_window),
        'weighted_reward': float(weighted_reward),
        'objective_without_constraint': float(objective_without_constraint),
        'target_satisfaction_rate': (
            target_satisfied_count / len(battlefield.targets)
            if battlefield.targets else 0.0
        ),
        'sync_violation_count': int(sync_violation_count),
        'max_sync_gap': float(max_sync_gap),
        'assigned_task_count': assigned_task_count,
        'active_uav_count': active_uav_count,
    }


def write_rows(rows: list[dict], output_path: str, fieldnames: list[str]) -> None:
    ensure_output_dir(output_path)
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_rows(scenario: str, rows: list[dict]) -> list[dict]:
    metrics = [
        'final_fitness',
        'distance_cost',
        'threat_cost',
        'time_window_penalty',
        'task_reward',
        'constraint_penalty',
        'weighted_distance',
        'weighted_threat',
        'weighted_time_window',
        'weighted_reward',
        'objective_without_constraint',
        'target_satisfaction_rate',
        'sync_violation_count',
        'max_sync_gap',
    ]

    summary_rows = []
    for metric in metrics:
        values = np.array([float(row[metric]) for row in rows], dtype=float)
        summary_rows.append(
            {
                'scenario': scenario,
                'metric': metric,
                'mean': float(mean(values)),
                'std': float(pstdev(values)),
                'min': float(np.min(values)),
                'p50': float(np.percentile(values, 50)),
                'p90': float(np.percentile(values, 90)),
                'max': float(np.max(values)),
            }
        )
    return summary_rows


def run_scenario(scenario: str, seeds: list[int], pso_params: dict) -> None:
    rows: list[dict] = []

    print(f'\n=== 场景: {scenario} ===')
    print(f'随机种子集合: {seeds}')
    print(f'PSO 参数: num_particles={pso_params["num_particles"]}, max_iter={pso_params["max_iter"]}')

    for seed in seeds:
        np.random.seed(seed)
        battlefield = load_scenario(scenario)
        _, _, curve, plan = run_pso(
            battlefield,
            WEIGHTS,
            pso_params,
            return_assignment_plan=True,
        )
        components = collect_objective_components(battlefield, plan, WEIGHTS)
        row = {
            'scenario': scenario,
            'seed': seed,
            'final_fitness': float(curve[-1]),
            **components,
        }
        rows.append(row)

        print(
            f'seed={seed}: '
            f'fitness={row["final_fitness"]:.4f}, '
            f'distance={row["distance_cost"]:.4f}, '
            f'threat={row["threat_cost"]:.4f}, '
            f'time={row["time_window_penalty"]:.6f}, '
            f'reward={row["task_reward"]:.4f}, '
            f'constraint={row["constraint_penalty"]:.1f}'
        )

    detail_output = os.path.join(RESULT_DIR, f'{scenario}_objective_components.csv')
    summary_output = os.path.join(RESULT_DIR, f'{scenario}_objective_components_summary.csv')
    write_rows(rows, detail_output, COMPONENT_FIELDS)
    write_rows(summarize_rows(scenario, rows), summary_output, SUMMARY_FIELDS)

    print('统计结果已保存到:')
    print(f'- {detail_output}')
    print(f'- {summary_output}')


def main() -> None:
    quick_mode = os.environ.get('OBJECTIVE_QUICK', '0').strip() == '1'
    scenarios = parse_scenarios()
    seeds = parse_seeds(quick_mode)
    pso_params = build_pso_params(quick_mode)

    print(f'运行模式: {"quick" if quick_mode else "formal"}')
    print(f'当前权重: {WEIGHTS}')

    for scenario in scenarios:
        run_scenario(scenario, seeds, pso_params)


if __name__ == '__main__':
    main()
