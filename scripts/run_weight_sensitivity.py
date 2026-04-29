"""
运行归一化目标函数权重灵敏度实验。

实验目标：
    对比不同权重组合在 medium/hard 场景下对距离、威胁、协同时间窗
    和最终归一化适应度的影响，为最终权重选择提供数据依据。
"""
import csv
import os
import sys
from pathlib import Path
from statistics import mean, pstdev

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPT_DIR)

from config.params import PSO, WEIGHTS
from analyze_objective_components import collect_objective_components, load_scenario
from src.pre_allocation.pso import run_pso
from src.visualization.common import ensure_output_dir


RESULT_DIR = Path('results/weight_analysis')
REFERENCE_PATH = RESULT_DIR / 'objective_normalization_refs.csv'
AHP_WEIGHTS_PATH = RESULT_DIR / 'ahp_weights.csv'

FORMAL_SEEDS = list(range(10))
QUICK_SEEDS = [0, 1]

DETAIL_FIELDS = [
    'variant',
    'scenario',
    'seed',
    'w1',
    'w2',
    'w3',
    'w4',
    'distance_ref',
    'threat_ref',
    'time_window_ref',
    'reward_ref',
    'final_fitness',
    'distance_cost',
    'threat_cost',
    'time_window_penalty',
    'task_reward',
    'constraint_penalty',
    'distance_norm',
    'threat_norm',
    'time_window_norm',
    'reward_norm',
    'weighted_distance_norm',
    'weighted_threat_norm',
    'weighted_time_window_norm',
    'weighted_reward_norm',
    'normalized_objective_without_constraint',
    'target_satisfaction_rate',
    'sync_violation_count',
    'max_sync_gap',
    'assigned_task_count',
    'active_uav_count',
]

SUMMARY_FIELDS = [
    'variant',
    'scenario',
    'metric',
    'mean',
    'std',
    'min',
    'p50',
    'p90',
    'max',
]


def parse_scenarios() -> list[str]:
    raw = os.environ.get('WEIGHT_SCENARIO', 'medium,hard')
    scenarios = [item.strip().lower() for item in raw.split(',') if item.strip()]
    if scenarios == ['all']:
        return ['small', 'medium', 'hard']
    return scenarios


def parse_seeds(quick_mode: bool) -> list[int]:
    raw = os.environ.get('WEIGHT_SEEDS')
    if raw:
        return [int(item.strip()) for item in raw.split(',') if item.strip()]
    return QUICK_SEEDS if quick_mode else FORMAL_SEEDS


def build_pso_params(quick_mode: bool) -> dict:
    params = dict(PSO)
    if quick_mode:
        params['num_particles'] = 30
        params['max_iter'] = 80
    return params


def read_baseline_refs() -> dict[str, float]:
    with open(REFERENCE_PATH, newline='', encoding='utf-8-sig') as csv_file:
        rows = list(csv.DictReader(csv_file))

    for row in rows:
        if row.get('scenario') == 'baseline':
            return {
                'distance_ref': float(row['distance_ref']),
                'threat_ref': float(row['threat_ref']),
                'time_window_ref': float(row['time_window_ref']),
                'reward_ref': float(row['reward_ref']),
            }
    raise ValueError(f'{REFERENCE_PATH} 中未找到 scenario=baseline 的统一归一化参考值')


def read_ahp_weights() -> dict[str, float]:
    weights = {}
    with open(AHP_WEIGHTS_PATH, newline='', encoding='utf-8-sig') as csv_file:
        for row in csv.DictReader(csv_file):
            weights[row['param_key']] = float(row['weight'])
    return {
        'w1': weights['w1'],
        'w2': weights['w2'],
        'w3': weights['w3'],
        'w4': weights['w4'],
    }


def build_weight_config(base_weights: dict[str, float], refs: dict[str, float]) -> dict:
    return {
        'w1': base_weights['w1'],
        'w2': base_weights['w2'],
        'w3': base_weights['w3'],
        'w4': base_weights['w4'],
        'alpha': WEIGHTS.get('alpha', 1.0),
        'sync_window': WEIGHTS.get('sync_window', 0.05),
        'objective_refs': refs,
    }


def build_variants(refs: dict[str, float]) -> list[dict]:
    ahp_weights = read_ahp_weights()
    variants = [
        {
            'label': '经验权重',
            'weights': {
                'w1': WEIGHTS['w1'],
                'w2': WEIGHTS['w2'],
                'w3': WEIGHTS['w3'],
                'w4': WEIGHTS['w4'],
            },
        },
        {
            'label': '均衡权重',
            'weights': {
                'w1': 0.25,
                'w2': 0.25,
                'w3': 0.25,
                'w4': 0.25,
            },
        },
        {
            'label': '威胁优先',
            'weights': {
                'w1': 0.20,
                'w2': 0.50,
                'w3': 0.20,
                'w4': 0.10,
            },
        },
        {
            'label': '协同优先',
            'weights': {
                'w1': 0.20,
                'w2': 0.30,
                'w3': 0.40,
                'w4': 0.10,
            },
        },
        {
            'label': 'AHP推荐权重',
            'weights': ahp_weights,
        },
        {
            'label': 'AHP-协同增强',
            'weights': {
                'w1': 0.20,
                'w2': 0.38,
                'w3': 0.32,
                'w4': 0.10,
            },
        },
        {
            'label': 'AHP-协同强增强',
            'weights': {
                'w1': 0.18,
                'w2': 0.36,
                'w3': 0.36,
                'w4': 0.10,
            },
        },
        {
            'label': 'AHP-安全协同折中',
            'weights': {
                'w1': 0.20,
                'w2': 0.42,
                'w3': 0.28,
                'w4': 0.10,
            },
        },
    ]

    for variant in variants:
        variant['config'] = build_weight_config(variant['weights'], refs)
    return variants


def add_normalized_terms(row: dict, refs: dict[str, float], weights: dict[str, float]) -> dict:
    distance_norm = float(row['distance_cost']) / refs['distance_ref']
    threat_norm = float(row['threat_cost']) / refs['threat_ref']
    time_window_norm = float(row['time_window_penalty']) / refs['time_window_ref']
    reward_norm = float(row['task_reward']) / refs['reward_ref']

    weighted_distance_norm = weights['w1'] * distance_norm
    weighted_threat_norm = weights['w2'] * threat_norm
    weighted_time_window_norm = weights['w3'] * time_window_norm
    weighted_reward_norm = -weights['w4'] * reward_norm

    return {
        **row,
        'distance_norm': float(distance_norm),
        'threat_norm': float(threat_norm),
        'time_window_norm': float(time_window_norm),
        'reward_norm': float(reward_norm),
        'weighted_distance_norm': float(weighted_distance_norm),
        'weighted_threat_norm': float(weighted_threat_norm),
        'weighted_time_window_norm': float(weighted_time_window_norm),
        'weighted_reward_norm': float(weighted_reward_norm),
        'normalized_objective_without_constraint': float(
            weighted_distance_norm
            + weighted_threat_norm
            + weighted_time_window_norm
            + weighted_reward_norm
        ),
    }


def write_rows(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    ensure_output_dir(str(path))
    with open(path, 'w', newline='', encoding='utf-8-sig') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_rows(rows: list[dict]) -> list[dict]:
    metrics = [
        'final_fitness',
        'distance_cost',
        'threat_cost',
        'time_window_penalty',
        'normalized_objective_without_constraint',
        'target_satisfaction_rate',
        'sync_violation_count',
        'max_sync_gap',
        'active_uav_count',
    ]

    summary_rows = []
    variant_labels = sorted({row['variant'] for row in rows})
    scenario_names = sorted({row['scenario'] for row in rows})
    for variant in variant_labels:
        for scenario in scenario_names:
            subset = [
                row for row in rows
                if row['variant'] == variant and row['scenario'] == scenario
            ]
            if not subset:
                continue
            for metric in metrics:
                values = np.array([float(row[metric]) for row in subset], dtype=float)
                summary_rows.append(
                    {
                        'variant': variant,
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


def run_experiment() -> list[dict]:
    quick_mode = os.environ.get('WEIGHT_SENSITIVITY_QUICK', '0').strip() == '1'
    scenarios = parse_scenarios()
    seeds = parse_seeds(quick_mode)
    pso_params = build_pso_params(quick_mode)
    refs = read_baseline_refs()
    variants = build_variants(refs)

    print(f'运行模式: {"quick" if quick_mode else "formal"}')
    print(f'场景集合: {scenarios}')
    print(f'随机种子集合: {seeds}')
    print(f'统一归一化参考值: {refs}')
    print(f'PSO 参数: num_particles={pso_params["num_particles"]}, max_iter={pso_params["max_iter"]}')

    rows: list[dict] = []
    for scenario in scenarios:
        print(f'\n=== 场景: {scenario} ===')
        for variant in variants:
            label = variant['label']
            config = variant['config']
            print(f'\n--- {label} ---')
            for seed in seeds:
                np.random.seed(seed)
                battlefield = load_scenario(scenario)
                _, _, curve, plan = run_pso(
                    battlefield,
                    config,
                    pso_params,
                    return_assignment_plan=True,
                )
                components = collect_objective_components(battlefield, plan, config)
                row = add_normalized_terms(components, refs, config)
                row = {
                    'variant': label,
                    'scenario': scenario,
                    'seed': seed,
                    'w1': config['w1'],
                    'w2': config['w2'],
                    'w3': config['w3'],
                    'w4': config['w4'],
                    'distance_ref': refs['distance_ref'],
                    'threat_ref': refs['threat_ref'],
                    'time_window_ref': refs['time_window_ref'],
                    'reward_ref': refs['reward_ref'],
                    'final_fitness': float(curve[-1]),
                    **row,
                }
                rows.append(row)
                print(
                    f'seed={seed}: fitness={row["final_fitness"]:.4f}, '
                    f'dist={row["distance_cost"]:.1f}, '
                    f'threat={row["threat_cost"]:.1f}, '
                    f'time={row["time_window_penalty"]:.5f}, '
                    f'violations={row["sync_violation_count"]}'
                )

    return rows


def main() -> None:
    rows = run_experiment()
    detail_path = RESULT_DIR / 'weight_sensitivity_detail.csv'
    summary_path = RESULT_DIR / 'weight_sensitivity_summary.csv'
    write_rows(rows, detail_path, DETAIL_FIELDS)
    write_rows(summarize_rows(rows), summary_path, SUMMARY_FIELDS)

    print('\n权重灵敏度实验结果已保存到:')
    print(f'- {detail_path}')
    print(f'- {summary_path}')


if __name__ == '__main__':
    main()
