"""
读取阶段 1 的目标函数分项统计结果，并生成归一化指标。

本脚本不重新运行 PSO，只基于 results/weight_analysis 下的
*_objective_components.csv 计算参考尺度和无量纲指标，用于后续 AHP
权重确定与权重灵敏度实验。
"""
import csv
import os
import sys
from pathlib import Path
from statistics import mean, pstdev

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config.params import WEIGHTS
from src.visualization.common import ensure_output_dir


INPUT_DIR = Path('results/weight_analysis')
OUTPUT_DIR = Path('results/weight_analysis')
EPS = 1e-12

RAW_COMPONENTS = [
    'distance_cost',
    'threat_cost',
    'time_window_penalty',
    'task_reward',
]

NORMALIZED_COMPONENTS = [
    'distance_norm',
    'threat_norm',
    'time_window_norm',
    'reward_norm',
]

REFERENCE_FIELDS = [
    'scenario',
    'reference_method',
    'distance_ref',
    'threat_ref',
    'time_window_ref',
    'reward_ref',
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


def parse_scenarios() -> list[str]:
    raw = os.environ.get('OBJECTIVE_SCENARIO', 'medium,hard')
    scenarios = [item.strip().lower() for item in raw.split(',') if item.strip()]
    if scenarios == ['all']:
        return [
            path.name.removesuffix('_objective_components.csv')
            for path in sorted(INPUT_DIR.glob('*_objective_components.csv'))
            if not path.name.endswith('_summary.csv')
        ]
    return scenarios


def load_rows(path: Path) -> list[dict]:
    with open(path, newline='', encoding='utf-8-sig') as csv_file:
        return list(csv.DictReader(csv_file))


def write_rows(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    ensure_output_dir(str(path))
    with open(path, 'w', newline='', encoding='utf-8-sig') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def numeric_values(rows: list[dict], key: str) -> np.ndarray:
    return np.array([float(row[key]) for row in rows], dtype=float)


def reference_value(values: np.ndarray, method: str) -> float:
    abs_values = np.abs(values.astype(float))
    non_zero_values = abs_values[abs_values > EPS]
    if non_zero_values.size == 0:
        return 1.0

    if method == 'max':
        return float(np.max(non_zero_values))
    if method == 'mean':
        return float(np.mean(non_zero_values))
    if method == 'p90':
        return float(np.percentile(non_zero_values, 90))
    raise ValueError(f'Unsupported reference method: {method}')


def build_references(rows: list[dict], scenario: str, method: str) -> dict[str, float | str]:
    return {
        'scenario': scenario,
        'reference_method': method,
        'distance_ref': reference_value(numeric_values(rows, 'distance_cost'), method),
        'threat_ref': reference_value(numeric_values(rows, 'threat_cost'), method),
        'time_window_ref': reference_value(numeric_values(rows, 'time_window_penalty'), method),
        'reward_ref': reference_value(numeric_values(rows, 'task_reward'), method),
    }


def normalize_row(row: dict, refs: dict[str, float | str]) -> dict:
    distance_norm = float(row['distance_cost']) / float(refs['distance_ref'])
    threat_norm = float(row['threat_cost']) / float(refs['threat_ref'])
    time_window_norm = float(row['time_window_penalty']) / float(refs['time_window_ref'])
    reward_norm = float(row['task_reward']) / float(refs['reward_ref'])

    weighted_distance_norm = WEIGHTS['w1'] * distance_norm
    weighted_threat_norm = WEIGHTS['w2'] * threat_norm
    weighted_time_window_norm = WEIGHTS['w3'] * time_window_norm
    weighted_reward_norm = -WEIGHTS['w4'] * reward_norm
    normalized_objective_without_constraint = (
        weighted_distance_norm
        + weighted_threat_norm
        + weighted_time_window_norm
        + weighted_reward_norm
    )

    return {
        **row,
        'distance_norm': distance_norm,
        'threat_norm': threat_norm,
        'time_window_norm': time_window_norm,
        'reward_norm': reward_norm,
        'weighted_distance_norm': weighted_distance_norm,
        'weighted_threat_norm': weighted_threat_norm,
        'weighted_time_window_norm': weighted_time_window_norm,
        'weighted_reward_norm': weighted_reward_norm,
        'normalized_objective_without_constraint': normalized_objective_without_constraint,
    }


def summarize_normalized_rows(scenario: str, rows: list[dict]) -> list[dict]:
    metrics = [
        *RAW_COMPONENTS,
        *NORMALIZED_COMPONENTS,
        'weighted_distance_norm',
        'weighted_threat_norm',
        'weighted_time_window_norm',
        'weighted_reward_norm',
        'normalized_objective_without_constraint',
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


def rounded_baseline_value(value: float, base: float) -> float:
    return float(np.ceil(value / base) * base)


def build_baseline_reference(reference_rows: list[dict]) -> dict[str, float | str]:
    max_distance = max(float(row['distance_ref']) for row in reference_rows)
    max_threat = max(float(row['threat_ref']) for row in reference_rows)
    max_time_window = max(float(row['time_window_ref']) for row in reference_rows)
    max_reward = max(float(row['reward_ref']) for row in reference_rows)

    return {
        'scenario': 'baseline',
        'reference_method': 'max_scenario_ref_rounded',
        'distance_ref': rounded_baseline_value(max_distance, 100.0),
        'threat_ref': rounded_baseline_value(max_threat, 50.0),
        'time_window_ref': rounded_baseline_value(max_time_window, 0.01),
        'reward_ref': rounded_baseline_value(max_reward, 10.0),
    }


def run_scenario(scenario: str, method: str) -> tuple[dict, list[dict]]:
    input_path = INPUT_DIR / f'{scenario}_objective_components.csv'
    if not input_path.exists():
        raise FileNotFoundError(f'阶段 1 明细文件不存在: {input_path}')

    rows = load_rows(input_path)
    refs = build_references(rows, scenario, method)
    normalized_rows = [normalize_row(row, refs) for row in rows]

    output_path = OUTPUT_DIR / f'{scenario}_objective_components_normalized.csv'
    fieldnames = list(normalized_rows[0].keys())
    write_rows(normalized_rows, output_path, fieldnames)

    summary_rows = summarize_normalized_rows(scenario, normalized_rows)
    summary_path = OUTPUT_DIR / f'{scenario}_objective_components_normalized_summary.csv'
    write_rows(summary_rows, summary_path, SUMMARY_FIELDS)

    print(f'\n=== 场景: {scenario} ===')
    print(
        '参考尺度: '
        f'distance={refs["distance_ref"]:.6f}, '
        f'threat={refs["threat_ref"]:.6f}, '
        f'time={refs["time_window_ref"]:.6f}, '
        f'reward={refs["reward_ref"]:.6f}'
    )
    print('归一化结果已保存到:')
    print(f'- {output_path}')
    print(f'- {summary_path}')

    return refs, summary_rows


def main() -> None:
    method = os.environ.get('OBJECTIVE_REF_METHOD', 'p90').strip().lower()
    scenarios = parse_scenarios()

    print(f'归一化参考值方法: {method}')
    print(f'场景集合: {scenarios}')
    print(f'当前权重: {WEIGHTS}')

    reference_rows = []
    combined_summary_rows = []
    for scenario in scenarios:
        refs, summary_rows = run_scenario(scenario, method)
        reference_rows.append(refs)
        combined_summary_rows.extend(summary_rows)

    if reference_rows:
        baseline_reference = build_baseline_reference(reference_rows)
        reference_rows.append(baseline_reference)
        print(
            '\n统一基准参考值: '
            f'distance={baseline_reference["distance_ref"]:.6f}, '
            f'threat={baseline_reference["threat_ref"]:.6f}, '
            f'time={baseline_reference["time_window_ref"]:.6f}, '
            f'reward={baseline_reference["reward_ref"]:.6f}'
        )

    reference_path = OUTPUT_DIR / 'objective_normalization_refs.csv'
    combined_summary_path = OUTPUT_DIR / 'objective_components_normalized_summary.csv'
    write_rows(reference_rows, reference_path, REFERENCE_FIELDS)
    write_rows(combined_summary_rows, combined_summary_path, SUMMARY_FIELDS)

    print('\n汇总结果已保存到:')
    print(f'- {reference_path}')
    print(f'- {combined_summary_path}')


if __name__ == '__main__':
    main()
