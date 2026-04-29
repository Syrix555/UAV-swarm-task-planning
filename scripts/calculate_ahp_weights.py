"""
基于 AHP 计算归一化目标函数权重。

输入：
    results/weight_analysis/objective_normalization_refs.csv 中的 baseline 行。

输出：
    1. AHP 判断矩阵
    2. AHP 权重结果
    3. 一致性检验结果

本脚本只生成权重依据，不修改 PSO/MCHA 主流程。
"""
import csv
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.visualization.common import ensure_output_dir


INPUT_PATH = Path('results/weight_analysis/objective_normalization_refs.csv')
OUTPUT_DIR = Path('results/weight_analysis')

CRITERIA = [
    {
        'key': 'distance',
        'param_key': 'w1',
        'label': '距离代价',
        'ref_key': 'distance_ref',
    },
    {
        'key': 'threat',
        'param_key': 'w2',
        'label': '威胁代价',
        'ref_key': 'threat_ref',
    },
    {
        'key': 'time_window',
        'param_key': 'w3',
        'label': '协同时间窗惩罚',
        'ref_key': 'time_window_ref',
    },
    {
        'key': 'reward',
        'param_key': 'w4',
        'label': '任务收益',
        'ref_key': 'reward_ref',
    },
]

# 指标顺序：距离代价、威胁代价、协同时间窗惩罚、任务收益。
# 含义：
# - 威胁规避比距离和时间协同更重要；
# - 距离代价与协同时间窗接近；
# - 任务收益在当前场景中区分度较弱，因此重要性最低。
PAIRWISE_MATRIX = np.array(
    [
        [1.0, 1 / 2, 1.0, 2.0],
        [2.0, 1.0, 2.0, 4.0],
        [1.0, 1 / 2, 1.0, 3.0],
        [1 / 2, 1 / 4, 1 / 3, 1.0],
    ],
    dtype=float,
)

RI_TABLE = {
    1: 0.00,
    2: 0.00,
    3: 0.58,
    4: 0.90,
    5: 1.12,
    6: 1.24,
    7: 1.32,
    8: 1.41,
    9: 1.45,
    10: 1.49,
}


def read_baseline_reference(path: Path) -> dict[str, str]:
    with open(path, newline='', encoding='utf-8-sig') as csv_file:
        rows = list(csv.DictReader(csv_file))

    for row in rows:
        if row.get('scenario') == 'baseline':
            return row

    raise ValueError(f'{path} 中未找到 scenario=baseline 的统一归一化参考值')


def calculate_ahp(matrix: np.ndarray) -> tuple[np.ndarray, float, float, float, float]:
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    max_index = int(np.argmax(eigenvalues.real))
    lambda_max = float(eigenvalues[max_index].real)
    weights = eigenvectors[:, max_index].real
    weights = weights / np.sum(weights)

    n = matrix.shape[0]
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    ri = RI_TABLE.get(n)
    if ri is None:
        raise ValueError(f'RI_TABLE 未配置 n={n} 的随机一致性指标')
    cr = ci / ri if ri > 0 else 0.0
    return weights, lambda_max, ci, ri, cr


def write_pairwise_matrix(matrix: np.ndarray, output_path: Path) -> None:
    ensure_output_dir(str(output_path))
    fieldnames = ['criterion', *[criterion['key'] for criterion in CRITERIA]]
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, criterion in enumerate(CRITERIA):
            row = {'criterion': criterion['key']}
            for col_index, col_criterion in enumerate(CRITERIA):
                row[col_criterion['key']] = float(matrix[row_index, col_index])
            writer.writerow(row)


def write_weights(
    weights: np.ndarray,
    baseline_ref: dict[str, str],
    output_path: Path,
) -> list[dict[str, str | float]]:
    rows = []
    for criterion, weight in zip(CRITERIA, weights):
        rows.append(
            {
                'criterion': criterion['key'],
                'criterion_label': criterion['label'],
                'param_key': criterion['param_key'],
                'weight': float(weight),
                'normalization_ref': float(baseline_ref[criterion['ref_key']]),
            }
        )

    ensure_output_dir(str(output_path))
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as csv_file:
        fieldnames = [
            'criterion',
            'criterion_label',
            'param_key',
            'weight',
            'normalization_ref',
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return rows


def write_consistency(
    lambda_max: float,
    ci: float,
    ri: float,
    cr: float,
    output_path: Path,
) -> None:
    ensure_output_dir(str(output_path))
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as csv_file:
        fieldnames = ['n', 'lambda_max', 'CI', 'RI', 'CR', 'is_consistent']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                'n': len(CRITERIA),
                'lambda_max': lambda_max,
                'CI': ci,
                'RI': ri,
                'CR': cr,
                'is_consistent': cr < 0.1,
            }
        )


def main() -> None:
    baseline_ref = read_baseline_reference(INPUT_PATH)
    weights, lambda_max, ci, ri, cr = calculate_ahp(PAIRWISE_MATRIX)

    matrix_output = OUTPUT_DIR / 'ahp_pairwise_matrix.csv'
    weights_output = OUTPUT_DIR / 'ahp_weights.csv'
    consistency_output = OUTPUT_DIR / 'ahp_consistency.csv'

    write_pairwise_matrix(PAIRWISE_MATRIX, matrix_output)
    weight_rows = write_weights(weights, baseline_ref, weights_output)
    write_consistency(lambda_max, ci, ri, cr, consistency_output)

    print('AHP 权重计算完成')
    print(f'lambda_max={lambda_max:.6f}, CI={ci:.6f}, RI={ri:.2f}, CR={cr:.6f}')
    print(f'一致性检验: {"通过" if cr < 0.1 else "未通过"}')
    print('权重结果:')
    for row in weight_rows:
        print(
            f'- {row["criterion_label"]}({row["param_key"]}): '
            f'{row["weight"]:.6f}, ref={row["normalization_ref"]}'
        )
    print('结果已保存到:')
    print(f'- {matrix_output}')
    print(f'- {weights_output}')
    print(f'- {consistency_output}')


if __name__ == '__main__':
    main()
