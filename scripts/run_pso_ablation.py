"""
运行 PSO 预分配消融实验。

输出内容：
1. 初始种群分布对比图
2. PSO 收敛曲线消融图
3. 最终适应度消融图
4. 消融实验统计表 CSV
"""
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config.params import PSO, WEIGHTS
from data.scenario_hard import create_hard_scenario
from data.scenario_medium import create_medium_scenario
from data.scenario_small import create_small_scenario
from src.pre_allocation.pso import run_pso
from src.visualization.common import ensure_output_dir
from src.visualization.preallocation import (
    plot_convergence_ablation,
    plot_final_fitness_ablation,
    plot_initial_population_comparison,
)


RESULT_DIR = 'results/pre_allocation/ablation'
FORMAL_SEEDS = list(range(20))
QUICK_SEEDS = list(range(2))

VARIANT_CONFIGS = [
    {
        'label': '基础 PSO',
        'init_method': 'random',
        'inertia_strategy': 'linear',
    },
    {
        'label': '仅改初始化',
        'init_method': 'logistic',
        'inertia_strategy': 'linear',
    },
    {
        'label': '仅改权重',
        'init_method': 'random',
        'inertia_strategy': 'cosine',
    },
    {
        'label': '完整改进 PSO',
        'init_method': 'logistic',
        'inertia_strategy': 'cosine',
    },
]


def load_scenario(name: str):
    if name == 'small':
        return create_small_scenario()
    if name == 'hard':
        return create_hard_scenario()
    return create_medium_scenario()


def build_pso_params(quick_mode: bool) -> dict:
    pso_params = dict(PSO)
    if quick_mode:
        pso_params['num_particles'] = 30
        pso_params['max_iter'] = 80
    return pso_params


def run_variant(
    battlefield,
    pso_params: dict,
    init_method: str,
    inertia_strategy: str,
    seed: int,
    return_initial_population: bool = False,
):
    np.random.seed(seed)
    return run_pso(
        battlefield,
        WEIGHTS,
        pso_params,
        init_method=init_method,
        inertia_strategy=inertia_strategy,
        return_initial_population=return_initial_population,
        return_diagnostics=True,
    )


def summarize_variant(
    label: str,
    init_method: str,
    inertia_strategy: str,
    diagnostics_list: list[dict],
    final_values: list[float],
) -> dict:
    return {
        '方法名称': label,
        '初始化方式': init_method,
        '惯性权重策略': inertia_strategy,
        '初始最优适应度均值': float(np.mean([item['initial_best_fitness'] for item in diagnostics_list])),
        '初始平均适应度均值': float(np.mean([item['initial_mean_fitness'] for item in diagnostics_list])),
        '初始不可行粒子数量均值': float(np.mean([item['initial_infeasible_count'] for item in diagnostics_list])),
        '最终适应度均值': float(np.mean(final_values)),
        '最终适应度标准差': float(np.std(final_values)),
        '最优最终适应度': float(np.min(final_values)),
    }


def write_summary_csv(rows: list[dict], output_path: str) -> None:
    ensure_output_dir(output_path)
    fieldnames = [
        '方法名称',
        '初始化方式',
        '惯性权重策略',
        '初始最优适应度均值',
        '初始平均适应度均值',
        '初始不可行粒子数量均值',
        '最终适应度均值',
        '最终适应度标准差',
        '最优最终适应度',
    ]
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    scenario_name = os.environ.get('PSO_SCENARIO', 'medium').strip().lower()
    quick_mode = os.environ.get('PSO_ABLATION_QUICK', '0').strip() == '1'
    seeds = QUICK_SEEDS if quick_mode else FORMAL_SEEDS
    pso_params = build_pso_params(quick_mode)
    battlefield = load_scenario(scenario_name)

    print(f'场景名称: {scenario_name}')
    print(f'运行模式: {"quick" if quick_mode else "formal"}')
    print(f'随机种子集合: {seeds}')
    print(f'PSO 参数: num_particles={pso_params["num_particles"]}, max_iter={pso_params["max_iter"]}')
    print(f'场景规模: {len(battlefield.uavs)} 架 UAV, {len(battlefield.targets)} 个目标, {len(battlefield.threats)} 个威胁区')

    first_seed = seeds[0]
    _, _, _, random_population, _ = run_variant(
        battlefield,
        pso_params,
        init_method='random',
        inertia_strategy='linear',
        seed=first_seed,
        return_initial_population=True,
    )
    _, _, _, logistic_population, _ = run_variant(
        battlefield,
        pso_params,
        init_method='logistic',
        inertia_strategy='linear',
        seed=first_seed,
        return_initial_population=True,
    )

    curve_groups: dict[str, list[list[float]]] = {}
    final_fitness_groups: dict[str, list[float]] = {}
    summary_rows: list[dict] = []

    for variant in VARIANT_CONFIGS:
        label = variant['label']
        init_method = variant['init_method']
        inertia_strategy = variant['inertia_strategy']
        curves = []
        final_values = []
        diagnostics_list = []

        print(f'\n--- {label} ({init_method}, {inertia_strategy}) ---')
        for seed in seeds:
            _, _, curve, diagnostics = run_variant(
                battlefield,
                pso_params,
                init_method=init_method,
                inertia_strategy=inertia_strategy,
                seed=seed,
            )
            curve = [float(value) for value in curve]
            final_value = float(curve[-1])
            curves.append(curve)
            final_values.append(final_value)
            diagnostics_list.append(diagnostics)

            print(
                f'seed={seed}: '
                f'init_best={diagnostics["initial_best_fitness"]:.4f}, '
                f'init_mean={diagnostics["initial_mean_fitness"]:.4f}, '
                f'init_infeasible={diagnostics["initial_infeasible_count"]}, '
                f'final={final_value:.4f}'
            )

        curve_groups[label] = curves
        final_fitness_groups[label] = final_values
        summary_rows.append(
            summarize_variant(
                label,
                init_method,
                inertia_strategy,
                diagnostics_list,
                final_values,
            )
        )
        print(
            f'{label}: mean={np.mean(final_values):.4f}, '
            f'std={np.std(final_values):.4f}, best={np.min(final_values):.4f}'
        )

    initial_population_output = os.path.join(RESULT_DIR, 'initial_population_comparison.png')
    convergence_output = os.path.join(RESULT_DIR, 'pso_convergence_ablation.png')
    final_fitness_output = os.path.join(RESULT_DIR, 'pso_final_fitness_ablation.png')
    summary_output = os.path.join(RESULT_DIR, 'pso_ablation_summary.csv')

    plot_initial_population_comparison(
        random_population,
        logistic_population,
        title=f'初始种群分布对比（{scenario_name}, seed={first_seed}）',
        output_path=initial_population_output,
    )
    plot_convergence_ablation(
        curve_groups,
        title=f'PSO 收敛曲线消融对比（{scenario_name}，越低越好）',
        output_path=convergence_output,
    )
    plot_final_fitness_ablation(
        final_fitness_groups,
        title=f'PSO 最终适应度消融对比（{scenario_name}，越低越好）',
        output_path=final_fitness_output,
    )
    write_summary_csv(summary_rows, summary_output)

    print('\n消融实验图表已保存到:')
    print(f'- {initial_population_output}')
    print(f'- {convergence_output}')
    print(f'- {final_fitness_output}')
    print(f'- {summary_output}')

    plt.close('all')


if __name__ == '__main__':
    main()
