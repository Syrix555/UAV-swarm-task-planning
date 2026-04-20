"""
预分配可视化与稳定性诊断脚本
输出以下图表：
1. 初始种群分布热力图对比（随机初始化 vs Logistic混沌初始化）
2. 收敛曲线对比（基础PSO / 仅混沌初始化 / 仅余弦权重 / 完整改进PSO）
3. 最终任务分配结果图
4. ETA协同分布图
5. 目标需求满足情况图

同时输出多随机种子下的诊断信息：
- 初始最优适应度
- 初始平均适应度
- 初始不可行粒子数量
- 最终适应度
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from config.params import WEIGHTS, PSO
from data.scenario_hard import create_hard_scenario
from data.scenario_medium import create_medium_scenario
from src.pre_allocation.pso import run_pso
from src.visualization.common import (
    ensure_output_dir,
    plot_assignment_map,
    plot_eta_distribution,
    plot_target_loads,
)


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

RESULT_DIR = 'results/pre_allocation'
SEEDS = [7, 21, 42, 84, 168]
SCENARIO_NAME = os.environ.get('PSO_SCENARIO', 'medium').strip().lower()


def plot_initial_population_comparison(random_population, logistic_population, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    im1 = axes[0].imshow(random_population, aspect='auto', cmap='viridis')
    axes[0].set_title('随机初始化种群分布')
    axes[0].set_xlabel('粒子维度')
    axes[0].set_ylabel('粒子编号')
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label='无人机编号')

    im2 = axes[1].imshow(logistic_population, aspect='auto', cmap='viridis')
    axes[1].set_title('Logistic混沌初始化种群分布')
    axes[1].set_xlabel('粒子维度')
    axes[1].set_ylabel('粒子编号')
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label='无人机编号')

    ensure_output_dir(output_path)
    fig.savefig(output_path, dpi=180)
    return fig, axes


def plot_convergence_comparison(curve_groups, output_path):
    fig, ax = plt.subplots(figsize=(9, 6))
    labels = [
        '基础PSO（随机+线性）',
        '仅改初始化（混沌+线性）',
        '仅改权重（随机+余弦）',
        '完整改进（混沌+余弦）',
    ]
    colors = ['#7f7f7f', '#1f77b4', '#ff7f0e', '#d62728']

    for curves, label, color in zip(curve_groups, labels, colors):
        mean_curve = np.mean(np.array(curves), axis=0)
        ax.plot(mean_curve, label=label, linewidth=1.9, color=color)

    ax.set_xlabel('迭代次数')
    ax.set_ylabel('平均适应度值')
    ax.set_title('PSO适应度收敛曲线对比（多次实验均值）')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ensure_output_dir(output_path)
    fig.savefig(output_path, dpi=180)
    return fig, ax


def run_variant(battlefield, init_method, inertia_strategy, seed, return_initial_population=False):
    np.random.seed(seed)
    return run_pso(
        battlefield,
        WEIGHTS,
        PSO,
        init_method=init_method,
        inertia_strategy=inertia_strategy,
        return_initial_population=return_initial_population,
        return_diagnostics=True,
    )


def main():
    if SCENARIO_NAME == 'hard':
        battlefield = create_hard_scenario()
    else:
        battlefield = create_medium_scenario()
    print(f"场景名称: {SCENARIO_NAME}")
    print(f"场景: {len(battlefield.uavs)}架无人机, {len(battlefield.targets)}个目标, {len(battlefield.threats)}个威胁区")
    print(f"地图尺寸: {battlefield.map_size[0]}x{battlefield.map_size[1]} km")
    print('=' * 60)
    print(f"随机种子集合: {SEEDS}")

    _, _, _, random_population, _ = run_variant(
        battlefield,
        init_method='random',
        inertia_strategy='linear',
        seed=SEEDS[0],
        return_initial_population=True,
    )
    _, _, _, logistic_population, _ = run_variant(
        battlefield,
        init_method='logistic',
        inertia_strategy='linear',
        seed=SEEDS[0],
        return_initial_population=True,
    )

    variant_configs = [
        ('基础PSO（随机+线性）', 'random', 'linear'),
        ('仅改初始化（混沌+线性）', 'logistic', 'linear'),
        ('仅改权重（随机+余弦）', 'random', 'cosine'),
        ('完整改进（混沌+余弦）', 'logistic', 'cosine'),
    ]

    all_curves = []
    best_improved = None
    best_improved_fitness = float('inf')

    print('正在运行多随机种子对比实验...')
    for label, init_method, inertia_strategy in variant_configs:
        curves = []
        final_values = []
        print(f"\n--- {label} ---")
        for seed in SEEDS:
            assignment, etas, curve, diagnostics = run_variant(
                battlefield,
                init_method=init_method,
                inertia_strategy=inertia_strategy,
                seed=seed,
                return_initial_population=False,
            )
            curves.append(curve)
            final_values.append(curve[-1])

            print(
                f"seed={seed}: init_best={diagnostics['initial_best_fitness']:.4f}, "
                f"init_mean={diagnostics['initial_mean_fitness']:.4f}, "
                f"init_infeasible={diagnostics['initial_infeasible_count']}, "
                f"final={diagnostics['final_best_fitness']:.4f}"
            )

            if label == '完整改进（混沌+余弦）' and curve[-1] < best_improved_fitness:
                best_improved_fitness = curve[-1]
                best_improved = (assignment, etas, curve, seed)

        all_curves.append(curves)
        print(f"{label}: mean={np.mean(final_values):.4f}, std={np.std(final_values):.4f}, best={np.min(final_values):.4f}")

    if best_improved is None:
        raise RuntimeError('未能获得完整改进PSO的有效结果')

    improved_assignment, improved_etas, improved_curve, best_seed = best_improved

    plot_initial_population_comparison(
        random_population,
        logistic_population,
        os.path.join(RESULT_DIR, 'initial_population_comparison.png'),
    )

    plot_convergence_comparison(
        all_curves,
        os.path.join(RESULT_DIR, 'convergence_comparison.png'),
    )

    plot_assignment_map(
        battlefield,
        improved_assignment,
        f'改进PSO预分配结果图（seed={best_seed}）',
        os.path.join(RESULT_DIR, 'improved_assignment_map.png'),
    )

    plot_eta_distribution(
        battlefield,
        improved_etas,
        f'协同攻击ETA分布图（seed={best_seed}）',
        os.path.join(RESULT_DIR, 'eta_distribution.png'),
    )

    plot_target_loads(
        battlefield,
        improved_assignment,
        f'目标需求满足情况图（seed={best_seed}）',
        os.path.join(RESULT_DIR, 'target_loads.png'),
    )

    print('\n=== 完整改进PSO代表性结果 ===')
    print(f"最佳随机种子: {best_seed}")
    print(f"最佳最终适应度: {improved_curve[-1]:.4f}")

    print('\n图表已保存到 results/pre_allocation/ 目录：')
    print('- initial_population_comparison.png')
    print('- convergence_comparison.png')
    print('- improved_assignment_map.png')
    print('- eta_distribution.png')
    print('- target_loads.png')

    plt.show()


if __name__ == '__main__':
    main()
