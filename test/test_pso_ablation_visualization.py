"""
PSO 预分配消融实验可视化测试。
"""
import os
import sys
import tempfile

import matplotlib
import numpy as np

matplotlib.use('Agg')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.visualization.preallocation import (
    plot_convergence_ablation,
    plot_final_fitness_ablation,
    plot_initial_population_comparison,
)


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def test_plot_initial_population_comparison_saves_file():
    random_population = np.array([
        [0, 1, 2, 0],
        [1, 1, 0, 2],
        [2, 0, 1, 1],
    ])
    logistic_population = np.array([
        [0, 2, 1, 0],
        [2, 1, 0, 1],
        [1, 0, 2, 2],
    ])

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = os.path.join(tmp_dir, 'initial_population_comparison.png')
        fig, axes = plot_initial_population_comparison(
            random_population,
            logistic_population,
            title='初始种群分布对比',
            output_path=output_path,
        )

        assert_true(os.path.exists(output_path), '初始种群分布图应保存到指定路径')
        assert_true(os.path.getsize(output_path) > 0, '输出图片文件不应为空')
        assert_true(len(axes) == 2, '初始种群分布图应包含随机初始化和 Logistic 初始化两个子图')
        fig.clf()


def test_plot_convergence_ablation_saves_file():
    curve_groups = {
        '基础 PSO': [
            [120.0, 100.0, 90.0],
            [130.0, 105.0, 92.0],
        ],
        '完整改进 PSO': [
            [110.0, 88.0, 75.0],
            [115.0, 84.0, 73.0],
        ],
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = os.path.join(tmp_dir, 'pso_convergence_ablation.png')
        fig, ax = plot_convergence_ablation(
            curve_groups,
            title='PSO 收敛曲线消融对比',
            output_path=output_path,
        )

        assert_true(os.path.exists(output_path), 'PSO 收敛曲线消融图应保存到指定路径')
        assert_true(os.path.getsize(output_path) > 0, '输出图片文件不应为空')
        assert_true(len(ax.lines) == len(curve_groups), '每个消融组应绘制一条均值收敛曲线')
        assert_true(len(ax.collections) >= len(curve_groups), '收敛曲线图应绘制标准差阴影')
        fig.clf()


def test_plot_final_fitness_ablation_saves_file():
    final_fitness_groups = {
        '基础 PSO': [90.0, 92.0, 88.0],
        '完整改进 PSO': [75.0, 73.0, 76.0],
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = os.path.join(tmp_dir, 'pso_final_fitness_ablation.png')
        fig, ax = plot_final_fitness_ablation(
            final_fitness_groups,
            title='PSO 最终适应度消融对比',
            output_path=output_path,
        )

        assert_true(os.path.exists(output_path), 'PSO 最终适应度消融图应保存到指定路径')
        assert_true(os.path.getsize(output_path) > 0, '输出图片文件不应为空')
        assert_true(len(ax.patches) >= len(final_fitness_groups), '每个消融组应绘制一个均值柱')
        assert_true(len(ax.collections) >= 1, '最终适应度图应绘制各 seed 的散点')
        fig.clf()


if __name__ == '__main__':
    test_plot_initial_population_comparison_saves_file()
    test_plot_convergence_ablation_saves_file()
    test_plot_final_fitness_ablation_saves_file()
    print('通过 3 项 PSO 消融实验可视化测试')
