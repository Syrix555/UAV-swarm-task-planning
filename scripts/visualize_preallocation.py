"""
生成预分配任务序列论文图。
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config.params import WEIGHTS, PSO
from data.scenario_hard import create_hard_scenario
from data.scenario_medium import create_medium_scenario
from data.scenario_small import create_small_scenario
from src.pre_allocation.pso import run_pso
from src.visualization.preallocation import (
    plot_cooperative_arrival_windows,
    plot_target_loads,
    plot_task_sequence_assignment_map,
)


RESULT_DIR = 'results/pre_allocation'


def load_scenario(name: str):
    if name == 'small':
        return create_small_scenario()
    if name == 'hard':
        return create_hard_scenario()
    return create_medium_scenario()


def main():
    scenario_name = os.environ.get('PSO_SCENARIO', 'medium').strip().lower()
    seed = int(os.environ.get('PSO_SEED', '42'))
    np.random.seed(seed)

    battlefield = load_scenario(scenario_name)
    print(f'场景名称: {scenario_name}')
    print(f'随机种子: {seed}')
    print(f'场景规模: {len(battlefield.uavs)} 架 UAV, {len(battlefield.targets)} 个目标, {len(battlefield.threats)} 个威胁区')

    assignment, etas, curve, plan = run_pso(
        battlefield,
        WEIGHTS,
        PSO,
        return_assignment_plan=True,
    )
    del assignment, etas

    sequence_output_path = os.path.join(RESULT_DIR, f'{scenario_name}_task_sequence_assignment.png')
    plot_task_sequence_assignment_map(
        battlefield,
        plan,
        title=f'预分配任务序列结果（{scenario_name}, seed={seed}）',
        output_path=sequence_output_path,
    )

    target_loads_output_path = os.path.join(RESULT_DIR, f'{scenario_name}_target_loads.png')
    plot_target_loads(
        battlefield,
        plan,
        title=f'预分配目标需求满足情况（{scenario_name}, seed={seed}）',
        output_path=target_loads_output_path,
    )

    cooperative_output_path = os.path.join(RESULT_DIR, f'{scenario_name}_cooperative_arrival_windows.png')
    plot_cooperative_arrival_windows(
        battlefield,
        plan,
        title=f'多无人机协同打击到达时间分布图（{scenario_name}, seed={seed}）',
        output_path=cooperative_output_path,
        sync_window=WEIGHTS.get('sync_window', 0.05),
    )

    print(f'最终适应度: {curve[-1]:.4f}')
    print('图表已保存到:')
    print(f'- {sequence_output_path}')
    print(f'- {target_loads_output_path}')
    print(f'- {cooperative_output_path}')
    plt.show()


if __name__ == '__main__':
    main()
