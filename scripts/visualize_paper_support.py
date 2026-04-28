"""
生成论文支撑类说明图。
"""
import os
import sys

import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data.scenario_reallocation import create_reallocation_scenario
from src.visualization.paper_support import (
    plot_scenario_elements,
    plot_system_workflow,
)


RESULT_DIR = 'results/paper_support'


def main():
    battlefield = create_reallocation_scenario()

    workflow_output_path = os.path.join(RESULT_DIR, 'system_workflow.png')
    scenario_elements_output_path = os.path.join(RESULT_DIR, 'scenario_elements.png')

    plot_system_workflow(
        title='无人集群协同打击任务规划完整流程',
        output_path=workflow_output_path,
    )
    plot_scenario_elements(
        battlefield,
        title='战场场景要素建模图',
        output_path=scenario_elements_output_path,
    )

    print('论文支撑图已保存到:')
    print(f'- {workflow_output_path}')
    print(f'- {scenario_elements_output_path}')

    plt.show()


if __name__ == '__main__':
    main()
