"""
PSO预分配测试脚本
运行小规模场景，验证改进PSO算法的基本功能和效果
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import matplotlib.pyplot as plt
from data.scenario_small import create_small_scenario
from data.scenario_medium import create_medium_scenario
from src.pre_allocation.pso import run_pso
from config.params import WEIGHTS, PSO

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def main():
    # 加载场景
    # bf = create_small_scenario()
    bf = create_medium_scenario()
    print(f"场景: {len(bf.uavs)}架无人机, {len(bf.targets)}个目标, {len(bf.threats)}个威胁区")
    print(f"地图尺寸: {bf.map_size[0]}x{bf.map_size[1]} km")
    print("=" * 50)

    # 运行PSO
    print("正在运行改进PSO算法...")
    assignment, etas, curve = run_pso(bf, WEIGHTS)
    print("PSO运行完成!\n")

    # 打印分配结果
    print("=== 任务分配结果 ===")
    for j in range(len(bf.targets)):
        assigned = np.where(assignment[:, j] == 1)[0]
        uav_ids = ", ".join([f"UAV-{i}" for i in assigned])
        t = bf.targets[j]
        print(f"  目标{j} (价值={t.value}, 需{t.required_uavs}架, "
              f"位置=({t.x},{t.y})) ← {uav_ids}")

    print("\n=== 各无人机负载 ===")
    for i in range(len(bf.uavs)):
        assigned = np.where(assignment[i] == 1)[0]
        u = bf.uavs[i]
        print(f"  UAV-{i} (位置=({u.x},{u.y}), 弹药={u.ammo}): "
              f"分配了{len(assigned)}个目标 {list(assigned)}")

    print(f"\n最终适应度值: {curve[-1]:.4f}")

    # 画图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 图1: 收敛曲线
    ax1 = axes[0]
    ax1.plot(curve, linewidth=1.5)
    ax1.set_xlabel('迭代次数')
    ax1.set_ylabel('适应度值')
    ax1.set_title('改进PSO收敛曲线')
    ax1.grid(True, alpha=0.3)

    # 图2: 任务分配可视化
    ax2 = axes[1]

    # 画威胁区
    for threat in bf.threats:
        circle = plt.Circle((threat.x, threat.y), threat.radius,
                             color='red', alpha=0.15, label='威胁区' if threat.id == 0 else None)
        ax2.add_patch(circle)
        ax2.plot(threat.x, threat.y, 'rx', markersize=8)

    # 画无人机
    for uav in bf.uavs:
        ax2.plot(uav.x, uav.y, 'b^', markersize=10)
        ax2.annotate(f'U{uav.id}', (uav.x, uav.y), textcoords="offset points",
                     xytext=(5, 5), fontsize=8)

    # 画目标（标注需求量）
    for t in bf.targets:
        ax2.plot(t.x, t.y, 'rs', markersize=10)
        ax2.annotate(f'T{t.id}(v={t.value},n={t.required_uavs})',
                     (t.x, t.y), textcoords="offset points",
                     xytext=(5, -10), fontsize=7)

    # 画分配连线
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i in range(len(bf.uavs)):
        for j in range(len(bf.targets)):
            if assignment[i, j] == 1:
                uav = bf.uavs[i]
                target = bf.targets[j]
                ax2.plot([uav.x, target.x], [uav.y, target.y],
                         '--', color=colors[i % len(colors)], linewidth=1.2, alpha=0.7)

    ax2.set_xlim(0, bf.map_size[0])
    ax2.set_ylim(0, bf.map_size[1])
    ax2.set_xlabel('X (km)')
    ax2.set_ylabel('Y (km)')
    ax2.set_title('任务分配结果')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs('results/pre_allocation', exist_ok=True)
    plt.savefig('results/pre_allocation/pso_test_result.png', dpi=150)
    print(f"\n图表已保存至 results/pre_allocation/pso_test_result.png")
    plt.show()


if __name__ == '__main__':
    main()
