from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from src.core.models import Battlefield
from src.visualization.common import ensure_output_dir


def plot_scenario_elements(
    battlefield: Battlefield,
    title: str,
    output_path: Optional[str] = None,
):
    """绘制论文支撑用的战场场景要素建模图。"""
    fig, ax = plt.subplots(figsize=(9.8, 7.2), constrained_layout=True)

    for threat in battlefield.threats:
        circle = plt.Circle(
            (threat.x, threat.y),
            threat.radius,
            color='#c75146',
            alpha=0.13,
            linewidth=1.2,
            zorder=1,
        )
        ax.add_patch(circle)
        ax.scatter(
            threat.x,
            threat.y,
            marker='x',
            s=52,
            color='#9c2f2f',
            linewidths=1.7,
            zorder=5,
        )
        ax.text(
            threat.x,
            threat.y + threat.radius + 1.2,
            f'Z{threat.id}\nr={threat.radius:g}',
            ha='center',
            va='bottom',
            fontsize=7.6,
            color='#7a2828',
            zorder=6,
        )

    ax.scatter(
        [uav.x for uav in battlefield.uavs],
        [uav.y for uav in battlefield.uavs],
        marker='^',
        s=86,
        color='#2f5f8f',
        edgecolors='white',
        linewidths=1.0,
        zorder=8,
        label='UAV 初始位置',
    )
    for uav in battlefield.uavs:
        ax.annotate(
            f'U{uav.id}\nammo={uav.ammo}',
            (uav.x, uav.y),
            textcoords='offset points',
            xytext=(6, 6),
            fontsize=7.3,
            color='#17476f',
            zorder=9,
        )

    target_values = np.array([target.value for target in battlefield.targets], dtype=float)
    required_uavs = np.array([target.required_uavs for target in battlefield.targets], dtype=float)
    target_scatter = ax.scatter(
        [target.x for target in battlefield.targets],
        [target.y for target in battlefield.targets],
        marker='s',
        s=58 + required_uavs * 34,
        c=target_values,
        cmap='YlOrRd',
        edgecolors='white',
        linewidths=0.9,
        zorder=7,
        label='打击目标',
    )
    for target in battlefield.targets:
        ax.annotate(
            f'T{target.id}\nn={target.required_uavs}',
            (target.x, target.y),
            textcoords='offset points',
            xytext=(6, -12),
            fontsize=7.2,
            color='#4a1f22',
            zorder=9,
        )

    colorbar = fig.colorbar(target_scatter, ax=ax, fraction=0.032, pad=0.018)
    colorbar.set_label('目标价值')
    colorbar.outline.set_alpha(0.35)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_handles = [
        Line2D([0], [0], marker='^', color='none', markerfacecolor='#2f5f8f', markeredgecolor='white', markersize=9, label='UAV 初始位置'),
        Line2D([0], [0], marker='s', color='none', markerfacecolor='#e36f47', markeredgecolor='white', markersize=9, label='打击目标'),
        Patch(facecolor='#c75146', edgecolor='#c75146', alpha=0.18, label='威胁区'),
        Line2D([0], [0], marker='s', color='none', markerfacecolor='#e36f47', markeredgecolor='white', markersize=7, label='n=1'),
        Line2D([0], [0], marker='s', color='none', markerfacecolor='#e36f47', markeredgecolor='white', markersize=10, label='n=2+'),
    ]
    ax.legend(
        handles=legend_handles,
        loc='upper left',
        bbox_to_anchor=(1.08, 1.0),
        borderaxespad=0.0,
        fontsize=8.2,
        title='场景要素',
        title_fontsize=9.0,
        frameon=True,
        framealpha=0.95,
        edgecolor='#dddddd',
    )

    total_demand = int(sum(target.required_uavs for target in battlefield.targets))
    total_ammo = int(sum(uav.ammo for uav in battlefield.uavs))
    value_min = float(np.min(target_values)) if len(target_values) else 0.0
    value_max = float(np.max(target_values)) if len(target_values) else 0.0
    summary = (
        f'UAV {len(battlefield.uavs)} 架 | 目标 {len(battlefield.targets)} 个 | '
        f'总打击需求 {total_demand} | 总弹药 {total_ammo} | '
        f'威胁区 {len(battlefield.threats)} 个 | 目标价值 {value_min:g}-{value_max:g}'
    )
    ax.text(
        0.5,
        1.02,
        summary,
        transform=ax.transAxes,
        ha='center',
        va='bottom',
        fontsize=9.0,
        color='#2f5f8f',
    )

    ax.set_xlim(0, battlefield.map_size[0])
    ax.set_ylim(0, battlefield.map_size[1])
    ax.set_xlabel('X 坐标 (km)')
    ax.set_ylabel('Y 坐标 (km)')
    ax.set_title(title, pad=24)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.18, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines['top'].set_alpha(0.45)
    ax.spines['right'].set_alpha(0.45)

    if output_path is not None:
        ensure_output_dir(output_path)
        fig.savefig(output_path, dpi=180, bbox_inches='tight')

    return fig, ax


def plot_system_workflow(
    title: str,
    output_path: Optional[str] = None,
):
    """绘制论文支撑用的无人集群协同任务规划完整流程示意图。"""
    fig, ax = plt.subplots(figsize=(12.8, 6.8))
    ax.set_xlim(0, 0.84)
    ax.set_ylim(0, 1)
    ax.axis('off')

    steps = [
        {'label': '场景输入', 'detail': 'UAV / 目标 / 威胁区\n弹药、航程、目标需求', 'xy': (0.08, 0.64), 'color': '#d8e6f2'},
        {'label': '打击任务目标建模', 'detail': '目标价值、需求数量\n协同到达时间窗', 'xy': (0.25, 0.64), 'color': '#e8f1dc'},
        {'label': '任务预分配', 'detail': '改进 PSO\n生成 UAV 任务序列', 'xy': (0.42, 0.64), 'color': '#f6e6c9'},
        {'label': '动态事件触发', 'detail': 'UAV 损失 / 新增目标\n需求变化 / 新增威胁', 'xy': (0.59, 0.64), 'color': '#f2d5d1'},
        {'label': '任务重分配', 'detail': 'MCHA 启发式拍卖\n开放任务需求修复', 'xy': (0.76, 0.64), 'color': '#dde6f6'},
        {'label': '航迹规划与仿真展示', 'detail': 'A* 路径搜索\nB 样条平滑', 'xy': (0.76, 0.30), 'color': '#e9ddf3'},
        {'label': '实验结果输出', 'detail': '任务序列、需求满足\n代价变化、可视化图表', 'xy': (0.59, 0.30), 'color': '#dcebe7'},
    ]

    box_width = 0.145
    box_height = 0.19

    def draw_box(step: dict) -> None:
        x, y = step['xy']
        box = plt.Rectangle(
            (x - box_width / 2, y - box_height / 2),
            box_width,
            box_height,
            facecolor=step['color'],
            edgecolor='#6b7280',
            linewidth=1.15,
            zorder=2,
        )
        ax.add_patch(box)
        ax.text(x, y + 0.045, step['label'], ha='center', va='center', fontsize=10.8, weight='bold', color='#222222', zorder=3)
        ax.text(x, y - 0.038, step['detail'], ha='center', va='center', fontsize=8.0, color='#3f3f46', linespacing=1.18, zorder=3)

    def draw_arrow(start: tuple[float, float], end: tuple[float, float], text: str | None = None) -> None:
        ax.annotate(
            '',
            xy=end,
            xytext=start,
            arrowprops={
                'arrowstyle': '->',
                'lw': 1.7,
                'color': '#4b5563',
                'shrinkA': 5,
                'shrinkB': 5,
                'mutation_scale': 14,
            },
            zorder=1,
        )
        if text:
            mid_x = (start[0] + end[0]) / 2.0
            mid_y = (start[1] + end[1]) / 2.0
            ax.text(
                mid_x,
                mid_y + 0.03,
                text,
                ha='center',
                va='bottom',
                fontsize=8.2,
                color='#4b5563',
                bbox={'boxstyle': 'round,pad=0.18', 'facecolor': 'white', 'edgecolor': '#e5e7eb', 'alpha': 0.95},
                zorder=4,
            )

    for step in steps:
        draw_box(step)

    for left, right in zip(steps[:4], steps[1:5]):
        left_x, left_y = left['xy']
        right_x, right_y = right['xy']
        draw_arrow((left_x + box_width / 2, left_y), (right_x - box_width / 2, right_y))

    draw_arrow(
        (steps[4]['xy'][0], steps[4]['xy'][1] - box_height / 2),
        (steps[5]['xy'][0], steps[5]['xy'][1] + box_height / 2),
        text='形成可执行任务链',
    )
    draw_arrow(
        (steps[5]['xy'][0] - box_width / 2, steps[5]['xy'][1]),
        (steps[6]['xy'][0] + box_width / 2, steps[6]['xy'][1]),
    )

    ax.annotate(
        '',
        xy=(steps[2]['xy'][0], steps[2]['xy'][1] - box_height / 2),
        xytext=(steps[6]['xy'][0] - box_width / 2, steps[6]['xy'][1] + 0.015),
        arrowprops={
            'arrowstyle': '->',
            'lw': 1.2,
            'color': '#8c6d31',
            'linestyle': '--',
            'connectionstyle': 'arc3,rad=-0.22',
            'mutation_scale': 12,
        },
        zorder=1,
    )
    ax.text(
        0.39,
        0.41,
        '实验分析反馈参数设置',
        ha='center',
        va='center',
        fontsize=8.2,
        color='#7a5c20',
        bbox={'boxstyle': 'round,pad=0.22', 'facecolor': '#fffdf5', 'edgecolor': '#e6d8a8', 'alpha': 0.95},
        zorder=4,
    )

    ax.text(0.42, 0.90, title, ha='center', va='center', fontsize=15.0, weight='bold', color='#111827')
    ax.text(
        0.42,
        0.84,
        '以任务序列为核心数据结构，贯通预分配、动态重分配与最终航迹展示',
        ha='center',
        va='center',
        fontsize=10.0,
        color='#2f5f8f',
    )

    for x0, x1, y, label in [
        (0.005, 0.495, 0.49, '任务规划阶段'),
        (0.515, 0.835, 0.17, '动态调整与展示阶段'),
    ]:
        ax.plot([x0, x1], [y, y], color='#d1d5db', linewidth=1.0, alpha=0.9, zorder=0)
        ax.text((x0 + x1) / 2, y - 0.035, label, ha='center', va='center', fontsize=8.5, color='#6b7280')

    if output_path is not None:
        ensure_output_dir(output_path)
        fig.savefig(output_path, dpi=180, bbox_inches='tight')

    return fig, ax
