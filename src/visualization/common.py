import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from src.core.models import Battlefield


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def ensure_output_dir(output_path: str) -> None:
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def draw_battlefield(ax, battlefield: Battlefield, title: str = "") -> None:
    """绘制战场基础元素：威胁区、无人机、目标。"""
    for threat in battlefield.threats:
        circle = plt.Circle(
            (threat.x, threat.y),
            threat.radius,
            color='red',
            alpha=0.15,
            label='威胁区' if threat.id == 0 else None,
        )
        ax.add_patch(circle)
        ax.plot(threat.x, threat.y, 'rx', markersize=8)

    for uav in battlefield.uavs:
        ax.plot(uav.x, uav.y, 'b^', markersize=9)
        ax.annotate(
            f'U{uav.id}',
            (uav.x, uav.y),
            textcoords='offset points',
            xytext=(5, 5),
            fontsize=8,
        )

    for target in battlefield.targets:
        ax.plot(target.x, target.y, 'rs', markersize=9)
        ax.annotate(
            f'T{target.id}(v={target.value},n={target.required_uavs})',
            (target.x, target.y),
            textcoords='offset points',
            xytext=(5, -10),
            fontsize=7,
        )

    ax.set_xlim(0, battlefield.map_size[0])
    ax.set_ylim(0, battlefield.map_size[1])
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)


def draw_assignment_lines(
    ax,
    battlefield: Battlefield,
    assignment: np.ndarray,
    linestyle: str = '--',
    linewidth: float = 1.4,
    alpha: float = 0.75,
    color_by_uav: bool = True,
) -> None:
    """绘制任务分配连线。"""
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for i in range(assignment.shape[0]):
        for j in range(assignment.shape[1]):
            if assignment[i, j] == 1:
                uav = battlefield.get_uav(i)
                target = battlefield.get_target(j)
                line_color = colors[i % len(colors)] if color_by_uav else '#1f77b4'
                ax.plot(
                    [uav.x, target.x],
                    [uav.y, target.y],
                    linestyle,
                    color=line_color,
                    linewidth=linewidth,
                    alpha=alpha,
                )


def plot_assignment_map(
    battlefield: Battlefield,
    assignment: np.ndarray,
    title: str,
    output_path: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    draw_battlefield(ax, battlefield, title)
    draw_assignment_lines(ax, battlefield, assignment)
    fig.tight_layout()

    if output_path is not None:
        ensure_output_dir(output_path)
        fig.savefig(output_path, dpi=180)

    return fig, ax


def plot_eta_distribution(
    battlefield: Battlefield,
    etas: np.ndarray,
    title: str,
    output_path: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for target in battlefield.targets:
        assigned_uavs = np.where(etas[:, target.id] > 0)[0]
        for uav_id in assigned_uavs:
            ax.scatter(target.id, etas[uav_id, target.id], color='#1f77b4', s=45)
            ax.annotate(
                f'U{uav_id}',
                (target.id, etas[uav_id, target.id]),
                textcoords='offset points',
                xytext=(5, 4),
                fontsize=8,
            )

    ax.set_xlabel('目标编号')
    ax.set_ylabel('预计到达时间 (h)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if output_path is not None:
        ensure_output_dir(output_path)
        fig.savefig(output_path, dpi=180)

    return fig, ax




def draw_path_lines(
    ax,
    path_points,
    color: str,
    label: str,
    linestyle: str = '-',
    linewidth: float = 1.8,
    alpha: float = 0.9,
) -> None:
    """绘制单条路径折线。"""
    if path_points is None or len(path_points) < 2:
        return

    xs = [point[0] for point in path_points]
    ys = [point[1] for point in path_points]
    ax.plot(
        xs,
        ys,
        linestyle=linestyle,
        linewidth=linewidth,
        color=color,
        alpha=alpha,
        label=label,
    )
