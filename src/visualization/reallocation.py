import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from src.core.models import Battlefield
from src.visualization.common import (
    draw_assignment_lines,
    draw_battlefield,
    ensure_output_dir,
)


def plot_reallocation_before_after(
    battlefield_before: Battlefield,
    assignment_before: np.ndarray,
    battlefield_after: Battlefield,
    assignment_after: np.ndarray,
    title_before: str,
    title_after: str,
    output_path: Optional[str] = None,
):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)

    draw_battlefield(axes[0], battlefield_before, title_before)
    draw_assignment_lines(axes[0], battlefield_before, assignment_before)

    draw_battlefield(axes[1], battlefield_after, title_after)
    draw_assignment_lines(axes[1], battlefield_after, assignment_after)

    if output_path is not None:
        ensure_output_dir(output_path)
        fig.savefig(output_path, dpi=180)

    return fig, axes


def plot_assignment_diff(
    battlefield: Battlefield,
    assignment_before: np.ndarray,
    assignment_after: np.ndarray,
    title: str,
    output_path: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    draw_battlefield(ax, battlefield, title)

    num_uavs, num_targets = assignment_before.shape
    for uav_id in range(num_uavs):
        for target_id in range(num_targets):
            before = int(assignment_before[uav_id, target_id])
            after = int(assignment_after[uav_id, target_id])
            if before == after:
                continue

            uav = battlefield.get_uav(uav_id)
            target = battlefield.get_target(target_id)
            if before == 1 and after == 0:
                ax.plot(
                    [uav.x, target.x],
                    [uav.y, target.y],
                    '-',
                    color='red',
                    linewidth=2.0,
                    alpha=0.85,
                )
            elif before == 0 and after == 1:
                ax.plot(
                    [uav.x, target.x],
                    [uav.y, target.y],
                    '-',
                    color='green',
                    linewidth=2.0,
                    alpha=0.85,
                )

    if output_path is not None:
        ensure_output_dir(output_path)
        fig.savefig(output_path, dpi=180)

    return fig, ax


def plot_reallocation_target_loads(
    battlefield_before: Battlefield,
    assignment_before: np.ndarray,
    battlefield_after: Battlefield,
    assignment_after: np.ndarray,
    title: str,
    output_path: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(10, 5.5))

    target_ids = [target.id for target in battlefield_after.targets]
    before_counts = []
    after_counts = []
    required_counts = []

    for target in battlefield_after.targets:
        target_id = target.id
        if target_id < assignment_before.shape[1]:
            before_counts.append(int(np.sum(assignment_before[:, target_id])))
        else:
            before_counts.append(0)
        after_counts.append(int(np.sum(assignment_after[:, target_id])))
        required_counts.append(target.required_uavs)

    x = np.arange(len(target_ids))
    width = 0.28
    ax.bar(x - width, before_counts, width=width, label='事件前分配数', color='#7f7f7f')
    ax.bar(x, after_counts, width=width, label='重分配后分配数', color='#1f77b4')
    ax.bar(x + width, required_counts, width=width, label='需求数量', color='#ff7f0e')

    ax.set_xticks(x)
    ax.set_xticklabels(target_ids)
    ax.set_xlabel('目标编号')
    ax.set_ylabel('无人机数量')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    if output_path is not None:
        ensure_output_dir(output_path)
        fig.savefig(output_path, dpi=180)

    return fig, ax
