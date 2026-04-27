from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from src.core.models import AssignmentPlan, Battlefield, UavTaskSequence
from src.core.sequence_eval import evaluate_uav_task_sequence
from src.visualization.common import ensure_output_dir


SEQUENCE_COLORS = [
    '#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
    '#9467bd',
    '#8c564b',
    '#e377c2',
    '#bcbd22',
    '#17becf',
    '#7f7f7f',
]


def draw_compact_battlefield(ax, battlefield: Battlefield, title: str = "") -> None:
    """绘制论文图用的紧凑战场底图。"""
    for threat in battlefield.threats:
        circle = plt.Circle(
            (threat.x, threat.y),
            threat.radius,
            color='#c44e52',
            alpha=0.10,
            linewidth=1.1,
        )
        ax.add_patch(circle)
        ax.scatter(threat.x, threat.y, marker='x', s=36, color='#c44e52', linewidths=1.5, zorder=4)

    for uav in battlefield.uavs:
        ax.scatter(uav.x, uav.y, marker='^', s=62, color='#1f77b4', edgecolors='white', linewidths=0.9, zorder=6)
        ax.annotate(f'U{uav.id}', (uav.x, uav.y), textcoords='offset points', xytext=(5, 5), fontsize=8, color='#174a7c')

    for target in battlefield.targets:
        ax.scatter(target.x, target.y, marker='s', s=52, color='#c44e52', edgecolors='white', linewidths=0.9, zorder=6)
        ax.annotate(f'T{target.id}', (target.x, target.y), textcoords='offset points', xytext=(5, -10), fontsize=8, color='#4a1f22')

    ax.set_xlim(0, battlefield.map_size[0])
    ax.set_ylim(0, battlefield.map_size[1])
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.18, linewidth=0.8)
    ax.spines['top'].set_alpha(0.45)
    ax.spines['right'].set_alpha(0.45)


def _sequence_points(battlefield: Battlefield, sequence: UavTaskSequence) -> list[tuple[float, float]]:
    uav = battlefield.get_uav(sequence.uav_id)
    points = [(uav.x, uav.y)]
    for task in sequence.tasks:
        target = battlefield.get_target(task.target_id)
        points.append((target.x, target.y))
    return points


def draw_task_sequence_lines(
    ax,
    battlefield: Battlefield,
    plan: AssignmentPlan,
    *,
    linewidth: float = 2.1,
    alpha: float = 0.78,
    show_sequence_labels: bool = False,
) -> None:
    """按 UAV -> T1 -> T2 的顺序绘制任务链。"""
    for color_index, uav_id in enumerate(sorted(plan.uav_task_sequences)):
        sequence = plan.uav_task_sequences[uav_id]
        if sequence.task_count() == 0:
            continue

        color = SEQUENCE_COLORS[color_index % len(SEQUENCE_COLORS)]
        points = _sequence_points(battlefield, sequence)
        label = f'U{uav_id}: ' + '→'.join(f'T{target_id}' for target_id in sequence.target_ids())

        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=alpha, label=label, zorder=2)

        for order, (start, end) in enumerate(zip(points[:-1], points[1:]), start=1):
            ax.annotate(
                '',
                xy=end,
                xytext=start,
                arrowprops={
                    'arrowstyle': '->',
                    'color': color,
                    'lw': linewidth,
                    'alpha': alpha,
                    'shrinkA': 8,
                    'shrinkB': 9,
                    'mutation_scale': 12,
                },
                zorder=3,
            )

            if show_sequence_labels:
                mid_x = (start[0] + end[0]) / 2.0
                mid_y = (start[1] + end[1]) / 2.0
                ax.text(
                    mid_x,
                    mid_y,
                    str(order),
                    fontsize=7,
                    color=color,
                    ha='center',
                    va='center',
                    bbox={'boxstyle': 'circle,pad=0.18', 'facecolor': 'white', 'edgecolor': color, 'linewidth': 0.8},
                    zorder=6,
                )


def plot_task_sequence_assignment_map(
    battlefield: Battlefield,
    plan: AssignmentPlan,
    title: str,
    output_path: Optional[str] = None,
):
    """绘制预分配任务序列结果图。"""
    fig, ax = plt.subplots(figsize=(9.6, 7.0))
    draw_compact_battlefield(ax, battlefield, title)
    draw_task_sequence_lines(ax, battlefield, plan)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            loc='upper left',
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            fontsize=7.5,
            title='任务序列',
            title_fontsize=8.5,
            frameon=True,
            framealpha=0.92,
            edgecolor='#dddddd',
        )
    fig.tight_layout()

    if output_path is not None:
        ensure_output_dir(output_path)
        fig.savefig(output_path, dpi=180, bbox_inches='tight')

    return fig, ax


def _target_assignment_counts(
    battlefield: Battlefield,
    assignment_or_plan: np.ndarray | AssignmentPlan,
) -> list[int]:
    if isinstance(assignment_or_plan, AssignmentPlan):
        return [
            len(assignment_or_plan.target_assignees.get(target.id, []))
            for target in battlefield.targets
        ]

    assignment = assignment_or_plan
    return [
        int(np.sum(assignment[:, target.id])) if target.id < assignment.shape[1] else 0
        for target in battlefield.targets
    ]


def plot_target_loads(
    battlefield: Battlefield,
    assignment_or_plan: np.ndarray | AssignmentPlan,
    title: str,
    output_path: Optional[str] = None,
):
    """绘制预分配后各目标的需求满足情况。"""
    target_ids = [target.id for target in battlefield.targets]
    assigned_counts = _target_assignment_counts(battlefield, assignment_or_plan)
    required_counts = [target.required_uavs for target in battlefield.targets]
    satisfied = [
        assigned >= required
        for assigned, required in zip(assigned_counts, required_counts)
    ]
    satisfied_rate = sum(satisfied) / len(satisfied) if satisfied else 0.0

    x = np.arange(len(target_ids))
    bar_colors = ['#5b8cc0' if ok else '#c44e52' for ok in satisfied]

    fig_width = max(9.0, len(target_ids) * 0.44)
    fig, ax = plt.subplots(figsize=(fig_width, 4.9))
    bars = ax.bar(
        x,
        assigned_counts,
        width=0.58,
        color=bar_colors,
        alpha=0.92,
        edgecolor='white',
        linewidth=0.7,
        label='实际分配 UAV 数',
    )
    ax.scatter(
        x,
        required_counts,
        color='#f58518',
        marker='D',
        s=28,
        zorder=4,
        label='目标需求数量',
    )
    for idx, required in enumerate(required_counts):
        ax.hlines(
            required,
            idx - 0.32,
            idx + 0.32,
            colors='#f58518',
            linewidth=1.2,
            alpha=0.72,
            zorder=3,
        )

    for idx, (bar, assigned, required) in enumerate(zip(bars, assigned_counts, required_counts)):
        if assigned != required:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                assigned + 0.06,
                str(assigned),
                ha='center',
                va='bottom',
                fontsize=8,
                color='#333333',
            )
        if assigned < required:
            ax.text(
                idx,
                max(assigned, required) + 0.28,
                '未满足',
                ha='center',
                va='bottom',
                fontsize=8,
                color='#c44e52',
            )

    ax.text(
        0.99,
        0.93,
        f'任务满足率 {satisfied_rate:.0%} | 目标数 {len(target_ids)}',
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=9.5,
        color='#2f5f8f' if satisfied_rate == 1.0 else '#9c2f2f',
        bbox={'boxstyle': 'round,pad=0.32', 'facecolor': 'white', 'edgecolor': '#dddddd', 'alpha': 0.96},
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f'T{target_id}' for target_id in target_ids], rotation=0)
    ax.set_xlabel('目标编号')
    ax.set_ylabel('UAV 数量')
    ax.set_title(title)
    ax.set_ylim(0, max(max(assigned_counts, default=0), max(required_counts, default=0)) + 0.9)
    ax.set_axisbelow(True)
    ax.grid(True, axis='y', alpha=0.18, linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_alpha(0.6)
    ax.spines['bottom'].set_alpha(0.6)
    ax.legend(
        loc='upper left',
        ncol=2,
        frameon=True,
        framealpha=0.95,
        edgecolor='#dddddd',
        fontsize=9,
    )
    fig.tight_layout()

    if output_path is not None:
        ensure_output_dir(output_path)
        fig.savefig(output_path, dpi=180, bbox_inches='tight')

    return fig, ax


def collect_target_arrivals(
    battlefield: Battlefield,
    plan: AssignmentPlan,
) -> dict[int, list[tuple[int, float]]]:
    """收集 AssignmentPlan 中每个目标的 UAV 累计到达时间。"""
    target_arrivals: dict[int, list[tuple[int, float]]] = {}

    for sequence in plan.uav_task_sequences.values():
        evaluated = evaluate_uav_task_sequence(battlefield, sequence)
        for task in evaluated.evaluated_sequence.tasks:
            target_arrivals.setdefault(task.target_id, []).append(
                (sequence.uav_id, task.planned_arrival_time)
            )

    for arrivals in target_arrivals.values():
        arrivals.sort(key=lambda item: item[1])
    return target_arrivals


def plot_cooperative_arrival_windows(
    battlefield: Battlefield,
    plan: AssignmentPlan,
    title: str,
    output_path: Optional[str] = None,
    sync_window: float = 0.05,
):
    """绘制多 UAV 协同打击目标的到达时间窗图。"""
    time_scale = 60.0
    time_unit = 'min'
    sync_window_display = sync_window * time_scale
    target_arrivals = collect_target_arrivals(battlefield, plan)
    cooperative_items = [
        (target_id, arrivals)
        for target_id, arrivals in sorted(target_arrivals.items())
        if len(arrivals) >= 2
    ]

    fig_height = max(3.2, 0.72 * max(1, len(cooperative_items)) + 1.9)
    fig, ax = plt.subplots(figsize=(9.6, fig_height))

    if not cooperative_items:
        ax.text(
            0.5,
            0.5,
            '当前预分配结果中无多 UAV 协同打击目标',
            transform=ax.transAxes,
            ha='center',
            va='center',
            fontsize=11,
            color='#555555',
        )
        ax.set_axis_off()
        if output_path is not None:
            ensure_output_dir(output_path)
            fig.savefig(output_path, dpi=180, bbox_inches='tight')
        return fig, ax

    y_positions = np.arange(len(cooperative_items))
    max_delta = 0.0
    all_times: list[float] = []
    satisfied_count = 0

    for y, (target_id, arrivals) in zip(y_positions, cooperative_items):
        arrival_times = [arrival_time * time_scale for _, arrival_time in arrivals]
        min_eta = min(arrival_times)
        max_eta = max(arrival_times)
        t_syn = float(np.mean(arrival_times))
        delta_t = max_eta - min_eta
        satisfied = delta_t <= sync_window_display + 1e-12
        if satisfied:
            satisfied_count += 1
        max_delta = max(max_delta, delta_t)
        all_times.extend(arrival_times)

        band_color = '#d8e9f7' if satisfied else '#f7d8d8'
        point_color = '#2f5f8f' if satisfied else '#b33a3a'
        line_color = '#36566f' if satisfied else '#8f2f2f'

        ax.broken_barh(
            [(min_eta, sync_window_display)],
            (y - 0.23, 0.46),
            facecolors=band_color,
            edgecolors='none',
            alpha=0.92,
            label='同步时间窗' if y == 0 else None,
        )
        ax.vlines(
            t_syn,
            y - 0.28,
            y + 0.28,
            colors=line_color,
            linestyles='--',
            linewidth=1.35,
            label='同步基准时刻' if y == 0 else None,
        )

        for uav_id, arrival_time_hour in arrivals:
            arrival_time = arrival_time_hour * time_scale
            ax.scatter(
                arrival_time,
                y,
                s=48,
                color=point_color,
                edgecolors='white',
                linewidths=0.8,
                zorder=4,
                label='UAV 到达时间' if y == 0 and arrival_time == arrival_times[0] else None,
            )
            ax.annotate(
                f'U{uav_id}',
                (arrival_time, y),
                textcoords='offset points',
                xytext=(4, 5),
                fontsize=8,
                color=point_color,
            )

        ax.text(
            max_eta + max(sync_window_display * 0.18, 0.36),
            y,
            f'Δt={delta_t:.1f}{time_unit}',
            ha='left',
            va='center',
            fontsize=8.5,
            color=point_color,
        )

    satisfied_rate = satisfied_count / len(cooperative_items)
    margin = max(sync_window_display * 0.8, max_delta * 0.5, 1.8)
    ax.set_xlim(min(all_times) - margin, max(all_times) + sync_window_display + margin)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f'T{target_id}' for target_id, _ in cooperative_items])
    ax.invert_yaxis()
    ax.set_xlabel(f'预计到达时间 ETA ({time_unit})')
    ax.set_ylabel('协同打击目标')
    ax.set_title(title)
    ax.grid(True, axis='x', alpha=0.22, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_alpha(0.6)
    ax.spines['bottom'].set_alpha(0.6)
    ax.text(
        0.99,
        0.94,
        f'同步窗口 {sync_window_display:.1f} {time_unit} | 满足率 {satisfied_rate:.0%}',
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=9.2,
        color='#2f5f8f' if satisfied_rate == 1.0 else '#9c2f2f',
        bbox={'boxstyle': 'round,pad=0.32', 'facecolor': 'white', 'edgecolor': '#dddddd', 'alpha': 0.96},
    )
    ax.legend(
        loc='lower right',
        ncol=3,
        frameon=True,
        framealpha=0.95,
        edgecolor='#dddddd',
        fontsize=8.5,
    )
    fig.tight_layout()

    if output_path is not None:
        ensure_output_dir(output_path)
        fig.savefig(output_path, dpi=180, bbox_inches='tight')

    return fig, ax
