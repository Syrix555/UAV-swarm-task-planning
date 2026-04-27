import csv
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


def collect_preallocation_metrics(
    battlefield: Battlefield,
    plan: AssignmentPlan,
    *,
    final_fitness: float | None = None,
    sync_window: float = 0.05,
    alpha: float = 1.0,
) -> dict[str, float | int]:
    """汇总预分配结果的论文展示指标。"""
    total_distance = 0.0
    total_travel_time = 0.0
    explicit_time_window_penalty = 0.0
    threat_cost = 0.0
    assigned_target_ids: set[int] = set()

    for sequence in plan.uav_task_sequences.values():
        uav = battlefield.get_uav(sequence.uav_id)
        evaluated = evaluate_uav_task_sequence(
            battlefield,
            sequence,
            alpha=alpha,
        )
        total_distance += evaluated.total_distance
        total_travel_time += evaluated.total_travel_time
        explicit_time_window_penalty += evaluated.time_window_penalty

        current_x, current_y = uav.x, uav.y
        for task in sequence.tasks:
            target = battlefield.get_target(task.target_id)
            threat_cost += battlefield.threat_cost_on_line(
                current_x,
                current_y,
                target.x,
                target.y,
            )
            current_x, current_y = target.x, target.y
            assigned_target_ids.add(task.target_id)

    target_satisfied_count = 0
    for target in battlefield.targets:
        assigned_count = len(plan.target_assignees.get(target.id, []))
        if assigned_count >= target.required_uavs:
            target_satisfied_count += 1

    task_counts = [
        plan.uav_task_sequences.get(uav.id, UavTaskSequence(uav.id)).task_count()
        for uav in battlefield.uavs
    ]
    assigned_task_count = int(sum(task_counts))
    active_uav_count = int(sum(1 for count in task_counts if count > 0))
    required_task_count = int(sum(target.required_uavs for target in battlefield.targets))
    target_count = len(battlefield.targets)

    target_arrivals = collect_target_arrivals(battlefield, plan)
    cooperative_targets = [
        arrivals for arrivals in target_arrivals.values()
        if len(arrivals) >= 2
    ]
    sync_violation_count = 0
    max_sync_gap = 0.0
    for arrivals in cooperative_targets:
        arrival_times = [arrival_time for _, arrival_time in arrivals]
        gap = max(arrival_times) - min(arrival_times)
        max_sync_gap = max(max_sync_gap, gap)
        if gap > sync_window + 1e-12:
            sync_violation_count += 1

    sync_satisfied_rate = (
        1.0 - sync_violation_count / len(cooperative_targets)
        if cooperative_targets else 1.0
    )

    total_target_value = float(
        sum(battlefield.get_target(target_id).value for target_id in assigned_target_ids)
    )

    metrics: dict[str, float | int] = {
        'uav_count': len(battlefield.uavs),
        'target_count': target_count,
        'active_uav_count': active_uav_count,
        'assigned_task_count': assigned_task_count,
        'required_task_count': required_task_count,
        'target_satisfied_count': target_satisfied_count,
        'target_satisfaction_rate': target_satisfied_count / target_count if target_count else 0.0,
        'average_tasks_per_uav': assigned_task_count / len(battlefield.uavs) if battlefield.uavs else 0.0,
        'average_tasks_per_active_uav': assigned_task_count / active_uav_count if active_uav_count else 0.0,
        'max_task_chain_length': max(task_counts, default=0),
        'total_distance': float(total_distance),
        'total_travel_time': float(total_travel_time),
        'threat_cost': float(threat_cost),
        'explicit_time_window_penalty': float(explicit_time_window_penalty),
        'cooperative_target_count': len(cooperative_targets),
        'sync_violation_count': sync_violation_count,
        'sync_satisfied_rate': float(sync_satisfied_rate),
        'max_sync_gap': float(max_sync_gap),
        'total_target_value': total_target_value,
    }
    if final_fitness is not None:
        metrics['final_fitness'] = float(final_fitness)
    elif plan.total_cost:
        metrics['final_fitness'] = float(plan.total_cost)
    else:
        metrics['final_fitness'] = 0.0

    return metrics


def plot_uav_task_loads(
    battlefield: Battlefield,
    plan: AssignmentPlan,
    title: str,
    output_path: Optional[str] = None,
):
    """绘制每架 UAV 的任务链长度与 ammo 容量对比。"""
    uav_ids = [uav.id for uav in battlefield.uavs]
    task_counts = [
        plan.uav_task_sequences.get(uav.id, UavTaskSequence(uav.id)).task_count()
        for uav in battlefield.uavs
    ]
    ammo_limits = [uav.ammo for uav in battlefield.uavs]
    active_count = sum(1 for count in task_counts if count > 0)
    avg_load = float(np.mean(task_counts)) if task_counts else 0.0
    max_load = max(task_counts, default=0)

    x = np.arange(len(uav_ids))
    bar_colors = [
        '#c44e52' if count > ammo else '#5b8cc0'
        for count, ammo in zip(task_counts, ammo_limits)
    ]

    fig_width = max(8.8, len(uav_ids) * 0.58)
    fig, ax = plt.subplots(figsize=(fig_width, 4.9))
    bars = ax.bar(
        x,
        task_counts,
        width=0.58,
        color=bar_colors,
        alpha=0.92,
        edgecolor='white',
        linewidth=0.7,
        label='任务链长度',
    )
    ax.scatter(
        x,
        ammo_limits,
        color='#f58518',
        marker='D',
        s=30,
        zorder=4,
        label='ammo 容量',
    )
    for idx, ammo in enumerate(ammo_limits):
        ax.hlines(
            ammo,
            idx - 0.32,
            idx + 0.32,
            colors='#f58518',
            linewidth=1.2,
            alpha=0.72,
            zorder=3,
        )

    for bar, count in zip(bars, task_counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            count + 0.05,
            str(count),
            ha='center',
            va='bottom',
            fontsize=8,
            color='#333333',
        )

    ax.text(
        0.99,
        0.93,
        f'活跃 UAV {active_count}/{len(uav_ids)} | 平均任务 {avg_load:.2f} | 最大链长 {max_load}',
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=9.2,
        color='#2f5f8f',
        bbox={'boxstyle': 'round,pad=0.32', 'facecolor': 'white', 'edgecolor': '#dddddd', 'alpha': 0.96},
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f'U{uav_id}' for uav_id in uav_ids], rotation=0)
    ax.set_xlabel('UAV 编号')
    ax.set_ylabel('任务数量')
    ax.set_title(title)
    ax.set_ylim(0, max(max(task_counts, default=0), max(ammo_limits, default=0)) + 0.9)
    ax.grid(True, axis='y', alpha=0.18, linewidth=0.8)
    ax.set_axisbelow(True)
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
        fontsize=8.8,
    )
    fig.tight_layout()

    if output_path is not None:
        ensure_output_dir(output_path)
        fig.savefig(output_path, dpi=180, bbox_inches='tight')

    return fig, ax


def _format_metric_value(key: str, value: float | int) -> str:
    if key.endswith('_rate'):
        return f'{float(value):.1%}'
    if key in {'total_travel_time', 'max_sync_gap'}:
        return f'{float(value) * 60.0:.2f} min'
    if isinstance(value, int):
        return str(value)
    return f'{float(value):.3f}'


def _write_metrics_csv(metrics: dict[str, float | int], output_path: str) -> None:
    ensure_output_dir(output_path)
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['指标', '数值'])
        for key, label in _METRIC_TABLE_ITEMS:
            if key in metrics:
                writer.writerow([label, _format_metric_value(key, metrics[key])])


_METRIC_TABLE_ITEMS = [
    ('uav_count', 'UAV 总数'),
    ('target_count', '目标总数'),
    ('active_uav_count', '活跃 UAV 数'),
    ('assigned_task_count', '已分配任务槽数'),
    ('required_task_count', '目标需求任务槽数'),
    ('target_satisfied_count', '满足需求目标数'),
    ('target_satisfaction_rate', '目标满足率'),
    ('average_tasks_per_uav', '单机平均任务数'),
    ('average_tasks_per_active_uav', '活跃单机平均任务数'),
    ('max_task_chain_length', '最大任务链长度'),
    ('total_distance', '任务链总航程 km'),
    ('total_travel_time', '任务链总飞行时间'),
    ('threat_cost', '威胁代价'),
    ('cooperative_target_count', '协同打击目标数'),
    ('sync_violation_count', '同步窗口超限目标数'),
    ('sync_satisfied_rate', '同步窗口满足率'),
    ('max_sync_gap', '最大协同到达间隔'),
    ('total_target_value', '覆盖目标总价值'),
    ('final_fitness', '最终适应度'),
]


def plot_preallocation_metrics_table(
    metrics: dict[str, float | int],
    title: str,
    output_path: Optional[str] = None,
    csv_output_path: Optional[str] = None,
):
    """绘制预分配综合指标表，并可同步导出 CSV。"""
    table_rows = [
        [label, _format_metric_value(key, metrics[key])]
        for key, label in _METRIC_TABLE_ITEMS
        if key in metrics
    ]

    fig_height = max(4.8, 0.33 * len(table_rows) + 1.3)
    fig, ax = plt.subplots(figsize=(8.2, fig_height))
    ax.set_axis_off()
    ax.set_title(title, fontsize=13, pad=12)

    table = ax.table(
        cellText=table_rows,
        colLabels=['指标', '数值'],
        loc='center',
        cellLoc='center',
        colLoc='center',
        colWidths=[0.58, 0.32],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.2)
    table.scale(1.0, 1.26)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#d9d9d9')
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor('#e9f1f7')
            cell.set_text_props(weight='bold', color='#24445c')
        elif row % 2 == 0:
            cell.set_facecolor('#f7f9fb')
        else:
            cell.set_facecolor('white')
        if col == 0 and row > 0:
            cell.set_text_props(ha='left')

    fig.tight_layout()

    if output_path is not None:
        ensure_output_dir(output_path)
        fig.savefig(output_path, dpi=180, bbox_inches='tight')
    if csv_output_path is not None:
        _write_metrics_csv(metrics, csv_output_path)

    return fig, ax


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


def plot_initial_population_comparison(
    random_population: np.ndarray,
    logistic_population: np.ndarray,
    title: str,
    output_path: Optional[str] = None,
):
    """绘制随机初始化与 Logistic 混沌初始化的初始种群分布对比。"""
    vmax = int(max(np.max(random_population), np.max(logistic_population)))
    vmin = int(min(np.min(random_population), np.min(logistic_population)))

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8), constrained_layout=True)
    fig.suptitle(title, fontsize=13.5, y=1.02)

    panels = [
        ('随机初始化', random_population),
        ('Logistic 混沌初始化', logistic_population),
    ]
    images = []
    for ax, (panel_title, population) in zip(axes, panels):
        image = ax.imshow(
            population,
            aspect='auto',
            cmap='viridis',
            vmin=vmin,
            vmax=vmax,
            interpolation='nearest',
        )
        images.append(image)
        ax.set_title(panel_title, fontsize=11)
        ax.set_xlabel('任务槽位')
        ax.set_ylabel('粒子编号')
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_alpha(0.55)
        ax.spines['bottom'].set_alpha(0.55)

    cbar = fig.colorbar(images[-1], ax=axes, fraction=0.035, pad=0.025)
    cbar.set_label('UAV 编号')

    if output_path is not None:
        ensure_output_dir(output_path)
        fig.savefig(output_path, dpi=180, bbox_inches='tight')

    return fig, axes


def _curve_group_to_array(curves: list[list[float]] | list[np.ndarray]) -> np.ndarray:
    if not curves:
        raise ValueError('curve group must not be empty')

    lengths = {len(curve) for curve in curves}
    if len(lengths) != 1:
        raise ValueError('all convergence curves in one group must have the same length')

    return np.asarray(curves, dtype=float)


def plot_convergence_ablation(
    curve_groups: dict[str, list[list[float]] | list[np.ndarray]],
    title: str,
    output_path: Optional[str] = None,
):
    """绘制四组 PSO 消融实验的平均收敛曲线和标准差阴影。"""
    colors = ['#4c78a8', '#f58518', '#54a24b', '#d95f5f', '#7f7f7f']

    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    for index, (label, curves) in enumerate(curve_groups.items()):
        curve_array = _curve_group_to_array(curves)
        mean_curve = np.mean(curve_array, axis=0)
        std_curve = np.std(curve_array, axis=0)
        iterations = np.arange(len(mean_curve))
        color = colors[index % len(colors)]

        ax.plot(
            iterations,
            mean_curve,
            label=label,
            color=color,
            linewidth=2.0,
        )
        ax.fill_between(
            iterations,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color=color,
            alpha=0.14,
            linewidth=0,
        )

    ax.set_xlabel('迭代次数')
    ax.set_ylabel('适应度值（越低越好）')
    ax.set_title(title)
    ax.grid(True, alpha=0.22, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_alpha(0.6)
    ax.spines['bottom'].set_alpha(0.6)
    ax.legend(
        loc='upper right',
        frameon=True,
        framealpha=0.95,
        edgecolor='#dddddd',
        fontsize=8.8,
    )
    fig.tight_layout()

    if output_path is not None:
        ensure_output_dir(output_path)
        fig.savefig(output_path, dpi=180, bbox_inches='tight')

    return fig, ax


def plot_final_fitness_ablation(
    final_fitness_groups: dict[str, list[float] | np.ndarray],
    title: str,
    output_path: Optional[str] = None,
):
    """绘制最终适应度均值、标准差和各随机种子散点。"""
    labels = list(final_fitness_groups.keys())
    values = [np.asarray(final_fitness_groups[label], dtype=float) for label in labels]
    means = np.array([np.mean(group) for group in values])
    stds = np.array([np.std(group) for group in values])

    x = np.arange(len(labels))
    colors = ['#4c78a8', '#f58518', '#54a24b', '#d95f5f', '#7f7f7f']

    fig_width = max(8.8, len(labels) * 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, 5.4))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        width=0.58,
        color=[colors[index % len(colors)] for index in range(len(labels))],
        alpha=0.88,
        edgecolor='white',
        linewidth=0.8,
        label='最终适应度均值 ± 标准差',
    )

    for index, group in enumerate(values):
        if len(group) == 1:
            jitter = np.array([0.0])
        else:
            jitter = np.linspace(-0.15, 0.15, len(group))
        ax.scatter(
            np.full(len(group), x[index]) + jitter,
            group,
            s=38,
            color='#333333',
            alpha=0.76,
            zorder=4,
            label='单个 seed 结果' if index == 0 else None,
        )

        ax.text(
            bars[index].get_x() + bars[index].get_width() / 2.0,
            means[index] + stds[index] + max(means) * 0.015,
            f'{means[index]:.2f}',
            ha='center',
            va='bottom',
            fontsize=8.5,
            color='#333333',
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha='right')
    ax.set_ylabel('最终适应度值（越低越好）')
    ax.set_title(title)
    ax.grid(True, axis='y', alpha=0.22, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_alpha(0.6)
    ax.spines['bottom'].set_alpha(0.6)
    ax.legend(
        loc='upper right',
        frameon=True,
        framealpha=0.95,
        edgecolor='#dddddd',
        fontsize=8.8,
    )
    fig.tight_layout()

    if output_path is not None:
        ensure_output_dir(output_path)
        fig.savefig(output_path, dpi=180, bbox_inches='tight')

    return fig, ax
