import csv
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.core.models import Battlefield
from src.route_planning.planner import AssignmentRoutePlan, RouteSegment
from src.visualization.common import ensure_output_dir
from src.visualization.preallocation import SEQUENCE_COLORS, draw_compact_battlefield


def _draw_inflated_threats(ax, battlefield: Battlefield, safety_margin: float) -> None:
    if safety_margin <= 0:
        return

    for threat in battlefield.threats:
        inflated = plt.Circle(
            (threat.x, threat.y),
            threat.radius + safety_margin,
            color='#f58518',
            alpha=0.055,
            linewidth=0.9,
            linestyle='--',
        )
        ax.add_patch(inflated)


def _draw_route_arrow(ax, path_points: list[tuple[float, float]], color: str, linewidth: float, alpha: float) -> None:
    if len(path_points) < 2:
        return

    start = path_points[-2]
    end = path_points[-1]
    ax.annotate(
        '',
        xy=end,
        xytext=start,
        arrowprops={
            'arrowstyle': '->',
            'color': color,
            'lw': linewidth,
            'alpha': alpha,
            'shrinkA': 5,
            'shrinkB': 8,
            'mutation_scale': 9,
        },
        zorder=5,
    )


def _draw_failed_segment(ax, segment: RouteSegment) -> None:
    ax.plot(
        [segment.start_xy[0], segment.end_xy[0]],
        [segment.start_xy[1], segment.end_xy[1]],
        color='#b33f3f',
        linestyle=(0, (4, 3)),
        linewidth=1.7,
        alpha=0.75,
        zorder=3,
    )
    ax.scatter(
        [segment.end_xy[0]],
        [segment.end_xy[1]],
        marker='x',
        s=72,
        color='#b33f3f',
        linewidths=1.8,
        zorder=8,
    )


def _route_style(route, color: str) -> tuple[str, float, float]:
    if len(route.target_ids) >= 2:
        return color, 1.75, 0.72
    return '#6d8299', 1.15, 0.42


def _build_route_legend(has_failed_segments: bool) -> list[Line2D]:
    legend_handles = [
        Line2D([0], [0], color='#1f77b4', linewidth=1.8, alpha=0.72, label='多任务 UAV 航迹'),
        Line2D([0], [0], color='#6d8299', linewidth=1.2, alpha=0.48, label='单任务 UAV 航迹'),
        Line2D([0], [0], color='#c44e52', marker='s', linestyle='None', markersize=6, label='目标点'),
        Line2D([0], [0], color='#1f77b4', marker='^', linestyle='None', markersize=7, label='UAV 起点'),
        Line2D([0], [0], color='#c44e52', linestyle='-', linewidth=7, alpha=0.10, label='威胁区'),
    ]
    if has_failed_segments:
        legend_handles.append(
            Line2D([0], [0], color='#b33f3f', linestyle=(0, (4, 3)), linewidth=1.7, label='失败航段')
        )
    return legend_handles


def plot_assignment_route_plan(
    battlefield: Battlefield,
    route_plan: AssignmentRoutePlan,
    title: str,
    output_path: Optional[str] = None,
    *,
    safety_margin: float = 0.0,
):
    """绘制 AssignmentPlan 对应的最终多 UAV 航迹规划图。"""
    fig, ax = plt.subplots(figsize=(10.2, 7.1))
    draw_compact_battlefield(ax, battlefield, title)
    _draw_inflated_threats(ax, battlefield, safety_margin)

    for color_index, uav_id in enumerate(sorted(route_plan.uav_route_plans)):
        route = route_plan.uav_route_plans[uav_id]
        if not route.active:
            continue

        color, linewidth, alpha = _route_style(
            route,
            SEQUENCE_COLORS[color_index % len(SEQUENCE_COLORS)],
        )
        full_path = route.full_path
        if len(full_path) >= 2:
            xs = [point[0] for point in full_path]
            ys = [point[1] for point in full_path]
            ax.plot(
                xs,
                ys,
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                zorder=4,
            )
            for segment in route.segments:
                if segment.success:
                    _draw_route_arrow(ax, segment.final_path, color, linewidth=linewidth, alpha=alpha)

        for failed_segment in route.failed_segments:
            _draw_failed_segment(ax, failed_segment)

    status_color = '#2f5f8f' if route_plan.success else '#9c2f2f'
    success_rate = (
        1.0 - len(route_plan.failed_segments) / route_plan.segment_count
        if route_plan.segment_count else 1.0
    )
    ax.text(
        0.99,
        0.02,
        (
            f'航迹成功率 {success_rate:.0%}\n'
            f'任务航段 {route_plan.segment_count} | 总航迹 {route_plan.total_path_length:.1f} km'
        ),
        transform=ax.transAxes,
        ha='right',
        va='bottom',
        fontsize=9.4,
        color=status_color,
        bbox={'boxstyle': 'round,pad=0.32', 'facecolor': 'white', 'edgecolor': '#dddddd', 'alpha': 0.96},
        zorder=10,
    )

    ax.legend(
        handles=_build_route_legend(bool(route_plan.failed_segments)),
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=7.8,
        title='图例',
        title_fontsize=8.6,
        frameon=True,
        framealpha=0.92,
        edgecolor='#dddddd',
    )

    fig.tight_layout()
    if output_path is not None:
        ensure_output_dir(output_path)
        fig.savefig(output_path, dpi=180, bbox_inches='tight')

    return fig, ax


def write_route_plan_summary_csv(route_plan: AssignmentRoutePlan, output_path: str) -> None:
    """输出最终航迹规划摘要，便于论文表格整理。"""
    ensure_output_dir(output_path)
    fieldnames = [
        'uav_id',
        'target_sequence',
        'segment_count',
        'success',
        'total_path_length',
        'failed_segments',
    ]
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for uav_id in sorted(route_plan.uav_route_plans):
            route = route_plan.uav_route_plans[uav_id]
            failed = [
                f'{segment.start_kind}:{segment.start_id}->T{segment.end_target_id}({segment.failure_reason})'
                for segment in route.failed_segments
            ]
            writer.writerow({
                'uav_id': uav_id,
                'target_sequence': '->'.join(f'T{target_id}' for target_id in route.target_ids),
                'segment_count': len(route.segments),
                'success': int(route.success),
                'total_path_length': f'{route.total_path_length:.6f}',
                'failed_segments': '; '.join(failed),
            })
