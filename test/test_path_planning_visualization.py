"""
路径规划可视化脚本
展示多组具有明显绕障效果的路径规划结果。
"""
import math
import os
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from config.params import ASTAR
from data.scenario_medium import create_medium_scenario
from src.route_planning.geometry import estimate_min_turn_radius, path_length, segment_intersects_inflated_threat
from src.route_planning.planner import PathPlanningResult, plan_path_for_uav
from src.visualization.common import ensure_output_dir, draw_battlefield, draw_path_lines


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

RESULT_DIR = 'results/path_planning'
MAX_CASES = 4


@dataclass(frozen=True)
class VisualizationCase:
    uav_id: int
    target_id: int
    result: PathPlanningResult
    threat_cross_count: int
    detour_ratio: float


def count_straight_line_threat_crossings(battlefield, uav_id: int, target_id: int, safety_margin: float) -> int:
    uav = battlefield.get_uav(uav_id)
    target = battlefield.get_target(target_id)
    start = (uav.x, uav.y)
    end = (target.x, target.y)

    return sum(
        1
        for threat in battlefield.threats
        if segment_intersects_inflated_threat(start, end, threat, safety_margin)
    )


def build_visualization_cases(battlefield, max_cases: int) -> list[VisualizationCase]:
    cases: list[VisualizationCase] = []
    safety_margin = ASTAR['safety_margin']

    for uav in battlefield.uavs:
        for target in battlefield.targets:
            threat_cross_count = count_straight_line_threat_crossings(
                battlefield,
                uav.id,
                target.id,
                safety_margin,
            )
            if threat_cross_count == 0:
                continue

            result = plan_path_for_uav(battlefield, uav.id, target.id, params=ASTAR)
            if not result.success or len(result.final_path) < 2:
                continue

            straight_distance = math.hypot(target.x - uav.x, target.y - uav.y)
            if straight_distance == 0.0:
                continue

            detour_ratio = result.path_length / straight_distance
            cases.append(
                VisualizationCase(
                    uav_id=uav.id,
                    target_id=target.id,
                    result=result,
                    threat_cross_count=threat_cross_count,
                    detour_ratio=detour_ratio,
                )
            )

    cases.sort(
        key=lambda case: (
            case.threat_cross_count,
            case.detour_ratio,
            path_length(case.result.original_path) - path_length(case.result.los_path),
        ),
        reverse=True,
    )
    return cases[:max_cases]


def draw_inflated_threats(ax, battlefield, safety_margin: float) -> None:
    for threat in battlefield.threats:
        inflated = plt.Circle(
            (threat.x, threat.y),
            threat.radius + safety_margin,
            color='orange',
            alpha=0.08,
            linestyle='--',
            fill=True,
        )
        ax.add_patch(inflated)


def draw_los_waypoints(ax, path_points: list[tuple[float, float]]) -> None:
    if len(path_points) < 2:
        return

    xs = [point[0] for point in path_points]
    ys = [point[1] for point in path_points]
    ax.scatter(
        xs,
        ys,
        s=22,
        c='#1f77b4',
        edgecolors='white',
        linewidths=0.8,
        zorder=7,
        label='LOS 折点',
    )


def print_case_summary(case: VisualizationCase) -> None:
    result = case.result
    estimated_radius = estimate_min_turn_radius(result.final_path)
    print(
        f'U{case.uav_id} -> T{case.target_id}: '
        f'threat_cross_count={case.threat_cross_count}, '
        f'detour_ratio={case.detour_ratio:.3f}, '
        f'min_turn_radius={ASTAR["min_turn_radius"]}, '
        f'estimated_min_turn_radius={estimated_radius:.3f}, '
        f'reported_min_turn_radius={result.estimated_min_turn_radius:.3f}, '
        f'kinematic_mode={result.kinematic_mode}, '
        f'original_points={len(result.original_path)}, '
        f'los_points={len(result.los_path)}, '
        f'kinematic_points={len(result.kinematic_path)}, '
        f'smoothed_points={len(result.smoothed_path)}, '
        f'final_points={len(result.final_path)}, '
        f'used_kinematic_constraint={result.used_kinematic_constraint}, '
        f'used_smoothing={result.used_smoothing}, '
        f'fallback_reason={result.fallback_reason}, '
        f'failure_reason={result.failure_reason}, '
        f'path_length={result.path_length:.4f}'
    )


def plot_cases(battlefield, cases: list[VisualizationCase], output_path: str) -> None:
    columns = 2
    rows = math.ceil(len(cases) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(14, 5.8 * rows))
    axes_list = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for ax, case in zip(axes_list, cases):
        result = case.result
        title = (
            f'U{case.uav_id} → T{case.target_id} | '
            f'穿越威胁数={case.threat_cross_count} | '
            f'绕行比={case.detour_ratio:.2f} | '
            f'模式={result.kinematic_mode}'
        )
        draw_battlefield(ax, battlefield, title)
        draw_inflated_threats(ax, battlefield, ASTAR['safety_margin'])

        draw_path_lines(ax, result.original_path, '#7f7f7f', 'A* 原始路径', linestyle='--', linewidth=1.0, alpha=0.35)
        draw_path_lines(ax, result.los_path, '#1f77b4', 'LOS 简化路径', linestyle=(0, (6, 3)), linewidth=2.2, alpha=0.95)
        draw_los_waypoints(ax, result.los_path)
        draw_path_lines(ax, result.kinematic_path, '#9467bd', '运动学约束路径', linestyle='-.', linewidth=2.4, alpha=0.95)

        if result.used_smoothing and result.smoothed_path == result.final_path:
            draw_path_lines(ax, result.smoothed_path, '#2ca02c', '局部 B 样条骨架', linestyle=(0, (1, 2)), linewidth=1.4, alpha=0.9)
        else:
            draw_path_lines(ax, result.smoothed_path, '#2ca02c', '平滑路径', linestyle=':', linewidth=2.0, alpha=0.9)

        draw_path_lines(ax, result.final_path, '#d62728', '最终采用路径', linestyle='-', linewidth=3.0, alpha=0.9)
        ax.legend(loc='upper left', fontsize=8)

    for ax in axes_list[len(cases):]:
        ax.axis('off')

    fig.tight_layout()
    ensure_output_dir(output_path)
    fig.savefig(output_path, dpi=180)
    print(f'图表已保存到 {output_path}')
    plt.show()


def main():
    battlefield = create_medium_scenario()
    cases = build_visualization_cases(battlefield, MAX_CASES)

    if not cases:
        print('未找到具有明显绕障效果的可行组合。')
        return

    print(f'共筛选出 {len(cases)} 组可视化组合：')
    for case in cases:
        print_case_summary(case)

    output_path = os.path.join(RESULT_DIR, 'path_planning_demo_multi.png')
    plot_cases(battlefield, cases, output_path)


if __name__ == '__main__':
    main()
