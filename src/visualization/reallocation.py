import csv
import os
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from src.core.models import AssignmentPlan, Battlefield
from src.core.sequence_eval import evaluate_uav_task_sequence
from src.re_allocation.events import Event, EventType
from src.re_allocation.mcha import BidResult, BidRoundLog
from src.visualization.common import (
    draw_assignment_lines,
    draw_battlefield,
    ensure_output_dir,
)
from src.visualization.preallocation import (
    SEQUENCE_COLORS,
    draw_compact_battlefield,
)


@dataclass(frozen=True)
class SequenceSegment:
    """任务链中的一条航段。"""

    uav_id: int
    start_kind: str
    start_id: int
    end_target_id: int


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


def _add_sequence_legend(ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return

    ax.legend(
        handles,
        labels,
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=7.5,
        title='变化任务链',
        title_fontsize=8.5,
        frameon=True,
        framealpha=0.92,
        edgecolor='#dddddd',
    )


def _draw_event_badge(ax, text: str, color: str = '#9c2f2f') -> None:
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=9.2,
        color=color,
        bbox={'boxstyle': 'round,pad=0.32', 'facecolor': 'white', 'edgecolor': color, 'alpha': 0.94},
        zorder=10,
    )


def _sequence_targets(plan: AssignmentPlan, uav_id: int) -> tuple[int, ...]:
    sequence = plan.uav_task_sequences.get(uav_id)
    if sequence is None:
        return ()
    return tuple(sequence.target_ids())


def _plan_segments(plan: AssignmentPlan) -> set[SequenceSegment]:
    segments: set[SequenceSegment] = set()
    for uav_id, sequence in plan.uav_task_sequences.items():
        previous_kind = 'uav'
        previous_id = uav_id
        for task in sequence.tasks:
            segments.add(
                SequenceSegment(
                    uav_id=uav_id,
                    start_kind=previous_kind,
                    start_id=previous_id,
                    end_target_id=task.target_id,
                )
            )
            previous_kind = 'target'
            previous_id = task.target_id
    return segments


def _changed_uav_ids(plan_before: AssignmentPlan, plan_after: AssignmentPlan) -> set[int]:
    uav_ids = set(plan_before.uav_task_sequences) | set(plan_after.uav_task_sequences)
    return {
        uav_id
        for uav_id in uav_ids
        if _sequence_targets(plan_before, uav_id) != _sequence_targets(plan_after, uav_id)
    }


def _sequence_points(battlefield: Battlefield, plan: AssignmentPlan, uav_id: int) -> list[tuple[float, float]]:
    sequence = plan.uav_task_sequences.get(uav_id)
    if sequence is None or sequence.task_count() == 0:
        return []

    uav = battlefield.get_uav(uav_id)
    points = [(uav.x, uav.y)]
    for task in sequence.tasks:
        target = battlefield.get_target(task.target_id)
        points.append((target.x, target.y))
    return points


def _draw_sequence_line(
    ax,
    battlefield: Battlefield,
    plan: AssignmentPlan,
    uav_id: int,
    *,
    color: str,
    linewidth: float,
    alpha: float,
    zorder: int,
    label: str | None = None,
) -> None:
    points = _sequence_points(battlefield, plan, uav_id)
    if len(points) < 2:
        return

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    ax.plot(
        xs,
        ys,
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        label=label,
        zorder=zorder,
    )

    for start, end in zip(points[:-1], points[1:]):
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
                'mutation_scale': 11,
            },
            zorder=zorder + 1,
        )


def _segment_points(
    battlefield: Battlefield,
    segment: SequenceSegment,
) -> tuple[tuple[float, float], tuple[float, float]]:
    if segment.start_kind == 'uav':
        start_uav = battlefield.get_uav(segment.start_id)
        start = (start_uav.x, start_uav.y)
    else:
        start_target = battlefield.get_target(segment.start_id)
        start = (start_target.x, start_target.y)

    end_target = battlefield.get_target(segment.end_target_id)
    end = (end_target.x, end_target.y)
    return start, end


def _draw_segment(
    ax,
    battlefield: Battlefield,
    segment: SequenceSegment,
    *,
    color: str,
    linestyle: str,
    linewidth: float,
    alpha: float,
    zorder: int,
    label: str | None = None,
) -> None:
    start, end = _segment_points(battlefield, segment)
    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        linestyle=linestyle,
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        label=label,
        zorder=zorder,
    )
    ax.annotate(
        '',
        xy=end,
        xytext=start,
        arrowprops={
            'arrowstyle': '->',
            'color': color,
            'lw': linewidth,
            'alpha': alpha,
            'linestyle': linestyle,
            'shrinkA': 8,
            'shrinkB': 9,
            'mutation_scale': 12,
        },
        zorder=zorder + 1,
    )


def _draw_reallocation_sequence_lines(
    ax,
    battlefield: Battlefield,
    plan: AssignmentPlan,
    *,
    changed_uav_ids: set[int],
    show_unchanged_label: bool = False,
) -> None:
    uav_ids = sorted(plan.uav_task_sequences)
    unchanged_labeled = False

    for uav_id in uav_ids:
        if uav_id in changed_uav_ids:
            continue
        label = '未变化任务链' if show_unchanged_label and not unchanged_labeled else None
        _draw_sequence_line(
            ax,
            battlefield,
            plan,
            uav_id,
            color='#b8b8b8',
            linewidth=1.25,
            alpha=0.32,
            zorder=1,
            label=label,
        )
        unchanged_labeled = unchanged_labeled or label is not None

    for uav_id in uav_ids:
        if uav_id not in changed_uav_ids:
            continue
        targets = _sequence_targets(plan, uav_id)
        if not targets:
            continue
        color = SEQUENCE_COLORS[uav_id % len(SEQUENCE_COLORS)]
        label = f'U{uav_id}: ' + '→'.join(f'T{target_id}' for target_id in targets)
        _draw_sequence_line(
            ax,
            battlefield,
            plan,
            uav_id,
            color=color,
            linewidth=2.35,
            alpha=0.88,
            zorder=3,
            label=label,
        )


def _highlight_uav(ax, battlefield: Battlefield, uav_id: int, *, after_event: bool) -> None:
    uav = battlefield.get_uav(uav_id)
    if after_event:
        ax.scatter(
            uav.x,
            uav.y,
            marker='x',
            s=145,
            color='#555555',
            linewidths=2.2,
            zorder=9,
        )
        _draw_event_badge(ax, f'UAV_LOST: U{uav_id}', '#555555')
    else:
        ax.scatter(
            uav.x,
            uav.y,
            marker='o',
            s=220,
            facecolors='none',
            edgecolors='#d95f5f',
            linewidths=1.8,
            zorder=8,
        )
        _draw_event_badge(ax, f'损失 UAV 预览：U{uav_id}', '#9c2f2f')


def _target_assignee_ids(plan: AssignmentPlan | None, target_id: int) -> list[int]:
    if plan is None:
        return []

    assignees = plan.target_assignees.get(target_id)
    if assignees:
        return sorted(assignees)

    found = []
    for uav_id, sequence in plan.uav_task_sequences.items():
        if target_id in sequence.target_ids():
            found.append(uav_id)
    return sorted(found)


def _format_assignee_label(assignee_ids: list[int], target_id: int, action: str) -> str:
    if not assignee_ids:
        return ''
    uav_text = '、'.join(f'U{uav_id}' for uav_id in assignee_ids)
    return f'{uav_text} {action} T{target_id}'


def _annotate_target_assignees(
    ax,
    target,
    assignee_ids: list[int],
    *,
    action: str = '接入',
) -> None:
    label = _format_assignee_label(assignee_ids, target.id, action)
    if not label:
        return

    ax.annotate(
        label,
        xy=(target.x, target.y),
        xytext=(1.02, 0.47),
        xycoords='data',
        textcoords=ax.transAxes,
        ha='left',
        va='center',
        fontsize=8.8,
        color='#1f6f2a',
        bbox={'boxstyle': 'round,pad=0.28', 'facecolor': 'white', 'edgecolor': '#2ca02c', 'alpha': 0.96},
        arrowprops={
            'arrowstyle': '->',
            'color': '#2ca02c',
            'lw': 1.2,
            'shrinkA': 4,
            'shrinkB': 5,
        },
        annotation_clip=False,
        zorder=12,
    )


def _highlight_target(
    ax,
    target,
    *,
    after_event: bool,
    assignee_ids: list[int] | None = None,
) -> None:
    ax.scatter(
        target.x,
        target.y,
        marker='s',
        s=150 if after_event else 130,
        facecolors='none',
        edgecolors='#f58518',
        linewidths=2.0,
        zorder=9,
    )
    ax.annotate(
        f'T{target.id}',
        (target.x, target.y),
        textcoords='offset points',
        xytext=(7, 8),
        fontsize=8.5,
        color='#a65f00',
        zorder=10,
    )
    label = f'新增目标：T{target.id}' if after_event else f'新增目标预览：T{target.id}'
    _draw_event_badge(ax, label, '#a65f00')
    if after_event:
        _annotate_target_assignees(ax, target, assignee_ids or [])


def _new_assignee_ids(
    before_plan: AssignmentPlan | None,
    after_plan: AssignmentPlan | None,
    target_id: int,
) -> list[int]:
    before_assignees = set(_target_assignee_ids(before_plan, target_id))
    after_assignees = set(_target_assignee_ids(after_plan, target_id))
    return sorted(after_assignees - before_assignees)


def _highlight_target_demand_increased(
    ax,
    battlefield: Battlefield,
    event: Event,
    *,
    after_event: bool,
    before_plan: AssignmentPlan | None = None,
    after_plan: AssignmentPlan | None = None,
) -> None:
    target_id = event.data['target_id']
    target = battlefield.get_target(target_id)
    new_required = event.data['new_required_uavs']
    old_required = event.data.get('old_required_uavs')

    ax.scatter(
        target.x,
        target.y,
        marker='s',
        s=155,
        facecolors='none',
        edgecolors='#f58518',
        linewidths=2.0,
        zorder=9,
    )
    demand_text = (
        f'T{target_id}: {old_required}->{new_required}架'
        if old_required is not None
        else f'T{target_id}: 需求->{new_required}架'
    )
    label = f'需求增加：{demand_text}' if after_event else f'需求增加预览：{demand_text}'
    _draw_event_badge(ax, label, '#a65f00')

    if after_event:
        _annotate_target_assignees(
            ax,
            target,
            _new_assignee_ids(before_plan, after_plan, target_id),
            action='补充',
        )


def _highlight_threat(ax, threat, *, after_event: bool) -> None:
    circle = plt.Circle(
        (threat.x, threat.y),
        threat.radius,
        fill=after_event,
        facecolor='#d95f5f' if after_event else 'none',
        edgecolor='#9c2f2f',
        alpha=0.15 if after_event else 0.55,
        linewidth=1.8 if after_event else 1.6,
        linestyle='-' if after_event else '--',
        zorder=7,
    )
    ax.add_patch(circle)
    ax.scatter(
        threat.x,
        threat.y,
        marker='x',
        s=55,
        color='#9c2f2f',
        linewidths=1.8,
        zorder=9,
    )
    label = f'新增威胁：Threat-{threat.id}' if after_event else f'新增威胁预览：Threat-{threat.id}'
    _draw_event_badge(ax, label, '#9c2f2f')


def _highlight_event(
    ax,
    battlefield: Battlefield,
    event: Event | None,
    *,
    after_event: bool,
    plan: AssignmentPlan | None = None,
    before_plan: AssignmentPlan | None = None,
) -> None:
    if event is None:
        return

    if event.type == EventType.UAV_LOST:
        _highlight_uav(ax, battlefield, event.data['uav_id'], after_event=after_event)
        return

    if event.type == EventType.TARGET_ADDED:
        target = event.data['target']
        _highlight_target(
            ax,
            target,
            after_event=after_event,
            assignee_ids=_target_assignee_ids(plan, target.id) if after_event else None,
        )
        return

    if event.type == EventType.TARGET_DEMAND_INCREASED:
        _highlight_target_demand_increased(
            ax,
            battlefield,
            event,
            after_event=after_event,
            before_plan=before_plan,
            after_plan=plan,
        )
        return

    if event.type == EventType.THREAT_ADDED:
        _highlight_threat(ax, event.data['threat'], after_event=after_event)


def plot_plan_reallocation_before_after(
    battlefield_before: Battlefield,
    plan_before: AssignmentPlan,
    battlefield_after: Battlefield,
    plan_after: AssignmentPlan,
    title_before: str,
    title_after: str,
    output_path: Optional[str] = None,
    event: Event | None = None,
):
    """绘制任务序列版重分配前后对比图。"""
    changed_uav_ids = _changed_uav_ids(plan_before, plan_after)
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 7.2), constrained_layout=True)

    draw_compact_battlefield(axes[0], battlefield_before, title_before)
    _draw_reallocation_sequence_lines(
        axes[0],
        battlefield_before,
        plan_before,
        changed_uav_ids=changed_uav_ids,
    )
    _highlight_event(
        axes[0],
        battlefield_before,
        event,
        after_event=False,
        plan=plan_before,
        before_plan=plan_before,
    )
    _add_sequence_legend(axes[0])

    draw_compact_battlefield(axes[1], battlefield_after, title_after)
    _draw_reallocation_sequence_lines(
        axes[1],
        battlefield_after,
        plan_after,
        changed_uav_ids=changed_uav_ids,
        show_unchanged_label=True,
    )
    _highlight_event(
        axes[1],
        battlefield_after,
        event,
        after_event=True,
        plan=plan_after,
        before_plan=plan_before,
    )
    _add_sequence_legend(axes[1])

    fig.suptitle('任务序列重分配前后对比', fontsize=14, y=1.02)

    if output_path is not None:
        ensure_output_dir(output_path)
        fig.savefig(output_path, dpi=180, bbox_inches='tight')

    return fig, axes


def plot_plan_reallocation_diff(
    battlefield: Battlefield,
    plan_before: AssignmentPlan,
    plan_after: AssignmentPlan,
    title: str,
    output_path: Optional[str] = None,
    event: Event | None = None,
):
    """绘制任务链航段变化图：保持、释放、新增。"""
    before_segments = _plan_segments(plan_before)
    after_segments = _plan_segments(plan_after)

    kept_segments = sorted(
        before_segments & after_segments,
        key=lambda item: (item.uav_id, item.start_kind, item.start_id, item.end_target_id),
    )
    released_segments = sorted(
        before_segments - after_segments,
        key=lambda item: (item.uav_id, item.start_kind, item.start_id, item.end_target_id),
    )
    added_segments = sorted(
        after_segments - before_segments,
        key=lambda item: (item.uav_id, item.start_kind, item.start_id, item.end_target_id),
    )

    fig, ax = plt.subplots(figsize=(9.8, 7.2))
    draw_compact_battlefield(ax, battlefield, title)

    labels_used = set()
    for segment in kept_segments:
        label = '保持不变' if 'kept' not in labels_used else None
        _draw_segment(
            ax,
            battlefield,
            segment,
            color='#b8b8b8',
            linestyle='-',
            linewidth=1.2,
            alpha=0.34,
            zorder=1,
            label=label,
        )
        labels_used.add('kept')

    for segment in released_segments:
        label = '事件释放' if 'released' not in labels_used else None
        _draw_segment(
            ax,
            battlefield,
            segment,
            color='#c44e52',
            linestyle='--',
            linewidth=2.15,
            alpha=0.88,
            zorder=3,
            label=label,
        )
        labels_used.add('released')

    for segment in added_segments:
        label = '新增接入' if 'added' not in labels_used else None
        _draw_segment(
            ax,
            battlefield,
            segment,
            color='#2ca02c',
            linestyle='-',
            linewidth=2.25,
            alpha=0.9,
            zorder=4,
            label=label,
        )
        labels_used.add('added')

    _highlight_event(
        ax,
        battlefield,
        event,
        after_event=True,
        plan=plan_after,
        before_plan=plan_before,
    )
    ax.text(
        0.99,
        0.02,
        f'保持 {len(kept_segments)} | 释放 {len(released_segments)} | 新增 {len(added_segments)}',
        transform=ax.transAxes,
        ha='right',
        va='bottom',
        fontsize=9,
        color='#333333',
        bbox={'boxstyle': 'round,pad=0.32', 'facecolor': 'white', 'edgecolor': '#dddddd', 'alpha': 0.95},
        zorder=10,
    )
    ax.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=8.5,
        title='任务链变化',
        title_fontsize=9.2,
        frameon=True,
        framealpha=0.94,
        edgecolor='#dddddd',
    )
    fig.tight_layout()

    if output_path is not None:
        ensure_output_dir(output_path)
        fig.savefig(output_path, dpi=180, bbox_inches='tight')

    return fig, ax


def _target_by_id(battlefield: Battlefield, target_id: int):
    try:
        return battlefield.get_target(target_id)
    except ValueError:
        return None


def _assignment_count_for_target(plan: AssignmentPlan, target_id: int) -> int:
    return len(_target_assignee_ids(plan, target_id))


def _event_target_ids(event: Event | None) -> set[int]:
    if event is None:
        return set()
    if event.type == EventType.TARGET_ADDED:
        return {event.data['target'].id}
    if event.type == EventType.TARGET_DEMAND_INCREASED:
        return {event.data['target_id']}
    return set()


def _target_load_rows(
    battlefield_before: Battlefield,
    plan_before: AssignmentPlan,
    battlefield_after: Battlefield,
    plan_after: AssignmentPlan,
    event: Event | None,
) -> list[dict]:
    target_ids = sorted({
        target.id for target in battlefield_before.targets
    } | {
        target.id for target in battlefield_after.targets
    })
    event_target_ids = _event_target_ids(event)

    rows = []
    for target_id in target_ids:
        before_target = _target_by_id(battlefield_before, target_id)
        after_target = _target_by_id(battlefield_after, target_id)

        before_required = before_target.required_uavs if before_target is not None else 0
        after_required = after_target.required_uavs if after_target is not None else 0
        before_count = _assignment_count_for_target(plan_before, target_id)
        after_count = _assignment_count_for_target(plan_after, target_id)
        changed = (
            before_count != after_count
            or before_required != after_required
            or target_id in event_target_ids
        )
        satisfied = after_count >= after_required

        rows.append({
            'target_id': target_id,
            'before_count': before_count,
            'after_count': after_count,
            'before_required': before_required,
            'after_required': after_required,
            'changed': changed,
            'satisfied': satisfied,
        })

    focused_rows = [row for row in rows if row['changed']]
    return focused_rows if focused_rows else rows


def plot_plan_reallocation_target_loads(
    battlefield_before: Battlefield,
    plan_before: AssignmentPlan,
    battlefield_after: Battlefield,
    plan_after: AssignmentPlan,
    title: str,
    output_path: Optional[str] = None,
    event: Event | None = None,
):
    """绘制任务序列版重分配前后目标需求满足变化图。"""
    rows = _target_load_rows(
        battlefield_before,
        plan_before,
        battlefield_after,
        plan_after,
        event,
    )

    target_labels = [f'T{row["target_id"]}' for row in rows]
    before_counts = [row['before_count'] for row in rows]
    after_counts = [row['after_count'] for row in rows]
    after_required = [row['after_required'] for row in rows]
    satisfied = [row['satisfied'] for row in rows]
    satisfied_count = sum(1 for item in satisfied if item)
    satisfied_rate = satisfied_count / len(rows) if rows else 1.0

    x = np.arange(len(rows))
    width = 0.24 if len(rows) <= 3 else 0.30
    fig_width = max(6.8, min(11.5, 4.8 + len(rows) * 0.55))
    fig, ax = plt.subplots(figsize=(fig_width, 5.1))

    ax.bar(
        x - width / 2,
        before_counts,
        width=width,
        color='#a9a9a9',
        alpha=0.74,
        edgecolor='white',
        linewidth=0.7,
        label='事件前分配数',
    )
    after_colors = ['#5b8cc0' if ok else '#c44e52' for ok in satisfied]
    bars = ax.bar(
        x + width / 2,
        after_counts,
        width=width,
        color=after_colors,
        alpha=0.92,
        edgecolor='white',
        linewidth=0.7,
        label='重分配后分配数',
    )

    ax.scatter(
        x,
        after_required,
        color='#f58518',
        marker='D',
        s=34,
        zorder=4,
        label='目标需求数量',
    )
    for idx, required in enumerate(after_required):
        ax.hlines(
            required,
            idx - 0.30,
            idx + 0.30,
            colors='#f58518',
            linewidth=1.25,
            alpha=0.76,
            zorder=3,
        )

    for idx, (bar, before, after, required) in enumerate(zip(bars, before_counts, after_counts, after_required)):
        if after != before:
            delta = after - before
            sign = '+' if delta > 0 else ''
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                max(after, required) + 0.12,
                f'{sign}{delta}',
                ha='center',
                va='bottom',
                fontsize=8.4,
                color='#2f5f8f' if delta >= 0 else '#9c2f2f',
            )
        if after < required:
            ax.text(
                idx,
                required + 0.42,
                '未满足',
                ha='center',
                va='bottom',
                fontsize=8.3,
                color='#c44e52',
            )

    event_hint = ''
    if event is not None and event.type == EventType.TARGET_DEMAND_INCREASED:
        event_hint = f" | T{event.data['target_id']}需求 {event.data.get('old_required_uavs', '?')}->{event.data['new_required_uavs']}"
    elif event is not None and event.type == EventType.TARGET_ADDED:
        event_hint = f" | 新增 T{event.data['target'].id}"

    ax.text(
        0.99,
        0.93,
        f'展示变化目标 {len(rows)} 个 | 满足率 {satisfied_rate:.0%}{event_hint}',
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=9.1,
        color='#2f5f8f' if satisfied_rate == 1.0 else '#9c2f2f',
        bbox={'boxstyle': 'round,pad=0.32', 'facecolor': 'white', 'edgecolor': '#dddddd', 'alpha': 0.96},
    )

    ax.set_xticks(x)
    ax.set_xticklabels(target_labels, rotation=0)
    if len(rows) == 1:
        ax.set_xlim(-0.55, 0.55)
    ax.set_xlabel('目标编号')
    ax.set_ylabel('UAV 数量')
    ax.set_title(title)
    ax.set_ylim(0, max(max(before_counts, default=0), max(after_counts, default=0), max(after_required, default=0)) + 1.0)
    ax.grid(True, axis='y', alpha=0.18, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_alpha(0.6)
    ax.spines['bottom'].set_alpha(0.6)
    ax.legend(
        loc='upper left',
        ncol=3,
        frameon=True,
        framealpha=0.95,
        edgecolor='#dddddd',
        fontsize=8.7,
    )
    fig.tight_layout()

    if output_path is not None:
        ensure_output_dir(output_path)
        fig.savefig(output_path, dpi=180, bbox_inches='tight')

    return fig, ax


def _uav_task_counts(
    battlefield: Battlefield,
    plan: AssignmentPlan,
) -> list[int]:
    counts = []
    for uav in battlefield.uavs:
        sequence = plan.uav_task_sequences.get(uav.id)
        counts.append(sequence.task_count() if sequence is not None else 0)
    return counts


def plot_plan_reallocation_uav_loads(
    battlefield_before: Battlefield,
    plan_before: AssignmentPlan,
    battlefield_after: Battlefield,
    plan_after: AssignmentPlan,
    title: str,
    output_path: Optional[str] = None,
):
    """绘制任务序列版重分配前后 UAV 任务负载变化图。"""
    del battlefield_before

    uav_ids = [uav.id for uav in battlefield_after.uavs]
    before_counts = _uav_task_counts(battlefield_after, plan_before)
    after_counts = _uav_task_counts(battlefield_after, plan_after)
    ammo_limits = [uav.ammo for uav in battlefield_after.uavs]
    deltas = [after - before for before, after in zip(before_counts, after_counts)]
    changed_count = sum(1 for delta in deltas if delta != 0)
    active_after = sum(1 for count in after_counts if count > 0)
    avg_after = float(np.mean(after_counts)) if after_counts else 0.0
    max_after = max(after_counts, default=0)

    y = np.arange(len(uav_ids))
    fig_height = max(4.9, 0.43 * len(uav_ids) + 1.8)
    fig, ax = plt.subplots(figsize=(8.8, fig_height))

    for idx, (before, after, delta) in enumerate(zip(before_counts, after_counts, deltas)):
        if delta > 0:
            color = '#2ca02c'
            alpha = 0.86
            linewidth = 2.2
        elif delta < 0:
            color = '#c44e52'
            alpha = 0.86
            linewidth = 2.2
        else:
            color = '#c9c9c9'
            alpha = 0.55
            linewidth = 1.4

        ax.plot(
            [before, after],
            [idx, idx],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=2,
        )
        if delta != 0:
            sign = '+' if delta > 0 else ''
            ax.text(
                max(before, after) + 0.10,
                idx,
                f'{sign}{delta}',
                ha='left',
                va='center',
                fontsize=8.5,
                color=color,
            )

    ax.scatter(
        before_counts,
        y,
        s=42,
        color='#9a9a9a',
        edgecolors='white',
        linewidths=0.7,
        zorder=4,
        label='事件前任务数',
    )
    ax.scatter(
        after_counts,
        y,
        s=50,
        color='#5b8cc0',
        edgecolors='white',
        linewidths=0.7,
        zorder=5,
        label='重分配后任务数',
    )
    ax.scatter(
        ammo_limits,
        y,
        s=34,
        color='#f58518',
        marker='D',
        edgecolors='white',
        linewidths=0.6,
        zorder=6,
        label='ammo 容量',
    )

    for idx, ammo in enumerate(ammo_limits):
        ax.vlines(
            ammo,
            idx - 0.25,
            idx + 0.25,
            colors='#f58518',
            linewidth=1.0,
            alpha=0.55,
            zorder=1,
        )

    ax.text(
        0.99,
        0.985,
        f'变化 UAV {changed_count} 架 | 活跃 UAV {active_after}/{len(uav_ids)} | 平均任务 {avg_after:.2f} | 最大链长 {max_after}',
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=9.0,
        color='#2f5f8f',
        bbox={'boxstyle': 'round,pad=0.32', 'facecolor': 'white', 'edgecolor': '#dddddd', 'alpha': 0.96},
    )

    ax.set_yticks(y)
    ax.set_yticklabels([f'U{uav_id}' for uav_id in uav_ids])
    ax.set_ylim(len(uav_ids) - 0.55, -0.85)
    ax.set_xlabel('任务链长度')
    ax.set_ylabel('UAV 编号')
    ax.set_title(title)
    max_x = max(max(before_counts, default=0), max(after_counts, default=0), max(ammo_limits, default=0))
    ax.set_xlim(-0.15, max_x + 0.85)
    ax.set_xticks(np.arange(0, max_x + 1))
    ax.grid(True, axis='x', alpha=0.18, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_alpha(0.6)
    ax.spines['bottom'].set_alpha(0.6)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        frameon=True,
        framealpha=0.95,
        edgecolor='#dddddd',
        fontsize=8.6,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    if output_path is not None:
        ensure_output_dir(output_path)
        fig.savefig(output_path, dpi=180, bbox_inches='tight')

    return fig, ax


def plot_mcha_winning_bids(
    selected_bids: list[BidResult],
    title: str,
    output_path: Optional[str] = None,
):
    """绘制 MCHA 重分配过程中的中标结果。"""
    fig_height = max(3.8, 0.42 * max(1, len(selected_bids)) + 1.8)
    fig, ax = plt.subplots(figsize=(9.2, fig_height))

    if not selected_bids:
        ax.text(
            0.5,
            0.5,
            '本事件未产生 MCHA 中标结果',
            transform=ax.transAxes,
            ha='center',
            va='center',
            fontsize=11,
            color='#555555',
        )
        ax.set_axis_off()
        ax.set_title(title)
        if output_path is not None:
            ensure_output_dir(output_path)
            fig.savefig(output_path, dpi=180, bbox_inches='tight')
        return fig, ax

    labels = [
        f'{idx}. U{bid.uav_id} → T{bid.target_id}'
        for idx, bid in enumerate(selected_bids, start=1)
    ]
    scores = np.array([float(bid.score) for bid in selected_bids])
    y = np.arange(len(selected_bids))

    min_score = float(np.min(scores))
    max_score = float(np.max(scores))
    score_span = max(max_score - min_score, 1e-9)
    normalized = (scores - min_score) / score_span
    colors = [
        plt.cm.Blues(0.45 + 0.42 * value)
        for value in normalized
    ]

    bars = ax.barh(
        y,
        scores,
        color=colors,
        alpha=0.92,
        edgecolor='white',
        linewidth=0.8,
    )
    ax.axvline(0.0, color='#777777', linewidth=1.0, alpha=0.55)

    x_min, x_max, offset, _ = _mcha_score_axis_scale(scores)
    for bar, score in zip(bars, scores):
        if score >= 0:
            x_text = score + offset
            ha = 'left'
        else:
            x_text = score - offset
            ha = 'right'
        ax.text(
            x_text,
            bar.get_y() + bar.get_height() / 2.0,
            _format_mcha_score(float(score)),
            ha=ha,
            va='center',
            fontsize=8.5,
            color='#333333',
        )

    unique_targets = sorted({bid.target_id for bid in selected_bids})
    unique_uavs = sorted({bid.uav_id for bid in selected_bids})
    ax.text(
        1.02,
        0.99,        f'中标 {len(selected_bids)} 次 | 覆盖目标 {len(unique_targets)} 个 | 中标 UAV {len(unique_uavs)} 架',
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=9.0,
        color='#2f5f8f',
        bbox={'boxstyle': 'round,pad=0.32', 'facecolor': 'white', 'edgecolor': '#dddddd', 'alpha': 0.96},
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel('MCHA 边际竞标得分（越高越优）')
    ax.set_ylabel('中标顺序与任务')
    ax.set_title(title)

    ax.set_xlim(x_min, x_max)
    ax.grid(True, axis='x', alpha=0.18, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_alpha(0.6)
    ax.spines['bottom'].set_alpha(0.6)
    fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))

    if output_path is not None:
        ensure_output_dir(output_path)
        fig.savefig(output_path, dpi=180, bbox_inches='tight')

    return fig, ax


def _bid_key(bid: BidResult) -> tuple[int, int]:
    return bid.uav_id, bid.target_id


def _format_mcha_score(score: float) -> str:
    """按归一化后的小量级得分自适应显示精度。"""
    if abs(score) < 5e-5:
        score = 0.0
    abs_score = abs(score)
    if abs_score >= 1.0:
        return f'{score:.2f}'
    if abs_score >= 0.1:
        return f'{score:.3f}'
    return f'{score:.4f}'


def _mcha_score_axis_scale(scores: np.ndarray) -> tuple[float, float, float, float]:
    """为归一化 MCHA 得分计算坐标范围和标注偏移量。"""
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))
    scale = max(abs(min_score), abs(max_score), max_score - min_score, 1e-3)
    margin = scale * 0.18
    offset = scale * 0.035
    x_min = min(min_score, 0.0) - margin
    x_max = max(max_score, 0.0) + margin
    return x_min, x_max, offset, scale


def plot_mcha_candidate_bid_scores(
    bid_round_logs: list[BidRoundLog],
    title: str,
    output_path: Optional[str] = None,
):
    """绘制 MCHA 每轮候选投标得分与中标结果。"""
    plot_rows = []
    for round_log in bid_round_logs:
        accepted_keys = {_bid_key(bid) for bid in round_log.accepted_bids}
        for bid in sorted(round_log.candidate_bids, key=lambda item: item.score, reverse=True):
            plot_rows.append({
                'iteration': round_log.iteration,
                'uav_id': bid.uav_id,
                'target_id': bid.target_id,
                'score': float(bid.score),
                'accepted': _bid_key(bid) in accepted_keys,
            })

    fig_height = max(4.2, 0.34 * max(1, len(plot_rows)) + 1.9)
    fig, ax = plt.subplots(figsize=(10.2, fig_height))

    if not plot_rows:
        ax.text(
            0.5,
            0.5,
            '本事件未产生候选投标记录',
            transform=ax.transAxes,
            ha='center',
            va='center',
            fontsize=11,
            color='#555555',
        )
        ax.set_axis_off()
        ax.set_title(title)
        if output_path is not None:
            ensure_output_dir(output_path)
            fig.savefig(output_path, dpi=180, bbox_inches='tight')
        return fig, ax

    labels = [f"R{row['iteration']}  U{row['uav_id']} → T{row['target_id']}" for row in plot_rows]
    scores = np.array([row['score'] for row in plot_rows])
    accepted = [row['accepted'] for row in plot_rows]
    y = np.arange(len(plot_rows))

    round_ids = [row['iteration'] for row in plot_rows]
    for round_id in sorted(set(round_ids)):
        round_indices = [idx for idx, item in enumerate(round_ids) if item == round_id]
        if not round_indices:
            continue
        if round_id % 2 == 0:
            ax.axhspan(
                min(round_indices) - 0.45,
                max(round_indices) + 0.45,
                color='#f5f7fa',
                alpha=0.9,
                zorder=0,
            )

    candidate_y = [idx for idx, item in enumerate(accepted) if not item]
    winner_y = [idx for idx, item in enumerate(accepted) if item]
    if candidate_y:
        ax.scatter(
            scores[candidate_y],
            candidate_y,
            s=62,
            color='#b8c2cc',
            edgecolors='white',
            linewidths=0.7,
            alpha=0.92,
            zorder=4,
            label='候选投标',
        )
    if winner_y:
        ax.scatter(
            scores[winner_y],
            winner_y,
            s=92,
            color='#2f5f8f',
            edgecolors='#173f5f',
            linewidths=1.2,
            alpha=0.98,
            zorder=5,
            label='中标投标',
        )

    x_min, x_max, offset, _ = _mcha_score_axis_scale(scores)
    for row_y, score, is_accepted in zip(y, scores, accepted):
        x_text = score + offset if score >= 0 else score - offset
        ax.text(
            x_text,
            row_y,
            _format_mcha_score(float(score)) + ('  中标' if is_accepted else ''),
            ha='left' if score >= 0 else 'right',
            va='center',
            fontsize=8.2,
            color='#17476f' if is_accepted else '#555555',
            weight='bold' if is_accepted else 'normal',
        )

    candidate_count = len(plot_rows)
    accepted_count = sum(1 for item in accepted if item)
    candidate_uavs = sorted({row['uav_id'] for row in plot_rows})
    candidate_targets = sorted({row['target_id'] for row in plot_rows})
    summary_text = (
        f'候选投标 {candidate_count} 个 | 中标 {accepted_count} 个 | '
        f'候选 UAV {len(candidate_uavs)} 架 | 开放目标 {len(candidate_targets)} 个'
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel('MCHA 候选投标得分（越靠右越优）')
    ax.set_ylabel('轮次 / UAV / 目标')
    fig.suptitle(title, fontsize=13, y=0.93)
    ax.set_title(summary_text, fontsize=9.2, color='#2f5f8f', pad=6)
    ax.set_xlim(x_min, x_max)
    ax.grid(True, axis='x', alpha=0.18, linewidth=0.8)
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
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    if output_path is not None:
        ensure_output_dir(output_path)
        fig.savefig(output_path, dpi=180, bbox_inches='tight')

    return fig, ax


def plot_mcha_open_demand_repair(
    initial_remaining_demand: dict[int, int],
    bid_round_logs: list[BidRoundLog],
    title: str,
    output_path: Optional[str] = None,
):
    """绘制开放目标剩余需求随 MCHA 轮次被修复的过程。"""
    target_ids = sorted({
        target_id
        for target_id, demand in initial_remaining_demand.items()
        if demand > 0
    } | {
        bid.target_id
        for round_log in bid_round_logs
        for bid in round_log.accepted_bids
    } | {
        target_id
        for round_log in bid_round_logs
        for target_id in round_log.active_targets
    })

    if not target_ids:
        fig, ax = plt.subplots(figsize=(8.6, 3.6))
        ax.text(
            0.5,
            0.5,
            '本事件没有需要修复的开放任务需求',
            transform=ax.transAxes,
            ha='center',
            va='center',
            fontsize=11,
            color='#555555',
        )
        ax.set_axis_off()
        ax.set_title(title)
        if output_path is not None:
            ensure_output_dir(output_path)
            fig.savefig(output_path, dpi=180, bbox_inches='tight')
        return fig, ax

    demand_snapshots: list[dict[int, int]] = [dict(initial_remaining_demand)]
    current_remaining = dict(initial_remaining_demand)
    repaired_per_round: list[int] = []

    for round_log in bid_round_logs:
        repaired_count = 0
        for bid in round_log.accepted_bids:
            before = current_remaining.get(bid.target_id, 0)
            if before > 0:
                current_remaining[bid.target_id] = before - 1
                repaired_count += 1
        repaired_per_round.append(repaired_count)
        demand_snapshots.append(dict(current_remaining))

    demand_matrix = np.array([
        [snapshot.get(target_id, 0) for snapshot in demand_snapshots]
        for target_id in target_ids
    ], dtype=float)
    total_remaining = demand_matrix.sum(axis=0)
    initial_total = int(total_remaining[0]) if len(total_remaining) > 0 else 0
    final_total = int(total_remaining[-1]) if len(total_remaining) > 0 else 0
    repaired_total = initial_total - final_total
    repair_rate = 0.0 if initial_total == 0 else repaired_total / initial_total * 100.0

    fig_height = max(5.2, 0.32 * len(target_ids) + 2.7)
    fig, (ax_heatmap, ax_curve) = plt.subplots(
        2,
        1,
        figsize=(10.6, fig_height),
        gridspec_kw={'height_ratios': [max(2.4, 0.26 * len(target_ids)), 1.35], 'hspace': 0.2},
        constrained_layout=True,
    )

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        'demand_repair',
        ['#f7f9fb', '#d8e6f2', '#f2b56b', '#c75146'],
    )
    max_demand = max(float(np.max(demand_matrix)), 1.0)
    image = ax_heatmap.imshow(
        demand_matrix,
        aspect='auto',
        cmap=cmap,
        vmin=0,
        vmax=max_demand,
        zorder=1,
    )

    for row_idx, target_id in enumerate(target_ids):
        for col_idx, value in enumerate(demand_matrix[row_idx]):
            text_color = '#333333' if value <= max_demand * 0.55 else 'white'
            ax_heatmap.text(
                col_idx,
                row_idx,
                str(int(value)),
                ha='center',
                va='center',
                fontsize=8.4,
                color=text_color,
                weight='bold' if value > 0 else 'normal',
                zorder=3,
            )

    round_labels = ['R0\n事件后'] + [f'R{round_log.iteration}' for round_log in bid_round_logs]
    ax_heatmap.set_xticks(np.arange(len(round_labels)))
    ax_heatmap.set_xticklabels(round_labels)
    ax_heatmap.set_yticks(np.arange(len(target_ids)))
    ax_heatmap.set_yticklabels([f'T{target_id}' for target_id in target_ids])
    ax_heatmap.set_ylabel('开放目标')
    ax_heatmap.set_title(
        f'{title}\n'
        f'初始缺口 {initial_total} | 已修复 {repaired_total} | 剩余 {final_total} | 修复率 {repair_rate:.1f}%',
        fontsize=11.4,
        color='#222222',
        pad=10,
    )
    ax_heatmap.tick_params(axis='both', length=0)
    ax_heatmap.set_xticks(np.arange(-0.5, len(round_labels), 1), minor=True)
    ax_heatmap.set_yticks(np.arange(-0.5, len(target_ids), 1), minor=True)
    ax_heatmap.grid(which='minor', color='white', linewidth=1.1)
    ax_heatmap.tick_params(which='minor', bottom=False, left=False)
    for spine in ax_heatmap.spines.values():
        spine.set_visible(False)

    colorbar = fig.colorbar(image, ax=ax_heatmap, fraction=0.026, pad=0.018)
    colorbar.set_label('剩余需求')
    colorbar.outline.set_alpha(0.35)

    x = np.arange(len(round_labels))
    ax_curve.plot(
        x,
        total_remaining,
        color='#2f5f8f',
        linewidth=2.0,
        marker='o',
        markersize=5.8,
        markerfacecolor='white',
        markeredgewidth=1.7,
        zorder=4,
    )
    ax_curve.fill_between(
        x,
        total_remaining,
        color='#d8e6f2',
        alpha=0.55,
        zorder=1,
    )
    for idx, value in enumerate(total_remaining):
        ax_curve.text(
            idx,
            value + max(initial_total, 1) * 0.035,
            str(int(value)),
            ha='center',
            va='bottom',
            fontsize=8.4,
            color='#2f5f8f',
        )
    for idx, repaired_count in enumerate(repaired_per_round, start=1):
        if repaired_count <= 0:
            continue
        ax_curve.text(
            idx - 0.5,
            max(total_remaining[idx], 0) + max(initial_total, 1) * 0.1,
            f'-{repaired_count}',
            ha='center',
            va='bottom',
            fontsize=8.4,
            color='#2f5f8f',
            bbox={'boxstyle': 'round,pad=0.2', 'facecolor': 'white', 'edgecolor': '#d8e6f2', 'alpha': 0.95},
        )

    ax_curve.set_xticks(x)
    ax_curve.set_xticklabels(round_labels)
    ax_curve.set_ylabel('总剩余需求')
    ax_curve.set_xlabel('MCHA 重分配轮次')
    ax_curve.set_ylim(0, max(initial_total, 1) * 1.18)
    ax_curve.grid(True, axis='y', alpha=0.18, linewidth=0.8)
    ax_curve.set_axisbelow(True)
    ax_curve.spines['top'].set_visible(False)
    ax_curve.spines['right'].set_visible(False)
    ax_curve.spines['left'].set_alpha(0.6)
    ax_curve.spines['bottom'].set_alpha(0.6)

    if output_path is not None:
        ensure_output_dir(output_path)
        fig.savefig(output_path, dpi=180, bbox_inches='tight')

    return fig, (ax_heatmap, ax_curve)


def collect_reallocation_cost_metrics(
    battlefield: Battlefield,
    plan: AssignmentPlan,
    weights: dict,
) -> dict[str, float | int]:
    """汇总任务序列方案的重分配代价指标。"""
    total_distance = 0.0
    threat_cost = 0.0
    explicit_time_window_penalty = 0.0
    target_arrival_times: dict[int, list[float]] = {}
    assigned_target_ids: set[int] = set()

    for sequence in plan.uav_task_sequences.values():
        uav = battlefield.get_uav(sequence.uav_id)
        evaluated = evaluate_uav_task_sequence(
            battlefield,
            sequence,
            alpha=weights['alpha'],
        )
        total_distance += evaluated.total_distance
        explicit_time_window_penalty += evaluated.time_window_penalty

        for evaluated_task in evaluated.evaluated_sequence.tasks:
            target_arrival_times.setdefault(evaluated_task.target_id, []).append(
                evaluated_task.planned_arrival_time
            )

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

    sync_window = weights.get('sync_window', 0.05)
    cooperative_time_window_penalty = 0.0
    max_sync_gap = 0.0
    sync_violation_count = 0
    cooperative_target_count = 0
    for arrivals in target_arrival_times.values():
        if len(arrivals) < 2:
            continue
        cooperative_target_count += 1
        gap = max(arrivals) - min(arrivals)
        max_sync_gap = max(max_sync_gap, gap)
        if gap <= sync_window + 1e-12:
            continue
        sync_violation_count += 1
        sync_time = float(np.mean(arrivals))
        cooperative_time_window_penalty += float(
            sum(weights['alpha'] * (arrival - sync_time) ** 2 for arrival in arrivals)
        )

    time_window_penalty = explicit_time_window_penalty + cooperative_time_window_penalty
    target_value = float(sum(battlefield.get_target(target_id).value for target_id in assigned_target_ids))
    weighted_planning_cost = (
        weights['w1'] * total_distance
        + weights['w2'] * threat_cost
        + weights['w3'] * time_window_penalty
        - weights['w4'] * target_value
    )

    task_counts = [sequence.task_count() for sequence in plan.uav_task_sequences.values()]
    active_uav_count = int(sum(1 for count in task_counts if count > 0))
    assigned_task_count = int(sum(task_counts))

    target_satisfied_count = 0
    unmet_demand_count = 0
    for target in battlefield.targets:
        assigned_count = len(set(plan.target_assignees.get(target.id, [])))
        if assigned_count >= target.required_uavs:
            target_satisfied_count += 1
        unmet_demand_count += max(0, target.required_uavs - assigned_count)

    target_count = len(battlefield.targets)
    sync_satisfied_rate = (
        1.0 - sync_violation_count / cooperative_target_count
        if cooperative_target_count else 1.0
    )

    return {
        'total_distance': float(total_distance),
        'threat_cost': float(threat_cost),
        'explicit_time_window_penalty': float(explicit_time_window_penalty),
        'cooperative_time_window_penalty': float(cooperative_time_window_penalty),
        'time_window_penalty': float(time_window_penalty),
        'target_value': float(target_value),
        'weighted_planning_cost': float(weighted_planning_cost),
        'assigned_task_count': assigned_task_count,
        'active_uav_count': active_uav_count,
        'max_task_chain_length': max(task_counts, default=0),
        'target_satisfied_count': target_satisfied_count,
        'target_satisfaction_rate': target_satisfied_count / target_count if target_count else 0.0,
        'unmet_demand_count': int(unmet_demand_count),
        'cooperative_target_count': cooperative_target_count,
        'sync_violation_count': sync_violation_count,
        'sync_satisfied_rate': float(sync_satisfied_rate),
        'max_sync_gap': float(max_sync_gap),
    }


REALLOCATION_COST_METRIC_LABELS = [
    ('total_distance', '总航程代价'),
    ('threat_cost', '威胁代价'),
    ('explicit_time_window_penalty', '显式时间窗惩罚'),
    ('cooperative_time_window_penalty', '协同到达时间窗惩罚'),
    ('time_window_penalty', '总时间窗惩罚'),
    ('target_value', '目标收益'),
    ('weighted_planning_cost', '加权规划代价（越低越优）'),
    ('assigned_task_count', '已分配任务槽数量'),
    ('active_uav_count', '参与任务UAV数量'),
    ('max_task_chain_length', '最大任务链长度'),
    ('target_satisfied_count', '满足需求目标数量'),
    ('target_satisfaction_rate', '目标需求满足率'),
    ('unmet_demand_count', '未满足任务需求数量'),
    ('cooperative_target_count', '协同目标数量'),
    ('sync_violation_count', '同步窗口违反目标数量'),
    ('sync_satisfied_rate', '同步窗口满足率'),
    ('max_sync_gap', '最大协同到达时间差'),
]


def _format_csv_metric_value(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return f'{value:.6f}'


def write_reallocation_cost_change_csv(
    battlefield_before: Battlefield,
    plan_before: AssignmentPlan,
    battlefield_after: Battlefield,
    impacted_plan: AssignmentPlan,
    reallocated_plan: AssignmentPlan,
    weights: dict,
    output_path: str,
) -> list[dict[str, str]]:
    """写出重分配前后代价变化 CSV。"""
    before_metrics = collect_reallocation_cost_metrics(battlefield_before, plan_before, weights)
    impacted_metrics = collect_reallocation_cost_metrics(battlefield_after, impacted_plan, weights)
    reallocated_metrics = collect_reallocation_cost_metrics(battlefield_after, reallocated_plan, weights)

    rows: list[dict[str, str]] = []
    for key, label in REALLOCATION_COST_METRIC_LABELS:
        before_value = before_metrics[key]
        impacted_value = impacted_metrics[key]
        reallocated_value = reallocated_metrics[key]
        delta = float(reallocated_value) - float(impacted_value)
        change_rate = delta / abs(float(impacted_value)) if abs(float(impacted_value)) > 1e-12 else 0.0

        rows.append({
            '指标': label,
            '事件前预分配': _format_csv_metric_value(before_value),
            '事件后待修复': _format_csv_metric_value(impacted_value),
            '重分配后': _format_csv_metric_value(reallocated_value),
            '重分配变化量': _format_csv_metric_value(delta),
            '重分配变化率': f'{change_rate * 100.0:.2f}%',
        })

    ensure_output_dir(output_path)
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as csv_file:
        fieldnames = ['指标', '事件前预分配', '事件后待修复', '重分配后', '重分配变化量', '重分配变化率']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows


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
