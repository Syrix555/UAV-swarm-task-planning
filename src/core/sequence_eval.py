from dataclasses import dataclass

from src.core.models import Battlefield, TaskNode, UavTaskSequence


@dataclass(frozen=True)
class TaskSequenceEvaluation:
    """单架无人机任务序列的统一评估结果。"""

    evaluated_sequence: UavTaskSequence
    total_distance: float
    total_travel_time: float
    completion_time: float
    arrival_times: list[float]
    time_window_penalty: float
    is_ammo_feasible: bool
    is_range_feasible: bool
    is_feasible: bool


def _time_window_penalty(arrival_time: float, window_start: float | None, window_end: float | None, alpha: float) -> float:
    if window_start is not None and arrival_time < window_start:
        return alpha * (window_start - arrival_time)
    if window_end is not None and arrival_time > window_end:
        return alpha * (arrival_time - window_end)
    return 0.0


def evaluate_uav_task_sequence(
    battlefield: Battlefield,
    sequence: UavTaskSequence,
    *,
    start_position: tuple[float, float] | None = None,
    start_time: float = 0.0,
    alpha: float = 1.0,
) -> TaskSequenceEvaluation:
    """评估单架无人机按顺序执行任务链时的累计距离、时序和约束可行性。"""
    uav = battlefield.get_uav(sequence.uav_id)
    current_x, current_y = start_position if start_position is not None else (uav.x, uav.y)
    current_time = start_time

    total_distance = 0.0
    arrival_times: list[float] = []
    evaluated_tasks: list[TaskNode] = []
    time_window_penalty = 0.0

    for order, task in enumerate(sequence.tasks):
        target = battlefield.get_target(task.target_id)
        segment_distance = battlefield.distance(current_x, current_y, target.x, target.y)
        travel_time = segment_distance / uav.speed
        arrival_time = current_time + travel_time

        time_window_penalty += _time_window_penalty(
            arrival_time,
            target.time_window_start,
            target.time_window_end,
            alpha,
        )

        evaluated_tasks.append(
            TaskNode(
                target_id=task.target_id,
                order=order,
                planned_arrival_time=arrival_time,
                planned_service_time=target.service_time,
                estimated_path_length=segment_distance,
            )
        )
        arrival_times.append(arrival_time)
        total_distance += segment_distance

        current_x, current_y = target.x, target.y
        current_time = arrival_time + target.service_time

    total_travel_time = total_distance / uav.speed if sequence.tasks else 0.0
    is_ammo_feasible = len(sequence.tasks) <= uav.ammo
    is_range_feasible = total_distance <= uav.range_left + 1e-9

    return TaskSequenceEvaluation(
        evaluated_sequence=UavTaskSequence(uav_id=sequence.uav_id, tasks=evaluated_tasks),
        total_distance=total_distance,
        total_travel_time=total_travel_time,
        completion_time=current_time,
        arrival_times=arrival_times,
        time_window_penalty=time_window_penalty,
        is_ammo_feasible=is_ammo_feasible,
        is_range_feasible=is_range_feasible,
        is_feasible=is_ammo_feasible and is_range_feasible,
    )
