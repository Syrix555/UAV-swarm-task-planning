from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.core.models import AssignmentPlan, Battlefield
from src.route_planning.astar import astar_search
from src.route_planning.geometry import (
    adaptive_sample_polyline,
    estimate_min_turn_radius,
    generate_kinematic_path_details,
    path_length,
)
from src.route_planning.grid import build_grid_map
from src.route_planning.simplify import los_simplify
from src.route_planning.smoothing import BSplineSmoothingError, smooth_bspline
from src.route_planning.validation import validate_path


@dataclass
class PathPlanningResult:
    success: bool
    original_path: List[Tuple[float, float]]
    los_path: List[Tuple[float, float]]
    kinematic_path: List[Tuple[float, float]]
    smoothed_path: List[Tuple[float, float]]
    final_path: List[Tuple[float, float]]
    used_kinematic_constraint: bool
    used_smoothing: bool
    kinematic_mode: str
    fallback_reason: Optional[str]
    failure_reason: Optional[str]
    path_length: float
    estimated_min_turn_radius: float


@dataclass
class RouteSegment:
    """任务链中的一段航迹规划结果。"""

    uav_id: int
    segment_order: int
    start_kind: str
    start_id: int
    end_target_id: int
    start_xy: Tuple[float, float]
    end_xy: Tuple[float, float]
    result: PathPlanningResult

    @property
    def success(self) -> bool:
        return self.result.success

    @property
    def failure_reason(self) -> Optional[str]:
        return self.result.failure_reason

    @property
    def final_path(self) -> List[Tuple[float, float]]:
        return self.result.final_path

    @property
    def path_length(self) -> float:
        return self.result.path_length if self.success else 0.0


@dataclass
class UavRoutePlan:
    """单架无人机的任务链航迹规划结果。"""

    uav_id: int
    target_ids: List[int]
    segments: List[RouteSegment]

    @property
    def active(self) -> bool:
        return bool(self.target_ids)

    @property
    def success(self) -> bool:
        return all(segment.success for segment in self.segments)

    @property
    def failed_segments(self) -> List[RouteSegment]:
        return [segment for segment in self.segments if not segment.success]

    @property
    def total_path_length(self) -> float:
        return sum(segment.path_length for segment in self.segments)

    @property
    def full_path(self) -> List[Tuple[float, float]]:
        full_path: List[Tuple[float, float]] = []
        for segment in self.segments:
            if not segment.success or not segment.final_path:
                continue
            if full_path and full_path[-1] == segment.final_path[0]:
                full_path.extend(segment.final_path[1:])
            else:
                full_path.extend(segment.final_path)
        return full_path


@dataclass
class AssignmentRoutePlan:
    """一个 AssignmentPlan 对应的多无人机航迹规划结果。"""

    uav_route_plans: Dict[int, UavRoutePlan]
    source: str = 'assignment_plan'

    @property
    def success(self) -> bool:
        return all(route.success for route in self.uav_route_plans.values())

    @property
    def active_uav_count(self) -> int:
        return sum(1 for route in self.uav_route_plans.values() if route.active)

    @property
    def segment_count(self) -> int:
        return sum(len(route.segments) for route in self.uav_route_plans.values())

    @property
    def failed_segments(self) -> List[RouteSegment]:
        failed: List[RouteSegment] = []
        for route in self.uav_route_plans.values():
            failed.extend(route.failed_segments)
        return failed

    @property
    def failed_uavs(self) -> List[int]:
        return [
            route.uav_id
            for route in self.uav_route_plans.values()
            if route.failed_segments
        ]

    @property
    def total_path_length(self) -> float:
        return sum(route.total_path_length for route in self.uav_route_plans.values())


DEFAULT_PARAMS: Dict[str, float | int | bool] = {
    'grid_resolution': 1.0,
    'safety_margin': 2.0,
    'allow_diagonal': True,
    'bspline_degree': 3,
    'smoothing_factor': 1.0,
    'sample_step': 0.5,
    'min_turn_radius': 0.0,
    'enable_kinematic_path': True,
    'kinematic_sample_step': 0.25,
    'kinematic_preferred_radius_scale': 1.5,
    'enable_bspline_after_kinematic': True,
}


def _merge_params(params: dict | None) -> dict:
    merged = dict(DEFAULT_PARAMS)
    if params:
        merged.update(params)
    return merged


def _normalize_reason(reason: Optional[str]) -> Optional[str]:
    if reason in (None, 'ok'):
        return None
    return reason


def plan_path_for_uav(
    battlefield: Battlefield,
    uav_id: int,
    target_id: int,
    params: dict | None = None,
) -> PathPlanningResult:
    """规划单架无人机从初始位置到指定目标的路径。"""
    uav = battlefield.get_uav(uav_id)
    target = battlefield.get_target(target_id)
    return plan_path_between_points(
        battlefield,
        uav_id,
        (uav.x, uav.y),
        (target.x, target.y),
        params=params,
    )


def plan_routes_for_assignment_plan(
    battlefield: Battlefield,
    assignment_plan: AssignmentPlan,
    params: dict | None = None,
    source: str = 'assignment_plan',
) -> AssignmentRoutePlan:
    """将 AssignmentPlan 中的 UAV 任务序列逐段转换为航迹规划结果。"""
    uav_route_plans: Dict[int, UavRoutePlan] = {}

    for uav in battlefield.uavs:
        sequence = assignment_plan.uav_task_sequences.get(uav.id)
        target_ids = sequence.target_ids() if sequence is not None else []
        segments: List[RouteSegment] = []

        current_kind = 'uav'
        current_id = uav.id
        current_xy = (uav.x, uav.y)

        for segment_order, target_id in enumerate(target_ids):
            target = battlefield.get_target(target_id)
            goal_xy = (target.x, target.y)
            result = plan_path_between_points(
                battlefield,
                uav.id,
                current_xy,
                goal_xy,
                params=params,
            )
            segment = RouteSegment(
                uav_id=uav.id,
                segment_order=segment_order,
                start_kind=current_kind,
                start_id=current_id,
                end_target_id=target_id,
                start_xy=current_xy,
                end_xy=goal_xy,
                result=result,
            )
            segments.append(segment)

            if not result.success:
                break

            current_kind = 'target'
            current_id = target_id
            current_xy = goal_xy

        uav_route_plans[uav.id] = UavRoutePlan(
            uav_id=uav.id,
            target_ids=target_ids,
            segments=segments,
        )

    return AssignmentRoutePlan(
        uav_route_plans=uav_route_plans,
        source=source,
    )


def plan_path_between_points(
    battlefield: Battlefield,
    uav_id: int,
    start_xy: Tuple[float, float],
    goal_xy: Tuple[float, float],
    params: dict | None = None,
) -> PathPlanningResult:
    """规划单架无人机在任意两点之间的路径，用于后续任务链逐段规划。"""
    merged_params = _merge_params(params)
    uav = battlefield.get_uav(uav_id)

    grid_map = build_grid_map(
        battlefield,
        float(merged_params['grid_resolution']),
        float(merged_params['safety_margin']),
    )
    original_path = astar_search(
        grid_map,
        start_xy,
        goal_xy,
        bool(merged_params['allow_diagonal']),
    )
    if not original_path:
        return PathPlanningResult(
            success=False,
            original_path=[],
            los_path=[],
            kinematic_path=[],
            smoothed_path=[],
            final_path=[],
            used_kinematic_constraint=False,
            used_smoothing=False,
            kinematic_mode='disabled',
            fallback_reason=None,
            failure_reason='no_path_found',
            path_length=0.0,
            estimated_min_turn_radius=float('inf'),
        )

    is_valid, reason = validate_path(
        original_path,
        battlefield,
        uav,
        float(merged_params['safety_margin']),
        float(merged_params['min_turn_radius']),
    )
    if not is_valid and reason == 'out_of_range':
        return PathPlanningResult(
            success=False,
            original_path=original_path,
            los_path=[],
            kinematic_path=[],
            smoothed_path=[],
            final_path=[],
            used_kinematic_constraint=False,
            used_smoothing=False,
            kinematic_mode='disabled',
            fallback_reason=None,
            failure_reason='out_of_range',
            path_length=path_length(original_path),
            estimated_min_turn_radius=estimate_min_turn_radius(original_path),
        )

    los_path = los_simplify(original_path, battlefield.threats, float(merged_params['safety_margin']))
    kinematic_input_path = los_path
    kinematic_path = adaptive_sample_polyline(los_path, float(merged_params['sample_step']))
    used_kinematic_constraint = False
    kinematic_reason: Optional[str] = None
    kinematic_mode = 'disabled'

    if bool(merged_params.get('enable_kinematic_path', True)) and float(merged_params['min_turn_radius']) > 0.0:
        kinematic_result = generate_kinematic_path_details(
            los_path,
            float(merged_params['min_turn_radius']),
            float(merged_params.get('kinematic_sample_step', merged_params['sample_step'])),
            float(merged_params.get('kinematic_preferred_radius_scale', 1.5)),
        )
        candidate_kinematic_path = kinematic_result.path_points
        kinematic_mode = kinematic_result.mode
        kinematic_valid, kinematic_reason = validate_path(
            candidate_kinematic_path,
            battlefield,
            uav,
            float(merged_params['safety_margin']),
            float(merged_params['min_turn_radius']),
        )
        if kinematic_mode == 'disabled':
            kinematic_path = candidate_kinematic_path
            kinematic_reason = None
        elif kinematic_valid and kinematic_mode != 'fallback':
            kinematic_path = candidate_kinematic_path
            kinematic_input_path = candidate_kinematic_path
            used_kinematic_constraint = True
        else:
            kinematic_reason = kinematic_result.reason or kinematic_reason
            if kinematic_valid and kinematic_mode == 'fallback':
                kinematic_path = candidate_kinematic_path
            elif kinematic_reason is not None:
                kinematic_reason = f'turn_radius_generation_failed: {kinematic_reason}'

    smoothed_path: List[Tuple[float, float]] = []
    smoothed_sampled: List[Tuple[float, float]] = []
    smoothing_error: Optional[str] = None
    if bool(merged_params.get('enable_bspline_after_kinematic', True)):
        try:
            smoothed_path = smooth_bspline(
                kinematic_input_path,
                int(merged_params['bspline_degree']),
                float(merged_params['smoothing_factor']),
                float(merged_params['sample_step']),
                float(merged_params.get('corner_angle_threshold_deg', 15.0)),
                int(merged_params.get('corner_window_points', 3)),
            )
            smoothed_sampled = adaptive_sample_polyline(smoothed_path, float(merged_params['sample_step']))
        except BSplineSmoothingError as exc:
            smoothing_error = str(exc)
            smoothed_path = []
            smoothed_sampled = []

    if smoothed_sampled:
        smoothed_valid, smoothed_reason = validate_path(
            smoothed_sampled,
            battlefield,
            uav,
            float(merged_params['safety_margin']),
            float(merged_params['min_turn_radius']),
        )
        if smoothed_valid:
            return PathPlanningResult(
                success=True,
                original_path=original_path,
                los_path=los_path,
                kinematic_path=kinematic_path,
                smoothed_path=smoothed_sampled,
                final_path=smoothed_sampled,
                used_kinematic_constraint=used_kinematic_constraint,
                used_smoothing=True,
                kinematic_mode=kinematic_mode,
                fallback_reason=_normalize_reason(kinematic_reason),
                failure_reason=None,
                path_length=path_length(smoothed_sampled),
                estimated_min_turn_radius=estimate_min_turn_radius(smoothed_sampled),
            )
    else:
        smoothed_reason = 'smoothing_failed'

    if smoothing_error:
        smoothed_reason = f'smoothing_failed: {smoothing_error}'

    kinematic_valid, validated_kinematic_reason = validate_path(
        kinematic_path,
        battlefield,
        uav,
        float(merged_params['safety_margin']),
        float(merged_params['min_turn_radius']),
    )
    if kinematic_valid:
        return PathPlanningResult(
            success=True,
            original_path=original_path,
            los_path=los_path,
            kinematic_path=kinematic_path,
            smoothed_path=smoothed_sampled,
            final_path=kinematic_path,
            used_kinematic_constraint=used_kinematic_constraint,
            used_smoothing=False,
            kinematic_mode=kinematic_mode,
            fallback_reason=_normalize_reason(smoothed_reason if smoothed_sampled or smoothing_error else kinematic_reason),
            failure_reason=None,
            path_length=path_length(kinematic_path),
            estimated_min_turn_radius=estimate_min_turn_radius(kinematic_path),
        )

    los_valid, los_reason = validate_path(
        los_path,
        battlefield,
        uav,
        float(merged_params['safety_margin']),
        float(merged_params['min_turn_radius']),
    )
    if los_valid:
        return PathPlanningResult(
            success=True,
            original_path=original_path,
            los_path=los_path,
            kinematic_path=kinematic_path,
            smoothed_path=smoothed_sampled,
            final_path=los_path,
            used_kinematic_constraint=False,
            used_smoothing=False,
            kinematic_mode=kinematic_mode,
            fallback_reason=_normalize_reason(validated_kinematic_reason if validated_kinematic_reason != 'ok' else smoothed_reason),
            failure_reason=None,
            path_length=path_length(los_path),
            estimated_min_turn_radius=estimate_min_turn_radius(los_path),
        )

    original_valid, original_reason = validate_path(
        original_path,
        battlefield,
        uav,
        float(merged_params['safety_margin']),
        float(merged_params['min_turn_radius']),
    )
    if original_valid:
        return PathPlanningResult(
            success=True,
            original_path=original_path,
            los_path=los_path,
            kinematic_path=kinematic_path,
            smoothed_path=smoothed_sampled,
            final_path=original_path,
            used_kinematic_constraint=False,
            used_smoothing=False,
            kinematic_mode=kinematic_mode,
            fallback_reason=_normalize_reason(los_reason),
            failure_reason=None,
            path_length=path_length(original_path),
            estimated_min_turn_radius=estimate_min_turn_radius(original_path),
        )

    failure_reason = original_reason
    if failure_reason == 'ok':
        failure_reason = los_reason if los_reason != 'ok' else smoothed_reason

    return PathPlanningResult(
        success=False,
        original_path=original_path,
        los_path=los_path,
        kinematic_path=kinematic_path,
        smoothed_path=smoothed_sampled,
        final_path=[],
        used_kinematic_constraint=False,
        used_smoothing=False,
        kinematic_mode=kinematic_mode,
        fallback_reason=_normalize_reason(smoothed_reason),
        failure_reason=failure_reason,
        path_length=0.0,
        estimated_min_turn_radius=float('inf'),
    )
