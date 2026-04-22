from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.core.models import Battlefield
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
    merged_params = _merge_params(params)
    uav = battlefield.get_uav(uav_id)
    target = battlefield.get_target(target_id)

    grid_map = build_grid_map(
        battlefield,
        float(merged_params['grid_resolution']),
        float(merged_params['safety_margin']),
    )
    original_path = astar_search(
        grid_map,
        (uav.x, uav.y),
        (target.x, target.y),
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
