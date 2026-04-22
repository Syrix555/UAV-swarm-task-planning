import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from src.core.models import Threat

Point = Tuple[float, float]
ARC_MIN_DEFLECTION_RAD = math.radians(5.0)
DEFAULT_PREFERRED_RADIUS_SCALE = 1.5


@dataclass(frozen=True)
class CornerPlan:
    index: int
    radius: float
    tangent_distance: float
    deflection: float
    turn_sign: float
    reason: str
    active: bool


@dataclass(frozen=True)
class KinematicPathResult:
    path_points: List[Point]
    mode: str
    reason: str | None
    applied_radii: List[float]


def point_in_inflated_threat(x: float, y: float, threats: Sequence[Threat], safety_margin: float) -> bool:
    for threat in threats:
        radius = threat.radius + safety_margin
        if (x - threat.x) ** 2 + (y - threat.y) ** 2 < radius ** 2:
            return True
    return False


def segment_intersects_circle(start: Point, end: Point, center: Point, radius: float) -> bool:
    x1, y1 = start
    x2, y2 = end
    cx, cy = center

    dx = x2 - x1
    dy = y2 - y1
    if dx == 0.0 and dy == 0.0:
        return (x1 - cx) ** 2 + (y1 - cy) ** 2 < radius ** 2

    t = ((cx - x1) * dx + (cy - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    return (closest_x - cx) ** 2 + (closest_y - cy) ** 2 < radius ** 2


def segment_intersects_inflated_threat(
    start: Point,
    end: Point,
    threat: Threat,
    safety_margin: float,
) -> bool:
    return segment_intersects_circle(
        start,
        end,
        (threat.x, threat.y),
        threat.radius + safety_margin,
    )


def path_intersects_any_threat(
    path_points: Sequence[Point],
    threats: Sequence[Threat],
    safety_margin: float,
) -> bool:
    if len(path_points) < 2:
        return False

    for x, y in path_points:
        if point_in_inflated_threat(x, y, threats, safety_margin):
            return True

    for start, end in zip(path_points[:-1], path_points[1:]):
        for threat in threats:
            if segment_intersects_inflated_threat(start, end, threat, safety_margin):
                return True
    return False


def path_length(path_points: Sequence[Point]) -> float:
    if len(path_points) < 2:
        return 0.0
    total = 0.0
    for start, end in zip(path_points[:-1], path_points[1:]):
        total += math.hypot(end[0] - start[0], end[1] - start[1])
    return total


def adaptive_sample_polyline(path_points: Sequence[Point], sample_step: float) -> List[Point]:
    if len(path_points) < 2:
        return list(path_points)

    sampled: List[Point] = [path_points[0]]
    for start, end in zip(path_points[:-1], path_points[1:]):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        segment_length = math.hypot(dx, dy)
        if segment_length == 0.0:
            continue
        steps = max(1, int(math.ceil(segment_length / sample_step)))
        for k in range(1, steps + 1):
            t = k / steps
            sampled.append((start[0] + t * dx, start[1] + t * dy))
    return sampled


def _deduplicate_points(path_points: Sequence[Point]) -> List[Point]:
    if not path_points:
        return []

    deduplicated = [path_points[0]]
    for point in path_points[1:]:
        if point != deduplicated[-1]:
            deduplicated.append(point)
    return deduplicated


def _normalize(vector: Point) -> Point | None:
    length = math.hypot(vector[0], vector[1])
    if length == 0.0:
        return None
    return (vector[0] / length, vector[1] / length)


def _left_normal(vector: Point) -> Point:
    return (-vector[1], vector[0])


def _right_normal(vector: Point) -> Point:
    return (vector[1], -vector[0])


def _signed_turn(v_in: Point, v_out: Point) -> float:
    return v_in[0] * v_out[1] - v_in[1] * v_out[0]


def _sample_arc(
    center: Point,
    start_point: Point,
    end_point: Point,
    radius: float,
    turn_sign: float,
    sample_step: float,
) -> List[Point]:
    start_angle = math.atan2(start_point[1] - center[1], start_point[0] - center[0])
    end_angle = math.atan2(end_point[1] - center[1], end_point[0] - center[0])

    if turn_sign > 0.0:
        while end_angle <= start_angle:
            end_angle += 2.0 * math.pi
    else:
        while end_angle >= start_angle:
            end_angle -= 2.0 * math.pi

    sweep = end_angle - start_angle
    arc_length = abs(sweep) * radius
    steps = max(2, int(math.ceil(arc_length / sample_step)) + 1)

    arc_points: List[Point] = []
    for index in range(steps):
        ratio = index / (steps - 1)
        angle = start_angle + sweep * ratio
        arc_points.append((center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle)))
    return arc_points


def _build_corner_geometry(prev_point: Point, corner_point: Point, next_point: Point) -> dict | None:
    incoming = _normalize((corner_point[0] - prev_point[0], corner_point[1] - prev_point[1]))
    outgoing = _normalize((next_point[0] - corner_point[0], next_point[1] - corner_point[1]))
    if incoming is None or outgoing is None:
        return None

    dot_value = max(-1.0, min(1.0, incoming[0] * outgoing[0] + incoming[1] * outgoing[1]))
    deflection = math.acos(dot_value)
    if deflection < ARC_MIN_DEFLECTION_RAD or abs(math.pi - deflection) < 1e-6:
        return None

    tan_half = math.tan(deflection / 2.0)
    if tan_half <= 0.0:
        return None

    turn_sign = _signed_turn(incoming, outgoing)
    if abs(turn_sign) < 1e-9:
        return None

    in_length = math.hypot(corner_point[0] - prev_point[0], corner_point[1] - prev_point[1])
    out_length = math.hypot(next_point[0] - corner_point[0], next_point[1] - corner_point[1])
    max_radius = min(in_length, out_length) / tan_half
    return {
        'incoming': incoming,
        'outgoing': outgoing,
        'deflection': deflection,
        'tan_half': tan_half,
        'turn_sign': turn_sign,
        'in_length': in_length,
        'out_length': out_length,
        'max_radius': max_radius,
    }


def _build_corner_arc(
    prev_point: Point,
    corner_point: Point,
    next_point: Point,
    radius: float,
    sample_step: float,
) -> List[Point] | None:
    geometry = _build_corner_geometry(prev_point, corner_point, next_point)
    if geometry is None or radius <= 0.0 or sample_step <= 0.0:
        return None

    tangent_distance = radius * geometry['tan_half']
    if tangent_distance <= 0.0:
        return None
    if tangent_distance >= geometry['in_length'] or tangent_distance >= geometry['out_length']:
        return None

    tangent_start = (
        corner_point[0] - geometry['incoming'][0] * tangent_distance,
        corner_point[1] - geometry['incoming'][1] * tangent_distance,
    )
    tangent_end = (
        corner_point[0] + geometry['outgoing'][0] * tangent_distance,
        corner_point[1] + geometry['outgoing'][1] * tangent_distance,
    )

    normal = _left_normal(geometry['incoming']) if geometry['turn_sign'] > 0.0 else _right_normal(geometry['incoming'])
    center = (
        tangent_start[0] + normal[0] * radius,
        tangent_start[1] + normal[1] * radius,
    )
    arc_points = _sample_arc(center, tangent_start, tangent_end, radius, geometry['turn_sign'], sample_step)
    if not arc_points:
        return None
    arc_points[0] = tangent_start
    arc_points[-1] = tangent_end
    return arc_points


def _initial_corner_plan(
    path_points: Sequence[Point],
    min_turn_radius: float,
    preferred_radius_scale: float,
) -> List[CornerPlan]:
    plans: List[CornerPlan] = []
    preferred_radius = min_turn_radius * preferred_radius_scale

    for index in range(1, len(path_points) - 1):
        geometry = _build_corner_geometry(path_points[index - 1], path_points[index], path_points[index + 1])
        if geometry is None:
            plans.append(CornerPlan(index, 0.0, 0.0, 0.0, 0.0, 'collinear', False))
            continue

        if geometry['max_radius'] < min_turn_radius:
            plans.append(
                CornerPlan(
                    index,
                    0.0,
                    0.0,
                    geometry['deflection'],
                    geometry['turn_sign'],
                    'turn_radius_segment_too_short',
                    False,
                )
            )
            continue

        radius = min(geometry['max_radius'], preferred_radius)
        tangent_distance = radius * geometry['tan_half']
        plans.append(
            CornerPlan(
                index,
                radius,
                tangent_distance,
                geometry['deflection'],
                geometry['turn_sign'],
                'adaptive',
                True,
            )
        )
    return plans


def _resolve_adjacent_corner_windows(
    path_points: Sequence[Point],
    plans: Sequence[CornerPlan],
    min_turn_radius: float,
) -> tuple[List[CornerPlan], bool]:
    resolved = list(plans)
    used_windowed = False

    for plan_index in range(len(resolved) - 1):
        left = resolved[plan_index]
        right = resolved[plan_index + 1]
        if not left.active or not right.active:
            continue
        if right.index != left.index + 1:
            continue

        shared_segment_length = math.hypot(
            path_points[right.index][0] - path_points[left.index][0],
            path_points[right.index][1] - path_points[left.index][1],
        )
        if left.tangent_distance + right.tangent_distance <= shared_segment_length:
            continue

        used_windowed = True
        left_extra = max(0.0, left.radius - min_turn_radius)
        right_extra = max(0.0, right.radius - min_turn_radius)
        left_tan_half = left.tangent_distance / left.radius if left.radius > 0.0 else 0.0
        right_tan_half = right.tangent_distance / right.radius if right.radius > 0.0 else 0.0
        overflow = left.tangent_distance + right.tangent_distance - shared_segment_length
        reducible = left_extra * left_tan_half + right_extra * right_tan_half

        if reducible > 0.0 and overflow <= reducible + 1e-9:
            shrink_ratio = overflow / reducible
            new_left_radius = left.radius - left_extra * shrink_ratio
            new_right_radius = right.radius - right_extra * shrink_ratio
            resolved[plan_index] = CornerPlan(
                left.index,
                new_left_radius,
                new_left_radius * left_tan_half,
                left.deflection,
                left.turn_sign,
                'windowed',
                True,
            )
            resolved[plan_index + 1] = CornerPlan(
                right.index,
                new_right_radius,
                new_right_radius * right_tan_half,
                right.deflection,
                right.turn_sign,
                'windowed',
                True,
            )
            continue

        disable_left = left.deflection < right.deflection
        if abs(left.deflection - right.deflection) < 1e-6:
            disable_left = False

        if disable_left:
            resolved[plan_index] = CornerPlan(left.index, 0.0, 0.0, left.deflection, left.turn_sign, 'turn_radius_window_overlap', False)
        else:
            resolved[plan_index + 1] = CornerPlan(right.index, 0.0, 0.0, right.deflection, right.turn_sign, 'turn_radius_window_overlap', False)

    return resolved, used_windowed


def build_corner_fillet(
    prev_point: Point,
    corner_point: Point,
    next_point: Point,
    radius: float,
    sample_step: float,
) -> List[Point] | None:
    return _build_corner_arc(prev_point, corner_point, next_point, radius, sample_step)


def generate_kinematic_path_details(
    path_points: Sequence[Point],
    min_turn_radius: float,
    sample_step: float,
    preferred_radius_scale: float = DEFAULT_PREFERRED_RADIUS_SCALE,
) -> KinematicPathResult:
    deduplicated = _deduplicate_points(path_points)
    if len(deduplicated) < 3 or min_turn_radius <= 0.0:
        sampled = adaptive_sample_polyline(deduplicated, sample_step) if sample_step > 0.0 else list(deduplicated)
        return KinematicPathResult(sampled, 'disabled', None, [])

    initial_plans = _initial_corner_plan(deduplicated, min_turn_radius, preferred_radius_scale)
    resolved_plans, used_windowed = _resolve_adjacent_corner_windows(deduplicated, initial_plans, min_turn_radius)
    plan_by_index = {plan.index: plan for plan in resolved_plans}

    merged: List[Point] = [deduplicated[0]]
    applied_radii: List[float] = []

    for index in range(1, len(deduplicated) - 1):
        plan = plan_by_index[index]
        corner_point = deduplicated[index]
        if not plan.active:
            if corner_point != merged[-1]:
                merged.append(corner_point)
            continue

        arc_points = _build_corner_arc(deduplicated[index - 1], corner_point, deduplicated[index + 1], plan.radius, sample_step)
        if arc_points is None:
            if corner_point != merged[-1]:
                merged.append(corner_point)
            continue

        applied_radii.append(plan.radius)
        if arc_points[0] != merged[-1]:
            merged.append(arc_points[0])
        for point in arc_points[1:]:
            if point != merged[-1]:
                merged.append(point)

    if deduplicated[-1] != merged[-1]:
        merged.append(deduplicated[-1])

    sampled = adaptive_sample_polyline(merged, sample_step)
    if used_windowed:
        mode = 'windowed'
    elif applied_radii:
        mode = 'adaptive'
    else:
        mode = 'fallback'

    failure_reasons = [plan.reason for plan in resolved_plans if not plan.active and plan.reason not in ('collinear',)]
    reason = failure_reasons[0] if failure_reasons else None
    return KinematicPathResult(sampled, mode, reason, applied_radii)


def generate_kinematic_path(
    path_points: Sequence[Point],
    min_turn_radius: float,
    sample_step: float,
) -> List[Point]:
    return generate_kinematic_path_details(path_points, min_turn_radius, sample_step).path_points


def estimate_min_turn_radius(path_points: Sequence[Point]) -> float:
    if len(path_points) < 3:
        return float('inf')

    radii: List[float] = []
    for p0, p1, p2 in zip(path_points[:-2], path_points[1:-1], path_points[2:]):
        a = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        b = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        c = math.hypot(p2[0] - p0[0], p2[1] - p0[1])
        if a == 0.0 or b == 0.0 or c == 0.0:
            continue

        area_twice = abs(
            p0[0] * (p1[1] - p2[1])
            + p1[0] * (p2[1] - p0[1])
            + p2[0] * (p0[1] - p1[1])
        )
        if area_twice == 0.0:
            continue

        radius = (a * b * c) / (2.0 * area_twice)
        radii.append(radius)

    if not radii:
        return float('inf')
    return min(radii)
