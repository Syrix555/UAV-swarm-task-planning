from typing import Sequence, Tuple

from src.core.models import Battlefield, UAV
from src.route_planning.geometry import estimate_min_turn_radius, path_intersects_any_threat, path_length


def validate_path(
    path_points: Sequence[Tuple[float, float]],
    battlefield: Battlefield,
    uav: UAV,
    safety_margin: float,
    min_turn_radius: float | None = None,
) -> tuple[bool, str]:
    if not path_points or len(path_points) < 2:
        return False, 'empty_path'

    if path_intersects_any_threat(path_points, battlefield.threats, safety_margin):
        return False, 'intersects_threat'

    total_length = path_length(path_points)
    if total_length > uav.range_left:
        return False, 'out_of_range'

    if min_turn_radius is not None and min_turn_radius > 0:
        estimated_radius = estimate_min_turn_radius(path_points)
        if estimated_radius < min_turn_radius:
            return False, 'turn_radius_violation'

    return True, 'ok'
