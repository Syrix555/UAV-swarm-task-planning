from typing import List, Sequence, Tuple

from src.core.models import Threat
from src.route_planning.geometry import path_intersects_any_threat

Point = Tuple[float, float]


def los_simplify(path_points: Sequence[Point], threats: Sequence[Threat], safety_margin: float) -> List[Point]:
    if len(path_points) <= 2:
        return list(path_points)

    simplified: List[Point] = [path_points[0]]
    anchor_idx = 0
    n = len(path_points)

    while anchor_idx < n - 1:
        next_idx = anchor_idx + 1
        furthest_visible = next_idx
        while next_idx < n:
            candidate_segment = [path_points[anchor_idx], path_points[next_idx]]
            if path_intersects_any_threat(candidate_segment, threats, safety_margin):
                break
            furthest_visible = next_idx
            next_idx += 1

        simplified.append(path_points[furthest_visible])
        anchor_idx = furthest_visible

    return simplified
