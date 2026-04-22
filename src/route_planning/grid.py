from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from src.core.models import Battlefield, Threat


@dataclass
class GridMap:
    width: int
    height: int
    resolution: float
    occupancy: np.ndarray
    map_size: Tuple[float, float]

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width

    def is_blocked(self, row: int, col: int) -> bool:
        return bool(self.occupancy[row, col])


def point_in_inflated_threat(x: float, y: float, threats: List[Threat], safety_margin: float) -> bool:
    for threat in threats:
        effective_radius = threat.radius + safety_margin
        if (x - threat.x) ** 2 + (y - threat.y) ** 2 < effective_radius ** 2:
            return True
    return False


def world_to_grid(x: float, y: float, resolution: float) -> Tuple[int, int]:
    col = int(round(x / resolution))
    row = int(round(y / resolution))
    return row, col


def grid_to_world(row: int, col: int, resolution: float) -> Tuple[float, float]:
    return col * resolution, row * resolution


def build_grid_map(battlefield: Battlefield, resolution: float, safety_margin: float) -> GridMap:
    width = int(round(battlefield.map_size[0] / resolution)) + 1
    height = int(round(battlefield.map_size[1] / resolution)) + 1
    occupancy = np.zeros((height, width), dtype=bool)

    for row in range(height):
        for col in range(width):
            x, y = grid_to_world(row, col, resolution)
            occupancy[row, col] = point_in_inflated_threat(x, y, battlefield.threats, safety_margin)

    return GridMap(
        width=width,
        height=height,
        resolution=resolution,
        occupancy=occupancy,
        map_size=battlefield.map_size,
    )
