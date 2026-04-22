import heapq
import math
from typing import Dict, List, Optional, Tuple

from src.route_planning.grid import GridMap, grid_to_world, world_to_grid

GridIndex = Tuple[int, int]


def heuristic(node: GridIndex, goal: GridIndex, resolution: float) -> float:
    return math.hypot(goal[0] - node[0], goal[1] - node[1]) * resolution


def get_neighbors(node: GridIndex, grid_map: GridMap, allow_diagonal: bool = True):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if allow_diagonal:
        directions += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    for dr, dc in directions:
        nr = node[0] + dr
        nc = node[1] + dc
        if not grid_map.in_bounds(nr, nc):
            continue
        if grid_map.is_blocked(nr, nc):
            continue
        step_cost = math.hypot(dr, dc) * grid_map.resolution
        yield (nr, nc), step_cost


def reconstruct_path(came_from: Dict[GridIndex, GridIndex], current: GridIndex, grid_map: GridMap) -> List[Tuple[float, float]]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return [grid_to_world(row, col, grid_map.resolution) for row, col in path]


def astar_search(
    grid_map: GridMap,
    start_xy: Tuple[float, float],
    goal_xy: Tuple[float, float],
    allow_diagonal: bool = True,
) -> List[Tuple[float, float]]:
    start = world_to_grid(start_xy[0], start_xy[1], grid_map.resolution)
    goal = world_to_grid(goal_xy[0], goal_xy[1], grid_map.resolution)

    if not grid_map.in_bounds(*start) or not grid_map.in_bounds(*goal):
        return []
    if grid_map.is_blocked(*start) or grid_map.is_blocked(*goal):
        return []

    open_heap: List[Tuple[float, GridIndex]] = []
    heapq.heappush(open_heap, (0.0, start))
    came_from: Dict[GridIndex, GridIndex] = {}
    g_score: Dict[GridIndex, float] = {start: 0.0}
    f_score: Dict[GridIndex, float] = {start: heuristic(start, goal, grid_map.resolution)}
    closed_set = set()

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in closed_set:
            continue
        if current == goal:
            return reconstruct_path(came_from, current, grid_map)

        closed_set.add(current)
        for neighbor, step_cost in get_neighbors(current, grid_map, allow_diagonal):
            tentative_g = g_score[current] + step_cost
            if tentative_g >= g_score.get(neighbor, float('inf')):
                continue
            came_from[neighbor] = current
            g_score[neighbor] = tentative_g
            total_cost = tentative_g + heuristic(neighbor, goal, grid_map.resolution)
            f_score[neighbor] = total_cost
            heapq.heappush(open_heap, (total_cost, neighbor))

    return []
