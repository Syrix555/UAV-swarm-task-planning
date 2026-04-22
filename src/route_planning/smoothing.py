from typing import List, Sequence, Tuple

import numpy as np
from scipy.interpolate import splprep, splev

Point = Tuple[float, float]

DEFAULT_CORNER_ANGLE_THRESHOLD_DEG = 15.0
DEFAULT_CORNER_WINDOW_POINTS = 3
DEFAULT_TRIM_RATIO = 0.2


class BSplineSmoothingError(RuntimeError):
    pass


def _deduplicate_points(path_points: Sequence[Point]) -> List[Point]:
    if not path_points:
        return []

    deduplicated = [path_points[0]]
    for point in path_points[1:]:
        if point != deduplicated[-1]:
            deduplicated.append(point)
    return deduplicated


def _build_chord_parameters(path_points: Sequence[Point]) -> np.ndarray:
    if len(path_points) < 2:
        return np.array([0.0], dtype=float)

    distances = [0.0]
    for start, end in zip(path_points[:-1], path_points[1:]):
        distances.append(float(np.hypot(end[0] - start[0], end[1] - start[1])))

    cumulative = np.cumsum(distances, dtype=float)
    total_length = cumulative[-1]
    if total_length <= 0.0:
        raise BSplineSmoothingError('路径长度为零，无法进行 B 样条参数化')
    return cumulative / total_length


def _resample_path(path_points: Sequence[Point], sample_step: float) -> List[Point]:
    if len(path_points) < 2:
        return list(path_points)

    if sample_step <= 0.0:
        raise BSplineSmoothingError('sample_step 必须大于 0')

    cumulative = [0.0]
    for start, end in zip(path_points[:-1], path_points[1:]):
        cumulative.append(cumulative[-1] + float(np.hypot(end[0] - start[0], end[1] - start[1])))

    total_length = cumulative[-1]
    if total_length == 0.0:
        return [path_points[0], path_points[-1]]

    num_samples = max(2, int(np.ceil(total_length / sample_step)) + 1)
    target_distances = np.linspace(0.0, total_length, num_samples)

    resampled: List[Point] = []
    segment_idx = 0
    for distance in target_distances:
        while segment_idx < len(cumulative) - 2 and cumulative[segment_idx + 1] < distance:
            segment_idx += 1

        start = path_points[segment_idx]
        end = path_points[segment_idx + 1]
        segment_start = cumulative[segment_idx]
        segment_end = cumulative[segment_idx + 1]
        if segment_end == segment_start:
            resampled.append(start)
            continue

        ratio = (distance - segment_start) / (segment_end - segment_start)
        x = start[0] + ratio * (end[0] - start[0])
        y = start[1] + ratio * (end[1] - start[1])
        resampled.append((float(x), float(y)))

    return resampled


def _compute_turn_angle(prev_point: Point, corner_point: Point, next_point: Point) -> float:
    vector_in = np.array([corner_point[0] - prev_point[0], corner_point[1] - prev_point[1]], dtype=float)
    vector_out = np.array([next_point[0] - corner_point[0], next_point[1] - corner_point[1]], dtype=float)
    norm_in = float(np.linalg.norm(vector_in))
    norm_out = float(np.linalg.norm(vector_out))
    if norm_in == 0.0 or norm_out == 0.0:
        return 0.0

    cosine = float(np.dot(vector_in, vector_out) / (norm_in * norm_out))
    cosine = max(-1.0, min(1.0, cosine))
    return float(np.degrees(np.arccos(cosine)))


def _is_corner(prev_point: Point, corner_point: Point, next_point: Point, threshold_deg: float) -> bool:
    turn_angle = _compute_turn_angle(prev_point, corner_point, next_point)
    bend_angle = 180.0 - turn_angle
    return bend_angle >= threshold_deg


def _extract_corner_window(path_points: Sequence[Point], corner_index: int, window_points: int) -> List[Point]:
    if window_points < 3:
        raise BSplineSmoothingError('corner_window_points 必须至少为 3')

    half_window = max(1, window_points // 2)
    start_index = max(0, corner_index - half_window)
    end_index = min(len(path_points), corner_index + half_window + 1)
    window = list(path_points[start_index:end_index])
    if len(window) < 3:
        raise BSplineSmoothingError('局部平滑窗口点数不足，无法构造 B 样条')
    return window


def _point_along(start: Point, end: Point, distance_from_start: float) -> Point:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    segment_length = float(np.hypot(dx, dy))
    if segment_length == 0.0:
        return start
    ratio = distance_from_start / segment_length
    return (start[0] + dx * ratio, start[1] + dy * ratio)


def _build_trimmed_corner_window(
    prev_point: Point,
    corner_point: Point,
    next_point: Point,
    smoothing_factor: float,
) -> List[Point]:
    in_length = float(np.hypot(corner_point[0] - prev_point[0], corner_point[1] - prev_point[1]))
    out_length = float(np.hypot(next_point[0] - corner_point[0], next_point[1] - corner_point[1]))
    if in_length == 0.0 or out_length == 0.0:
        raise BSplineSmoothingError('拐点相邻线段长度为零，无法局部平滑')

    trim_ratio = min(0.35, DEFAULT_TRIM_RATIO + 0.05 * max(0.0, smoothing_factor - 1.0))
    trim_distance = min(in_length, out_length) * trim_ratio
    if trim_distance <= 0.0:
        raise BSplineSmoothingError('局部平滑截短长度非法')

    entry = _point_along(prev_point, corner_point, in_length - trim_distance)
    exit_point = _point_along(corner_point, next_point, trim_distance)
    return [entry, corner_point, exit_point]


def _smooth_with_scipy(
    path_points: Sequence[Point],
    degree: int,
    smoothing_factor: float,
    sample_step: float,
) -> List[Point]:
    deduplicated = _deduplicate_points(path_points)
    if len(deduplicated) < 2:
        return deduplicated
    if len(deduplicated) == 2:
        return _resample_path(deduplicated, sample_step)

    clamped_degree = max(1, min(int(degree), len(deduplicated) - 1))
    parameters = _build_chord_parameters(deduplicated)
    coordinates = np.array(deduplicated, dtype=float)
    sample_count = max(50, len(deduplicated) * 20)

    try:
        tck, _ = splprep(
            [coordinates[:, 0], coordinates[:, 1]],
            u=parameters,
            k=clamped_degree,
            s=0.0,
        )
        dense_u = np.linspace(0.0, 1.0, sample_count)
        xs, ys = splev(dense_u, tck)
    except ImportError as exc:
        raise BSplineSmoothingError('当前环境未安装 scipy，无法生成严格 B 样条') from exc
    except ValueError as exc:
        raise BSplineSmoothingError(f'B 样条拟合失败：{exc}') from exc

    dense_points = [(float(x), float(y)) for x, y in zip(xs, ys)]
    if dense_points:
        dense_points[0] = deduplicated[0]
        dense_points[-1] = deduplicated[-1]
    return _resample_path(dense_points, sample_step)


def _smooth_corner_with_scipy(
    prev_point: Point,
    corner_point: Point,
    next_point: Point,
    degree: int,
    smoothing_factor: float,
    sample_step: float,
) -> List[Point]:
    trimmed_window = _build_trimmed_corner_window(prev_point, corner_point, next_point, smoothing_factor)
    return _smooth_with_scipy(trimmed_window, min(degree, 2), smoothing_factor, sample_step)


def _smooth_locally(
    path_points: Sequence[Point],
    degree: int,
    smoothing_factor: float,
    sample_step: float,
    corner_angle_threshold_deg: float,
    corner_window_points: int,
) -> List[Point]:
    deduplicated = _deduplicate_points(path_points)
    if len(deduplicated) < 3:
        return deduplicated

    if corner_window_points < 3:
        raise BSplineSmoothingError('corner_window_points 必须至少为 3')

    merged: List[Point] = [deduplicated[0]]
    last_smoothed_index = -10

    for index in range(1, len(deduplicated) - 1):
        prev_point = deduplicated[index - 1]
        corner_point = deduplicated[index]
        next_point = deduplicated[index + 1]

        if index <= last_smoothed_index + 1:
            if corner_point != merged[-1]:
                merged.append(corner_point)
            continue

        if _is_corner(prev_point, corner_point, next_point, corner_angle_threshold_deg):
            local_curve = _smooth_corner_with_scipy(
                prev_point,
                corner_point,
                next_point,
                degree,
                smoothing_factor,
                sample_step,
            )
            for point in local_curve[1:-1]:
                if point != merged[-1]:
                    merged.append(point)
            last_smoothed_index = index
        else:
            if corner_point != merged[-1]:
                merged.append(corner_point)

    if deduplicated[-1] != merged[-1]:
        merged.append(deduplicated[-1])
    return _resample_path(merged, sample_step)


def smooth_bspline(
    path_points: Sequence[Point],
    degree: int,
    smoothing_factor: float,
    sample_step: float,
    corner_angle_threshold_deg: float = DEFAULT_CORNER_ANGLE_THRESHOLD_DEG,
    corner_window_points: int = DEFAULT_CORNER_WINDOW_POINTS,
) -> List[Point]:
    if len(path_points) < 3 or smoothing_factor <= 0.0:
        return list(path_points)

    return _smooth_locally(
        path_points,
        degree,
        smoothing_factor,
        sample_step,
        corner_angle_threshold_deg,
        corner_window_points,
    )
