from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.core.models import Battlefield, Target, Threat


class EventType(Enum):
    """触发MCHA重分配的事件类型。"""

    UAV_LOST = "uav_lost"
    THREAT_ADDED = "threat_added"
    TARGET_ADDED = "target_added"
    TARGET_REMOVED = "target_removed"
    TARGET_DEMAND_CHANGED = "target_demand_changed"
    TARGET_VALUE_CHANGED = "target_value_changed"


@dataclass
class Event:
    """战场动态事件。"""

    type: EventType
    data: dict


@dataclass
class ReallocationState:
    """MCHA重分配输入状态。"""

    locked_assignment: np.ndarray
    open_targets: List[int]
    available_uavs: List[int]
    remaining_demand: Dict[int, int]


def apply_event_to_battlefield(event: Event, battlefield: Battlefield) -> None:
    """
    将事件应用到当前战场状态。

    当前主要用于确保 THREAT_ADDED 等事件不仅影响“释放判断”，
    也会影响后续 MCHA 重分配阶段的代价计算。
    """
    if event.type == EventType.THREAT_ADDED:
        new_threat = event.data['threat']
        if all(existing.id != new_threat.id for existing in battlefield.threats):
            battlefield.threats.append(new_threat)
        return

    if event.type == EventType.TARGET_ADDED:
        new_target = event.data['target']
        if all(existing.id != new_target.id for existing in battlefield.targets):
            battlefield.targets.append(new_target)
        return

    if event.type == EventType.TARGET_REMOVED:
        target_id = event.data['target_id']
        battlefield.targets = [target for target in battlefield.targets if target.id != target_id]
        return

    if event.type == EventType.TARGET_DEMAND_CHANGED:
        target = battlefield.get_target(event.data['target_id'])
        target.required_uavs = event.data['new_required_uavs']
        return

    if event.type == EventType.TARGET_VALUE_CHANGED:
        target = battlefield.get_target(event.data['target_id'])
        target.value = event.data['new_value']
        return


def analyze_event_impact(
    event: Event,
    battlefield: Battlefield,
    assignment: np.ndarray,
    etas: np.ndarray,
) -> ReallocationState:
    """
    根据事件分析原分配方案，生成MCHA需要处理的开放子问题。
    """
    if event.type == EventType.UAV_LOST:
        return handle_uav_lost(event.data['uav_id'], battlefield, assignment)

    if event.type == EventType.THREAT_ADDED:
        return handle_threat_added(
            event.data['threat'],
            battlefield,
            assignment,
            etas,
            event.data.get('threat_threshold', 0.0),
        )

    if event.type == EventType.TARGET_ADDED:
        return handle_target_added(event.data['target'], battlefield, assignment)

    if event.type == EventType.TARGET_REMOVED:
        return handle_target_removed(event.data['target_id'], battlefield, assignment)

    if event.type == EventType.TARGET_DEMAND_CHANGED:
        return handle_target_demand_changed(
            event.data['target_id'],
            event.data['new_required_uavs'],
            battlefield,
            assignment,
        )

    if event.type == EventType.TARGET_VALUE_CHANGED:
        target_id = event.data['target_id']
        locked_assignment = assignment.copy()
        available_uavs = get_available_uavs(battlefield, locked_assignment)
        return ReallocationState(
            locked_assignment=locked_assignment,
            open_targets=[target_id],
            available_uavs=available_uavs,
            remaining_demand=compute_remaining_demand(
                battlefield,
                locked_assignment,
                [target_id],
            ),
        )

    raise ValueError(f"不支持的事件类型: {event.type}")


def handle_uav_lost(
    uav_id: int,
    battlefield: Battlefield,
    assignment: np.ndarray,
) -> ReallocationState:
    """
    无人机损失事件：
    - 释放该无人机承担的任务
    - 从可用无人机集合中移除该无人机
    """
    released_pairs = [(uav_id, target_id) for _, target_id in assignment_pairs(assignment)
                      if _ == uav_id]
    open_targets = sorted({target_id for _, target_id in released_pairs})
    locked_assignment = build_locked_assignment(assignment, released_pairs)
    remaining_demand = compute_remaining_demand(
        battlefield,
        locked_assignment,
        open_targets,
    )
    available_uavs = get_available_uavs(
        battlefield,
        locked_assignment,
        excluded_uavs=[uav_id],
    )
    return ReallocationState(
        locked_assignment=locked_assignment,
        open_targets=open_targets,
        available_uavs=available_uavs,
        remaining_demand=remaining_demand,
    )


def handle_threat_added(
    new_threat: Threat,
    battlefield: Battlefield,
    assignment: np.ndarray,
    etas: np.ndarray,
    threat_threshold: float = 0.0,
) -> ReallocationState:
    """
    新增威胁事件：
    - 检查哪些已有分配路径受新增威胁显著影响
    - 释放这些分配对，保留其余锁定分配
    """
    del etas

    released_pairs: List[Tuple[int, int]] = []
    for uav_id, target_id in assignment_pairs(assignment):
        uav = battlefield.get_uav(uav_id)
        target = battlefield.get_target(target_id)
        added_cost = threat_cost_on_line_for_single_threat(
            new_threat,
            uav.x,
            uav.y,
            target.x,
            target.y,
        )
        if added_cost > threat_threshold:
            released_pairs.append((uav_id, target_id))

    open_targets = sorted({target_id for _, target_id in released_pairs})
    locked_assignment = build_locked_assignment(assignment, released_pairs)
    remaining_demand = compute_remaining_demand(
        battlefield,
        locked_assignment,
        open_targets,
    )
    available_uavs = get_available_uavs(battlefield, locked_assignment)
    return ReallocationState(
        locked_assignment=locked_assignment,
        open_targets=open_targets,
        available_uavs=available_uavs,
        remaining_demand=remaining_demand,
    )


def handle_target_added(
    new_target: Target,
    battlefield: Battlefield,
    assignment: np.ndarray,
) -> ReallocationState:
    """
    新增目标事件：
    - 原分配全部锁定
    - 新目标进入开放任务池
    """
    del battlefield

    locked_assignment = assignment.copy()
    available_uavs = list(range(assignment.shape[0]))
    return ReallocationState(
        locked_assignment=locked_assignment,
        open_targets=[new_target.id],
        available_uavs=available_uavs,
        remaining_demand={new_target.id: new_target.required_uavs},
    )


def handle_target_removed(
    target_id: int,
    battlefield: Battlefield,
    assignment: np.ndarray,
) -> ReallocationState:
    """
    目标移除事件：
    - 删除该目标相关分配
    - 释放对应无人机资源
    """
    del battlefield

    released_pairs = [(uav_id, j) for uav_id, j in assignment_pairs(assignment) if j == target_id]
    locked_assignment = build_locked_assignment(assignment, released_pairs)
    available_uavs = list(range(assignment.shape[0]))
    return ReallocationState(
        locked_assignment=locked_assignment,
        open_targets=[],
        available_uavs=available_uavs,
        remaining_demand={},
    )


def handle_target_demand_changed(
    target_id: int,
    new_required_uavs: int,
    battlefield: Battlefield,
    assignment: np.ndarray,
) -> ReallocationState:
    """
    目标需求变化事件：
    - 需求增加时补齐缺口
    - 需求减少时释放多余分配
    """
    current_assigned = [uav_id for uav_id, j in assignment_pairs(assignment) if j == target_id]
    current_count = len(current_assigned)

    if new_required_uavs >= current_count:
        locked_assignment = assignment.copy()
        remaining = new_required_uavs - current_count
        return ReallocationState(
            locked_assignment=locked_assignment,
            open_targets=[target_id] if remaining > 0 else [],
            available_uavs=get_available_uavs(battlefield, locked_assignment),
            remaining_demand={target_id: remaining} if remaining > 0 else {},
        )

    released_uavs = sorted(current_assigned, reverse=True)[:current_count - new_required_uavs]
    released_pairs = [(uav_id, target_id) for uav_id in released_uavs]
    locked_assignment = build_locked_assignment(assignment, released_pairs)
    return ReallocationState(
        locked_assignment=locked_assignment,
        open_targets=[],
        available_uavs=get_available_uavs(battlefield, locked_assignment),
        remaining_demand={},
    )


def build_locked_assignment(
    assignment: np.ndarray,
    released_pairs: List[Tuple[int, int]],
) -> np.ndarray:
    """
    根据需要释放的分配对，构造锁定分配矩阵。
    """
    locked_assignment = assignment.copy()
    for uav_id, target_id in released_pairs:
        if 0 <= uav_id < locked_assignment.shape[0] and 0 <= target_id < locked_assignment.shape[1]:
            locked_assignment[uav_id, target_id] = 0
    return locked_assignment


def compute_remaining_demand(
    battlefield: Battlefield,
    locked_assignment: np.ndarray,
    target_ids: List[int],
) -> Dict[int, int]:
    """
    计算在锁定分配基础上，各目标仍需补充的无人机数量。
    """
    remaining_demand: Dict[int, int] = {}
    for target_id in target_ids:
        target = battlefield.get_target(target_id)
        assigned_count = int(np.sum(locked_assignment[:, target_id]))
        remaining = max(0, target.required_uavs - assigned_count)
        if remaining > 0:
            remaining_demand[target_id] = remaining
    return remaining_demand


def get_available_uavs(
    battlefield: Battlefield,
    locked_assignment: np.ndarray,
    excluded_uavs: Optional[List[int]] = None,
) -> List[int]:
    """
    获取当前仍可参与MCHA竞标的无人机ID列表。
    """
    del battlefield

    excluded_set = set(excluded_uavs or [])
    available_uavs: List[int] = []
    for uav_id in range(locked_assignment.shape[0]):
        if uav_id in excluded_set:
            continue
        available_uavs.append(uav_id)
    return available_uavs


def assignment_pairs(assignment: np.ndarray) -> List[Tuple[int, int]]:
    """
    将分配矩阵转换为 (uav_id, target_id) 列表。
    """
    rows, cols = np.where(assignment == 1)
    return list(zip(rows.tolist(), cols.tolist()))


def threat_cost_on_line_for_single_threat(
    threat: Threat,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    num_samples: int = 20,
) -> float:
    """计算单个威胁区在一条直线路径上的威胁积分。"""
    total = 0.0
    for k in range(num_samples + 1):
        t = k / num_samples
        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)
        total += threat.threat_cost_at(px, py)

    seg_len = battlefield_distance(x1, y1, x2, y2)
    return total * seg_len / (num_samples + 1)


def battlefield_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """欧氏距离工具函数。"""
    return float(np.hypot(x1 - x2, y1 - y2))
