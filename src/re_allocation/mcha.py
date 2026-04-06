from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from src.core.models import Battlefield, Target, UAV
from src.re_allocation.events import ReallocationState


@dataclass
class BidResult:
    """单架无人机对单个开放目标的竞标结果。"""

    uav_id: int
    target_id: int
    score: float


@dataclass
class MCHAResult:
    """MCHA重分配结果。"""

    assignment: np.ndarray
    etas: np.ndarray
    selected_bids: List[BidResult]
    remaining_demand: Dict[int, int]
    iterations: int


def run_mcha(
    battlefield: Battlefield,
    weights: dict,
    state: ReallocationState,
    mcha_params: dict = None,
) -> MCHAResult:
    """
    在锁定分配基础上，对开放目标执行多维代价启发式拍卖重分配。
    """
    if mcha_params is None:
        mcha_params = {}

    max_iter = mcha_params.get('max_iter', 50)
    min_score = mcha_params.get('min_score', float('-inf'))

    current_assignment = state.locked_assignment.copy()
    open_targets = list(state.open_targets)
    available_uavs = list(state.available_uavs)
    remaining_demand = dict(state.remaining_demand)
    selected_bids: List[BidResult] = []
    iterations = 0

    for iteration in range(max_iter):
        iterations = iteration + 1

        active_targets = [
            target_id for target_id in open_targets
            if remaining_demand.get(target_id, 0) > 0
        ]
        if not active_targets or not available_uavs:
            break

        bids = generate_bids(
            battlefield,
            weights,
            current_assignment,
            available_uavs,
            active_targets,
            remaining_demand,
        )
        bids = [bid for bid in bids if bid.score >= min_score]
        if not bids:
            break

        accepted_bids, _ = resolve_bids(bids, remaining_demand)
        if not accepted_bids:
            break

        current_assignment = apply_bids_to_assignment(current_assignment, accepted_bids)
        selected_bids.extend(accepted_bids)

        for bid in accepted_bids:
            remaining_demand[bid.target_id] = max(
                0,
                remaining_demand.get(bid.target_id, 0) - 1,
            )

        available_uavs = update_available_uavs_after_round(
            battlefield,
            current_assignment,
            available_uavs,
        )
        open_targets = [
            target_id for target_id in active_targets
            if remaining_demand.get(target_id, 0) > 0
        ]

    etas = compute_eta_matrix(battlefield, current_assignment)
    return MCHAResult(
        assignment=current_assignment,
        etas=etas,
        selected_bids=selected_bids,
        remaining_demand=remaining_demand,
        iterations=iterations,
    )


def marginal_score(
    uav: UAV,
    target: Target,
    current_assignment: np.ndarray,
    battlefield: Battlefield,
    weights: dict,
) -> float:
    """
    MCHA边际得分：
    score = w4 * reward - w1 * distance - w2 * threat - w3 * time_increment
    """
    if not is_feasible_for_uav(uav, target, current_assignment, battlefield):
        return float('-inf')

    distance_cost = uav.distance_to(target.x, target.y)
    threat_cost = battlefield.threat_cost_on_line(uav.x, uav.y, target.x, target.y)
    time_increment = time_window_increment(
        uav,
        target,
        current_assignment,
        battlefield,
        weights['alpha'],
    )
    reward = target.value

    return (
        weights['w4'] * reward
        - weights['w1'] * distance_cost
        - weights['w2'] * threat_cost
        - weights['w3'] * time_increment
    )


def time_window_increment(
    uav: UAV,
    target: Target,
    current_assignment: np.ndarray,
    battlefield: Battlefield,
    alpha: float,
) -> float:
    """
    计算将无人机加入目标当前攻击编队后的时间窗惩罚增量。
    """
    assigned_uav_ids = np.where(current_assignment[:, target.id] == 1)[0].tolist()
    old_etas = [
        battlefield.get_uav(uav_id).eta_to(target.x, target.y)
        for uav_id in assigned_uav_ids
    ]

    new_eta = uav.eta_to(target.x, target.y)
    new_etas = old_etas + [new_eta]

    old_penalty = synchronized_penalty(old_etas, alpha)
    new_penalty = synchronized_penalty(new_etas, alpha)
    return new_penalty - old_penalty


def generate_bids(
    battlefield: Battlefield,
    weights: dict,
    current_assignment: np.ndarray,
    available_uavs: List[int],
    open_targets: List[int],
    remaining_demand: Dict[int, int],
) -> List[BidResult]:
    """
    每轮为每架可用无人机生成一个当前最优投标。
    """
    bids: List[BidResult] = []

    for uav_id in available_uavs:
        uav = battlefield.get_uav(uav_id)
        best_bid = None

        for target_id in open_targets:
            if remaining_demand.get(target_id, 0) <= 0:
                continue

            target = battlefield.get_target(target_id)
            score = marginal_score(uav, target, current_assignment, battlefield, weights)
            if best_bid is None or score > best_bid.score:
                best_bid = BidResult(uav_id=uav_id, target_id=target_id, score=score)

        if best_bid is not None and np.isfinite(best_bid.score):
            bids.append(best_bid)

    return bids


def resolve_bids(
    bids: List[BidResult],
    remaining_demand: Dict[int, int],
) -> Tuple[List[BidResult], List[int]]:
    """
    对每个目标保留若干最高分投标，解决同轮竞标冲突。

    Returns:
        accepted_bids: 本轮中标结果
        rejected_uavs: 本轮落选无人机ID列表
    """
    sorted_bids = sorted(bids, key=lambda bid: bid.score, reverse=True)
    accepted_bids: List[BidResult] = []
    rejected_uavs: List[int] = []
    used_uavs = set()
    accepted_count_by_target: Dict[int, int] = {}

    for bid in sorted_bids:
        if bid.uav_id in used_uavs:
            rejected_uavs.append(bid.uav_id)
            continue

        accepted_count = accepted_count_by_target.get(bid.target_id, 0)
        demand = remaining_demand.get(bid.target_id, 0)
        if accepted_count >= demand:
            rejected_uavs.append(bid.uav_id)
            continue

        accepted_bids.append(bid)
        used_uavs.add(bid.uav_id)
        accepted_count_by_target[bid.target_id] = accepted_count + 1

    all_uavs = {bid.uav_id for bid in bids}
    rejected_uavs.extend(sorted(all_uavs - used_uavs - set(rejected_uavs)))
    return accepted_bids, sorted(set(rejected_uavs))


def apply_bids_to_assignment(
    assignment: np.ndarray,
    accepted_bids: List[BidResult],
) -> np.ndarray:
    """
    将本轮中标结果写入分配矩阵。
    """
    updated_assignment = assignment.copy()
    for bid in accepted_bids:
        updated_assignment[bid.uav_id, bid.target_id] = 1
    return updated_assignment


def compute_eta_matrix(
    battlefield: Battlefield,
    assignment: np.ndarray,
) -> np.ndarray:
    """
    根据分配矩阵重算ETA矩阵。
    """
    num_uavs, num_targets = assignment.shape
    etas = np.zeros((num_uavs, num_targets))

    for uav_id in range(num_uavs):
        uav = battlefield.get_uav(uav_id)
        for target_id in range(num_targets):
            if assignment[uav_id, target_id] == 1:
                target = battlefield.get_target(target_id)
                etas[uav_id, target_id] = uav.eta_to(target.x, target.y)

    return etas


def is_feasible_for_uav(
    uav: UAV,
    target: Target,
    current_assignment: np.ndarray,
    battlefield: Battlefield,
) -> bool:
    """
    检查无人机是否满足新增该任务后的弹药与航程约束。
    """
    current_targets = np.where(current_assignment[uav.id] == 1)[0]
    if target.id in current_targets:
        return False

    current_task_count = len(current_targets)
    if current_task_count + 1 > uav.ammo:
        return False

    current_distance = sum(
        uav.distance_to(battlefield.get_target(target_id).x, battlefield.get_target(target_id).y)
        for target_id in current_targets
    )
    new_distance = current_distance + uav.distance_to(target.x, target.y)
    if new_distance > uav.range_left:
        return False

    return True


def update_available_uavs_after_round(
    battlefield: Battlefield,
    current_assignment: np.ndarray,
    available_uavs: List[int],
) -> List[int]:
    """
    在一轮竞标结束后更新下一轮可继续投标的无人机集合。
    """
    next_available: List[int] = []
    for uav_id in available_uavs:
        uav = battlefield.get_uav(uav_id)
        assigned_count = int(np.sum(current_assignment[uav_id]))
        if assigned_count < uav.ammo:
            next_available.append(uav_id)
    return next_available


def synchronized_penalty(etas: List[float], alpha: float) -> float:
    """计算一组到达时间的协同惩罚。"""
    if len(etas) < 2:
        return 0.0

    t_syn = float(np.mean(etas))
    return float(sum(alpha * (eta - t_syn) ** 2 for eta in etas))
