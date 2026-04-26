from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from src.core.models import AssignmentPlan, Battlefield, Target, TaskNode, UAV, UavTaskSequence
from src.core.sequence_eval import evaluate_uav_task_sequence
from src.re_allocation.events import PlanReallocationState, ReallocationState, copy_assignment_plan


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


@dataclass
class PlanMCHAResult:
    """任务序列版 MCHA 重分配结果。"""

    assignment_plan: AssignmentPlan
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


def run_mcha_for_plan(
    battlefield: Battlefield,
    weights: dict,
    state: PlanReallocationState,
    mcha_params: dict = None,
) -> PlanMCHAResult:
    """
    在 AssignmentPlan 基础上，对开放目标执行链尾追加式 MCHA 重分配。
    """
    if mcha_params is None:
        mcha_params = {}

    max_iter = mcha_params.get('max_iter', 50)
    min_score = mcha_params.get('min_score', float('-inf'))

    current_plan = copy_assignment_plan(state.locked_plan)
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

        bids = generate_plan_bids(
            battlefield,
            weights,
            current_plan,
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

        current_plan = apply_bids_to_plan(current_plan, accepted_bids)
        selected_bids.extend(accepted_bids)

        for bid in accepted_bids:
            remaining_demand[bid.target_id] = max(
                0,
                remaining_demand.get(bid.target_id, 0) - 1,
            )

        available_uavs = update_available_uavs_after_plan_round(
            battlefield,
            current_plan,
            available_uavs,
        )
        open_targets = [
            target_id for target_id in active_targets
            if remaining_demand.get(target_id, 0) > 0
        ]

    num_uavs = len(battlefield.uavs)
    num_targets = len(battlefield.targets)
    assignment = current_plan.to_assignment_matrix(num_uavs, num_targets)
    etas = compute_eta_matrix_for_plan(battlefield, current_plan)
    current_plan.total_cost = float(sum(bid.score for bid in selected_bids))

    return PlanMCHAResult(
        assignment_plan=current_plan,
        assignment=assignment,
        etas=etas,
        selected_bids=selected_bids,
        remaining_demand=remaining_demand,
        iterations=iterations,
    )


def generate_plan_bids(
    battlefield: Battlefield,
    weights: dict,
    current_plan: AssignmentPlan,
    available_uavs: List[int],
    open_targets: List[int],
    remaining_demand: Dict[int, int],
) -> List[BidResult]:
    """每轮为每架可用 UAV 生成一个链尾追加的当前最优投标。"""
    bids: List[BidResult] = []

    for uav_id in available_uavs:
        uav = battlefield.get_uav(uav_id)
        best_bid = None

        for target_id in open_targets:
            if remaining_demand.get(target_id, 0) <= 0:
                continue

            target = battlefield.get_target(target_id)
            score = marginal_score_for_plan(uav, target, current_plan, battlefield, weights)
            if best_bid is None or score > best_bid.score:
                best_bid = BidResult(uav_id=uav_id, target_id=target_id, score=score)

        if best_bid is not None and np.isfinite(best_bid.score):
            bids.append(best_bid)

    return bids


def marginal_score_for_plan(
    uav: UAV,
    target: Target,
    current_plan: AssignmentPlan,
    battlefield: Battlefield,
    weights: dict,
) -> float:
    """
    任务序列版 MCHA 边际得分：
    将目标追加到 UAV 任务链尾部，比较追加前后的链式代价变化。
    """
    sequence = current_plan.uav_task_sequences.get(uav.id, UavTaskSequence(uav_id=uav.id))
    if target.id in sequence.target_ids():
        return float('-inf')

    candidate_tasks = list(sequence.tasks)
    candidate_tasks.append(TaskNode(target_id=target.id, order=len(candidate_tasks)))
    candidate_sequence = UavTaskSequence(uav_id=uav.id, tasks=candidate_tasks)

    current_eval = evaluate_uav_task_sequence(
        battlefield,
        sequence,
        alpha=weights['alpha'],
    )
    candidate_eval = evaluate_uav_task_sequence(
        battlefield,
        candidate_sequence,
        alpha=weights['alpha'],
    )
    if not candidate_eval.is_feasible:
        return float('-inf')

    distance_delta = candidate_eval.total_distance - current_eval.total_distance
    threat_delta = sequence_threat_cost(battlefield, candidate_sequence) - sequence_threat_cost(battlefield, sequence)
    time_delta = candidate_eval.time_window_penalty - current_eval.time_window_penalty

    return (
        weights['w4'] * target.value
        - weights['w1'] * distance_delta
        - weights['w2'] * threat_delta
        - weights['w3'] * time_delta
    )


def sequence_threat_cost(
    battlefield: Battlefield,
    sequence: UavTaskSequence,
) -> float:
    """计算任务链各航段的直线威胁代价估计。"""
    if not sequence.tasks:
        return 0.0

    uav = battlefield.get_uav(sequence.uav_id)
    current_x, current_y = uav.x, uav.y
    total = 0.0
    for task in sequence.tasks:
        target = battlefield.get_target(task.target_id)
        total += battlefield.threat_cost_on_line(current_x, current_y, target.x, target.y)
        current_x, current_y = target.x, target.y
    return total


def apply_bids_to_plan(
    plan: AssignmentPlan,
    accepted_bids: List[BidResult],
) -> AssignmentPlan:
    """将本轮中标结果追加到对应 UAV 任务链尾部。"""
    updated_plan = copy_assignment_plan(plan)

    for bid in accepted_bids:
        sequence = updated_plan.uav_task_sequences.setdefault(
            bid.uav_id,
            UavTaskSequence(uav_id=bid.uav_id),
        )
        sequence.append_target(bid.target_id)
        assignees = updated_plan.target_assignees.setdefault(bid.target_id, [])
        if bid.uav_id not in assignees:
            assignees.append(bid.uav_id)
            assignees.sort()

    return updated_plan


def update_available_uavs_after_plan_round(
    battlefield: Battlefield,
    current_plan: AssignmentPlan,
    available_uavs: List[int],
) -> List[int]:
    """在一轮任务序列竞标结束后更新仍有 ammo 余量的 UAV。"""
    next_available: List[int] = []
    for uav_id in available_uavs:
        uav = battlefield.get_uav(uav_id)
        sequence = current_plan.uav_task_sequences.get(uav_id)
        task_count = sequence.task_count() if sequence is not None else 0
        if task_count < uav.ammo:
            next_available.append(uav_id)
    return next_available


def compute_eta_matrix_for_plan(
    battlefield: Battlefield,
    plan: AssignmentPlan,
) -> np.ndarray:
    """根据任务链累计到达时刻生成 ETA 矩阵。"""
    num_uavs = len(battlefield.uavs)
    num_targets = len(battlefield.targets)
    etas = np.zeros((num_uavs, num_targets))

    for sequence in plan.uav_task_sequences.values():
        evaluated = evaluate_uav_task_sequence(battlefield, sequence)
        for task in evaluated.evaluated_sequence.tasks:
            if 0 <= sequence.uav_id < num_uavs and 0 <= task.target_id < num_targets:
                etas[sequence.uav_id, task.target_id] = task.planned_arrival_time

    return etas


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
