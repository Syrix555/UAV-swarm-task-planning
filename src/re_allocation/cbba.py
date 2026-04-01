from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from src.core.models import Battlefield, Target, UAV
from src.re_allocation.events import ReallocationState


@dataclass
class BidResult:
    """单架无人机对单个目标的竞标结果。"""

    uav_id: int
    target_id: int
    score: float


@dataclass
class CBBAResult:
    """CBBA重分配结果。"""

    assignment: np.ndarray
    etas: np.ndarray
    selected_bids: List[BidResult]
    remaining_demand: Dict[int, int]
    iterations: int


def run_cbba(
    battlefield: Battlefield,
    weights: dict,
    state: ReallocationState,
    cbba_params: dict = None,
) -> CBBAResult:
    """
    在锁定分配基础上，对开放目标执行改进CBBA重分配。
    """
    raise NotImplementedError("run_cbba 尚未实现")


def marginal_score(
    uav: UAV,
    target: Target,
    current_assignment: np.ndarray,
    battlefield: Battlefield,
    weights: dict,
) -> float:
    """
    改进CBBA边际得分：
    score = w4 * reward - w1 * distance - w2 * threat - w3 * time_increment
    """
    raise NotImplementedError("marginal_score 尚未实现")


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
    raise NotImplementedError("time_window_increment 尚未实现")


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
    raise NotImplementedError("generate_bids 尚未实现")


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
    raise NotImplementedError("resolve_bids 尚未实现")


def apply_bids_to_assignment(
    assignment: np.ndarray,
    accepted_bids: List[BidResult],
) -> np.ndarray:
    """
    将中标结果写入分配矩阵。
    """
    raise NotImplementedError("apply_bids_to_assignment 尚未实现")


def compute_eta_matrix(
    battlefield: Battlefield,
    assignment: np.ndarray,
) -> np.ndarray:
    """
    根据分配矩阵重算ETA矩阵。
    """
    raise NotImplementedError("compute_eta_matrix 尚未实现")


def is_feasible_for_uav(
    uav: UAV,
    target: Target,
    current_assignment: np.ndarray,
    battlefield: Battlefield,
) -> bool:
    """
    检查无人机是否满足新增该任务后的弹药与航程约束。
    """
    raise NotImplementedError("is_feasible_for_uav 尚未实现")


def update_available_uavs_after_round(
    battlefield: Battlefield,
    current_assignment: np.ndarray,
    available_uavs: List[int],
) -> List[int]:
    """
    在一轮竞标结束后更新下一轮可继续投标的无人机集合。
    """
    raise NotImplementedError("update_available_uavs_after_round 尚未实现")
