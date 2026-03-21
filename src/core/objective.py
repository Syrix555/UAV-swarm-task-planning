import numpy as np
from typing import List
from src.core.models import UAV, Target, Battlefield


def cost_distance(assignment: np.ndarray, battlefield: Battlefield) -> float:
    """距离代价：所有分配对的直线距离总和"""
    total = 0.0
    for i, uav in enumerate(battlefield.uavs):
        for j, target in enumerate(battlefield.targets):
            if assignment[i, j] == 1:
                total += uav.distance_to(target.x, target.y)
    return total


def cost_threat(assignment: np.ndarray, battlefield: Battlefield) -> float:
    """威胁代价：所有分配对的直线威胁积分总和（预估）"""
    total = 0.0
    for i, uav in enumerate(battlefield.uavs):
        for j, target in enumerate(battlefield.targets):
            if assignment[i, j] == 1:
                total += battlefield.threat_cost_on_line(
                    uav.x, uav.y, target.x, target.y)
    return total


def penalty_time_window(assignment: np.ndarray, battlefield: Battlefield,
                        alpha: float) -> float:
    """时间窗协同惩罚代价，T_syn取分配给同一目标的无人机到达时间平均值"""
    total = 0.0
    num_targets = assignment.shape[1]
    for j in range(num_targets):
        assigned = np.where(assignment[:, j] == 1)[0]
        if len(assigned) < 2:
            continue
        target = battlefield.targets[j]
        etas = [battlefield.uavs[i].eta_to(target.x, target.y)
                for i in assigned]
        t_syn = np.mean(etas)
        for eta in etas:
            total += alpha * (eta - t_syn) ** 2
    return total


def reward_task(assignment: np.ndarray, battlefield: Battlefield) -> float:
    """任务收益：被分配目标的价值总和"""
    total = 0.0
    for j, target in enumerate(battlefield.targets):
        if np.any(assignment[:, j] == 1):
            total += target.value
    return total


def objective_function(assignment: np.ndarray, battlefield: Battlefield,
                       weights: dict) -> float:
    """
    综合目标函数:
    F = w1*C_distance + w2*C_threat + w3*P_time - w4*R_task
    """
    c_dist = cost_distance(assignment, battlefield)
    c_threat = cost_threat(assignment, battlefield)
    p_time = penalty_time_window(assignment, battlefield, weights['alpha'])
    r_task = reward_task(assignment, battlefield)

    return (weights['w1'] * c_dist
            + weights['w2'] * c_threat
            + weights['w3'] * p_time
            - weights['w4'] * r_task)
