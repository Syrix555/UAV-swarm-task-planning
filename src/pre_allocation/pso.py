"""
改进粒子群算法(PSO)实现无人机集群任务预分配

创新点：
1. Logistic混沌映射初始化种群 —— 提高初始解的多样性和覆盖率
2. 余弦函数自适应惯性权重 —— 平衡全局搜索与局部开发能力

离散化策略：
  标准PSO是连续优化算法，本实现通过Sigmoid概率映射将其适配到离散任务分配问题。
  核心思想：将连续的"速度"值通过Sigmoid函数映射为[0,1]区间的概率值，
  以该概率决定粒子在每个维度上是否向全局最优解学习，从而实现离散位置更新。

编码方式（多槽位）：
  每个目标根据其 required_uavs 属性占据多个槽位。例如目标0需要2架、目标1需要3架，
  则粒子为 [t0_slot0, t0_slot1, t1_slot0, t1_slot1, t1_slot2, ...]。
  每个槽位的值为无人机编号，表示该无人机被分配去打击对应目标。
  这种编码天然支持饱和攻击（多架无人机打同一目标），时间窗协同惩罚因此生效。
"""

import math
import numpy as np
from collections import defaultdict
from typing import Dict, Tuple, List
from src.core.models import AssignmentPlan, Battlefield
from src.core.sequence_eval import evaluate_uav_task_sequence
from config.params import PSO as PSO_PARAMS


# ============================================================
# 槽位映射工具
# ============================================================

def build_slot_mapping(battlefield: Battlefield) -> Tuple[int, List[int]]:
    """
    根据每个目标的 required_uavs 构建槽位到目标的映射

    Returns:
        total_slots: 粒子总维度
        slot_to_target: 长度为 total_slots 的列表，slot_to_target[d] = 目标编号j
    """
    slot_to_target = []
    for target in battlefield.targets:
        for _ in range(target.required_uavs):
            slot_to_target.append(target.id)
    return len(slot_to_target), slot_to_target


# ============================================================
# 创新点一：Logistic 混沌映射初始化
# ============================================================
# Logistic映射公式: z_{n+1} = μ * z_n * (1 - z_n)
# 当 μ=4 时，映射处于完全混沌状态，序列具有遍历性（即序列值会均匀
# 覆盖(0,1)区间），利用这一特性替代伪随机数生成初始种群，
# 可使初始粒子在解空间中分布更加均匀，避免算法因初始解集中而过早收敛。

def _default_capacities(num_uavs: int) -> List[int]:
    return [1 for _ in range(num_uavs)]


def _validate_capacities(dim: int, num_uavs: int, uav_capacities: List[int]) -> None:
    if len(uav_capacities) != num_uavs:
        raise ValueError('uav_capacities 长度必须等于 num_uavs')
    if any(capacity < 0 for capacity in uav_capacities):
        raise ValueError('uav_capacities 不能包含负数')
    if dim > sum(uav_capacities):
        raise ValueError('任务槽位数超过 UAV 总 ammo 容量，请检查场景配置')


def _capacity_pool(num_uavs: int, uav_capacities: List[int]) -> List[int]:
    pool: List[int] = []
    for uav_id in range(num_uavs):
        pool.extend([uav_id] * uav_capacities[uav_id])
    return pool


def logistic_init(
    num_particles: int,
    dim: int,
    num_uavs: int,
    uav_capacities: List[int] | None = None,
) -> np.ndarray:
    """
    使用Logistic混沌映射生成初始种群。

    针对任务序列场景，初始化允许同一 UAV 在多个任务槽位中出现，
    但出现次数不能超过其 ammo 容量。若不传入 uav_capacities，则保持
    原有 ammo=1 的排列型初始化语义。

    Parameters:
        num_particles: 粒子数量
        dim: 粒子维度（等于所有目标的 required_uavs 之和）
        num_uavs: 无人机数量N

    Returns:
        shape=(num_particles, dim) 的整数数组，每个元素为无人机编号 [0, N-1]
    """
    capacities = uav_capacities if uav_capacities is not None else _default_capacities(num_uavs)
    _validate_capacities(dim, num_uavs, capacities)
    pool = _capacity_pool(num_uavs, capacities)

    fixed_points = [0.0, 0.25, 0.5, 0.75, 1.0]
    z = np.random.rand()
    while any(abs(z - fp) < 0.01 for fp in fixed_points):
        z = np.random.rand()

    particles = []
    for _ in range(num_particles):
        chaotic_values = []
        for uav_id in pool:
            z = 4.0 * z * (1 - z)
            chaotic_values.append((z, uav_id))

        chaotic_values.sort(key=lambda item: item[0])
        row = [uav_id for _, uav_id in chaotic_values[:dim]]
        particles.append(row)

    return np.array(particles, dtype=int)


def random_init(
    num_particles: int,
    dim: int,
    num_uavs: int,
    uav_capacities: List[int] | None = None,
) -> np.ndarray:
    """使用容量约束下的无放回随机采样生成初始种群。"""
    capacities = uav_capacities if uav_capacities is not None else _default_capacities(num_uavs)
    _validate_capacities(dim, num_uavs, capacities)
    pool = np.array(_capacity_pool(num_uavs, capacities), dtype=int)

    particles = []
    for _ in range(num_particles):
        row = np.random.permutation(pool)[:dim]
        particles.append(row)
    return np.array(particles, dtype=int)


# 传统PSO使用线性递减惯性权重 w(t) = w_start - (w_start-w_end)*t/T，
# 这种线性策略在中期过渡不够平滑。
# 余弦策略: w(t) = w_end + 0.5*(w_start - w_end)*(1 + cos(π*t/T))
# - 迭代初期(t≈0): cos≈1, w≈w_start（大值），增强全局搜索能力
# - 迭代中期(t≈T/2): cos≈0, w≈(w_start+w_end)/2，平滑过渡
# - 迭代后期(t≈T): cos≈-1, w≈w_end（小值），增强局部精细开发能力

def cosine_weight(t: int, T: int, w_start: float, w_end: float) -> float:
    """
    余弦自适应惯性权重

    Parameters:
        t: 当前迭代次数
        T: 最大迭代次数
        w_start: 惯性权重初始值（较大，如0.9）
        w_end: 惯性权重终值（较小，如0.4）

    Returns:
        当前迭代的惯性权重值
    """
    return w_end + 0.5 * (w_start - w_end) * (1 + math.cos(math.pi * t / T))


def linear_weight(t: int, T: int, w_start: float, w_end: float) -> float:
    """线性递减惯性权重，用于与余弦策略做对比实验。"""
    return w_start - (w_start - w_end) * t / T


def swap_to_match(particle: np.ndarray, position_idx: int, target_uav: int) -> None:
    """通过交换操作，将目标无人机移动到指定位置，保持排列合法。"""
    if particle[position_idx] == target_uav:
        return

    match_idx = np.where(particle == target_uav)[0]
    if len(match_idx) == 0:
        return

    other_idx = int(match_idx[0])
    particle[position_idx], particle[other_idx] = particle[other_idx], particle[position_idx]


def repair_permutation(particle: np.ndarray, num_uavs: int) -> np.ndarray:
    """修复粒子中的重复编号，保持无人机编号不重复。"""
    seen = set()
    duplicates = []
    for idx, value in enumerate(particle):
        if int(value) in seen:
            duplicates.append(idx)
        else:
            seen.add(int(value))

    missing = [uav_id for uav_id in range(num_uavs) if uav_id not in seen]
    for idx, new_value in zip(duplicates, missing):
        particle[idx] = new_value

    return particle


def repair_capacity(particle: np.ndarray, uav_capacities: List[int]) -> np.ndarray:
    """修复粒子，使每架 UAV 出现次数不超过 ammo 容量。"""
    num_uavs = len(uav_capacities)
    if len(particle) > sum(uav_capacities):
        raise ValueError('任务槽位数超过 UAV 总 ammo 容量，无法修复粒子')

    repaired = particle.copy()
    counts = np.zeros(num_uavs, dtype=int)
    overflow_indices: List[int] = []

    for idx, value in enumerate(repaired):
        uav_id = int(value)
        if uav_id < 0 or uav_id >= num_uavs or counts[uav_id] >= uav_capacities[uav_id]:
            overflow_indices.append(idx)
            continue
        counts[uav_id] += 1

    available: List[int] = []
    for uav_id, capacity in enumerate(uav_capacities):
        available.extend([uav_id] * int(capacity - counts[uav_id]))

    if len(available) < len(overflow_indices):
        raise ValueError('没有足够的 UAV ammo 容量修复粒子')

    np.random.shuffle(available)
    for idx, uav_id in zip(overflow_indices, available):
        repaired[idx] = uav_id

    return repaired


def decode(particle: np.ndarray, num_uavs: int, num_targets: int,
           slot_to_target: List[int]) -> np.ndarray:
    """
    将多槽位粒子编码解码为分配矩阵

    多槽位编码: 长度为 sum(required_uavs) 的整数数组
      每个槽位对应一个目标，值为无人机编号
      同一目标的多个槽位允许分配不同的无人机（饱和攻击）

    分配矩阵: N×M 的 0-1 矩阵，X[i][j]=1 表示无人机 i 被分配攻击目标 j

    Parameters:
        particle: 粒子编码数组
        num_uavs: 无人机数量N
        num_targets: 目标数量M
        slot_to_target: 槽位到目标的映射

    Returns:
        N×M 的分配矩阵
    """
    assignment = np.zeros((num_uavs, num_targets), dtype=int)
    for d, target_id in enumerate(slot_to_target):
        uav_id = particle[d]
        assignment[uav_id, target_id] = 1
    return assignment


def decode_to_assignment_plan(
    particle: np.ndarray,
    battlefield: Battlefield,
    slot_to_target: List[int],
    total_cost: float = 0.0,
) -> AssignmentPlan:
    """将粒子按链尾追加规则解码为 AssignmentPlan。"""
    uav_ids = [uav.id for uav in battlefield.uavs]
    plan = AssignmentPlan.empty(uav_ids)
    target_assignee_sets: dict[int, set[int]] = defaultdict(set)

    for slot_idx, target_id in enumerate(slot_to_target):
        uav_id = int(particle[slot_idx])
        if uav_id not in plan.uav_task_sequences:
            continue
        plan.uav_task_sequences[uav_id].append_target(target_id)
        target_assignee_sets[target_id].add(uav_id)

    plan.target_assignees = {
        target_id: sorted(assignees)
        for target_id, assignees in target_assignee_sets.items()
        if assignees
    }
    plan.total_cost = total_cost
    return plan


def assignment_plan_to_eta_matrix(battlefield: Battlefield, plan: AssignmentPlan) -> np.ndarray:
    """根据任务链累计到达时刻生成兼容旧流程的 ETA 矩阵。"""
    num_uavs = len(battlefield.uavs)
    num_targets = len(battlefield.targets)
    etas = np.zeros((num_uavs, num_targets))

    for sequence in plan.uav_task_sequences.values():
        evaluated = evaluate_uav_task_sequence(battlefield, sequence)
        for task in evaluated.evaluated_sequence.tasks:
            if 0 <= sequence.uav_id < num_uavs and 0 <= task.target_id < num_targets:
                etas[sequence.uav_id, task.target_id] = task.planned_arrival_time

    return etas


def cooperative_time_window_penalty_from_arrivals(
    target_arrival_times: Dict[int, List[float]],
    *,
    alpha: float,
    sync_window: float,
) -> float:
    """
    计算协同打击时间窗惩罚。

    对同一目标的多架 UAV，若最晚到达时间与最早到达时间之差
    超过允许同步窗口 sync_window，则按建模中的平方偏差项：
        sum(alpha * (eta - T_syn)^2)
    计入惩罚，其中 T_syn 为该目标所有到达时间的平均值。
    """
    total = 0.0
    for arrivals in target_arrival_times.values():
        if len(arrivals) < 2:
            continue

        spread = max(arrivals) - min(arrivals)
        if spread <= sync_window + 1e-12:
            continue

        t_syn = float(np.mean(arrivals))
        total += float(sum(alpha * (arrival - t_syn) ** 2 for arrival in arrivals))

    return total


def cooperative_time_window_penalty(
    battlefield: Battlefield,
    plan: AssignmentPlan,
    *,
    alpha: float,
    sync_window: float,
) -> float:
    """根据 AssignmentPlan 的任务链累计到达时间计算协同时间窗惩罚。"""
    target_arrival_times: Dict[int, List[float]] = defaultdict(list)

    for sequence in plan.uav_task_sequences.values():
        evaluated = evaluate_uav_task_sequence(
            battlefield,
            sequence,
            alpha=alpha,
        )
        for task in evaluated.evaluated_sequence.tasks:
            target_arrival_times[task.target_id].append(task.planned_arrival_time)

    return cooperative_time_window_penalty_from_arrivals(
        target_arrival_times,
        alpha=alpha,
        sync_window=sync_window,
    )


def _objective_terms(
    distance_cost: float,
    threat_cost: float,
    time_window_penalty: float,
    task_reward: float,
    weights: dict,
) -> tuple[float, float, float, float]:
    """根据权重配置计算目标函数四个加权项，可选启用归一化。"""
    objective_refs = weights.get('objective_refs')
    if objective_refs:
        distance_cost = distance_cost / max(float(objective_refs['distance_ref']), 1e-12)
        threat_cost = threat_cost / max(float(objective_refs['threat_ref']), 1e-12)
        time_window_penalty = time_window_penalty / max(float(objective_refs['time_window_ref']), 1e-12)
        task_reward = task_reward / max(float(objective_refs['reward_ref']), 1e-12)

    return (
        weights['w1'] * distance_cost,
        weights['w2'] * threat_cost,
        weights['w3'] * time_window_penalty,
        -weights['w4'] * task_reward,
    )


# ============================================================
# 适应度评估（含惩罚函数）
# ============================================================

def evaluate_fitness(particle: np.ndarray, battlefield: Battlefield,
                     weights: dict, slot_to_target: List[int]) -> float:
    """
    计算粒子的适应度值

    适应度 = 目标函数值 + 惩罚项
    惩罚项通过对违反约束的程度施加大惩罚值（PENALTY=1e6），
    使不可行解在竞争中自然被淘汰，引导粒子群飞向可行解空间。
    相比"违反约束就回退"的硬修复策略，惩罚函数方法：
    - 保留了不可行解携带的部分有用信息（方向性）
    - 允许粒子在不可行域边界探索，可能找到更好的可行解
    - 实现更简洁，不需要额外的修复逻辑

    Parameters:
        particle: 粒子编码
        battlefield: 战场环境
        weights: 权重参数字典
        slot_to_target: 槽位到目标的映射

    Returns:
        适应度值（越小越好）
    """
    PENALTY = 1e6
    plan = decode_to_assignment_plan(particle, battlefield, slot_to_target)

    distance_cost = 0.0
    threat_cost = 0.0
    time_window_penalty = 0.0
    task_reward = 0.0
    constraint_penalty = 0.0
    target_arrival_times: Dict[int, List[float]] = defaultdict(list)

    assigned_target_ids = set()
    for sequence in plan.uav_task_sequences.values():
        uav = battlefield.get_uav(sequence.uav_id)
        evaluated = evaluate_uav_task_sequence(
            battlefield,
            sequence,
            alpha=weights['alpha'],
        )
        distance_cost += evaluated.total_distance
        time_window_penalty += evaluated.time_window_penalty
        for evaluated_task in evaluated.evaluated_sequence.tasks:
            target_arrival_times[evaluated_task.target_id].append(evaluated_task.planned_arrival_time)

        if not evaluated.is_ammo_feasible:
            constraint_penalty += PENALTY * (sequence.task_count() - uav.ammo)
        if not evaluated.is_range_feasible:
            constraint_penalty += PENALTY * (evaluated.total_distance - uav.range_left)

        current_x, current_y = uav.x, uav.y
        for task in sequence.tasks:
            target = battlefield.get_target(task.target_id)
            threat_cost += battlefield.threat_cost_on_line(
                current_x,
                current_y,
                target.x,
                target.y,
            )
            current_x, current_y = target.x, target.y
            assigned_target_ids.add(task.target_id)

    for target_id in assigned_target_ids:
        task_reward += battlefield.get_target(target_id).value

    # 饱和攻击惩罚：检查每个目标是否分配到了足够数量的不同无人机
    target_uav_sets: dict[int, set[int]] = defaultdict(set)
    for d, target_id in enumerate(slot_to_target):
        target_uav_sets[target_id].add(int(particle[d]))
    for target in battlefield.targets:
        actual = len(target_uav_sets[target.id])
        if actual < target.required_uavs:
            # 不足的无人机数量越多，惩罚越大
            constraint_penalty += PENALTY * (target.required_uavs - actual)

    time_window_penalty += cooperative_time_window_penalty_from_arrivals(
        target_arrival_times,
        alpha=weights['alpha'],
        sync_window=weights.get('sync_window', 0.05),
    )

    objective_terms = _objective_terms(
        distance_cost,
        threat_cost,
        time_window_penalty,
        task_reward,
        weights,
    )
    return sum(objective_terms) + constraint_penalty


# ============================================================
# 离散PSO主算法
# ============================================================
# 标准PSO的速度-位置更新公式适用于连续实数空间：
#   v_new = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
#   x_new = x + v_new
#
# 但任务分配是离散问题（无人机编号是整数），不能直接相加。
# 解决方案：Sigmoid概率映射（Kennedy & Eberhart, 1997）
#
# 核心思想：
# 1) 速度仍按连续公式计算，保留PSO的搜索动力学特性
# 2) 将速度通过Sigmoid函数 σ(v)=1/(1+e^{-v}) 映射为概率
# 3) 以该概率决定每个维度的更新行为（保留/学pbest/学gbest/随机探索）
#
# 为什么这样做有效：
# - 速度绝对值大 → σ(v)接近0或1 → 高确定性更新 → 快速收敛
# - 速度绝对值小 → σ(v)接近0.5 → 近似随机 → 保持多样性
# - 惯性权重w控制上一代速度的"记忆"程度，余弦递减实现前期探索、后期收敛
# - 认知项c1使粒子记住自身历史最优，社会项c2使粒子学习群体最优

def run_pso(battlefield: Battlefield, weights: dict,
            pso_params: dict = None,
            init_method: str = 'logistic',
            inertia_strategy: str = 'cosine',
            return_initial_population: bool = False,
            return_diagnostics: bool = False,
            return_assignment_plan: bool = False) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    运行改进PSO算法求解任务预分配

    Parameters:
        battlefield: 战场环境
        weights: 目标函数权重
        pso_params: PSO算法参数，为None时使用config中的默认值

    Returns:
        best_assignment: N×M 最优分配矩阵
        best_etas: 各无人机到分配目标的预计到达时间矩阵（N×M）
        convergence_curve: 每代全局最优适应度值列表
        initial_population: 初始粒子种群（仅当 return_initial_population=True 时返回）
    """
    if pso_params is None:
        pso_params = PSO_PARAMS

    num_uavs = len(battlefield.uavs)
    num_targets = len(battlefield.targets)
    num_particles = pso_params['num_particles']
    max_iter = pso_params['max_iter']
    w_start = pso_params['w_start']
    w_end = pso_params['w_end']
    c1 = pso_params['c1']
    c2 = pso_params['c2']
    uav_capacities = [uav.ammo for uav in battlefield.uavs]

    # 构建槽位映射
    total_slots, slot_to_target = build_slot_mapping(battlefield)
    _validate_capacities(total_slots, num_uavs, uav_capacities)

    # === 第1步：初始化种群 ===
    if init_method == 'logistic':
        positions = logistic_init(num_particles, total_slots, num_uavs, uav_capacities)
    elif init_method == 'random':
        positions = random_init(num_particles, total_slots, num_uavs, uav_capacities)
    else:
        raise ValueError(f"Unsupported init_method: {init_method}")

    initial_population = positions.copy()

    # 初始化速度矩阵（连续值，初始为0）
    velocities = np.zeros((num_particles, total_slots), dtype=float)

    # === 第2步：评估初始适应度 ===
    fitness = np.array([
        evaluate_fitness(positions[p], battlefield, weights, slot_to_target)
        for p in range(num_particles)
    ])
    initial_best_fitness = float(np.min(fitness))
    initial_mean_fitness = float(np.mean(fitness))
    initial_infeasible_count = int(np.sum(fitness >= 1e6))

    # === 第3步：初始化个体最优和全局最优 ===
    pbest_positions = positions.copy()
    pbest_fitness = fitness.copy()

    gbest_idx = np.argmin(fitness)
    gbest_position = positions[gbest_idx].copy()
    gbest_fitness = fitness[gbest_idx]

    convergence_curve = [gbest_fitness]

    # === 第4步：迭代寻优 ===
    for t in range(max_iter):
        # (a) 计算当前迭代的惯性权重
        if inertia_strategy == 'cosine':
            w = cosine_weight(t, max_iter, w_start, w_end)
        elif inertia_strategy == 'linear':
            w = linear_weight(t, max_iter, w_start, w_end)
        else:
            raise ValueError(f"Unsupported inertia_strategy: {inertia_strategy}")

        for p in range(num_particles):
            r1 = np.random.rand(total_slots)
            r2 = np.random.rand(total_slots)

            # (b) 速度更新（连续计算）
            # 惯性项：保留上一代速度的方向记忆
            # 认知项：向自身历史最优位置学习
            # 社会项：向全局最优位置学习
            velocities[p] = (w * velocities[p]
                             + c1 * r1 * (pbest_positions[p] - positions[p])
                             + c2 * r2 * (gbest_position - positions[p]))

            # 限制速度范围，防止Sigmoid饱和导致概率恒为0或1
            velocities[p] = np.clip(velocities[p], -4.0, 4.0)

            # (c) Sigmoid概率映射 → 基于容量修复的离散更新
            probs = 1.0 / (1.0 + np.exp(-velocities[p]))

            for d in range(total_slots):
                rand_val = np.random.rand()
                threshold_keep = w * probs[d]
                threshold_pbest = threshold_keep + 0.3 * (1 - threshold_keep)
                threshold_gbest = 0.9

                if rand_val < threshold_keep:
                    continue
                elif rand_val < threshold_pbest:
                    positions[p, d] = int(pbest_positions[p, d])
                elif rand_val < threshold_gbest:
                    positions[p, d] = int(gbest_position[d])
                else:
                    positions[p, d] = np.random.randint(0, num_uavs)

            positions[p] = repair_capacity(positions[p], uav_capacities)

            # (d) 评估适应度（含惩罚函数）
            fit = evaluate_fitness(positions[p], battlefield, weights,
                                   slot_to_target)
            fitness[p] = fit

            # (e) 更新个体最优
            if fit < pbest_fitness[p]:
                pbest_fitness[p] = fit
                pbest_positions[p] = positions[p].copy()

        # (e) 更新全局最优
        gen_best_idx = np.argmin(pbest_fitness)
        if pbest_fitness[gen_best_idx] < gbest_fitness:
            gbest_fitness = pbest_fitness[gen_best_idx]
            gbest_position = pbest_positions[gen_best_idx].copy()

        # (f) 记录收敛曲线
        convergence_curve.append(gbest_fitness)

    # === 第5步：输出结果 ===
    best_plan = decode_to_assignment_plan(
        gbest_position,
        battlefield,
        slot_to_target,
        total_cost=float(convergence_curve[-1]),
    )
    best_assignment = best_plan.to_assignment_matrix(num_uavs, num_targets)
    best_etas = assignment_plan_to_eta_matrix(battlefield, best_plan)

    diagnostics = {
        'init_method': init_method,
        'inertia_strategy': inertia_strategy,
        'initial_best_fitness': initial_best_fitness,
        'initial_mean_fitness': initial_mean_fitness,
        'initial_infeasible_count': initial_infeasible_count,
        'final_best_fitness': float(convergence_curve[-1]),
    }

    outputs = [best_assignment, best_etas, convergence_curve]
    if return_assignment_plan:
        outputs.append(best_plan)
    if return_initial_population:
        outputs.append(initial_population)
    if return_diagnostics:
        outputs.append(diagnostics)

    return tuple(outputs)
