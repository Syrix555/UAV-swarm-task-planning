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
from typing import Tuple, List
from src.core.models import Battlefield
from src.core.objective import objective_function
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

def logistic_init(num_particles: int, dim: int, num_uavs: int) -> np.ndarray:
    """
    使用Logistic混沌映射生成初始种群。

    针对当前 ammo=1、总槽位数不超过无人机数量的场景，
    这里不再将每个槽位独立映射为无人机编号，而是先生成一组
    长度为 num_uavs 的混沌序列，再按序列值排序得到一个无人机排列，
    最后取前 dim 个元素作为粒子编码。这样可以显著降低初始粒子
    中“同一无人机重复占据多个槽位”导致的不可行性。

    Parameters:
        num_particles: 粒子数量
        dim: 粒子维度（等于所有目标的 required_uavs 之和）
        num_uavs: 无人机数量N

    Returns:
        shape=(num_particles, dim) 的整数数组，每个元素为无人机编号 [0, N-1]
    """
    if dim > num_uavs:
        raise ValueError('当前排列型初始化要求 dim <= num_uavs，请检查 ammo 或目标需求配置')

    fixed_points = [0.0, 0.25, 0.5, 0.75, 1.0]
    z = np.random.rand()
    while any(abs(z - fp) < 0.01 for fp in fixed_points):
        z = np.random.rand()

    particles = []
    for _ in range(num_particles):
        chaotic_values = []
        for uav_id in range(num_uavs):
            z = 4.0 * z * (1 - z)
            chaotic_values.append((z, uav_id))

        chaotic_values.sort(key=lambda item: item[0])
        row = [uav_id for _, uav_id in chaotic_values[:dim]]
        particles.append(row)

    return np.array(particles, dtype=int)


def random_init(num_particles: int, dim: int, num_uavs: int) -> np.ndarray:
    """使用无放回随机排列方式生成初始种群。"""
    if dim > num_uavs:
        raise ValueError('当前排列型初始化要求 dim <= num_uavs，请检查 ammo 或目标需求配置')

    particles = []
    for _ in range(num_particles):
        row = np.random.permutation(num_uavs)[:dim]
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
    num_uavs = len(battlefield.uavs)
    num_targets = len(battlefield.targets)
    assignment = decode(particle, num_uavs, num_targets, slot_to_target)

    # 基础目标函数值
    fitness = objective_function(assignment, battlefield, weights)

    # 惩罚项
    PENALTY = 1e6

    # 饱和攻击惩罚：检查每个目标是否分配到了足够数量的不同无人机
    # 从粒子编码中直接统计每个目标的不同无人机数（而非从assignment矩阵，
    # 因为assignment矩阵会将重复分配折叠为一个1）
    from collections import defaultdict
    target_uav_sets = defaultdict(set)
    for d, target_id in enumerate(slot_to_target):
        target_uav_sets[target_id].add(int(particle[d]))
    for target in battlefield.targets:
        actual = len(target_uav_sets[target.id])
        if actual < target.required_uavs:
            # 不足的无人机数量越多，惩罚越大
            fitness += PENALTY * (target.required_uavs - actual)

    # 无人机自身约束惩罚
    for i, uav in enumerate(battlefield.uavs):
        assigned_targets = np.where(assignment[i] == 1)[0]

        # 弹药超限惩罚：分配的目标数超过实时弹药量
        if len(assigned_targets) > uav.ammo:
            fitness += PENALTY * (len(assigned_targets) - uav.ammo)

        # 航程超限惩罚：到各目标的直线距离之和超过剩余航程
        if len(assigned_targets) > 0:
            total_dist = sum(
                uav.distance_to(battlefield.targets[j].x,
                                battlefield.targets[j].y)
                for j in assigned_targets
            )
            if total_dist > uav.range_left:
                fitness += PENALTY * (total_dist - uav.range_left)

    return fitness


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
            return_diagnostics: bool = False) -> Tuple[np.ndarray, np.ndarray, List[float]]:
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

    # 构建槽位映射
    total_slots, slot_to_target = build_slot_mapping(battlefield)

    # === 第1步：初始化种群 ===
    if init_method == 'logistic':
        positions = logistic_init(num_particles, total_slots, num_uavs)
    elif init_method == 'random':
        positions = random_init(num_particles, total_slots, num_uavs)
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

            # (c) Sigmoid概率映射 → 基于交换的排列更新
            probs = 1.0 / (1.0 + np.exp(-velocities[p]))

            for d in range(total_slots):
                rand_val = np.random.rand()
                threshold_keep = w * probs[d]
                threshold_pbest = threshold_keep + 0.3 * (1 - threshold_keep)
                threshold_gbest = 0.9

                if rand_val < threshold_keep:
                    continue
                elif rand_val < threshold_pbest:
                    swap_to_match(positions[p], d, int(pbest_positions[p, d]))
                elif rand_val < threshold_gbest:
                    swap_to_match(positions[p], d, int(gbest_position[d]))
                else:
                    swap_idx = np.random.randint(0, total_slots)
                    positions[p, d], positions[p, swap_idx] = positions[p, swap_idx], positions[p, d]

            positions[p] = repair_permutation(positions[p], num_uavs)

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
    best_assignment = decode(gbest_position, num_uavs, num_targets,
                             slot_to_target)

    # 计算ETA矩阵（无人机i到目标j的预计到达时间）
    best_etas = np.zeros((num_uavs, num_targets))
    for i, uav in enumerate(battlefield.uavs):
        for j, target in enumerate(battlefield.targets):
            if best_assignment[i, j] == 1:
                best_etas[i, j] = uav.eta_to(target.x, target.y)

    diagnostics = {
        'init_method': init_method,
        'inertia_strategy': inertia_strategy,
        'initial_best_fitness': initial_best_fitness,
        'initial_mean_fitness': initial_mean_fitness,
        'initial_infeasible_count': initial_infeasible_count,
        'final_best_fitness': float(convergence_curve[-1]),
    }

    if return_initial_population and return_diagnostics:
        return best_assignment, best_etas, convergence_curve, initial_population, diagnostics

    if return_initial_population:
        return best_assignment, best_etas, convergence_curve, initial_population

    if return_diagnostics:
        return best_assignment, best_etas, convergence_curve, diagnostics

    return best_assignment, best_etas, convergence_curve
