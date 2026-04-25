import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class UAV:
    id: int
    x: float
    y: float
    speed: float
    ammo: int
    range_left: float

    def distance_to(self, x: float, y: float) -> float:
        return math.hypot(self.x - x, self.y - y)

    def eta_to(self, x: float, y: float) -> float:
        return self.distance_to(x, y) / self.speed


@dataclass
class Target:
    id: int
    x: float
    y: float
    value: float
    required_uavs: int = 1  # 需要分配的无人机数量（饱和攻击）


@dataclass
class Threat:
    id: int
    x: float
    y: float
    radius: float

    def threat_cost_at(self, x: float, y: float) -> float:
        """二次威胁模型：越靠近威胁中心代价越大，半径外为0。"""
        dist = math.hypot(self.x - x, self.y - y)
        if dist >= self.radius:
            return 0.0
        s = 1.0 - dist / self.radius
        return 40.0 * (s ** 2)              # 威胁代价最大为40，且呈二次衰减


@dataclass(frozen=True)
class TaskNode:
    """任务序列中的单个目标节点。"""

    target_id: int
    order: int
    planned_arrival_time: float = 0.0
    planned_service_time: float = 0.0
    estimated_path_length: float = 0.0


@dataclass
class UavTaskSequence:
    """单架无人机的串行任务序列。"""

    uav_id: int
    tasks: List[TaskNode] = field(default_factory=list)

    def target_ids(self) -> List[int]:
        return [task.target_id for task in self.tasks]

    def task_count(self) -> int:
        return len(self.tasks)

    def append_target(self, target_id: int) -> None:
        self.tasks.append(TaskNode(target_id=target_id, order=len(self.tasks)))


@dataclass
class UavExecutionState:
    """重分配阶段需要的单机执行进度。"""

    uav_id: int
    completed_task_count: int = 0
    current_position: Tuple[float, float] = (0.0, 0.0)
    current_time: float = 0.0
    remaining_ammo: int = 0
    remaining_range: float = 0.0


@dataclass
class AssignmentPlan:
    """可表达任务序列的统一分配结果容器。"""

    uav_task_sequences: Dict[int, UavTaskSequence]
    target_assignees: Dict[int, List[int]]
    total_cost: float = 0.0

    @classmethod
    def empty(cls, uav_ids: List[int]) -> "AssignmentPlan":
        return cls(
            uav_task_sequences={uav_id: UavTaskSequence(uav_id=uav_id) for uav_id in uav_ids},
            target_assignees={},
            total_cost=0.0,
        )

    @classmethod
    def from_assignment_matrix(cls, assignment: np.ndarray) -> "AssignmentPlan":
        """兼容旧的 N×M 分配矩阵表示。"""
        num_uavs, num_targets = assignment.shape
        sequences = {
            uav_id: UavTaskSequence(
                uav_id=uav_id,
                tasks=[
                    TaskNode(target_id=target_id, order=order)
                    for order, target_id in enumerate(np.where(assignment[uav_id] == 1)[0].tolist())
                ],
            )
            for uav_id in range(num_uavs)
        }

        target_assignees: Dict[int, List[int]] = {}
        for target_id in range(num_targets):
            assigned_uavs = np.where(assignment[:, target_id] == 1)[0].tolist()
            if assigned_uavs:
                target_assignees[target_id] = assigned_uavs

        return cls(
            uav_task_sequences=sequences,
            target_assignees=target_assignees,
            total_cost=0.0,
        )

    def to_assignment_matrix(self, num_uavs: int, num_targets: int) -> np.ndarray:
        """将任务序列结果投影回兼容旧逻辑的分配矩阵。"""
        assignment = np.zeros((num_uavs, num_targets), dtype=int)
        for sequence in self.uav_task_sequences.values():
            for task in sequence.tasks:
                if 0 <= sequence.uav_id < num_uavs and 0 <= task.target_id < num_targets:
                    assignment[sequence.uav_id, task.target_id] = 1
        return assignment


class Battlefield:
    def __init__(self, uavs: List[UAV], targets: List[Target],
                 threats: List[Threat], map_size: Tuple[float, float]):
        self.uavs = uavs
        self.targets = targets
        self.threats = threats
        self.map_size = map_size

    def distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        return math.hypot(x1 - x2, y1 - y2)

    def threat_cost_on_line(self, x1: float, y1: float,
                            x2: float, y2: float,
                            num_samples: int = 20) -> float:
        """沿直线采样计算威胁代价积分（用于PSO/MCHA阶段的预估）"""
        total = 0.0
        for k in range(num_samples + 1):
            t = k / num_samples
            px = x1 + t * (x2 - x1)
            py = y1 + t * (y2 - y1)
            for threat in self.threats:
                total += threat.threat_cost_at(px, py)
        seg_len = self.distance(x1, y1, x2, y2)
        return total * seg_len / (num_samples + 1)

    def get_uav(self, uav_id: int) -> UAV:
        for u in self.uavs:
            if u.id == uav_id:
                return u
        raise ValueError(f"UAV {uav_id} not found")

    def get_target(self, target_id: int) -> Target:
        for t in self.targets:
            if t.id == target_id:
                return t
        raise ValueError(f"Target {target_id} not found")
