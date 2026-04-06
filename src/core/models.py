import math
from dataclasses import dataclass, field
from typing import List, Tuple
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
