"""中规模任务序列重分配测试场景：9架无人机、18个目标、5个威胁区"""
from src.core.models import UAV, Target, Threat, Battlefield
from config.params import UAV_MAX_RANGE, UAV_MAX_AMMO, UAV_SPEED, MAP_SIZE


def create_reallocation_scenario() -> Battlefield:
    """
    专用于MCHA动态重分配实验的中规模场景。

    设计原则：
    1. 目标总需求为22，低于9架无人机的总 ammo 容量27，保留一定冗余能力
    2. 无人机数量少于目标数量，每架无人机可承担多个任务点，体现任务序列重分配
    3. 威胁区分布在中部与右侧通道，便于测试威胁变化触发重分配
    """
    uavs = [
        UAV(id=i, x=10, y=10 + i * 9, speed=UAV_SPEED,
            ammo=UAV_MAX_AMMO, range_left=UAV_MAX_RANGE)
        for i in range(9)
    ]

    targets = [
        Target(id=0, x=74, y=14, value=8.0, required_uavs=1),
        Target(id=1, x=82, y=24, value=10.0, required_uavs=3),
        Target(id=2, x=88, y=36, value=7.5, required_uavs=1),
        Target(id=3, x=70, y=48, value=9.0, required_uavs=1),
        Target(id=4, x=92, y=56, value=6.5, required_uavs=1),
        Target(id=5, x=76, y=64, value=8.5, required_uavs=1),
        Target(id=6, x=86, y=72, value=10.0, required_uavs=2),
        Target(id=7, x=80, y=82, value=7.0, required_uavs=1),
        Target(id=8, x=68, y=88, value=9.0, required_uavs=1),
        Target(id=9, x=90, y=94, value=6.0, required_uavs=1),
        Target(id=10, x=72, y=20, value=8.5, required_uavs=1),
        Target(id=11, x=88, y=30, value=9.5, required_uavs=2),
        Target(id=12, x=78, y=42, value=7.0, required_uavs=1),
        Target(id=13, x=84, y=52, value=8.0, required_uavs=1),
        Target(id=14, x=66, y=60, value=7.5, required_uavs=1),
        Target(id=15, x=94, y=70, value=8.5, required_uavs=1),
        Target(id=16, x=72, y=78, value=9.0, required_uavs=1),
        Target(id=17, x=82, y=90, value=7.5, required_uavs=1),
    ]

    threats = [
        Threat(id=0, x=38, y=20, radius=11.0),
        Threat(id=1, x=52, y=34, radius=10.0),
        Threat(id=2, x=46, y=54, radius=13.0),
        Threat(id=3, x=60, y=74, radius=11.0),
        Threat(id=4, x=50, y=88, radius=12.0),
    ]

    return Battlefield(uavs, targets, threats, MAP_SIZE)
