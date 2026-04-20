"""高难度预分配测试场景：24架无人机、12个目标、8个威胁区"""
from src.core.models import UAV, Target, Threat, Battlefield
from config.params import UAV_MAX_RANGE, UAV_MAX_AMMO, UAV_SPEED, MAP_SIZE


def create_hard_scenario() -> Battlefield:
    """
    专用于放大PSO改进效果的高难度场景。

    设计原则：
    1. 目标数量增加到12个，总需求为24，与无人机数量相同，形成较强资源耦合
    2. 威胁区数量增加到8个，并布设在主要飞行通道附近，增强路径风险差异
    3. 目标位置分布更加不规则，避免对称结构过强导致多方法轻易收敛到同一方案
    """
    uavs = [
        UAV(id=i, x=8, y=6 + i * 3.8, speed=UAV_SPEED,
            ammo=UAV_MAX_AMMO, range_left=UAV_MAX_RANGE)
        for i in range(24)
    ]

    targets = [
        Target(id=0, x=72, y=12, value=8.5, required_uavs=2),
        Target(id=1, x=84, y=18, value=10.0, required_uavs=2),
        Target(id=2, x=78, y=28, value=7.5, required_uavs=2),
        Target(id=3, x=90, y=34, value=9.5, required_uavs=2),
        Target(id=4, x=68, y=46, value=8.0, required_uavs=2),
        Target(id=5, x=86, y=52, value=10.0, required_uavs=2),
        Target(id=6, x=74, y=62, value=8.5, required_uavs=2),
        Target(id=7, x=92, y=68, value=9.0, required_uavs=2),
        Target(id=8, x=80, y=76, value=7.5, required_uavs=2),
        Target(id=9, x=70, y=84, value=9.5, required_uavs=2),
        Target(id=10, x=88, y=90, value=8.0, required_uavs=2),
        Target(id=11, x=76, y=96, value=7.0, required_uavs=2),
    ]

    threats = [
        Threat(id=0, x=34, y=16, radius=10.0),
        Threat(id=1, x=58, y=36, radius=10.0),
        Threat(id=2, x=56, y=58, radius=11.0),
        Threat(id=3, x=44, y=70, radius=12.0),
        Threat(id=4, x=60, y=82, radius=10.5),
        Threat(id=5, x=52, y=92, radius=11.5),
    ]

    return Battlefield(uavs, targets, threats, MAP_SIZE)
