"""高难度任务序列预分配测试场景：12架无人机、28个目标、6个威胁区"""
from src.core.models import UAV, Target, Threat, Battlefield
from config.params import UAV_MAX_RANGE, UAV_MAX_AMMO, UAV_SPEED, MAP_SIZE


def create_hard_scenario() -> Battlefield:
    """
    专用于放大PSO改进效果的高难度场景。

    设计原则：
    1. 目标数量增加到28个，总需求为36，与12架无人机的总 ammo 容量相同，形成较强资源耦合
    2. 威胁区布设在主要飞行通道附近，增强路径风险差异
    3. 目标位置分布更加不规则，避免对称结构过强导致多方法轻易收敛到同一方案
    """
    uavs = [
        UAV(id=i, x=8, y=8 + i * 7.5, speed=UAV_SPEED,
            ammo=UAV_MAX_AMMO, range_left=UAV_MAX_RANGE)
        for i in range(12)
    ]

    targets = [
        Target(id=0, x=72, y=10, value=8.5, required_uavs=2),
        Target(id=1, x=84, y=14, value=10.0, required_uavs=1),
        Target(id=2, x=78, y=20, value=7.5, required_uavs=1),
        Target(id=3, x=92, y=24, value=9.5, required_uavs=2),
        Target(id=4, x=66, y=30, value=8.0, required_uavs=1),
        Target(id=5, x=86, y=36, value=10.0, required_uavs=1),
        Target(id=6, x=74, y=42, value=8.5, required_uavs=2),
        Target(id=7, x=94, y=48, value=9.0, required_uavs=1),
        Target(id=8, x=80, y=54, value=7.5, required_uavs=1),
        Target(id=9, x=68, y=60, value=9.5, required_uavs=2),
        Target(id=10, x=88, y=66, value=8.0, required_uavs=1),
        Target(id=11, x=76, y=72, value=7.0, required_uavs=1),
        Target(id=12, x=96, y=78, value=9.0, required_uavs=2),
        Target(id=13, x=70, y=84, value=8.5, required_uavs=1),
        Target(id=14, x=82, y=90, value=7.5, required_uavs=1),
        Target(id=15, x=90, y=96, value=8.0, required_uavs=2),
        Target(id=16, x=62, y=18, value=6.5, required_uavs=1),
        Target(id=17, x=58, y=28, value=7.0, required_uavs=1),
        Target(id=18, x=64, y=38, value=8.0, required_uavs=1),
        Target(id=19, x=60, y=50, value=7.5, required_uavs=1),
        Target(id=20, x=62, y=70, value=8.5, required_uavs=1),
        Target(id=21, x=58, y=80, value=7.0, required_uavs=1),
        Target(id=22, x=66, y=92, value=9.0, required_uavs=1),
        Target(id=23, x=98, y=34, value=8.0, required_uavs=2),
        Target(id=24, x=96, y=58, value=7.5, required_uavs=1),
        Target(id=25, x=94, y=86, value=8.5, required_uavs=1),
        Target(id=26, x=74, y=88, value=9.5, required_uavs=2),
        Target(id=27, x=54, y=64, value=6.5, required_uavs=1),
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
