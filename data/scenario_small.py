"""小规模任务序列测试场景：3架无人机、5个目标、3个威胁区"""
from src.core.models import UAV, Target, Threat, Battlefield
from config.params import UAV_MAX_RANGE, UAV_MAX_AMMO, UAV_SPEED, MAP_SIZE


def create_small_scenario() -> Battlefield:
    uavs = [
        UAV(id=0, x=10, y=10, speed=UAV_SPEED, ammo=UAV_MAX_AMMO, range_left=UAV_MAX_RANGE),
        UAV(id=1, x=10, y=45, speed=UAV_SPEED, ammo=UAV_MAX_AMMO, range_left=UAV_MAX_RANGE),
        UAV(id=2, x=10, y=80, speed=UAV_SPEED, ammo=UAV_MAX_AMMO, range_left=UAV_MAX_RANGE),
    ]

    targets = [
        Target(id=0, x=80, y=20, value=8.0, required_uavs=1),
        Target(id=1, x=85, y=45, value=10.0, required_uavs=1),
        Target(id=2, x=75, y=60, value=6.0, required_uavs=1),
        Target(id=3, x=90, y=75, value=9.0, required_uavs=1),
        Target(id=4, x=80, y=90, value=7.0, required_uavs=1),
    ]

    threats = [
        Threat(id=0, x=45, y=30, radius=12.0),
        Threat(id=1, x=55, y=55, radius=10.0),
        Threat(id=2, x=50, y=80, radius=15.0),
    ]

    return Battlefield(uavs, targets, threats, MAP_SIZE)
