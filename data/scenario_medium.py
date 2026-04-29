"""中规模任务序列论文场景：10架无人机、20个目标、5个威胁区"""
from src.core.models import UAV, Target, Threat, Battlefield
from config.params import UAV_MAX_RANGE, UAV_MAX_AMMO, UAV_SPEED, MAP_SIZE


def create_medium_scenario() -> Battlefield:
    uavs = [
        UAV(id=i, x=10, y=12 + i * 8, speed=UAV_SPEED,
            ammo=UAV_MAX_AMMO, range_left=UAV_MAX_RANGE)
        for i in range(10)
    ]

    targets = [
        Target(id=0, x=72, y=10, value=8.0, required_uavs=1),
        Target(id=1, x=84, y=16, value=10.0, required_uavs=2),
        Target(id=2, x=78, y=24, value=7.0, required_uavs=1),
        Target(id=3, x=90, y=30, value=9.0, required_uavs=1),
        Target(id=4, x=90, y=55, value=6.0, required_uavs=1),
        Target(id=5, x=74, y=40, value=8.5, required_uavs=1),
        Target(id=6, x=86, y=46, value=10.0, required_uavs=2),
        Target(id=7, x=80, y=52, value=7.5, required_uavs=1),
        Target(id=8, x=70, y=58, value=9.0, required_uavs=1),
        Target(id=9, x=90, y=95, value=6.5, required_uavs=1),
        Target(id=10, x=76, y=64, value=8.0, required_uavs=1),
        Target(id=11, x=88, y=70, value=9.5, required_uavs=2),
        Target(id=12, x=72, y=76, value=7.0, required_uavs=1),
        Target(id=13, x=84, y=82, value=8.5, required_uavs=1),
        Target(id=14, x=94, y=88, value=7.5, required_uavs=1),
        Target(id=15, x=68, y=92, value=9.0, required_uavs=1),
        Target(id=16, x=66, y=32, value=6.5, required_uavs=1),
        Target(id=17, x=72, y=68, value=8.0, required_uavs=1),
        Target(id=18, x=92, y=42, value=7.5, required_uavs=1),
        Target(id=19, x=82, y=90, value=8.5, required_uavs=1),
    ]

    threats = [
        Threat(id=0, x=40, y=20, radius=8.0),
        Threat(id=1, x=55, y=38, radius=9.0),
        Threat(id=2, x=35, y=55, radius=10.0),
        Threat(id=3, x=60, y=70, radius=9.0),
        Threat(id=4, x=50, y=90, radius=8.0),
    ]

    return Battlefield(uavs, targets, threats, MAP_SIZE)
