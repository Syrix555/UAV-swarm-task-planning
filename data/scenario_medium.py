"""中规模论文场景：20架无人机、10个目标、5个威胁区"""
from src.core.models import UAV, Target, Threat, Battlefield
from config.params import UAV_MAX_RANGE, UAV_MAX_AMMO, UAV_SPEED, MAP_SIZE


def create_medium_scenario() -> Battlefield:
    uavs = [
        UAV(id=i, x=10, y=10 + i * 4, speed=UAV_SPEED,
            ammo=UAV_MAX_AMMO, range_left=UAV_MAX_RANGE)
        for i in range(20)
    ]

    targets = [
        Target(id=0, x=75, y=15, value=8.0, required_uavs=2),
        Target(id=1, x=80, y=25, value=10.0, required_uavs=3),
        Target(id=2, x=85, y=40, value=7.0, required_uavs=2),
        Target(id=3, x=70, y=50, value=9.0, required_uavs=2),
        Target(id=4, x=90, y=55, value=6.0, required_uavs=1),
        Target(id=5, x=75, y=65, value=8.5, required_uavs=2),
        Target(id=6, x=85, y=70, value=10.0, required_uavs=3),
        Target(id=7, x=80, y=80, value=7.5, required_uavs=2),
        Target(id=8, x=70, y=85, value=9.0, required_uavs=2),
        Target(id=9, x=90, y=95, value=6.5, required_uavs=1),
    ]

    threats = [
        Threat(id=0, x=40, y=20, radius=8.0),
        Threat(id=1, x=55, y=35, radius=9.0),
        Threat(id=2, x=45, y=55, radius=10.0),
        Threat(id=3, x=60, y=75, radius=9.0),
        Threat(id=4, x=50, y=90, radius=8.0),
    ]

    return Battlefield(uavs, targets, threats, MAP_SIZE)
