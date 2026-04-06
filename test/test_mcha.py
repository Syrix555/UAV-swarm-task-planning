"""
MCHA重分配测试脚本
支持 UAV_LOST 与 THREAT_ADDED 两类事件测试。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from config.params import WEIGHTS, MCHA, MCHA_TEST
from data.scenario_reallocation import create_reallocation_scenario
from src.core.models import Threat
from src.pre_allocation.pso import run_pso
from src.re_allocation.mcha import run_mcha
from src.re_allocation.events import (
    Event,
    EventType,
    analyze_event_impact,
    apply_event_to_battlefield,
)


def print_assignment(title, battlefield, assignment):
    print(f"\n=== {title} ===")
    for target_id in range(assignment.shape[1]):
        assigned_uavs = np.where(assignment[:, target_id] == 1)[0].tolist()
        if target_id < len(battlefield.targets):
            target = battlefield.targets[target_id]
            print(
                f"目标{target_id} (value={target.value}, need={target.required_uavs}) "
                f"<- {assigned_uavs}"
            )
        else:
            print(f"目标{target_id} <- {assigned_uavs}")


def print_state(state):
    print("\n=== 事件分析结果 ===")
    print(f"open_targets: {state.open_targets}")
    print(f"available_uavs: {state.available_uavs}")
    print(f"remaining_demand: {state.remaining_demand}")
    print("locked_assignment:")
    print(state.locked_assignment)


def print_assignment_changes(before_assignment, after_assignment):
    print("\n=== 分配变化 ===")
    changes = []
    num_uavs, num_targets = before_assignment.shape
    for uav_id in range(num_uavs):
        for target_id in range(num_targets):
            before = int(before_assignment[uav_id, target_id])
            after = int(after_assignment[uav_id, target_id])
            if before != after:
                changes.append((uav_id, target_id, before, after))
                print(f"UAV-{uav_id} -> Target-{target_id}: {before} -> {after}")

    if not changes:
        print("无变化")
    return changes


def print_summary(event, state, result, battlefield, changes):
    print("\n=== 摘要统计 ===")
    print(f"事件类型: {event.type.value}")
    print(f"受影响目标数: {len(state.open_targets)}")
    print(f"可参与重分配无人机数: {len(state.available_uavs)}")
    print(f"中标次数: {len(result.selected_bids)}")
    print(f"分配变化数: {len(changes)}")

    recovered_targets = sum(1 for value in result.remaining_demand.values() if value == 0)
    total_open_targets = len(state.remaining_demand)
    print(f"已恢复目标数: {recovered_targets}/{total_open_targets}")

    satisfied_targets = 0
    for target in battlefield.targets:
        assigned_count = int(np.sum(result.assignment[:, target.id]))
        if assigned_count >= target.required_uavs:
            satisfied_targets += 1
    print(f"满足需求目标数: {satisfied_targets}/{len(battlefield.targets)}")


def build_uav_lost_event(lost_uav_id):
    return Event(
        type=EventType.UAV_LOST,
        data={"uav_id": lost_uav_id},
    )


def build_threat_added_event():
    new_threat = Threat(
        id=MCHA_TEST['new_threat_id'],
        x=MCHA_TEST['new_threat_x'],
        y=MCHA_TEST['new_threat_y'],
        radius=MCHA_TEST['new_threat_radius'],
    )
    return Event(
        type=EventType.THREAT_ADDED,
        data={
            "threat": new_threat,
            "threat_threshold": MCHA_TEST['threat_threshold'],
        },
    )


def describe_event(event):
    if event.type == EventType.UAV_LOST:
        return f"模拟 UAV-{event.data['uav_id']} 损失"
    if event.type == EventType.THREAT_ADDED:
        threat = event.data['threat']
        return (
            f"模拟新增威胁 Threat-{threat.id} at ({threat.x}, {threat.y}), "
            f"radius={threat.radius}"
        )
    return f"模拟事件 {event.type.value}"


def main():
    battlefield = create_reallocation_scenario()

    print("正在运行PSO预分配...")
    assignment, etas, curve = run_pso(battlefield, WEIGHTS)
    print(f"PSO完成，最终适应度: {curve[-1]:.4f}")
    print_assignment("PSO初始分配", battlefield, assignment)

    event_mode = os.environ.get("MCHA_EVENT", MCHA_TEST['default_event']).strip().lower()
    if event_mode == "threat_added":
        event = build_threat_added_event()
    else:
        event = build_uav_lost_event(MCHA_TEST['lost_uav_id'])

    print("\n=== 事件参数 ===")
    if event.type == EventType.UAV_LOST:
        print(f"lost_uav_id: {event.data['uav_id']}")
    if event.type == EventType.THREAT_ADDED:
        threat = event.data['threat']
        print(f"threat_id: {threat.id}")
        print(f"threat_center: ({threat.x}, {threat.y})")
        print(f"threat_radius: {threat.radius}")
        print(f"threat_threshold: {event.data['threat_threshold']}")

    apply_event_to_battlefield(event, battlefield)
    state = analyze_event_impact(event, battlefield, assignment, etas)
    print_state(state)

    print(f"\n正在执行MCHA重分配（{describe_event(event)}）...")
    result = run_mcha(battlefield, WEIGHTS, state, MCHA)

    print_assignment("MCHA重分配结果", battlefield, result.assignment)
    changes = print_assignment_changes(assignment, result.assignment)
    print("\n=== MCHA统计 ===")
    print(f"iterations: {result.iterations}")
    print(f"remaining_demand: {result.remaining_demand}")
    print("selected_bids:")
    for bid in result.selected_bids:
        print(f"  UAV-{bid.uav_id} -> Target-{bid.target_id}, score={bid.score:.4f}")

    print_summary(event, state, result, battlefield, changes)

    print("\n=== 结果校验 ===")
    if event.type == EventType.UAV_LOST:
        lost_uav_id = event.data['uav_id']
        print(f"损失无人机 UAV-{lost_uav_id} 是否已清空任务: {np.sum(result.assignment[lost_uav_id]) == 0}")
    if event.type == EventType.THREAT_ADDED:
        print(f"当前战场威胁区数量: {len(battlefield.threats)}")

    for target in battlefield.targets:
        assigned_count = int(np.sum(result.assignment[:, target.id]))
        print(
            f"目标{target.id}: assigned={assigned_count}, required={target.required_uavs}, "
            f"satisfied={assigned_count >= target.required_uavs}"
        )


if __name__ == '__main__':
    main()
