# ISSUES

## 1

  现在 available_uavs 在 decrease 场景下仍然会打印成全体 UAV：

  available_uavs: [0, 1, 2, ..., 23]

  这不影响正确性，但在“纯裁剪事件”里它的解释价值不大。
  不过这已经属于展示优化问题了，不是功能问题。

## 2

1. 矩阵版 TARGET_ADDED 只适合单个新目标，而且最好：

- required_uavs = 1
- id 紧接旧目标数

2. 还没做多目标同时新增
3. 任务序列版 TARGET_ADDED 已支持单个新目标的 required_uavs > 1，但矩阵版和可视化流程仍需继续验证
4. 还没做“删目标后再加目标”的复杂编号问题
5. 任务序列版 TARGET_ADDED 目前仍要求新增目标 id 连续，即 target.id == len(battlefield.targets) - 1
6. 任务序列版暂不支持非连续目标 id 映射、多目标同时新增、目标删除后复用编号等复杂场景

## 3

  预分配已经改成任务序列语义后，PSO 的代价评估按任务链累计计算：

  UAV 起点 -> T1 -> T2 -> T3

  也就是说，第二个任务的距离、威胁和到达时间应基于 T1 -> T2，而不是 UAV 起点 -> T2。

  但当前预分配可视化仍然使用旧的分配矩阵画法：

  UAV 起点 -> T1
  UAV 起点 -> T2
  UAV 起点 -> T3

  这会导致一架 UAV 执行多个任务时，图上从 UAV 起点发散出多条虚线，和真实任务序列不一致。

  后续应新增基于 AssignmentPlan / UavTaskSequence 的任务链绘图函数，例如：

  - UAV 起点连接到第一个任务
  - 第一个任务连接到第二个任务
  - 第二个任务连接到第三个任务

  旧的矩阵连线函数可以保留给重分配旧流程或对比图使用。

## 4

  当前任务序列版重分配还不是真正的“任务执行过程中”的实时重分配。

  目前实现更接近：

  预分配得到完整 AssignmentPlan
  -> 事件发生
  -> 在完整任务序列方案上做局部修复

  但还没有真正接入 UavExecutionState 中的执行进度信息：

  - completed_task_count
  - current_position
  - current_time
  - remaining_ammo
  - remaining_range

  因此当前 UAV_LOST 任务序列版采用的是“整链释放”：

  UAV-0: [T2, T5, T9]
  -> UAV-0 损失
  -> 释放 [T2, T5, T9]

  还没有实现更真实的“只释放未完成任务”：

  UAV-0: [T2, T5, T9]
  completed_task_count = 1
  -> 保留已完成 T2
  -> 释放未完成 [T5, T9]

  后续如果论文或演示要强调任务执行过程中的动态重分配，需要增加一个执行态扩展阶段：

  - 基于 UavExecutionState 更新每架 UAV 的当前位置、当前时间、剩余弹药和剩余航程
  - 事件分析只处理未完成任务链
  - MCHA 从 UAV 当前状态出发评估链尾追加代价
  - THREAT_ADDED 只检查未完成航段，而不是完整历史任务链

  在此之前，论文表述应避免写成已经完整实现“执行过程中实时重规划”，更准确的说法是“任务序列方案层面的事件触发式局部重分配”。
