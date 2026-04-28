# CLAUDE.md

本文件用于给后续接手本仓库的 AI 编程助手提供长期上下文，避免每次上下文压缩后丢失本科毕设背景、当前实现边界和开发约定。

## 项目背景

本仓库是本科毕业设计项目，题目为：

> 面向打击任务的无人集群协同任务规划

项目定位是 Python 仿真项目，目标不是做复杂工程系统，而是服务本科毕设论文、实验图表和答辩演示。当前主链路应围绕以下内容展开：

1. 打击任务目标建模
2. 任务预分配算法
3. 任务重分配算法
4. 路径规划与仿真展示
5. 实验结果可视化

开发原则：

- 优先保证流程完整、可运行、可展示。
- 算法复杂度要能在论文中解释清楚。
- 不要为了炫技引入过复杂实现或新依赖。
- 命名和概念尽量贴合论文表达。
- 论文重点在预分配和重分配，路径规划属于全链路展示补充。
- 静态图、CSV 和可复现实验优先级高于复杂动画系统。

## 运行环境

本仓库使用 conda 环境运行，环境名为 `uav`。

常用命令格式：

```bash
env MPLBACKEND=Agg MPLCONFIGDIR=/tmp conda run -n uav python <script_or_test>
```

说明：

- `MPLBACKEND=Agg` 用于无 GUI 环境下生成 matplotlib 图片。
- `MPLCONFIGDIR=/tmp` 用于避免 matplotlib 配置目录权限问题。
- 脚本中出现 `FigureCanvasAgg is non-interactive, and thus cannot be shown` 通常是无害警告。

## 主要目录

- `config/`
  - 全局参数配置，例如地图、UAV、PSO、MCHA、A* 参数。
  - 关键文件：`config/params.py`。

- `data/`
  - 场景数据构造。
  - `scenario_small.py`、`scenario_medium.py`、`scenario_hard.py` 用于预分配场景。
  - `scenario_reallocation.py` 用于重分配场景。

- `src/core/`
  - 核心数据模型和任务序列评估。
  - 关键模型包括 `UAV`、`Target`、`Threat`、`AssignmentPlan`、`UavTaskSequence`、`TaskNode`、`UavExecutionState`。
  - `sequence_eval.py` 负责按任务链累计计算距离、到达时间、显式时间窗惩罚、弹药和航程可行性。

- `src/pre_allocation/`
  - PSO 预分配算法。
  - 当前预分配已经从“一架 UAV 只能执行一个目标”改为“每架 UAV 可按 ammo 执行任务序列”。
  - 关键文件：`src/pre_allocation/pso.py`。

- `src/re_allocation/`
  - 动态事件分析与 MCHA 重分配。
  - 关键文件：`events.py`、`mcha.py`。

- `src/route_planning/`
  - A* 路径规划、威胁栅格、路径简化、B 样条/运动学平滑。
  - 路径规划不是论文主要创新点，当前定位是支撑最终航迹展示。

- `src/visualization/`
  - 论文图和实验图绘制模块。
  - `preallocation.py`：预分配结果图、协同到达时间窗图、PSO 消融图、指标表等。
  - `reallocation.py`：重分配结果对比、MCHA 分析图、代价变化 CSV 等。
  - `paper_support.py`：论文支撑类说明图，如场景要素建模图、完整流程示意图。

- `scripts/`
  - 外部调用脚本，负责运行算法并调用 `src/visualization/` 出图。
  - 重要脚本：
    - `scripts/visualize_preallocation.py`
    - `scripts/run_pso_ablation.py`
    - `scripts/visualize_reallocation.py`
    - `scripts/visualize_paper_support.py`

- `test/`
  - 测试和部分历史可视化脚本。
  - 后续新出图逻辑应优先放到 `src/visualization/`，再由 `scripts/` 调用，不建议继续把主要出图功能写在 `test/` 里。

- `results/`
  - 输出图片和 CSV。
  - 不要假设旧结果文件一定代表当前代码输出，必要时重新运行脚本生成。

- `doc/`
  - 论文建模、修改方案、重分配计划、问题记录和后续计划。
  - 重要文件：
    - `doc/建模.md`
    - `doc/修改方案.md`
    - `doc/重分配计划.md`
    - `doc/issues.md`
    - `doc/计划.md`

## 当前主流程

当前论文主链路应理解为：

```text
场景加载
-> PSO 预分配
-> 输出 AssignmentPlan 任务序列
-> 动态事件触发
-> 分析受影响任务链和开放目标需求
-> MCHA 链尾追加式局部重分配
-> 输出重分配后 AssignmentPlan
-> 路径规划与最终航迹展示
-> 图表、CSV 和论文说明图输出
```

注意：当前重分配更准确地说是“任务序列方案层面的事件触发式局部重分配”，不是完整真实执行过程中的实时重规划。

## 任务序列语义

本仓库已经完成从单任务分配到任务序列分配的核心改造。

关键语义：

- `ammo` 表示一架 UAV 最多可执行多少个目标任务。
- `AssignmentPlan.uav_task_sequences[uav_id]` 表示某架 UAV 的任务链。
- `UavTaskSequence.tasks` 中的任务按顺序执行。
- 第二个及后续任务的距离、威胁和到达时间应基于链尾目标计算，例如 `UAV -> T1 -> T2` 中，`T2` 的代价基于 `T1 -> T2`，不是 `UAV -> T2`。
- 当前 MCHA 重分配采用链尾追加策略，不做中间插入、链内重排或跨 UAV 交换优化。

论文表述时建议强调：

> 为保证重分配快速性和可解释性，本文在动态重分配阶段采用链尾追加式任务序列修复策略。

## 时间窗与协同打击建模

用户理解的“时间窗”重点不是每个目标都有绝对起止时间，而是协同打击目标中第一架 UAV 和最后一架 UAV 到达同一目标的时间差不能过大。

当前配置：

- `WEIGHTS['sync_window']` 表示协同打击允许的最大到达时间差，单位为小时。
- 当前值通常为 `0.05 h`，约等于 3 分钟。

当前实现中存在两类时间相关惩罚：

1. 显式时间窗惩罚
   - 来自目标字段 `time_window_start` / `time_window_end`。
   - 由 `src/core/sequence_eval.py` 按任务链累计到达时刻计算。

2. 协同到达时间窗惩罚
   - 用于同一目标由多架 UAV 协同打击时的到达时间一致性。
   - 按同一目标的最大到达时间与最小到达时间之差计算。
   - 若差值超过 `sync_window`，按建模中的平方偏差项计入适应度。

预分配适应度中已经接入同步时间窗惩罚。相关逻辑在 `src/pre_allocation/pso.py`。

## 预分配现状

预分配当前以改进 PSO 为主体，不应改成纯贪心。

当前能力：

- 支持 UAV 执行任务序列。
- 支持 `ammo > 1`，可处理 UAV 数小于目标数的场景。
- 支持按任务链累计计算距离、威胁、到达时间和时间窗惩罚。
- 支持 PSO 消融实验：
  - 基础 PSO：random + linear
  - 仅改初始化：logistic + linear
  - 仅改权重：random + cosine
  - 完整改进 PSO：logistic + cosine
- 消融实验当前使用多 seed，正式模式为 `range(20)`。

常用命令：

```bash
env MPLBACKEND=Agg MPLCONFIGDIR=/tmp conda run -n uav python scripts/visualize_preallocation.py
env MPLBACKEND=Agg MPLCONFIGDIR=/tmp conda run -n uav python scripts/run_pso_ablation.py
```

可选环境变量：

- `PSO_SCENARIO=small|medium|hard`
- `PSO_SEED=<int>`
- `PSO_ABLATION_QUICK=1`

预分配主要输出：

- `results/pre_allocation/<scenario>_task_sequence_assignment.png`
- `results/pre_allocation/<scenario>_target_loads.png`
- `results/pre_allocation/<scenario>_cooperative_arrival_windows.png`
- `results/pre_allocation/<scenario>_uav_task_loads.png`
- `results/pre_allocation/<scenario>_preallocation_metrics.png`
- `results/pre_allocation/<scenario>_preallocation_metrics.csv`
- `results/pre_allocation/ablation/*.png`
- `results/pre_allocation/ablation/pso_ablation_summary.csv`

## 重分配现状

当前任务序列版重分配主要支持以下事件：

- `UAV_LOST`
- `TARGET_ADDED`
- `TARGET_DEMAND_INCREASED`
- `THREAT_ADDED`

有意不作为主线实现的事件：

- `TARGET_REMOVED`
- `TARGET_VALUE_CHANGED`
- `TARGET_DEMAND_DECREASED`

其中 `TARGET_DEMAND_DECREASED` 优先级较低，已经记录在 `doc/issues.md`。原因是打击任务中执行中途发现“不需要这么多 UAV”发生概率和重分配收益偏低，而且会引入执行态问题。

重分配关键流程：

```text
apply_event_to_battlefield()
-> analyze_plan_event_impact()
-> run_mcha_for_plan()
```

关键设计：

- 事件分析阶段将受影响任务释放为开放目标需求。
- MCHA 对开放需求进行多轮启发式拍卖。
- 每架 UAV 每轮提交一个当前最优投标。
- 每个目标可按剩余需求接收多个中标 UAV，因此某些轮次可能出现多架 UAV 同时中标。
- MCHA 当前是启发式集中式修复流程，不是标准分布式 CBBA。

常用命令：

```bash
env MPLBACKEND=Agg MPLCONFIGDIR=/tmp MCHA_EVENT=uav_lost MCHA_SEED=42 conda run -n uav python scripts/visualize_reallocation.py
env MPLBACKEND=Agg MPLCONFIGDIR=/tmp MCHA_EVENT=target_added MCHA_SEED=42 conda run -n uav python scripts/visualize_reallocation.py
env MPLBACKEND=Agg MPLCONFIGDIR=/tmp MCHA_EVENT=target_demand_increased MCHA_SEED=42 conda run -n uav python scripts/visualize_reallocation.py
env MPLBACKEND=Agg MPLCONFIGDIR=/tmp MCHA_EVENT=threat_added MCHA_SEED=42 conda run -n uav python scripts/visualize_reallocation.py
```

重分配输出按事件分目录保存：

```text
results/reallocation/<event_name>/
```

典型输出：

- `task_sequence_before_after.png`
- `task_sequence_diff.png`
- `target_loads.png`
- `uav_loads.png`
- `winning_bids.png`
- `candidate_bids.png`
- `demand_repair.png`
- `cost_change.csv`

`cost_change.csv` 中的 `事件后待修复` 可能出现代价低于事件前的情况，这是因为部分任务被释放后方案不完整。解释时必须结合 `未满足任务需求数量` 和 `目标需求满足率`，不能单看总代价。

## THREAT_ADDED 边界

`THREAT_ADDED` 当前第一版闭环采用：

```text
新增威胁航段检测
-> 保留安全前缀
-> 释放受影响航段目标及后续任务
-> MCHA 局部修复开放目标需求
```

已解决的问题：

- 避免新增威胁后部分目标完全没人执行。
- 能通过重分配补齐目标需求。
- 可输出任务链变化、MCHA 投标过程、开放需求修复过程和代价变化表。

仍需注意：

- 未接入真实执行态 `UavExecutionState`。
- 不能区分已完成、正在执行和未执行任务。
- 新增威胁检查基于直线航段威胁积分，不结合 A* 绕行可达性。
- 释放后缀策略偏保守，可能释放较多任务。
- `forbidden_pairs` 只避免释放任务回到同一 UAV，不能保证新航段完全避开新增威胁。

论文中不要表述为“完整实时路径重规划”，应表述为“新增威胁下的任务序列方案层局部重分配”。

## 路径规划现状

路径规划模块位于 `src/route_planning/`。

当前定位：

- 支撑最终航迹规划展示。
- 使用简单可解释的 A* + B 样条/局部运动学平滑。
- 不应成为论文主要创新点。

相关测试和输出：

```bash
env MPLBACKEND=Agg MPLCONFIGDIR=/tmp conda run -n uav python test/test_path_planning_visualization.py
env MPLBACKEND=Agg MPLCONFIGDIR=/tmp conda run -n uav python test/test_route_planning_smoothing.py
```

典型输出：

- `results/path_planning/path_planning_demo.png`
- `results/path_planning/path_planning_demo_multi.png`

后续若继续路径规划，优先目标是让最终任务链逐段规划并稳定出图，不要扩展成复杂动态避障系统。

## 可视化架构约定

当前可视化已经从 `test/` 逐步迁移到：

```text
src/visualization/
scripts/
```

约定：

- 绘图函数放在 `src/visualization/`。
- 运行脚本放在 `scripts/`。
- 测试放在 `test/`。
- 论文支撑类说明图放在 `src/visualization/paper_support.py`，不要混在重分配模块里。
- 新增图片应尽量输出到清晰的子目录，不要把所有图堆在同一层。

审美要求：

- 图要服务论文，不要只追求“能画出来”。
- 标题、图例、标注不能遮挡核心数据。
- 能放到坐标系外的图例和说明，优先放到坐标系外。
- 颜色要克制，语义清晰。
- 不要让连线、编号、标注过密导致图面混乱。
- 任务序列图应体现 `UAV -> T1 -> T2` 的链式语义，而不是从 UAV 起点向多个目标发散。

论文支撑图命令：

```bash
env MPLBACKEND=Agg MPLCONFIGDIR=/tmp conda run -n uav python scripts/visualize_paper_support.py
```

输出：

- `results/paper_support/system_workflow.png`
- `results/paper_support/scenario_elements.png`

## 测试命令

常用测试：

```bash
env MPLBACKEND=Agg MPLCONFIGDIR=/tmp conda run -n uav python test/test_pso_task_sequence.py
env MPLBACKEND=Agg MPLCONFIGDIR=/tmp conda run -n uav python test/test_task_sequence_evaluation.py
env MPLBACKEND=Agg MPLCONFIGDIR=/tmp conda run -n uav python test/test_mcha_task_sequence.py
env MPLBACKEND=Agg MPLCONFIGDIR=/tmp conda run -n uav python test/test_preallocation_task_sequence_visualization.py
env MPLBACKEND=Agg MPLCONFIGDIR=/tmp conda run -n uav python test/test_reallocation_task_sequence_visualization.py
env MPLBACKEND=Agg MPLCONFIGDIR=/tmp conda run -n uav python test/test_paper_support_visualization.py
```

也可以按需运行具体脚本，不一定每次跑全量测试。

修改前后建议至少运行和本次改动相关的测试，并说明是否通过。

## 当前已知问题和论文表述边界

详见 `doc/issues.md`，但后续助手必须特别记住以下点：

1. 当前不是完整执行过程实时重分配。
   - 还没有真正用 `UavExecutionState` 接入当前位置、当前时间、剩余弹药、剩余航程和已完成任务。
   - 论文应避免写成完整实时重规划。

2. `TARGET_ADDED` 有边界。
   - 任务序列版支持单个新增目标且 `required_uavs > 1`。
   - 仍要求新增目标 id 连续。
   - 不支持非连续目标 id、多目标同时新增、删除后复用 id 等复杂工程场景。

3. `TARGET_DEMAND_DECREASED` 暂不作为主线。
   - 可在论文中说明本文重点关注更具代表性的增压型或风险型动态扰动。

4. MCHA 不是标准 CBBA。
   - 不要擅自改成标准分布式共识拍卖。
   - 当前更适合描述为“基于多维代价的启发式拍卖重分配方法”。

5. 任务序列 MCHA 采用链尾追加。
   - 不做中间插入、重排、交换优化。
   - 这是为了保证快速性、稳定性和可解释性。

6. 路径规划不是主要创新点。
   - 不要为了路径规划大改任务分配模型。
   - 当前 A* + 平滑足够支撑全链路展示。

## 后续开发优先级

最高优先级：

- 保证主流程稳定可复现。
- 保证预分配、重分配、路径规划和论文图表能支撑答辩。
- 修正影响论文图语义的问题。
- 整理实验图、CSV 和论文可引用指标。

可以做但要控制范围：

- TARGET_DEMAND_INCREASED 作为补充展示。
- 路径规划接最终任务链逐段展示。
- 静态图稳定后再考虑动图或视频。

不建议做：

- 完整执行态实时重分配。
- 任务链中间插入、重排、交换优化。
- 多目标同时新增、复杂 id 映射。
- 标准分布式 CBBA。
- THREAT_ADDED 与 A* 实时绕行深度耦合。

## Git 和修改约定

- 不要回滚用户未明确要求回滚的修改。
- 修改代码前先理解现有实现和相关文档。
- 对已有论文计划和问题记录，优先追加说明，不要随意删除历史上下文。
- 如果用户要求“先给方案”，不要直接改代码。
- 如果用户要求“同意修改”或明确让开始实现，再修改。
- 用户偏好中文 commit message，`feat:` 前缀可以保留。

最近主线提交已经包含：

- 任务序列版预分配改造。
- 同步时间窗惩罚接入预分配适应度。
- 预分配论文可视化和 PSO 消融实验。
- 任务序列版 MCHA 对多个重分配事件的支持。
- 重分配结果展示、算法分析图和代价变化 CSV。
- 论文支撑类可视化模块。

## 推荐提交信息风格

示例：

```text
feat: 优化重分配与论文支撑可视化输出

- 新增任务序列版重分配结果对比、差异、目标需求、UAV负载等论文图
- 新增MCHA中标结果、候选投标得分、开放任务需求修复过程等算法分析图
- 新增重分配代价变化CSV，支持论文表格整理
- 拆分论文支撑类可视化模块，新增场景要素建模图和完整流程示意图
- 调整重分配结果输出结构，按事件类型分目录保存图表和CSV
- 补充重分配与论文支撑类可视化测试
```

## 给后续 AI 助手的提醒

这个项目的核心不是“实现最强算法”，而是在有限时间内形成一条本科毕设能讲清楚、跑得通、图表好看、实验能复现的无人集群协同打击任务规划链路。

做任何改动前，先问自己：

- 这个改动是否服务论文主线？
- 是否会扩大不可控工程复杂度？
- 是否会破坏当前已经能出图和能测试的流程？
- 是否能用清楚的数学建模或流程图解释？

如果答案不明确，优先选择更小、更稳、更容易写进论文的实现。
