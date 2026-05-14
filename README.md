# UAV Swarm Task Planning

面向无人机集群协同打击任务规划的本科毕业设计配套代码。项目围绕“任务预分配 - 动态重分配 - 航迹规划 - 可视化分析”主链路展开，支持一架无人机执行多个目标任务，并以任务序列作为预分配、重分配和最终航迹生成之间的核心数据结构。

## 已实现功能

- **任务序列建模**
  - 使用 `AssignmentPlan`、`UavTaskSequence`、`TaskNode` 表示多 UAV 任务分配结果。
  - `ammo` 表示无人机可执行的目标数量，支持一架 UAV 执行多个任务点。
  - 任务代价按链式顺序累计，即 `UAV -> T1 -> T2 -> ...`。

- **任务预分配**
  - 基于改进粒子群优化算法（PSO）求解初始任务分配。
  - 支持目标多 UAV 需求，即 `Target.required_uavs`。
  - 适应度函数综合考虑距离代价、威胁代价、协同时间窗惩罚和任务收益。
  - 引入 Logistic 混沌初始化和余弦递减惯性权重，并提供消融实验脚本。

- **动态任务重分配**
  - 基于 MCHA（Multi-dimensional Cost Heuristic Auction，多维代价启发式拍卖）进行事件触发式局部重分配。
  - 主要支持 UAV 损失、新增目标、新增威胁等动态事件。
  - 基于开放任务需求生成候选投标，并通过链尾追加方式修复任务序列。

- **航迹规划**
  - 将任务序列逐段转换为点到点航迹。
  - 使用栅格地图和 A* 搜索规避威胁区。
  - 支持视距简化、局部运动学路径处理和 B 样条平滑。
  - 可输出预分配和重分配后的最终多 UAV 航迹图。

- **论文图表与实验分析**
  - 预分配任务序列图、目标需求满足图、协同到达时间窗图、UAV 负载图。
  - PSO 消融实验结果图与 CSV。
  - 重分配前后任务序列对比图、任务链变化图、中标结果图、需求修复图、代价变化表。
  - 场景要素建模图和完整流程示意图。

## 代码结构

```text
.
├── config/
│   └── params.py                    # 全局参数：目标函数权重、PSO、MCHA、A* 等
├── data/
│   ├── scenario_small.py            # 小规模预分配场景
│   ├── scenario_medium.py           # 中规模预分配场景
│   ├── scenario_hard.py             # 较大规模预分配场景
│   └── scenario_reallocation.py     # 动态重分配实验场景
├── doc/                             # 建模、修改方案、计划和问题记录
├── scripts/
│   ├── visualize_preallocation.py   # 生成预分配论文图和指标
│   ├── run_pso_ablation.py          # 运行 PSO 消融实验
│   ├── visualize_reallocation.py    # 生成重分配论文图和指标
│   ├── visualize_final_routes.py    # 生成最终任务链航迹图
│   ├── visualize_paper_support.py   # 生成论文支撑类说明图
│   ├── analyze_objective_components.py
│   ├── normalize_objective_components.py
│   ├── calculate_ahp_weights.py
│   └── run_weight_sensitivity.py
├── src/
│   ├── core/
│   │   ├── models.py                # UAV、Target、Threat、AssignmentPlan 等核心模型
│   │   ├── objective.py             # 目标函数相关工具
│   │   └── sequence_eval.py         # 任务序列代价与到达时间评估
│   ├── pre_allocation/
│   │   └── pso.py                   # 改进 PSO 预分配算法
│   ├── re_allocation/
│   │   ├── events.py                # 动态事件建模与影响分析
│   │   └── mcha.py                  # MCHA 任务重分配算法
│   ├── route_planning/
│   │   ├── astar.py                 # A* 点到点路径搜索
│   │   ├── grid.py                  # 栅格地图与威胁区占据建模
│   │   ├── planner.py               # AssignmentPlan 到多 UAV 航迹规划结果
│   │   ├── simplify.py              # 视距简化
│   │   ├── smoothing.py             # B 样条平滑
│   │   ├── geometry.py              # 几何与运动学辅助函数
│   │   └── validation.py            # 航迹可行性检查
│   └── visualization/
│       ├── preallocation.py         # 预分配可视化
│       ├── reallocation.py          # 重分配可视化
│       ├── route_planning.py        # 最终航迹可视化
│       ├── paper_support.py         # 论文支撑类说明图
│       └── common.py
├── test/                            # 功能测试与早期可视化验证脚本
├── requirements.txt                 # Python 依赖
└── README.md
```

## 环境安装

本项目开发和实验使用 conda 环境，环境名为 `uav`。推荐 Python 3.10 及以上版本。

```bash
conda create -n uav python=3.10
conda activate uav
pip install -r requirements.txt
```

核心依赖包括：

```text
numpy
matplotlib
scipy
pandas
```

如果在无图形界面的环境中运行绘图脚本，可以使用：

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp conda run -n uav python <script>
```

## 运行方式

### 1. 生成预分配结果图

默认运行中等规模场景：

```bash
conda run -n uav python scripts/visualize_preallocation.py
```

可通过环境变量指定场景和种子：

```bash
PSO_SCENARIO=medium PSO_SEED=42 conda run -n uav python scripts/visualize_preallocation.py
PSO_SCENARIO=hard PSO_SEED=42 conda run -n uav python scripts/visualize_preallocation.py
```

主要输出：

```text
results/pre_allocation/
```

### 2. 运行 PSO 消融实验

```bash
conda run -n uav python scripts/run_pso_ablation.py
```

主要输出：

```text
results/pre_allocation/ablation/
```

### 3. 生成重分配结果图

默认事件为 UAV 损失，也可指定事件：

```bash
conda run -n uav python scripts/visualize_reallocation.py
MCHA_EVENT=uav_lost conda run -n uav python scripts/visualize_reallocation.py
MCHA_EVENT=target_added conda run -n uav python scripts/visualize_reallocation.py
MCHA_EVENT=threat_added conda run -n uav python scripts/visualize_reallocation.py
```

主要输出：

```text
results/reallocation/<event_name>/
```

主要实现了以下三个事件：

- `uav_lost`
- `target_added`
- `threat_added`

### 4. 生成最终航迹规划图

生成预分配与重分配后的最终任务链航迹：

```bash
conda run -n uav python scripts/visualize_final_routes.py
```

只生成预分配航迹：

```bash
ROUTE_MODE=preallocation ROUTE_SCENARIO=medium conda run -n uav python scripts/visualize_final_routes.py
```

只生成某个重分配事件的航迹：

```bash
ROUTE_MODE=reallocation ROUTE_EVENTS=threat_added conda run -n uav python scripts/visualize_final_routes.py
```

主要输出：

```text
results/route_planning/preallocation/
results/route_planning/reallocation/<event_name>/
```

### 5. 生成论文支撑类说明图

```bash
conda run -n uav python scripts/visualize_paper_support.py
```

主要输出：

```text
results/paper_support/
```

### 6. 目标函数权重分析

权重分析相关脚本包括：

```bash
conda run -n uav python scripts/analyze_objective_components.py
conda run -n uav python scripts/normalize_objective_components.py
conda run -n uav python scripts/calculate_ahp_weights.py
conda run -n uav python scripts/run_weight_sensitivity.py
```

主要输出：

```text
results/weight_analysis/
```

## 测试

测试脚本位于 `test/` 目录。可以单独运行某个测试文件，例如：

```bash
conda run -n uav python test/test_pso_task_sequence.py
conda run -n uav python test/test_mcha_task_sequence.py
conda run -n uav python test/test_final_route_visualization.py
```

当前测试脚本以轻量级断言和可视化验证为主，不依赖 pytest。

## 结果文件说明

`results/` 目录用于保存运行脚本生成的图片和 CSV。由于实验结果文件数量较多，且可以通过脚本重新生成，默认不建议将完整 `results/` 推送到远程仓库。

若需要复现实验图表，请先安装依赖，然后运行 `scripts/` 下对应脚本。

## 实现边界

- 当前重分配属于事件触发后的方案级局部修复，不是完整执行过程中的实时在线重规划。
- 重分配阶段主要采用链尾追加方式修复任务序列，未对所有任务插入位置进行全局优化。
- 航迹规划主要面向二维离线仿真场景，尚未完整考虑三维飞行、多机实时避碰、机场起降和返航等工程约束。
- 实验主要用于验证本文方法在设定场景下的可行性，尚未系统展开与 GA、ACO、MILP 等其他方法的横向对比。

## 论文主链路

本仓库的论文实验主链路为：

```text
场景加载
-> 改进 PSO 任务预分配
-> 输出 UAV 任务序列
-> 动态事件触发
-> MCHA 局部任务重分配
-> 输出重分配后任务序列
-> A* + 平滑生成最终航迹
-> 输出论文图表和实验指标
```
