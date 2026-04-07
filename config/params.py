# 全局参数配置

# 地图
MAP_SIZE = (100.0, 100.0)  # km x km

# 无人机公共属性
UAV_MAX_RANGE = 200.0    # km，最大航程
UAV_MAX_AMMO = 1         # 最大弹药量
UAV_SPEED = 250.0        # km/h，巡航速度

# 目标函数权重（AHP初始值，后续实验调参）
WEIGHTS = {
    'w1': 0.25,    # 距离代价
    'w2': 0.6,   # 威胁代价
    'w3': 0.05,   # 时间窗惩罚
    'w4': 0.1,    # 任务收益
    'alpha': 1.0, # 时间惩罚系数
}

# PSO参数（待创新点确定后调整）
PSO = {
    'num_particles': 50,
    'max_iter': 200,
    'w_start': 0.9,       # 惯性权重初始值
    'w_end': 0.4,         # 惯性权重终值
    'c1': 2.0,            # 个体学习因子
    'c2': 2.0,            # 社会学习因子
}

# MCHA参数
MCHA = {
    'max_iter': 50,           # 最大重分配轮次
    'min_score': float('-inf')  # 最小可接受竞标分数
}

# MCHA测试参数
MCHA_TEST = {
    'default_event': 'uav_lost',
    'lost_uav_id': 0,
    'threat_threshold': 1.0,
    'new_threat_id': 99,
    'new_threat_x': 48.0,
    'new_threat_y': 68.0,
    'new_threat_radius': 12.0,
    'target_added_id': 10,
    'target_added_x': 84.0,
    'target_added_y': 50.0,
    'target_added_value': 9.5,
    'target_added_required_uavs': 1,
    'target_demand_increase_target_id': 6,
    'target_demand_increase_new_required_uavs': 4,
    'target_demand_decrease_target_id': 1,
    'target_demand_decrease_new_required_uavs': 2,
}

# A*参数（待创新点确定后调整）
ASTAR = {
    'grid_resolution': 1.0,  # km，栅格分辨率
}
