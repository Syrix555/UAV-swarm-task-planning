from src.core.models import AssignmentPlan
from src.pre_allocation.pso import run_pso


def assignment_plan_from_matrix(assignment):
    """将现有预分配矩阵结果适配为任务序列表示。"""
    return AssignmentPlan.from_assignment_matrix(assignment)


__all__ = [
    'assignment_plan_from_matrix',
    'run_pso',
]
