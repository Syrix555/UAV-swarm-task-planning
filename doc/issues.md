# ISSUES

## 1

  现在 available_uavs 在 decrease 场景下仍然会打印成全体 UAV：

  available_uavs: [0, 1, 2, ..., 23]

  这不影响正确性，但在“纯裁剪事件”里它的解释价值不大。
  不过这已经属于展示优化问题了，不是功能问题。
