# Tracker 局部修复：6D 目标、future 速度与合手保持

1012 抓刷进入 Stage 4 时，用户需要对五个可修复 link 直接指定世界位姿目标，并在页面看到半透明目标手/脚。每帧从当前真实 link 位姿算目标方向，再按用户给定线速度和角速度推进 future setpoint；接近目标时速度归零。旧修改曾让手不再朝用户指定目标移动，原因是把 future setpoint 的逐帧积累破坏了。修复后应先检查目标不变时的连续多帧推进，再看实际跟踪残差。

另一个反例是恢复到原参考时对手指开合做插值：已合拢的右手在恢复过程中松开，原本抓住的物体会滑落。手部应在恢复段保持明确的闭手命令，并单独核对实际手指角度与接触力；运动学的 100% 合手滑杆不能证明实际紧握。1012 手刷的后续高 `kp` 试验仍没有稳定提刷，因此这条是**控制与页面行为经验**，不是抓取正例。

通用实现见 [tracker/修复库](../../../robosimstudio/humanoid_real2sim/track/repair_studio.py)与[网页控制](../../../robosimstudio/humanoid_real2sim/track/web_ui.html)；[抓刷任务记录](../../../outputs/humanoid-real2sim/2026-10-20/brush_pick_1012/curated_snapshot/README.md)和[圆篮以外的篮子恢复诊断](../../../outputs/humanoid-real2sim/2026-10-20/kick_basket/curated_snapshot/README.md)保存具体失败模式。新任务应重测目标坐标、速度上限、恢复触发和物手滑移，不复用这些任务的局部修复位姿。

## 本仓库结果与审查证据

- **审查证据** [kick_basket_recovery_to_117.json](artifacts/kick_basket_recovery_to_117.json) — `control_diagnostic`；Tracker recovery diagnostic; does not show grasp success。
- **审查证据** [1012_hand_physics_failure.json](../grasp/artifacts/1012_hand_physics_failure.json) — `negative`；1012 brush physical replay; hand closure did not yield reliable grasp。

**可证明的范围与限制：**恢复诊断 JSON 可检查接管后的状态变化；刷子物理失败分析说明合手保持仍不能保证物体不滑落。
