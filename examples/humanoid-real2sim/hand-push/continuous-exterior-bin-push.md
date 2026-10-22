# 手推落地容器：双接触点、受力方向与跟随检验

## 可迁移动作

人用左手持续推白色垃圾桶。机器人参考使用拇指与食指接触桶的**外侧后缘**，桶的水平位移方向应大致落在两处接触可施力方向之间。先在运动学中检查整段接触、桶的地面支撑和手指扫掠，再用 tracker/RSS 物理回放区分“参考中贴着桶”与“实际持续推动”。这属于手推物体，不应把桶固定到手上。

## 正例范围与反例

用户已通过左手 HOI 运动学候选，允许作为 Stage 4 输入；具体候选、源映射和阶段状态见[任务记录](../../../outputs/humanoid-real2sim/2026-10-20/push_bin_1019/curated_snapshot/stage_ledger.json)。`fit_hoi.py` 中双指接触点与物体运动方向的约束可迁移到其他手推物体，但接触点位置和桶径必须重新测量。

物理反例是只凭 kinematic 接触就宣称推成：一次原始 ScaleTrack checkpoint 回放，左腕接触前平均误差约 10.9 cm，物理桶终点 `(0.406, 0.971) m` 明显偏离参考 `(0.771, 0.379) m`。另一次桌旁回放中，约第 360 帧手与桶已分离，早于桶抵达桌沿；第 375 帧物理接触点间隙约 9.7 cm，参考约 1.6 cm。桌子位置不是这次分离的充分解释。相应记录见[任务说明](../../../outputs/humanoid-real2sim/2026-10-20/push_bin_1019/curated_snapshot/README.md)和 `review/stage5_human_vs_physics_contact.json`。

## 代码和取用边界

- [手推几何拟合](../../../scripts/humanoid-real2sim/tasks/push_bin_1019/fit_hoi.py)、[物理回放](../../../scripts/humanoid-real2sim/tasks/push_bin_1019/stage5_replay.py)、[审查页](../../../scripts/humanoid-real2sim/tasks/push_bin_1019/review_stage5.py)。
- 反例的滚动物理缓存属于历史实验，按时间与任务保存在 `outputs/humanoid-real2sim/`；长期知识保存接触条件、关键数值与审查路径，不保留每次 rollout。
- 对新桶复用的是“外侧多点持续接触 + 位移方向 + 物理分离时刻”这组检查，不复制该桶的摩擦、质量或接触坐标。

## 本仓库结果与审查证据

- **结果（见限定范围）** [bin_push_hoi_motion.npz](results/bin_push_hoi_motion.npz) — `kinematic_approved_physics_pending`；1019 waste bin, 474 frames at 50 Hz。
- **结果配套的估计物体轨迹** [bin_object_estimate.npz](results/bin_object_estimate.npz) — 同为 474 帧，仅是目标桶轨迹；实际物理桶运动另看下面的回放诊断。
- **审查证据** [bin_hoi_kinematic_audit.json](artifacts/bin_hoi_kinematic_audit.json) — `kinematic_approved_physics_pending`；手指与桶的运动学接触审查。
- **审查证据** [bin_physics_top_view.png](artifacts/bin_physics_top_view.png) — `physics_diagnostic`；1019 waste bin path vs replay, thumb/index tracks and table。

**可证明的范围与限制：**motion 显示手指外侧推动的运动学候选；俯视图是不同摩擦设定的物理路径诊断，不能把该图当作完整通过。
