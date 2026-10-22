# 长柄刷扫桌：抬起、远端落刷、贴桌扫回、再次抬起

**反例：**v21 在 19–21 秒把刷子压在近乎固定的 X 位置，漏掉第二次扫桌行程。**正例（运动学几何）：**v22 恢复抬刷、远端落刷、扫回、再抬刷的时间顺序和水平行程；物理接触未验证。

## 人在做什么

1019 取刷后，人在右膝跪地、左脚支撑的姿态下重复扫桌。20 秒附近的第二次动作不是原地压刷：原始 human motion 的右手辅助点先从近端升起并朝桌面远端移动，约 19.6 秒到世界 X `1.142 m`；随后降低并在约 20.0–21.0 秒从 X `1.10 m` 扫回 `0.83 m`；到 21.4–21.8 秒又明显抬起。右手辅助点高度从 19.0 秒的约 `0.647 m` 降至扫回末端的约 `0.439 m`，再升至约 `0.603 m`。这些时间和位置由该 take 的 `human_motion_world.npz` 测得；同步视频的原接触日程将第二次扫桌粗标为 19.5–20.9 秒。

## 失败原因

先前 `fit_brush_v19.py` 在扫桌残差中把刷毛中心 X 强制拉向固定 `0.85 m`，同时主要拟合离桌高度。这能得到刷毛靠近桌面的单点几何，却把人的“抬起→到远端→落刷→水平扫回→抬起”压扁为局部摆动。v21 的刷子估计在 19–21 秒 X 基本停在 `0.85–0.89 m`，与人体 19.6 秒的远端 X `1.142 m` 明显不符。只看整段最低刷毛高度或“刷毛有多少帧在桌面上方”会漏掉时间结构和水平行程。

## 解决方案

在已经通过的 v21 掌侧抓取与右膝接地 motion 上，仅重解全局 18.0–23.2 秒的右臂。根据同源帧的人手辅助点构造刷子中心 XY 目标：以 18.0 秒现有相对位置标定人手到刷子的固定 XY 偏移，保持人手的远近行程；从人手高度估计抬起/下降，并在 20.0–21.0 秒给刷毛约 `4 mm` 的桌面几何间隙。用真实 G1 腕 FK、刷子固定手腕变换及刷毛网格求解，保留原刷子握持相对位姿；离开时窗前逐渐回接旧动作。对右臂时序插值和速度约束，根、腿、躯干、左臂、手指、取刷与放回窗口不变。这个规则可迁移的是**先分段认出抬/放/扫/抬的接触状态，再按源帧拟合物体轨迹**；XY 偏移、桌面高度和时窗要对新场景重测。

## 结果与边界

v22 候选在 19.6 秒刷子中心到 X `1.129 m`（v21 为 `0.856 m`）；20.0–21.0 秒刷子从 X `1.096 m` 扫到 `0.833 m`，行程约 `263 mm`，刷毛与桌面代理的间隙约 `-0.53..4.88 mm`。21.2–21.8 秒刷毛重新离桌约 `34..172 mm`。逐帧右手网格未进入桌面代理，右臂最大速度约 `6 rad/s`、腕轴不超过本候选 `35°`。膝盖/脚尖、左脚支撑审查保持通过；20 秒以外的取刷和放回关节动作未改。负的 `0.53 mm` 是网格容差内的运动学微穿透，不是物理接触力。Stage 4/5 按用户此前要求未执行，刷子轨迹仍为运动学估计。

## 代码、motion 与证据

- `scripts/humanoid-real2sim/tasks/brush_sweep_1019/refine_sweep_cycle_v22.py`：从 v21 生成右臂与刷子 v22；`audit_sweep_cycle_v22.py` 审查四段循环、桌面、速度和未改动作。
- 局部 motion：`retarget/brush_sweep_kinematic_v22_cycle/motion.npz`；物体：同目录 `object_estimate.npz`。
- 完整动作：`retarget/push_to_brush_kinematic_v5_cycle/motion.npz`，由既有 `splice_push_v3.py --new ... --out ...` 拼接。审查：`review/sweep_cycle_audit_v22.json`、`review/knee_audit_v22.json`。
- 反馈记录：`feedback/2026-10-20-brush-sweep-cycle.json`。原 v21/v4 和归档 v2 保留用于对照；本例的具体秒数、0.85 m 错误目标及 4 mm 间隙不应复制到其他桌面任务。

## 本仓库结果与审查证据

- **结果（见限定范围）** [brush_sweep_v22_motion.npz](results/brush_sweep_v22_motion.npz) — `kinematic_geometry_checked`；1019 brush sweep 8–28 s segment。
- **结果配套的估计物体轨迹** [brush_sweep_v22_object_estimate.npz](results/brush_sweep_v22_object_estimate.npz) — 与 v22 motion 同源同帧的刷子估计轨迹，未通过物理扫刷验证。
- **审查证据** [brush_source_body_candidate_20_5s.jpg](artifacts/brush_source_body_candidate_20_5s.jpg) — `comparison`；Source head camera, body prior, brush candidate at 20.5 s。
- **审查证据** [brush_sweep_v22_cycle_audit.json](artifacts/brush_sweep_v22_cycle_audit.json) — `kinematic_geometry_checked`；Brush sweep cycle audit。

**可证明的范围与限制：**motion、三栏对照图与扫刷审查记录支持动作顺序和局部运动学；抓力、扫地接触力与释放后静置需另查物理证据。
