# 单膝下跪：右膝、脚尖和脚跟的接触切换

**反例：**旧 body prior 让右膝悬空约 17–18 cm。**正例（运动学几何）：**`right_knee_ground_v3` 与 v21 组合候选使右膝、脚尖贴地，脚跟抬起；物理接触尚未验证。下文分别说明成因、改法和证据。

## 人和任务发生了什么

1019 取刷后扫桌的 16–22 秒，人不是普通蹲姿，而是**右膝跪地**。原始 human motion 的右膝在约 16.4–22.18 秒持续降低；稳定时膝关节中心平均世界 XY 约 `(0.563, -0.037) m`，最低高度约 `0.04–0.05 m`。左膝保持较高。右脚跟抬起，脚尖继续接触地面并小幅水平移动。不能因为某个脚底接触检测器把右脚标成“非站立”，就把脚尖与地面的实际支撑抹掉；也不能套普通站立的全脚不滑动锚点。

## 旧候选为什么失败

`brush_sweep_1019` 的 body prior 优先保持常规脚部不滑和人体骨架相似，未为此窗口显式建膝/脚尖接地模式。人膝已经接近地面时，G1 右膝可见网格仍离地约 `0.17–0.18 m`，右脚也曾整体离地数厘米。只看人体膝关节位置、脚底滑移或全身姿态图，都不能证明机器人的膝盖网格接触地面。

## 可迁移的求解方法

1. 先读原始 human world motion、同步视频和场景平面；确定**哪侧膝盖**、接触/过渡时窗、膝盖世界 XY 落点、脚尖轨迹。高度阈值只能作为候选检测，结合视频/时序补漏。保留原人体尺度和根的世界坐标约定。
2. 在接触模式中，把跪地侧从“整脚锁定”改为“膝盖可见网格贴地 + 前脚掌/脚尖贴地 + 脚跟抬起”。脚尖可以沿地面水平移动；另一侧站立脚仍保持锚点与不滑约束。入口和退出用连续门函数和速度限制，不逐帧把根压低。
3. 用真实 G1 膝、脚碰撞/可见网格的最低点到地面距离作为约束，检查整个腿部不穿地。人体膝的世界 XY 是位置目标；因人机腿长和双支撑限制，实际机器人落点可有残差，必须量化，不通过改场景或缩放人体掩盖。
4. 将新下身动作与已有上身交互**按时间组合**。保持已经审查的取刷、握持和扫桌世界轨迹；必要时用限速的肩/肘/腕 IK 补偿根部变化。对手、刷子、桌面和上下身接缝重新做几何审查。物体估计轨迹仍是运动学，不是物理抓取。

## 这次得到的动作与审查

右膝接地解保存在 `outputs/humanoid-real2sim/2026-10-20/brush_sweep_1019/curated_snapshot/retarget/right_knee_ground_v3/`；组合后的 8–28 秒候选是 `retarget/brush_sweep_kinematic_v21_knee/motion.npz`，拼接到推桶前缀后是 `retarget/push_to_brush_kinematic_v4_knee/motion.npz`。右膝核心段网格与地面间隙约 `0.98 mm`，右脚尖约 `-0.50 mm` 贴地，脚跟至少离地约 `144 mm`。右脚尖世界 XY 位移最大约 `13.7 mm`、路径长约 `37.3 mm`；左脚踝核心段漂移低于 `0.001 mm`。机器人膝 XY 与人体目标平均相差约 `23 mm`，最大约 `33 mm`；这是人机尺寸与多接触约束下的残差，不声称逐点完全重合。扫桌刷毛在桌面代理上方的最低间隙约 `5.7 mm`。

## 代码、motion 和边界

- 接地准备：`knee_case.yaml`，复用 ScaleTrack `retarget3.ground_stage`；`prepare_right_knee_ground_v3.py` 读取原始人体膝 XY，仅对机器人优化目标施加由 gate 渐入的 `(0.10, 0.30) m` 几何补偿，参数只适用于本场景。
- 时空组合：`refine_knee_v21.py`；将接地根/腿与已审查的上身动作合成，腕角各轴不超过本次设定的 `35°`，右臂关节速度不超过 `6 rad/s`。接地核心的上身世界轨迹保持；过渡段最大手腕轨迹误差另见候选 `review.json`，不隐瞒折中。
- 审查代码：`audit_knee_v21.py`；证据：`review/knee_audit_v21.json` 和 `retarget/brush_sweep_kinematic_v21_knee/review.json`。
- 运动文件：`retarget/brush_sweep_kinematic_v21_knee/motion.npz` 与 `retarget/push_to_brush_kinematic_v4_knee/motion.npz`；物体 sidecar 同目录。这里所有 contact 是运动学几何检查，没有接触力或 Stage 4/5 物理验证。

迁移到另一侧膝盖或另一任务时，应重算人体落点、接触平面、足尖路径和支持脚；不要照搬这次的 `16.4–22.18 s`、目标补偿量、脚跟高度或腕角数值。

## 本仓库结果与审查证据

- **结果（见限定范围）** [right_knee_v21_motion.npz](results/right_knee_v21_motion.npz) — `kinematic_geometry_checked`；1019 brush sweep 8–28 s segment, right knee。
- **审查证据** [right_knee_v21_audit.json](artifacts/right_knee_v21_audit.json) — `kinematic_geometry_checked`；Right knee, toe, heel ground-geometry audit。

**可证明的范围与限制：**motion 与审查 JSON 可核对右脚尖、脚跟、膝盖的几何关系；没有物理接触力结论。
