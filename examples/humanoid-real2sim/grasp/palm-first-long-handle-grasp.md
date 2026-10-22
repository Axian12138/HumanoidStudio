# 长柄物体：从掌侧接近并保护整手和手腕

**反例：**旧候选只对齐拳眼，手背穿入刷子/托盒且翻腕过大。**正例（运动学几何）：**v20 掌侧接近候选通过已记录的网格与腕角审查；物理夹持、滑移和释放未验证。

1019 的刷子取放是一个长柄抓取任务。旧候选把拳眼与刷柄轴向对齐，但手背先朝向托盒/刷子并穿入物体，腕部为了贴合终点产生明显翻折。用户两次指出：接触终点看似贴合，不能代替从外部接近物体的可执行路径。

**成功的运动学做法**是先用实际 Dex3 FK 和手部网格定义掌面法向、闭手拳眼与可握刷柄段；让掌侧朝刷柄、由盒外接近。把接近、首次触碰、合手、持握、释放与撤离作为一段连续路径检查。拳眼轴线、掌面与进入速度夹角、手背/手指/前臂与托盒的首次接触顺序、逐轴腕角都要审查。接触终点可用而路径不好时固定终点，只重做接近段。别把“手指网格贴近”解释成夹持力。

本任务的 v20 候选将右腕三个轴约束在约 `±35°`，把握点移到托盒侧壁外的可握段；50 Hz 的掌部顶点审查未见大于 `0.5 mm` 的刷柄穿透，接近方向与掌面夹角约 `19–20°`。这些数值是这一个模型和对象的结果，不是所有抓取的通用阈值。抓取力、刷子相对手掌滑移和真实释放仍未经过物理验证。

**代码和动作**：`scripts/humanoid-real2sim/tasks/brush_sweep_1019/fit_brush_v19.py`、`finalize_brush_v20.py`、`audit_pick_v20.py`；motion 为 `retarget/brush_sweep_kinematic_v20/motion.npz`，审查记录为 `review/pick_audit_v20.json`、`review/hand_audit_v20.json`。组合右膝跪地后的动作见 [任务 example](../composed-actions/brush-pick-sweep-return-1019.md)；抓取窗口在跪地前，应保证新下身优化不改坏已通过的取刷阶段。

## 本仓库结果与审查证据

| 旧版：把手终点对齐 | 改版：掌侧进入候选 |
| --- | --- |
| ![旧接近方式的手与刷柄近景](artifacts/1012_handle_first_closeup.png) | ![掌侧接近的手与刷柄近景](artifacts/1012_palm_approach_closeup.png) |

两张图只显示选定帧的几何；下列两份 motion 用于检查整个 6–12 秒接近路径。

- **审查证据** [1012_handle_first_closeup.png](artifacts/1012_handle_first_closeup.png) — `negative`；1012 handle-first hand/brush geometric diagnostic。
- **审查证据** [1012_palm_approach_closeup.png](artifacts/1012_palm_approach_closeup.png) — `kinematic_candidate`；1012 palm-approach hand/brush geometric diagnostic。
- **审查证据** [1012_handle_first_motion.npz](artifacts/1012_handle_first_motion.npz) — `negative`；1012 brush 0–35 s; inspect 6–12 s approach。
- **审查证据** [1012_palm_approach_motion.npz](artifacts/1012_palm_approach_motion.npz) — `kinematic_candidate`；1012 brush 0–35 s; inspect 6–12 s approach。
- **结果（见限定范围）** [1019_brush_v20_motion.npz](results/1019_brush_v20_motion.npz) — `kinematic_geometry_checked`；1019 brush, 8–28 s segment。
- **结果配套的估计物体轨迹** [1019_brush_v20_object_estimate.npz](results/1019_brush_v20_object_estimate.npz) — 与 v20 motion 同一时窗；是运动学估计，不代表物理抓取。
- **审查证据** [1019_brush_v20_hand_audit.json](artifacts/1019_brush_v20_hand_audit.json) — `kinematic_geometry_checked`；1019 brush hand mesh audit。

**可证明的范围与限制：**1012 的两张图片和两份 motion 供比较接近方向；1019 v20 motion 与手网格审查支持运动学结论。这些证据不等于稳定物理抓取。
