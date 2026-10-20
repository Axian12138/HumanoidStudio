# Round basket，Stage 3.2

源录制 `16-31-26` 的 69–76 秒，50 Hz，350 帧。以 `body_prior.npz` 为不可修改的 3.1 输入，只调整右臂七关节和右 Dex3 七关节；机器人模型为 `g1_29dof_dex3_offsethand.xml`。当前候选在 [`hoi_v2/motion.npz`](hoi_v2/motion.npz)，物体地面平移估计在 [`hoi_v2/object_estimate.npz`](hoi_v2/object_estimate.npz)，源人体骨架在 [`hoi_v2/human_motion_world.npz`](hoi_v2/human_motion_world.npz)。

## 交互假设

右手从篮子外侧接近左侧提手顶部，掌面朝提手，抓住后将篮子沿地面朝人拖动，再松手撤离。原 1012 场景篮子初始位置 `(0.16, -1.92, 0.001)`；依据 1019 右手可达性，当前播放场景暂放在 `(0.10, -1.92, 0.001)`，偏移 6 cm，仍需在 3.5 页面由人审阅。源网格最低点局部 `z=0`，世界最低点约 `z=0.001 m`。物体轨迹为平面估计：`x` 最多减 5 cm，`y` 增 33 cm，`z` 始终为 0.001 m；这不是动力学结果。

开手 70.60–71.85 秒；闭手命令 73.23–73.53 秒；释放 74.55–74.80 秒。时序固定，不作为 IK 优化参数。接触目标为 73.5 秒左右，物理首次接触和夹持力尚未测量。

## 数值审查

[`hoi_v2/audit.json`](hoi_v2/audit.json) 记录 50 Hz 和交互窗口半帧插值的手部网格对篮子原始 65 个凸包的检测：篮壁/篮沿穿入 **0 次**；片段首尾关节接缝 **0 rad**；关节限位超出 **0**；最大关节速度为 URDF 限值的 **46.7%**。交互窗口右腕 roll/pitch/yaw 均限制在 ±35°，但三个轴都有触及软限位的时刻。最终接近段 72.9–73.25 秒掌心法向与移动方向夹角约 35.6°；更早的 72.0–72.9 秒仍以水平寻位为主，不能解释为正面入手。

73.5 秒拇指/中指/食指网格到提手采样顶点的最近距离约 5.6/2.5/9.1 mm；74.3 秒约 1.6/10.0/24.5 mm。拖动时手指与提手的关系仍有误差，尤其食指，不能据此宣称稳定物理抓取。接触阶段掌心目标在稀疏结点的误差约 3–20 mm；前期寻位和撤离结点有更大误差。详见 [`hoi_v2/review.json`](hoi_v2/review.json)。

## Stage 3.5 播放

```bash
.venv/bin/python scripts/humanoid-real2sim/kinematic_viewer.py \
  --motion examples/humanoid-real2sim/sequence_1019/tasks/06_round_basket/hoi_v2/motion.npz \
  --human-motion examples/humanoid-real2sim/sequence_1019/tasks/06_round_basket/hoi_v2/human_motion_world.npz \
  --scene-file scripts/humanoid-real2sim/tasks/sequence_1019/tasks/06_round_basket/scene.py \
  --object-track examples/humanoid-real2sim/sequence_1019/tasks/06_round_basket/hoi_v2/object_estimate.npz \
  --object-name round_laundry_basket --source-offset-s 69 --port 8260
```

打开 `http://localhost:8261/`，在 72.9–74.8 秒重点审阅右掌、提手和篮壁。页面逐帧设置 `qpos` 与物体估计位姿，不推进物理或 tracker。当前状态是待人工 3.5 审阅的运动学候选，未进入 Stage 4。

复现：先运行 `export_basket_obj.py`（需要 Blender），再运行 `fit_hoi.py`、`audit_hoi.py`。
