# 抓取：闭手命令、真实接触力与稳定提起分别验证

1012 绿手刷放在白色托盒内，把手需要露出并通过自由静置证明不会自行倾倒。人手掌侧靠近把手，Dex3 的开/合时刻由动作阶段估计为明确命令，不当作 retarget 的自由优化参数。候选在 8.90 秒开始张开，10.36 秒开始合手，11.20 秒完成；具体时刻属于该录制，不是通用模板。[任务说明](../../../outputs/humanoid-real2sim/2026-10-20/brush_pick_1012/curated_snapshot/README.md)记录把手放置、候选版本和物理诊断。

**反例一：**只看参考指关节 target，闭手看似完成，但旧模型和实际 USD 的腕到手变换不同，指端与柄未真正对齐。先统一 `g1_29dof_dex3_offsethand` 的 FK/可视/碰撞模型，再审查指节网格。**反例二：**提高 `kp` 并令指尖碰到柄，拇/食指接触力峰值约 26.8/43.3 N，但刷子横向移动约 44 mm，没有稳定持握。一个短暂碰撞导致刷子抬高，也不能算成功提起；相对手掌滑移和提起后的保持才是验收量。当前 Stage 4/5 抓刷仍未通过。

可迁移顺序是：实际几何一致性 → 开口足够且掌侧进入 → 合手命令与实测关节角 → 指垫接触力 → 提起后的物手相对位姿和自由物体稳定。必要的预压需在物理中测量，不能通过可视网格穿透冒充抓力。

- [手型与轨迹准备](../../../scripts/humanoid-real2sim/tasks/brush_pick_1012/prepare_grip_boost.py)、[统一手模型的闭合候选](../../../scripts/humanoid-real2sim/tasks/brush_pick_1012/prepare_offsethand_closure.py)、[物理闭手诊断](../../../scripts/humanoid-real2sim/tasks/brush_pick_1012/probe_grip_closure.py)。
- 少量典型审查数值与候选留在[任务记录](../../../outputs/humanoid-real2sim/2026-10-20/brush_pick_1012/curated_snapshot/README.md)；完整失败搜索和 rollout 放 `outputs/humanoid-real2sim/`。

## 本仓库结果与审查证据

- **审查证据** [1012_hand_physics_failure.json](artifacts/1012_hand_physics_failure.json) — `negative`；1012 brush physical replay; hand closure did not yield reliable grasp。
- **审查证据** [1012_palm_approach_motion.npz](artifacts/1012_palm_approach_motion.npz) — `kinematic_candidate`；1012 brush 0–35 s; inspect 6–12 s approach。

**可证明的范围与限制：**物理 replay 分析记录抓取失败，掌侧运动学 motion 只说明接近姿态；闭手指令不能据此推断实际夹持力。
