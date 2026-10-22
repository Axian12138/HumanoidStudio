# 物体物理检查：自由静置、材质绑定与接触优先级

一个物体在运动学画面里位置正确，不证明在 RSS 重力下稳定。1012 手刷曾有更大把手倾角候选（约 25–45°）在托盒中滚倒或被弹出；最终把柄露出的放置先经过自由静置，但物理抓起仍未通过。[手刷任务记录](../../../outputs/humanoid-real2sim/2026-10-20/brush_pick_1012/curated_snapshot/README.md)保留放置与失败原因。

1019 垃圾桶的另一个反例是只改桶的摩擦系数却未检查运行时实际接触材质。场景地面摩擦为 1.0；同优先级接触时 MuJoCo 选择了较大的值，低重心/0.10 摩擦候选仍倾倒。将**桶的 26 个碰撞形状**设为优先级 1 后，实际接触采用桶的 0.10 摩擦，完整 474 帧回放中桶最大倾角约 0.85°、移动 0.836 m；这只是该场景物理候选，手/桶在餐桌附近的后段接触仍有问题。[桶任务说明](../../../outputs/humanoid-real2sim/2026-10-20/push_bin_1019/curated_snapshot/README.md)列出对照试验。

圆篮也提供一个反例：若把 65 个 Blender `Hull` 合为一张碰撞网格，会错误封住篮腔；可视网格与逐个碰撞凸包必须区分。篮子的质量、摩擦及重心试验只增加位移约 1 cm，仍未弥补左踝约 13.6 cm 的跟踪偏差。[篮子物理诊断](../../../outputs/humanoid-real2sim/2026-10-20/kick_basket/curated_snapshot/README.md)解释了不能靠物理参数掩盖动作错误。

复用时检查：初始位姿自由静置、实际绑定到每个碰撞体的材质/优先级、质量与重心、机器人真实接触误差、物体最终位姿；一次只改一个物理因素并保留基线。[垃圾桶接触场景](../../../scripts/humanoid-real2sim/tasks/push_bin_1019/scene_mu_0p10_com_0p03_contactfix.py)、[圆篮静置检查](../../../scripts/humanoid-real2sim/tasks/kick_basket/rest_check.py)、[手刷静置检查](../../../scripts/humanoid-real2sim/tasks/brush_pick_1012/rest_check.py)。

## 本仓库结果与审查证据

- **审查证据** [brush_release_rest_check.json](artifacts/brush_release_rest_check.json) — `physics_diagnostic`；Brush release and free rest check。
- **审查证据** [bin_physics_top_view.png](../hand-push/artifacts/bin_physics_top_view.png) — `physics_diagnostic`；1019 waste bin path vs replay, thumb/index tracks and table。

**可证明的范围与限制：**静置检查 JSON 与推桶物理俯视图是两个接触/支撑诊断；它们不是所有材质、质量和场景的通用参数。
