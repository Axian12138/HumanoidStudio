# 修复 motion 后复用已认可的运动学审阅页

**反例：**为膝盖修复另写播放器，丢失用户已认可的时间轴工作方式。**正例（页面验证）：**复用原播放器，只更新 motion 与 sidecar，已验证页面状态和跳帧；页面所示接触仍只是运动学。

## 人在审阅什么，页面承担什么

1019 推桶、取刷、右膝跪地扫桌、放回刷子的审阅依赖同一时间轴：用户拖动进度条，在原 RSS 场景中同时看原始 human 骨架、实体 humanoid 和估计刷子。此前的 `kinematic_viewer.py` 页面已被用户认可。右膝修复只产生新的机器人 motion 与刷子轨迹，没有改变用户要看的信息或操作方式。

## 这次反馈的原因

我误把“新修复需要显示膝盖”理解为“需要新页面”，另写 `view_knee_viser.py`，加入自己的控件和画面，并给出新端口。新页面虽然能显示膝盖，却丢掉了已经确定的工作方式，还制造第二套播放器代码。问题不是显示能力不足，而是没有在开始前查询任务 README、stage ledger、原播放器入口和正在运行的 `http://localhost:8411/api/state`。该状态已表明原页面有帧范围 470–1380、human、brush track、拖动进度条，而且不推进物理。

## 修复方案

停止并删除新写的播放器。保留 `scripts/humanoid-real2sim/kinematic_viewer.py`、原 `scene.py`、human world skeleton 和帧范围，只把 `--motion` 指向 `push_to_brush_kinematic_v4_knee/motion.npz`。旧播放器的 `--object-track` 接口要 `position_w` 与 `quaternion_xyzw`；新增的是**数据适配**：`splice_push_v3.py` 从原组合物体估计导出 `green_hand_brush_viewer.npz`，不改页面实现。新修复若只改变 motion/sidecar，应按此方式复用；确有原入口无法表达的新信息时，先具体说明缺失的接口，再扩展现有入口并验证旧任务仍可用。

## 结果和验证

新版使用原播放器启动在 `http://localhost:8421/`，其 `/api/state` 返回 50 Hz、帧范围 470–1380、`human_motion=true`、`object_track=green_hand_brush`、`sim_steps=0`。对 `/api/command` 发送 `seek=900` 后，`cursor=900`。v4 的 viewer sidecar 与 `objects_estimated.npz` 中刷子位置、四元数逐帧完全一致，1399 帧。旧 v20 页保留在 8411 供对照。页面只表示运动学，不证明刷子或膝盖的物理接触。

## 修复反馈后的页面交付（2026-10-20）

扫桌循环 v22/v5 修复时，用户先要求关掉旧 Viser，再指出约 20 秒动作缺失。我完成了 motion 修复并关闭旧页面，却没有打开新审阅页。原因是把“关闭旧页面”当成“修复后也不提供页面”，使用户必须再提醒一次。页面是反馈修复的交付环节：先停旧候选和对应转发，再沿用本 take 已接受的播放器加载新 motion、human、scene 与物体 sidecar；检查 HTTP 可访问、数据源正确、可跳到反馈时窗，最后直接交付 URL。这是流程反馈，已加入主 skill；其他 take 的播放器、端口和数据仍从各自样本与 ledger 查。

本例新页面为 `http://localhost:8431/`，对应 `push_to_brush_kinematic_v5_cycle/motion.npz` 和同目录 `green_hand_brush_viewer.npz`。网页返回 HTTP 200，状态为 50 Hz、帧 470–1380、`human_motion=true`、`object_track=green_hand_brush`、`sim_steps=0`；跳到帧 1000 后 `cursor=1000`。旧 8411、8421 等服务保持关闭。反馈记录：`feedback/2026-10-20-viser-after-feedback.json`。

仓库工具库整理后，从原 CLI 重启同一播放器于 `http://localhost:8441/`；8431 也已关闭。新实现仍返回 HTTP 200 和上述状态，seek 到帧 1000 通过；进度条与 human+humanoid 的行为未改。

统一命名为 humanoid-real2sim 后，从 `scripts/humanoid-real2sim/kinematic_viewer.py` 再次启动在 `http://localhost:8451/`。HTTP 200、human 与刷子叠加、seek=1000 均通过；8441 已关闭。实现只有一份，位于 `robosimstudio/humanoid_real2sim/visualization/`。

最终移除旧目录及别名后，唯一页面重启于 `http://localhost:8461/`；8451 已关闭。相同的 motion、human、刷子轨迹和进度条跳帧验证仍通过。

## 代码、数据和可迁移边界

- 播放器实现：`robosimstudio/humanoid_real2sim/visualization/kinematic.py` 与同目录 `kinematic_web_ui.html`；兼容启动入口：`scripts/humanoid-real2sim/kinematic_viewer.py`。
- 本次 scene、motion、human 和物体参数：`outputs/humanoid-real2sim/2026-10-20/brush_sweep_1019/curated_snapshot/README.md`。
- 数据适配：`scripts/humanoid-real2sim/tasks/brush_sweep_1019/splice_push_v3.py`；产物：`retarget/push_to_brush_kinematic_v4_knee/green_hand_brush_viewer.npz`。
- 反馈记录：`feedback/2026-10-20-visualization-reuse.json`。别的 take 应先确认其**实际获认可**的页面与数据契约，不照搬本例端口或帧号。

## 本仓库结果与审查证据

- **审查证据** [1012_palm_approach_motion.npz](../grasp/artifacts/1012_palm_approach_motion.npz) — `kinematic_candidate`；1012 brush 0–35 s; inspect 6–12 s approach。
- **审查证据** [1012_palm_approach_closeup.png](../grasp/artifacts/1012_palm_approach_closeup.png) — `kinematic_candidate`；1012 palm-approach hand/brush geometric diagnostic。

**可证明的范围与限制：**这里的 motion 和图片是播放器的可打开输入与预期内容，不是页面运行成功截图；播放器行为仍由实现代码和实际启动检查确认。


## 外部 NX1 录制接入（2026-10-22）

01_1014 与 02_1014 的两个原始样本在家具场景中行走并接近水果目标，分别有1031和615帧、50Hz。原始 `traj.npz` 同时含 NX1 `qpos/qpos_ref`、人体 `axis_skel_pos`、物体 `object_pose`；BVH也在同一episode。它们标记为 `teleop_replay`，不能称为实机执行日志。liusy目录中找到的现成结果是NX1到G1的转换，不应拿它代替用户要求的NX1。

本例仍使用共享KinematicStudio，只增加任务数据适配器。反例风险是把未变换的人体骨架直接叠加到已spawn的机器人，或看到旧scene.xml的G1模型名就误认运动类型；应以录制robot_model、32关节顺序和实际数组为准。固定刚体变换将qpos_ref全段根坐标对齐qpos，两个样本RMS均小于5e-8m，关节逐帧完全一致。版本匹配NX1 v1.0.1仓库的33个网格哈希与录制记录完全一致；录制的简化MJCF本身未恢复。

Room_assets原始USD的146/149个对象世界变换逐项与录制配置完全一致。精确哈希恢复包缺大部分资产，使用原USD恢复房间，保留简化的漫反射颜色，未加载纹理。水果原网格没有找到，因此只展示明确标注的橙色位置标记，不能据此审查水果接触几何。房间大网格的投影阴影曾使画面发白，任务入口关闭静态背景的cast/receive shadow后已通过浏览器可见性检查。

页面验证了两个样本的human+NX1+房间，以及0、500、1030、1031、1330、1645帧跳转和播放，sim_steps始终为0。两个独立片段拼接为审阅时间轴，第二段整套场景平移12m以便同页展示；全局1031帧对应第二段源0帧，不是连续动作，也不应把显示平移加入训练数据。没有做物理验证或取得用户动作验收，因此本次不建立results。

复现代码与命令：[任务入口](../../../scripts/humanoid-real2sim/tasks/wbc_hoi_1014/README.md)。全量数据与stage_ledger保留于`outputs/humanoid-real2sim/2026-10-22/wbc-hoi-1014/runs/14-02-09.180626+0800/`。最小审查证据：[来源与页面检查](artifacts/wbc_nx1_source_review.json)、[源帧0/500/1030的数据切片](artifacts/wbc_nx1_review_keyframes.npz)。后者是稀疏诊断帧而不是连续motion。可迁移结论是保持播放器不变，验证外部数据的关节语义、固定坐标变换、时钟和资产来源；显示成功不等于接触或训练验收。
