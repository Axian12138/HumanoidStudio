# 1019 从垃圾桶到刷子扫桌：运动学待审候选

## 20 秒扫桌循环修复（2026-10-20）

用户指出原人体有抬刷、远端落刷、贴桌扫回、再抬刷的循环，但 v21 仍使刷子在 19–21 秒几乎固定在 X≈0.85 m。`refine_sweep_cycle_v22.py` 在保留原抓取、下跪、左脚支撑与放回的前提下，仅重解 18.0–23.2 秒右臂；完整动作拼接为 `retarget/push_to_brush_kinematic_v5_cycle/motion.npz`。`review/sweep_cycle_audit_v22.json` 测得 19.6 秒刷子到 X=1.129 m，20.0–21.0 秒贴桌扫回 263 mm，刷毛离桌 -0.53..4.88 mm，21.2 秒后再抬起；手部无桌面穿模，右臂最大速度约 6 rad/s。`review/knee_audit_v22.json` 确认原右膝/脚尖支撑仍通过。这些是运动学几何结果，Stage 4/5 继续按用户要求跳过。原因、方法、结果和迁移边界见 `examples/humanoid-real2sim/cases/brush-lift-far-place-table-stroke-lift.md`。

新版运动学包归档于 `/home/galbot/workspace/ScaleTrack/data/1019test/push-bin-brush-sweep-kinematic-v3-cycle/`。这个新候选尚待用户审阅，包内 `kinematic_reference_accepted_by_user=false`、`accepted=false`、`training_data_ready=false`；旧 v1/v2 归档不覆盖。

此前旧的 Viser 页面和转发进程已经关闭。当前 v5 审阅页为 [新版 Viser](http://localhost:8461/)；从唯一入口 `scripts/humanoid-real2sim/kinematic_viewer.py` 启动，播放器实现在 `robosimstudio/humanoid_real2sim/visualization/`，加载 v5 motion、同源 human、原 `scene.py` 和 v5 `green_hand_brush_viewer.npz`。已验证 HTTP 200、50 Hz、human 与刷子轨迹、帧范围 470–1380，并跳到帧 1000（约 20 秒），`sim_steps=0`。服务重启后以实际打印 URL 为准。旧 v21/v4 与归档 v2 保留用于对照。

## 右膝修复与新版归档（2026-10-20）

用户指出取刷后单膝跪地时机器人膝盖悬空。原始 human motion 显示右膝在全局约 16.4–22.18 秒落地，稳定 XY 约 `(0.563, -0.037) m`。`retarget/right_knee_ground_v3/` 求解膝盖和脚尖接地、脚跟抬起；`refine_knee_v21.py` 将此下身与原掌侧取刷/持刷动作组合，`splice_push_v3.py --new ... --out ...` 保留推桶前缀。新版组合 motion 为 `retarget/push_to_brush_kinematic_v4_knee/motion.npz`；原 v20/v3 和旧归档 v1 均保留。

新版包在 `/home/galbot/workspace/ScaleTrack/data/1019test/push-bin-brush-sweep-kinematic-v2-knee/`。`review/knee_audit_v21.json` 测得右膝网格离地约 0.98 mm、右脚尖贴地、脚跟高度至少 144 mm、脚尖水平位移约 13.7 mm、左脚踝漂移低于 0.001 mm；人体膝 XY 与机器人平均相差 23 mm。组合过渡的右腕世界轨迹与 v20 最大残差约 41.7 mm、姿态 23.4°，见 v21 `review.json`。这是**运动学候选**；Stage 4/5 仍按用户要求跳过，`accepted=false`、`training_data_ready=false`。

历史审阅记录：v21 曾沿用已认可的 RSS/Viser 进度条播放器，在 8421 与原 v20 的 8411 页面对照；两个旧服务现已关闭。v21 当时加载 v4 motion，帧范围 470–1380，曾验证跳到帧 900、`sim_steps=0`。以下为历史重启命令，当前 v5 页面见本文件开头。

复用原播放器的新版启动命令：

```bash
.venv/bin/python scripts/humanoid-real2sim/kinematic_viewer.py --motion examples/humanoid-real2sim/brush_sweep_1019/retarget/push_to_brush_kinematic_v4_knee/motion.npz --human-motion examples/humanoid-real2sim/brush_sweep_1019/retarget/push_to_brush_kinematic_v3/human_motion_world.npz --scene-file scripts/humanoid-real2sim/tasks/brush_sweep_1019/scene.py --object-track examples/humanoid-real2sim/brush_sweep_1019/retarget/push_to_brush_kinematic_v4_knee/green_hand_brush_viewer.npz --object-name green_hand_brush --start-frame 470 --stop-frame 1380 --port 8420
```

播放器页面使用 `--port` 的下一端口。

新版复现顺序：`retarget3 ground prepare` 使用 `knee_case.yaml`，运行 `prepare_right_knee_ground_v3.py` 修正机器人膝目标，然后 ground solve 帧 370–830；运行 `refine_knee_v21.py`、`audit_knee_v21.py`、带 `--new/--out` 参数的 `splice_push_v3.py`，最后运行 `archive_kinematic.py --combined retarget/push_to_brush_kinematic_v4_knee --package-id hoi1019_push_bin_brush_sweep_kinematic_v2_knee --relative-path data/1019test/push-bin-brush-sweep-kinematic-v2-knee --output <空目录>`。具体动作语义和迁移边界见 `examples/humanoid-real2sim/cases/right-knee-toe-slide.md`。

2026-10-20：用户要求跳过 Stage 4/5，将当前运动学候选直接归档到
`ScaleTrack/data/1019test/push-bin-brush-sweep-kinematic-v1/`。该包是
**kinematic-only** 参考，包含估计的物体轨迹；未执行 tracker、物理接触/释放验收或新视觉 GT，
`accepted=false`、`training_data_ready=false`。归档复现脚本是 `archive_kinematic.py`。

录制：`ScaleTrack/data/1019test/16-31-26/1-2026-10-19-16-31-26_0.mcap`。工作范围为录制的约 9.5–27.0 秒；8–9.5 秒用于和已完成的垃圾桶动作接缝。旧的 0–9.46 秒推桶关节和根轨迹来自 `push_bin_1019/retarget/hoi_candidate_scene.npz`，保持原值。新动作没有套用 1012 的时间表；1012 只提供 IKEA 坐标系、场景资产和刷子把手的几何先验。

## 当前产物

- `retarget/body_prior_8_28/motion.npz`：同一新录制 8–28 秒的 999 帧全身先验，50 Hz。支撑审查为脚滑移最大约 `1.7e-7 mm`、脚底最低高度约 `0 mm`、躯干方向最大误差 `0.76°`；还未由操作者审阅。
- `retarget/brush_sweep_kinematic_v20/motion.npz`：重做 3.2 后的右手取刷、扫桌、放回运动学候选。根、腿、躯干和左臂沿用全身先验。把手抓取点移到托盒侧壁之外的可握段；掌面从盒外侧朝向把手，右腕三个轴限制在 ±35°。`object_estimate.npz` 仍是固定手刷变换的估计轨迹，不是测得的物理夹持。
- `retarget/push_to_brush_kinematic_v3/motion.npz`：原推桶段与新版刷子段相接的 1399 帧、43 关节/50 身体七字段 ScaleTrack 参考，时间 0–27.98 秒。接缝在全局帧 474、9.48 秒；位置和速度校正在随后 30 帧消退。`objects_estimated.npz` 中的桶和刷子轨迹均为运动学估计。`source_sync.npz` 保留每帧对应的 Motive 和相机帧。
- `scene.py`：继承 1012 白色托盒、绿色手刷和 1019 垃圾桶资产的 RSS 场景。桌面代理按 1012 静态场景 `hoi_batch_30` 的范围 `x=0.76..1.36, y=-0.71..0.69, top z=0.400 m` 放置。推桶部分沿用 0.3 kg、0.10 摩擦、30 mm 局部重心及 26 个碰撞形状的候选配置。场景已在 CPU 上构建，刷子初始及放回位姿各做 5 秒自由静置检查。
- `sidecars/contact_schedule.json`：源视频观察时段、误差范围与独立设计的 Dex3 开合命令。

## 看图审阅

旧版 v20 Viser 审阅页运行时为 `http://localhost:8411/`，可拖动时间轴观察人体骨架、机器人、刷子估计轨迹。`review/candidate_v19_close_12.4s.jpg`、`candidate_v19_close_12.8s.jpg`、`candidate_v19_close_25.6s.jpg` 和 `candidate_v19_close_26.2s.jpg` 展示新版取刷、持握与放回；第 20 版只修正 22.58–23.04 秒的腕俯仰角，这些关键帧不变。画面均为**运动学**渲染，刷子轨迹仅是估计；原视频相机与模拟固定视角没有做像素级外参标定。

`review/kinematic_audit.json`、`review/hand_audit_v20.json` 和 `review/pick_audit_v20.json` 记录刷毛高度、关节速度、掌部/刷柄、手部/托盒及手部/桌面的网格检查。新版 15–22 秒的刷毛最低离桌约 `4.9 mm`；掌部和刷柄按 50 Hz 检查没有超过 `0.5 mm` 的顶点穿透，托盒和桌面按每两帧抽样没有发现手部穿透。闭手指节的刷柄网格深度在拟合关键帧约 `0.2 mm`。这些离散网格检查仍不能证明连续扫掠无碰撞、夹持力或实际桌面接触。放回位姿较原位偏移约 `18 mm`，还需物理静置回放。

## 阶段边界

用户已指定沿用场景与前两个阶段，不重新做。当前新的全身先验和第 20 版 HOI 候选尚待运动学画面审阅。之后需以 ScaleBridge WholeBody-14 和 RSS 场景重放，记录实际手指角度/接触力、刷子相对手掌滑移、桌面接触、放回稳定性以及机器人与参考的跟随误差。现有几何结果不能标成物理成功或训练数据。

## 复现

在仓库根目录运行：

```bash
PYTHONPATH=/home/galbot/workspace/ScaleTrack/retarget3/src /home/galbot/workspace/ScaleTrack/.venv/bin/python scripts/humanoid-real2sim/tasks/brush_sweep_1019/prepare_body.py
PYTHONPATH=/home/galbot/workspace/ScaleTrack/retarget3/src /home/galbot/workspace/ScaleTrack/.venv/bin/python -m retarget3.cli solve --mode body_prior --case /home/galbot/workspace/robosimstudio/examples/humanoid-real2sim/brush_sweep_1019/body_case.yaml --output /home/galbot/workspace/robosimstudio/examples/humanoid-real2sim/brush_sweep_1019/retarget/body_prior_8_28
.venv/bin/python scripts/humanoid-real2sim/tasks/brush_sweep_1019/prepare_sync.py
.venv/bin/python scripts/humanoid-real2sim/tasks/brush_sweep_1019/fit_brush_v19.py
.venv/bin/python scripts/humanoid-real2sim/tasks/brush_sweep_1019/finalize_brush_v20.py
.venv/bin/python scripts/humanoid-real2sim/tasks/brush_sweep_1019/audit_candidate.py
.venv/bin/python scripts/humanoid-real2sim/tasks/brush_sweep_1019/audit_hand_v20.py
.venv/bin/python scripts/humanoid-real2sim/tasks/brush_sweep_1019/audit_pick_v20.py
.venv/bin/python scripts/humanoid-real2sim/tasks/brush_sweep_1019/splice_push_v3.py
PYTHONPATH=/home/galbot/workspace/ScaleTrack/retarget3/src MUJOCO_GL=egl /home/galbot/workspace/ScaleTrack/.venv/bin/python scripts/humanoid-real2sim/tasks/brush_sweep_1019/render_review_v19.py
```

本工作目录中的旧版均保留为中间候选；第 20 版仍是待审版本。`stage_ledger.json` 记录来源哈希和剩余验收项。
