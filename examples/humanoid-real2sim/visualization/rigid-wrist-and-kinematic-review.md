# 运动学播放器中叠加独立手腕 rigid tracker

圆篮 69 秒后人体骨架手腕看起来异常。仅看骨架或机器人不能判断是否动捕源有错：原 MCAP 另有 `/optitrack/left_hand_rigid`、`/optitrack/right_hand_rigid`。先按 Motive 帧号对齐，再变换至与人体一致的世界坐标；tracker 装在手上，其原点不等于骨架腕点，应从稳定时窗估计固定安装偏移，并标出无效采样。

在 59–64 秒拟合偏移后，左右估计腕点与骨架腕点中位残差约 3.2/2.5 mm；69–76 秒约 13.1/16.6 mm。后段差异增加提示需要继续核查，**不能单靠两路残差判定哪一路错误**。3.5 Viser 同屏显示原人体青色骨架、橙/粉两路 rigid 原始坐标轴、浅色估计腕点、机器人、篮子和桌子；单一进度条按源时钟跳帧，只 set qpos，不做 tracker 或物理步进。[任务页](../../../outputs/humanoid-real2sim/2026-10-20/sequence_1019/curated_snapshot/tasks/06_round_basket/README.md)记启动参数与颜色约定。

[rigid 提取与同步](../../../scripts/humanoid-real2sim/tasks/sequence_1019/tasks/06_round_basket/build_review_59_76.py)是任务数据处理；[通用播放器](../../../robosimstudio/humanoid_real2sim/visualization/kinematic.py)与[网页进度条](../../../robosimstudio/humanoid_real2sim/visualization/kinematic_web_ui.html)可复用。新录制须重新估安装偏移、有效性与世界变换，不能沿用这里的残差。

## 本仓库结果与审查证据

- **审查证据** [round_basket_rigid_wrist_track.npz](artifacts/round_basket_rigid_wrist_track.npz) — `diagnostic`；Independent MCAP rigidbody wrist track aligned to 59–76 s。
- **审查证据** [round_basket_59_76_v5_motion.npz](../foot-push/artifacts/round_basket_59_76_v5_motion.npz) — `pending_kinematic`；Full 59–76 s candidate; force direction and arm residual not passed。

**可证明的范围与限制：**独立腕部 rigidbody 轨迹可与完整窗口 motion 比较；两路位姿差异只提示动捕诊断，不自动判定骨架或 rigid tracker 哪个正确。
