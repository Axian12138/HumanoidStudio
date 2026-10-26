# 圆柱帽顶的独立 Stage3.2

复用原 `black_ribbon_hat_grasped.usd`，而不是另外添加一个不可见抓点。原资产位于 ScaleTrack 的 `source/scaletrack/scaletrack/assets/objects/1012/`，帽冠上部替换为直径 70 mm、长度 205 mm、中心 `(0,0,0.110)`、轴向 X 的圆柱。视觉与碰撞来自同一 USD。原资产与整段 motion 保持不变。

窗口为 sequence_1019 源时间 35–50 秒，751 帧、50 Hz。仅更改右臂与右手；保留根、躯干、双腿、左臂、左手、人体和源时钟。**用户要求右拇指轴向根关节 `right_hand_thumb_0_joint` 全帧为零，仅 thumb_1 / thumb_2 弯曲。**此限制包括接近、闭合、释放与恢复，不得用根关节转动绕开碰撞。

程序：

- `prepare.py --run RUN`：用 ScaleTrack Python/pxr 导出原资产的每个视觉和碰撞部件，并截取原始 sidecar；需要 ScaleTrack 环境。
- `fit.py --run RUN --version VERSION`：固定手型族与闭合时序，拟合圆柱中心及右臂；包括手指相互避碰和肘—躯干避碰约束。输出仍为待审候选。
- `probe_open.py --run RUN`：在零轴向旋转约束下检查拇指弯曲开合路径。
- `retract.py --run RUN`：由 v6 生成 v7，释放后使用匹配边界位置/速度的五次关节曲线，避免 Cartesian IK 撤离换支造成肘部穿躯干。
- `lock_thumb.py --run RUN`：由 v7 生成 v8，硬性锁定拇指轴向根关节，并投影三帧约 0.00077 rad 的腕滚转边界过冲。
- `audit.py` / `audit_self.py --run RUN --version VERSION`：100 Hz 手帽几何和自碰撞回归。
- `bash view.sh RUN [VERSION]`：现有 Stage3 播放器；默认 v8，用户入口 http://localhost:8101/ 。

另使用 `sequence_1019/audit_joint_contract.py` 检查关节限位/相邻帧速度，`sequence_1019/audit_chair_brace_connection.py --static-room-only` 检查房间网格双向边界穿越。具体命令与结论保存在运行目录。

这是运动学候选。持握期间帽子按固定手—物变换生成估计轨迹；该约束不是抓取力的证据。未运行 Stage4 tracker、自由物体夹持或释放后重力验证，不可作为物理成功或训练验收结论。

## 用户纠正的横向圆柱版本

`rotate_asset.py`在Blender中只旋转视觉与碰撞圆柱90°；原USD保留。`probe_axis90.py`筛选握姿；`fit.py`支持圆柱yaw、掌部roll、腕中立权重和`--multistart`；`smooth_axis90.py`平滑多初值候选、预弯远端开合、固定释放腕位姿并撤离。新版本axis90_v11详见2026-10-23/sequence_1019__04_hat_cylinder_axis90运行README。最终各项几何/关节检查通过，用户审阅与物理验证仍待完成。
