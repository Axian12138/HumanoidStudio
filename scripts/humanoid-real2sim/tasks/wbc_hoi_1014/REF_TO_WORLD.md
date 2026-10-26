# NX1 ref → world 转换（不需要模型或场景资产）

入口：`export_ref_world.py`。依赖NumPy、SciPy。输入原始`traj.npz`，也可直接从服务器已有ZIP读取指定member，无需解压整套数据。

根部字段都是`xyz + quaternion(wxyz)`：`qpos[:,:7]`是场景world根位姿，`qpos_ref[:,:7]`是参考系根位姿。采用首帧恢复固定spawn变换：

```python
R = Rotation.from_quat(qpos[0, [4,5,6,3]]) * Rotation.from_quat(qpos_ref[0, [4,5,6,3]]).inv()
t = qpos[0, :3] - R.apply(qpos_ref[0, :3])
human_world = axis_skel_pos @ R.as_matrix().T + t
```

等价于`T_world_from_ref = T_world_root(0) @ inverse(T_ref_root(0))`。若有参考朝向，则`R_world = R @ R_ref`。全段固定同一个变换，不逐帧重新对齐；不改32个关节值。`object_pose`已经是world，不能重复应用此变换。

在服务器执行：

```bash
cd /mnt/home/xuehan/robosimstudio
.venv/bin/python scripts/humanoid-real2sim/tasks/wbc_hoi_1014/export_ref_world.py \
  /mnt/home/xuehan/huangjy/datasets/01_1014.zip \
  --member '20261008_01_pm_obstacleavoidance_reaching_familyscene_liushuaiqi/scence_25/task_00/20261008_172423_003/traj.npz' \
  --output outputs/humanoid-real2sim/2026-10-22/wbc-hoi-1014/runs/ref-world-check-01
```

输出目录必须不存在，防止覆盖旧结果。输出`ref_to_world.json`含4×4矩阵与全段位置/朝向残差；`human_world.npz`含原始帧顺序的世界人体坐标、关节名、valid、fps、原t_sim/t_wall及矩阵。输出不依赖pickle；读取原始object型关节名时只应使用已信任的原数据。

两份已核实的样本根位置分量RMS约4.53e-8m、3.57e-8m。其他样本必须检查自己的全段残差、有效帧、坐标和时间合同，不能照抄这两例的数值。异常时隔离调查，不逐帧拟合掩盖变化。

**本脚本没有+12m显示平移，也没有拼接两个episode的时钟。**原prepare.py为双场景展示添加过这些变换，不应把它的展示产物直接当RSI输入。本脚本不依赖URDF、MuJoCo、GPU、Room_assets或水果几何。

RSI边界：这里从源机器人GT根位姿恢复标定，适合数据检查/整理，不是独立校准证据。test生成器不得读取原traj里的源机器人答案；正式test的标定来源和隔离方式按RSI计划核查，无法独立确认者暂留uncertain。
