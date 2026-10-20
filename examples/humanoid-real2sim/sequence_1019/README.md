# 1019 录制：任务切分与 Stage 3.1 全身 prior

源录制为 `ScaleTrack/data/1019test/16-31-26/1-2026-10-19-16-31-26_0.mcap`，约 129.1 秒。时间均以录制首帧为 0 秒，区间为左闭右开，动作输出为 50 Hz。前两项沿用已有 1019 成果；新建 27–129.08 秒的**连续**全身运动后，再从同一轨迹裁切各任务，重叠区内数值完全一致。

| 任务 | 时间（秒） | 物体与动作 |
| --- | ---: | --- |
| 01 | 0–9.5 | 推白色垃圾桶；沿用 `push_bin_1019` |
| 02 | 8–28 | 拿刷子、刷地并放回；沿用 `brush_sweep_1019` |
| 03 | 27–70 | 左手从茶几下捡地面草莓，放入圆篮 |
| 04 | 35–49 | 右手从沙发拿帽子，送向柜边/衣帽架；终点待确认 |
| 05 | 48–55 | 右手操作抽屉 |
| 06 | 69–76 | 右手移动圆篮 |
| 07 | 75–91 | 双手拉灰色椅子；84.5–88.8 秒右手撑餐桌 |
| 08 | 90–99 | 拿餐桌上的另一颗草莓，放入长方篮 |
| 09 | 98–109 | 双手搬长方篮 |
| 10 | 108–129.08 | 移动电视柜旁的花束至小藤篮 |

03、04、05 在源动作中确实并行：左手一直拿着地面草莓，右手先拿帽子再操作抽屉，因此这几段的重叠超过 1 秒。其余相邻交互留约 1 秒重叠。完整切分、源帧号及每段输出见 [`segments.json`](segments.json)。每个新任务有 `tasks/<id>/body_prior.npz`、`source_sync.npz` 和 `human.npz`；`source_sync.npz` 含 Motive 帧号、录制时间和三个相机最近帧号。

## 场景复核

沿用 1012 v4 的统一世界坐标与大部分已测量物体位置。`white_storage_box` 是茶几上的刷子托盘；餐桌上搬走的是 `lined_rect_basket`。原场景只有一颗餐桌草莓，但视频中地面草莓已在 66–68 秒落入圆篮，82–93 秒餐桌上仍有另一颗。因此后续场景须新增 `floor_strawberry_1019` 实例；按 30.5 秒左手位置暂估初始中心约 `(0.65, -0.05, 0.06)` 米。长方篮的位置也应在 HOI 前以标定画面复核。依据和旧场景坐标列在 [`scene_audit.json`](scene_audit.json)，未改写旧 YAML。

## 全身求解与结果

先用 `retarget3.body_only` 从原始 Motive 连续重建 5104 帧全身动作。默认结果在两次下跪时膝盖离地 21–23 厘米，因此复用 `ground_stage` 接地约束，对 28–33 秒、70–76 秒局部重求；第二次人体膝高约 8–10 厘米，超过原自动检测的 7 厘米阈值，已显式加入。接缝处重投影站立右脚以保持锚点。随后在 84–90 秒以整只右手碰撞网格约束桌面支撑和退出时的桌面净空，保持其余身体轨迹。

最终候选为 [`supported_motion.npz`](retarget/ground_support_27_end/supported_motion.npz)，统计见 [`body_prior_quality.json`](retarget/ground_support_27_end/review/body_prior_quality.json)：两次左膝网格间隙约 0.98 毫米；站立脚滑移低于 0.001 毫米；右手撑桌核心段所有碰撞网格与 `z=0.754 m` 桌面间隙为 0.09–3.12 毫米，退出至 90 秒期间为 0.38–5.07 毫米；最大关节速度为限值的 29.1%。桌面的水平范围在数值核验中为**近似代理**，非已检验的整套家具碰撞网格。

## Stage 3.5 唯一可视化

从仓库根目录启动一个完整时间轴的运动学播放页：

```bash
.venv/bin/python scripts/humanoid-real2sim/kinematic_viewer.py \
  --motion examples/humanoid-real2sim/sequence_1019/retarget/ground_support_27_end/supported_motion.npz \
  --human-motion examples/humanoid-real2sim/sequence_1019/retarget/ground_support_27_end/sidecars/human_motion_world.npz \
  --scene-file scripts/humanoid-real2sim/tasks/sequence_1019/scene.py \
  --source-offset-s 27 \
  --segments-json examples/humanoid-real2sim/sequence_1019/segments.json \
  --port 8240
```

打开 `http://localhost:8241/`。进度条为 27–129.06 秒的原始录制时间；任务菜单可跳到每段起点，重叠动作仍在同一条连续轨迹中。青色半透明骨架为原始 Motive 人体，实体机器人为最终 body prior。`scene.py` 使用从 1012 v4 USD 导出的静态家具 OBJ；导出时按已审查的 `push_bin_1019/assets/coffee_table_visual.obj` 校准世界 XYZ：Blender 的 OBJ 原始坐标为 `(X+0.04, Z, -Y-0.17)`，逆变换后桌子 256 个对应顶点最大残差小于 1 微米，地面回到 `Z=0`。后续交互的物体动态轨迹尚未做 3.2，播放器只做逐帧 `qpos`/FK，不推进物理或 tracker。

已有刷地 prior 与新段在 27 秒重叠处根位置相差约 12.7 毫米、非手部关节最大约 0.10 弧度，留待拼接优化。Round basket 的独立 Stage 3.2 候选及专用 3.5 播放页见 [`tasks/06_round_basket/README.md`](tasks/06_round_basket/README.md)。其他任务仍以 Stage 3.1 为准；所有候选均未获准作为训练 motion。

## 复现

使用 ScaleTrack 的 `.venv` 和 `PYTHONPATH=/home/galbot/workspace/ScaleTrack/retarget3/src`：依次运行 `prepare_body.py`、`python -m retarget3.cli solve --case <本目录>/body_case.yaml --output <本目录>/retarget/body_prior_27_end --mode body_prior`、`python -m retarget3.cli prepare --case <本目录>/ground_case.yaml --output <本目录>/retarget/ground_support_27_end`、`refine_support.py`、`refine_table_support.py`、`split_body.py`、`audit_body.py`。保留 `sidecars/motion_order.json`，可从 `ScaleTrack/retarget3/assets/robot/motion_order.json` 复制。
