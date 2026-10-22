# Humanoid real2sim 工具库

这里是人形 real2sim 唯一的共享实现，包含 retarget 数据处理、RSS tracker 与可视化代码。依赖方向是 **skill 路由 → 本工具库 → `scripts/humanoid-real2sim/` 任务入口 → 运行结果 → 精选 examples**。一次使用的所有中间/最终结果、好坏候选、日志和审查写到被 Git 忽略的 `outputs/humanoid-real2sim/<date>/<task>/`；任务全量数据留在 outputs，[知识库](../../examples/humanoid-real2sim/KNOWLEDGE.md)按通用动作分类，每例携带最小直接证据并标明正反例及验证范围。不维护同一功能的第二套包或兼容别名。

| 子任务 | 真正执行的代码 | 任务层入口与边界 |
| --- | --- | --- |
| 3.1 人体/全身 retarget | ScaleTrack 的 `retarget3/src/retarget3/{body_only,ground_stage,solver,collision,quality}.py`；本仓库 `retarget/motion_fields.py` 负责 RSS 接受的 motion 字段与 FK | `scripts/humanoid-real2sim/tasks/<task>/prepare_*`、`fit_*` 是案例专属目标/约束，不作为通用求解器；retarget3 仍由 ScaleTrack 独立维护，避免复制出分叉 |
| 3.2 HOI retarget | ScaleTrack 的 `retarget3/src/retarget3/{grasp,solver,robot}.py` 与本仓库的 motion 字段工具 | 任务脚本中的 `fit_*`、`refine_*` 负责该物体的接触几何、目标与时窗；成熟的跨任务算子再提升至 `retarget/` |
| 4 tracker 和修复 | `track/{tracker,worker,raw_bfm,runtime,timeline,export,launcher,repair_studio}.py` 与 `track/web_ui.html`：ScaleBridge 推理进程、RSS 实际物理执行、回滚时间线、候选导出、编辑器 | `scripts/humanoid-real2sim/run.py` 是薄入口；场景脚本在 `scripts/humanoid-real2sim/tasks/` |
| 3.5 可视化 | `visualization/kinematic.py` 与 `kinematic_web_ui.html`：Viser/RSS 场景、human 骨架、物体轨迹、进度条和 seek API | `scripts/humanoid-real2sim/kinematic_viewer.py` 保留原 CLI 和启动命令 |

运动学页也接受 `--tracks-manifest tracks.json`，与原单物体 `--object-track`
二选一。JSON 的 `objects` 和 `joints` 数组各包含 `{ "name": "场景关节名",
"path": "相对于JSON的轨迹.npz" }`。自由物体文件含 `position_w[N,3]`、
`quaternion_xyzw[N,4]`；单自由度机构文件含 `position[N]`。两者都必须含
`fps` 和 `source_time_s[N]`，与 motion 的录制时间逐帧一致。加载时检查
四元数、有限值、场景坐标宽度，拒绝覆盖机器人关节或重复绑定。
`/api/state` 的 `scene_tracks` 给出当前各轨迹值与来源，跳帧仍只做 FK，
不执行物理步进。

物理接触由 RSS/Newton 执行；ScaleBridge worker 只返回策略目标。运动学审阅页只设置 pose/FK，不推进物理。两种页面输出不得混为同一验证结果。

新的可复用算法先放相应子包，任务脚本只传具体资产、时窗和约束。修改通用代码时，用原入口与现有示例验证兼容；不要把某个 take 的固定坐标、接触时窗或人为反馈规则写入通用算法。

## 可选离线动力学 refinement

[refinement](refinement/README.md) 提供独立的滚动窗口采样实验入口：RSS 编译场景、GPU 候选预测、CPU 连续物理执行、控制重放和实测 motion 导出。它不替换 Stage 4 tracker 或交互修复页。长方篮入口、初态投影和可选冻结 BFM 控制初值均写到新运行目录；候选是否完成搬运以物理审查为准，不自动进入训练集。
