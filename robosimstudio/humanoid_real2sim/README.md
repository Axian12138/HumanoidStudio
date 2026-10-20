# Humanoid real2sim 工具库

这里是人形 real2sim 唯一的共享实现，包含 retarget 数据处理、RSS tracker 与可视化代码。依赖方向是 **skill 路由 → 本工具库 → `scripts/humanoid-real2sim/` 任务入口 → `examples/humanoid-real2sim/` 样本数据**。`examples` 包含任务叙述、配置、原始/中间/最终结果和审查产物；其中 `cases/` 保存经过反馈提炼的语言、代码路径、motion 和结果。普通候选与日志写到被 git 忽略的 `outputs/humanoid-real2sim/<task>/`。不维护同一功能的第二套包或兼容别名。

| 子任务 | 真正执行的代码 | 任务层入口与边界 |
| --- | --- | --- |
| 3.1 人体/全身 retarget | ScaleTrack 的 `retarget3/src/retarget3/{body_only,ground_stage,solver,collision,quality}.py`；本仓库 `retarget/motion_fields.py` 负责 RSS 接受的 motion 字段与 FK | `scripts/humanoid-real2sim/tasks/<task>/prepare_*`、`fit_*` 是案例专属目标/约束，不作为通用求解器；retarget3 仍由 ScaleTrack 独立维护，避免复制出分叉 |
| 3.2 HOI retarget | ScaleTrack 的 `retarget3/src/retarget3/{grasp,solver,robot}.py` 与本仓库的 motion 字段工具 | 任务脚本中的 `fit_*`、`refine_*` 负责该物体的接触几何、目标与时窗；成熟的跨任务算子再提升至 `retarget/` |
| 4 tracker 和修复 | `track/{tracker,worker,raw_bfm,runtime,timeline,export,launcher,repair_studio}.py` 与 `track/web_ui.html`：ScaleBridge 推理进程、RSS 实际物理执行、回滚时间线、候选导出、编辑器 | `scripts/humanoid-real2sim/run.py` 是薄入口；场景脚本在 `scripts/humanoid-real2sim/tasks/` |
| 3.5 可视化 | `visualization/kinematic.py` 与 `kinematic_web_ui.html`：Viser/RSS 场景、human 骨架、物体轨迹、进度条和 seek API | `scripts/humanoid-real2sim/kinematic_viewer.py` 保留原 CLI 和启动命令 |

物理接触由 RSS/Newton 执行；ScaleBridge worker 只返回策略目标。运动学审阅页只设置 pose/FK，不推进物理。两种页面输出不得混为同一验证结果。

新的可复用算法先放相应子包，任务脚本只传具体资产、时窗和约束。修改通用代码时，用原入口与现有示例验证兼容；不要把某个 take 的固定坐标、接触时窗或人为反馈规则写入通用算法。
