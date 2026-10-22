# 工具与产物地图

通用实现位于 [RSS real2sim 工具库](../../../robosimstudio/humanoid_real2sim/README.md)，入口只装配参数、场景和任务约束。任务目录中的 `fit_*_vN.py`、`audit_*_vN.py` 是该 take 的固定候选和审查，不应当作通用库直接套到新物体。

| 阶段/子任务 | 现有入口 | 输入 → 产物 | 查看方式 |
| --- | --- | --- | --- |
| 1 输入/对齐 | `examples/live/ctl ref frames`；任务目录的 `prepare_*` 与源帧 sidecar | 视频、人体、标定 → 关键帧/映射/接触计划 | live 参考帧与时间表 |
| 2 场景/物体 | `examples/live/scene.py`、`./ctl status/up/snapshot/compare/check`；任务 `scene.py` 的 `build_sim(asset, motion)` | 尺寸/网格/资产 → RSS 场景 | live 页面与同相机对照 |
| 3.1 body prior/地面接触 | ScaleTrack `retarget3/{body_only,ground_stage,solver}.py`；RSS `humanoid_real2sim/retarget/motion_fields.py`；任务 `prepare_*` | 人体与场景 → 独立 motion、接触解、映射、质量报告 | 阶段 3.5 交互页；任务样本指定具体审查 |
| 3.2 HOI | ScaleTrack `retarget3/{grasp,solver}.py`；任务专属 `fit_*`、`audit_*` | body prior、目标物体/接触区 → HOI 候选与几何/运动审查 | 阶段 3.5 交互页；抓取看掌侧、手背、腕轴 |
| 3.5 运动学 | 实现 `robosimstudio/humanoid_real2sim/visualization/kinematic.py`；原 CLI `scripts/humanoid-real2sim/kinematic_viewer.py` | 新 motion、同源人骨架、可选物体估计 → 复用原页面的进度条与动画 | 只换数据；运动学证据不推断物理接触 |
| 4 tracker/修复 | 实现 `robosimstudio/humanoid_real2sim/track/`，含 `launcher.py`、`repair_studio.py` 和 `web_ui.html`；入口 `scripts/humanoid-real2sim/run.py` | 场景、motion、checkpoint → rollout/manifest/candidate | RSS 修复页、实体与参考虚影 |
| 5 物体优化/归档 | 任务场景和资产脚本；`robosimstudio/humanoid_real2sim/track/export.py`；任务 `stage_ledger.json` | 同一物理分支 → 物体轨迹、USD 变体、ScaleTrack 包 | RSS 回放、参数及逐帧校验 |

先确认当前 take 的 README、stage ledger 和已接受版本，再选择工具。详细命令和七字段 NPZ/导出约束见 [local-interfaces.md](local-interfaces.md)；模型和页面的能力边界见 [implementation-boundaries.md](implementation-boundaries.md)。

动作检索入口：`python scripts/humanoid-real2sim/feedback.py search --query '<动作与症状>'`，接着 `show <example-id>` 阅读语言、代码和 motion 证据。新用户反馈先入库；一次运行的全部实验结果留在 `outputs/humanoid-real2sim/<date>/<task>/`，审查后才把有迁移价值的正反例按通用动作提炼到 `examples/humanoid-real2sim/`。已通过范围内的 motion/物体轨迹/图片放类别 `results/`，失败或待验证候选与审查材料放 `artifacts/`，两类均记录哈希与验收范围。
