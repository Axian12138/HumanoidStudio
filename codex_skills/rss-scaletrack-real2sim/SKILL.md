---
name: rss-scaletrack-real2sim
description: 从视频/图像、人体动作和任务描述建立 RSS 场景，使用 ScaleTrack retarget 与 ScaleBridge tracker 交互修复人形机器人和物体动作，并将确定的 motion、物体轨迹、USD 物理参数归档为 ScaleTrack 数据包。用于五阶段 real2sim、暂停/回滚修复或修复后的最终归档；单独场景搭建或 retarget 可用各自工具。
---

# RSS × ScaleTrack humanoid real2sim

本 skill 只路由任务和维护验收流程。人形 pipeline 唯一名称是 **humanoid-real2sim**：可复用实现位于 [工具库](../../robosimstudio/humanoid_real2sim/README.md)，任务脚本与 `scene.py` 位于 [scripts/humanoid-real2sim](../../scripts/humanoid-real2sim/README.md)，经过用户反馈且可迁移的动作样本位于 [examples/humanoid-real2sim/cases](../../examples/humanoid-real2sim/cases/README.md)，普通候选和日志放 `outputs/humanoid-real2sim/`。检索由 `scripts/humanoid-real2sim/feedback.py` 执行；具体 take 的配置、资产、阶段证据和已归档数据保持在 example 任务目录及 ScaleTrack 数据包中。新增程序放包或 `scripts/`，example 只存叙述、代码路径/使用方法与样本结果；大二进制样本走 Git LFS，代码走普通 Git。

交付可再次交互的 RSS 环境和经过审查的机器人/物体数据。场景、运动学参考、tracker 实际执行、修复候选和最终数据分别留版本与证据；已有阶段被用户接受时复用，只重开受修改影响的检查。为每个 take 保留源资料、坐标/时间映射、资产与 checkpoint 版本、接触计划、源帧映射和哈希；使用独立 `scene/`、`retarget/`、`rollouts/`、`repairs/`、`review/`，不覆盖已接受版本。

## 选择子任务

| 当前任务 | 读取 | 完成条件 |
| --- | --- | --- |
| 1 输入对齐、2 场景重建、5 物体/物理参数与归档 | [场景与物体](references/scene-objects.md) | 坐标/尺度、同相机对照、静置/碰撞、物理回放及归档校验有记录 |
| 3.1 body prior、3.2 HOI retarget | [retarget](references/retarget.md) | 全身支撑和接缝、手/物接触路径、关节与速度审查通过 |
| 4 tracker 跟随、局部修复和恢复 | [track](references/track.md) | 实际物理状态、回滚/接管/恢复与候选映射可复核 |
| 3.5 运动学或 4 物理可视化 | [visualization](references/visualization.md) | 打开对应交互页面并交付可访问 URL；标清运动学估计与物理实测 |

反馈修复生成新候选后，沿用该 take 已认可的播放器；关闭旧候选的 Viser/转发进程，启动加载新 motion 与 sidecar 的页面，实际验证可访问、human/robot/物体叠加和跳帧，再交付新 URL。用户无需再次提醒。页面选择与具体参数查对应样本及 stage ledger。

按需查 [工具与产物地图](references/tool-map.md)、[本地接口和命令](references/local-interfaces.md) 与 [当前实现边界](references/implementation-boundaries.md)。第 3 阶段另读 ScaleTrack 的 `codex_skills/real2sim-human-object-retarget/SKILL.md` 及当前对象的引用。其 Isaac 物理验证不用于本流程：物理由 RSS 执行，ScaleBridge 负责 tracker 推理，MuJoCo 可作 FK/几何审查。

## 跨阶段验收门槛

- 1 → 2：人、物、支撑面、相机和时间轴能在同一坐标约定下解释；缺测尺度明确标注估计。
- 2 → 3：场景同相机对照与物体自由静置已审查。接触几何或位置发生实质变化时，重开受影响的 2/3 检查。
- 3.1 → 3.2：body prior 单独保存并在运动学页审阅。3.2 保留已通过的支撑和全身动作，只优化交互时窗。
- 3 → 4：同时有 body prior、HOI 候选、源帧映射和几何/运动学审查。运动学不等于物理抓取或 tracker 跟随成功。
- 4 ↔ 5：修改物体、接触或手部时序后重放受影响的物理段；最终 motion、物体轨迹和 USD 必须来自同一已审查分支。候选导出不自动代表训练验收。

## 样本检索与反馈吸收

开始任务或修复失败前，按任务、动作和症状检索 [样本库](../../examples/humanoid-real2sim/cases/README.md) 与 [反馈库](../../examples/humanoid-real2sim/cases/feedback/README.md)，阅读匹配样本的语言叙述、代码、motion 和验证证据。用户反馈先入库；修复后更新对应样本，必要时新增样本，并记录可迁移条件与局限。动作、对象和页面的具体规则留在样本及对应子任务资料；主 skill 只维护阶段路由和这套检索/更新机制。主动判断反馈是否针对 harness 的路由、复用、验证或交付流程；若是，直接修订主 skill，不等用户另行提醒。单次 take 的版本和批准状态以 stage ledger 为准。
