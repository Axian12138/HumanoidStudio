---
name: rss-scaletrack-real2sim
description: 从视频/图像、人体动作和任务描述建立 RSS 场景，使用 ScaleTrack retarget 与 ScaleBridge tracker 交互修复人形机器人和物体动作，并将确定的 motion、物体轨迹、USD 物理参数归档为 ScaleTrack 数据包。用于五阶段 real2sim、暂停/回滚修复或修复后的最终归档；单独场景搭建或 retarget 可用各自工具。
---

# RSS × ScaleTrack humanoid real2sim

本 skill 只路由任务和维护验收流程。人形 pipeline 唯一名称是 **humanoid-real2sim**：可复用实现位于 [工具库](../../robosimstudio/humanoid_real2sim/README.md)，任务脚本与 `scene.py` 位于 [scripts/humanoid-real2sim](../../scripts/humanoid-real2sim/README.md)。**一次 skill 使用的全量结果**（各阶段中间文件、好坏候选、日志、审查及这次的最终结果）放 `outputs/humanoid-real2sim/<date>/<task>/runs/<run-id>/`，日期为一级、任务为二级；不要因为文件名带 `final` 就直接当作 example，该目录被 Git 忽略。**Example 是事后提炼的可复核知识**：[正式定义](../../examples/humanoid-real2sim/KNOWLEDGE.md)要求按抓取、手推、脚推、下跪等机制分类，同时有完整语言叙述、实现代码路径、仓库内最小 motion/图片/审查证据、来源哈希、通过范围与未验证项。全量配置和资产仍留在 outputs；精选二进制证据在 examples 中走 Git LFS。全部已收录任务都有知识入口，不代表全部动作已通过。检索由 `scripts/humanoid-real2sim/feedback.py` 执行；已审查的最终 ScaleTrack 数据另按阶段 5 归档到版本化数据包。新增程序放包或 `scripts/`。

交付可再次交互的 RSS 环境和经过审查的机器人/物体数据。场景、运动学参考、tracker 实际执行、修复候选和最终数据分别留版本与证据；已有阶段被用户接受时复用，只重开受修改影响的检查。为每个 take 保留源资料、坐标/时间映射、资产与 checkpoint 版本、接触计划、源帧映射和哈希；使用独立 `scene/`、`retarget/`、`rollouts/`、`repairs/`、`review/`，不覆盖已接受版本。

## 实验结果与 example 的落盘门槛

1. 输出的一级目录只写本地日期 `YYYY-MM-DD`，二级目录是单层 `<task>`；同一天同一任务的多次运行由任务内 `runs/<run-id>/` 区分。本次 `scene/`、`retarget/`、`rollouts/`、`repairs/`、`review/` 和 `final/` 放在 `robosimstudio.humanoid_real2sim.paths.run_dir()` 返回的运行目录。嵌套 task id 中的 `/` 映射为 `__`。每个候选保留自己的版本和审查结论，失败文件与最后选定的文件都属于实验结果。
2. 审查后才从实验中提炼 example。正文写清可迁移的方法、情境与边界并链接代码。**已通过范围内的成功结果**放在类别的 `results/`，保存 motion NPZ，可附同帧估计物体轨迹或图片，并在 `results_manifest.json` 写通过阶段和仍未验证的部分；失败对照、待验证候选和诊断材料放 `artifacts/`，在 `evidence_manifest.json` 记来源、SHA-256、状态和含义。`index.json` 的 `result_ids`、`evidence_ids` 分别绑定两类文件；没有通过结果时 `result_ids` 为空，不伪造正例。运行目录、整套资产和未精选候选仍留在 outputs。新增或更新 example 后运行 `audit_knowledge.py`。
3. `stage_ledger.json` 记录某个 take 的版本和批准状态；案例的好坏结论须指向它或可复核的审查证据。归档到 ScaleTrack 不会自动生成 example，也不等于训练数据验收。

## 选择子任务

| 当前任务 | 读取 | 完成条件 |
| --- | --- | --- |
| 1 输入对齐、2 场景重建、5 物体/物理参数与归档 | [场景与物体](references/scene-objects.md) | 坐标/尺度、同相机对照、静置/碰撞、物理回放及归档校验有记录 |
| 3.1 body prior、3.2 HOI retarget | [retarget](references/retarget.md) | 全身支撑和接缝、手/物接触路径、关节与速度审查通过 |
| 4 tracker 跟随、局部修复和恢复 | [track](references/track.md) | 实际物理状态、回滚/接管/恢复与候选映射可复核 |
| 3.5 运动学或 4 物理可视化 | [visualization](references/visualization.md) | 只交付一个带时间轴的动画审阅 URL；Viser canvas 端口不作为第二入口；标清运动学估计与物理实测 |

反馈修复生成新候选后，沿用该 take 已认可的播放器；关闭旧候选的 Viser/转发进程，启动加载新 motion 与 sidecar 的页面，实际验证可访问、human/robot/物体叠加和跳帧，再交付新 URL。用户无需再次提醒。页面选择与具体参数查对应样本及 stage ledger。

按需查 [工具与产物地图](references/tool-map.md)、[本地接口和命令](references/local-interfaces.md) 与 [当前实现边界](references/implementation-boundaries.md)。第 3 阶段另读 ScaleTrack 的 `codex_skills/real2sim-human-object-retarget/SKILL.md` 及当前对象的引用。其 Isaac 物理验证不用于本流程：物理由 RSS 执行，ScaleBridge 负责 tracker 推理，MuJoCo 可作 FK/几何审查。

## 跨阶段验收门槛

- 1 → 2：人、物、支撑面、相机和时间轴能在同一坐标约定下解释；缺测尺度明确标注估计。
- 2 → 3：场景同相机对照与物体自由静置已审查。接触几何或位置发生实质变化时，重开受影响的 2/3 检查。
- 3.1 → 3.2：body prior 单独保存并在运动学页审阅。用任务切片的人体轨迹和录制时钟校验源时窗，先判定抓取、推、勾等接触类别与接触部位；3.2 保留已通过的支撑和全身动作，只优化交互时窗。
- 3 → 4：同时有 body prior、HOI 候选、源帧映射和几何/运动学审查。运动学不等于物理抓取或 tracker 跟随成功。
- 4 ↔ 5：修改物体、接触或手部时序后重放受影响的物理段；最终 motion、物体轨迹和 USD 必须来自同一已审查分支。候选导出不自动代表训练验收。

## 样本检索与反馈吸收

开始任务或修复失败前，按任务、动作和症状检索 [样本库](../../examples/humanoid-real2sim/README.md) 与 [反馈库](../../examples/humanoid-real2sim/feedback/README.md)，阅读匹配样本的语言叙述、代码、motion 和验证证据。收到反馈后区分错误、遗漏、新要求或尚待调查的问题；错误要修复实际产物，新要求要明确可观察的验收条件。按 [反馈处理与知识落点](references/feedback-routing.md) 分别判断是否改主 skill、共享工具和动作 example，并记下每项理由；三者可以同时更新。动作、对象的具体规则留在样本及阶段引用；主 skill 只维护 harness 机制。修复后补原因、方案、结果和迁移边界。单次 take 的版本和批准状态以 stage ledger 为准。
