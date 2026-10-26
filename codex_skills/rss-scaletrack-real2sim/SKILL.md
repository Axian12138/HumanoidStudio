---
name: rss-scaletrack-real2sim
description: 从图片、视频、人体 motion 和任务描述重建可交互 RSS 场景及 humanoid motion。用 n0 理解数据与语义，n1 场景和 n2 动作循环调试；支持运动学重建、可选动力学修复及按需 ScaleTrack 归档。
---

# RSS × ScaleTrack humanoid real2sim

流程是 **n0 → n1 ↔ n2**：先理解任务与数据，再联合调试场景和动作，使交互成立，同时尽量符合输入图片、motion 和任务本义。动力学是 n2 的可选分支；归档是按需交付，不是必须经过的重建阶段。

## n0：理解任务、数据与语义

确认数据读取方式、字段/模型顺序、单位、坐标与四元数约定、相机、源时窗及帧映射。由允许使用的输入识别物体身份、交互角色、接触类别与顺序、支撑关系和任务终点，区分观测、推断与未知。固定任务不可改变的语义、可调整的场景参数，以及本次交付范围（运动学或含动力学）。

产物是可复用的输入/变换/时间映射和语义约束记录；人、物、支撑面与相机须能共同解释。输入有矛盾先修 n0，不用场景或动作变形掩盖读取错误。详见[场景与物体](references/scene-objects.md)。

## n1：构建并修正适合交互的场景

统一处理场景初建与后续修正：位置、几何、碰撞代理、关节、支撑及必要的物理参数。用 RSS live page 对照输入，兼顾视觉/motion 一致性、机器人可交互性和任务语义；已有可靠场景直接复用。调整须记录依据、偏差和资产版本，不能靠改变物体身份、接触任务或目标终点凑成功。GT 冻结场景遵守调用方的只读约束。

详见[场景与物体](references/scene-objects.md)。场景可加载、坐标一致且相关接触几何可解释后即可进入 n2；自由静置和物理参数检查按动力学交付范围执行，不作为纯运动学重建的强制门槛。

## n2：重建 humanoid motion

- **运动学**：先保存并审阅全身 body prior，再在其上细化 HOI，保留已通过的支撑和动作。检查接触类别/部位/时窗、可达性、穿透、限位、速度与接缝。读取 [retarget](references/retarget.md) 和[可视化](references/visualization.md)；沿用该 take 已认可的播放器，更新候选后验证 human、robot、物体同步及跳帧，交付一个可访问的动画审阅 URL。
- **动力学（可选）**：需要实际执行、物理验证或动力学修复时，且模型、控制器和资源可用，读取 [track](references/track.md)。ScaleBridge 做 tracker 推理，RSS/Newton 推进物理；MuJoCo 可做 FK/几何审查。缺依赖时报告缺口，不以 smoke test 冒充 rollout，也不阻塞独立的运动学交付。

运动学拟合不证明稳定抓取、承重或 tracker 成功。未运行动力学时明确标记未验证；若用户要求动力学而尚未完成，应报告该范围未完成，不能以“可选”替代要求。

## n1 ↔ n2：按问题循环

接触不合理时先定位原因：数据/语义问题回 n0；场景位置、碰撞或参数问题回 n1；姿态、接触时序或控制问题在 n2 修正。每轮绑定场景和 motion 版本，复用已接受结果，仅重验受改动影响的几何、动作及接缝；启用动力学时再重跑受影响的物理段。达到 n0 约定的语义和交付范围后结束，未通过项明确保留。

历史对应：旧阶段 1 → n0；旧阶段 2 与 5 的场景/物体修正 → n1；旧阶段 3/3.5 → n2 运动学及审阅；旧阶段 4 → n2 可选动力学。旧阶段 5 的归档独立为按需交付。历史文件名、CLI 和 `stage_ledger.json` 不因命名调整而迁移或改写批准记录。

## 产物、交付与知识

共享实现位于[工具库](../../robosimstudio/humanoid_real2sim/README.md)，任务入口位于 [scripts/humanoid-real2sim](../../scripts/humanoid-real2sim/README.md)。全量候选、失败结果、日志和最终结果放 `outputs/humanoid-real2sim/<date>/<task>/runs/<run-id>/`（由 `paths.run_dir()` 创建）；保留源映射、接触计划、版本、哈希和 take 的 stage ledger，不覆盖已接受版本。

按需[归档 ScaleTrack 数据包](references/scene-objects.md#按需归档到-scaletrack-数据包)：motion、物体轨迹和资产须来自同一已审查版本组合；运动学估计与物理实测分别注明来源，跳过动力学无需伪造 rollout。归档直接引用 ScaleTrack `source/scaletrack/scaletrack/assets` 中已有初始 USD，不重新导出或复制 USD；位姿与尺度写入配置，资产与审阅代理的差异须标明验证边界。归档成功、运动学通过、动力学通过和训练验收分别记录。

开始任务或修复前，用 `scripts/humanoid-real2sim/feedback.py search --query '<任务 动作 症状>'` 检索[样本与证据](../../reference-knowledge/humanoid-real2sim/README.md)。用户反馈按[反馈路由](references/feedback-routing.md)记录类型、原因、修复和验证，并分别决定 skill、工具、example 是否更新。动作规则留在样本，主 skill 只维护 harness。按[知识规范](../../reference-knowledge/humanoid-real2sim/KNOWLEDGE.md)提炼文字参考知识：适用条件、观察与判断依据、方案取舍、迁移边界和有阅读目的的参考资料。简洁决策说明不写成内部思考记录或版本流水账。代码、motion和artifact是可选追溯，不是新条目的门槛；已保留的证据仍校验哈希和声明范围。运行 `audit_knowledge.py`；全量实验数据仍留outputs。参考文献不能代替本次验证，RSI晋升仍需隔离实验依据。

具体操作按需读取[工具地图](references/tool-map.md)、[本地接口](references/local-interfaces.md)与[实现边界](references/implementation-boundaries.md)。使用 ScaleTrack retarget 时再读取其 `codex_skills/real2sim-human-object-retarget/SKILL.md`；其中 Isaac 物理步骤由本流程的可选 RSS 动力学替代。
