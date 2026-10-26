---
name: evolve-humanoid-hoi-knowledge
description: 在已有场景、human 和 humanoid HOI 数据上长期进化 RoboSimStudio motion real2sim 知识。支持从有 URDF 的其他机器人学习，以 G1 在隔离 test 上验证跨机器人接触与支撑规律，维护独立候选和验证知识库及每日反馈。用于 RSI 知识实验，不用于数据下载部署或单次动作修复。
---

# Whole body HOI 知识进化

从 train 的人机交互中积累可迁移知识，用 test 判断冻结知识在 **Unitree G1** 上是否有用。源机器人可以不同；G1 是固定验证目标。调用 real2sim 的 **n0 → n1 ↔ n2**：n0 核验输入与任务语义，n1 复用只读 GT 场景，n2 重建 body prior 和 HOI motion，动力学可选。

## 执行入口

- 每次开始或恢复实验，读取 [数据划分与长期协议](references/protocol.md)，核实 split、代码/知识版本、已消费 test 和每日报告状态。
- 源机器人不是目标 G1，或 G1 的手模型/关节契约不同时，必须读取 [跨机器人学习与 G1 验证](references/cross-embodiment.md)。同模型也可用其语义指标补充 GT 相似度。
- 从显式配置或当前 checkout 定位 `REPO_ROOT`，再读其 `AGENTS.md` 和 `codex_skills/rss-scaletrack-real2sim/SKILL.md`。调用既有 retarget/审阅工具；没有工具或资产时记录具体缺口，不假定本 skill 已提供实现。

本 skill 定义实验执行契约，不自带 URDF 特征提取器、跨机器人评分器或后台 worker。执行时先实现和验证缺少的接口。仅创建或编辑本 skill 不代表启动实验、安排每日任务或授权下载数据。

## real2sim 调用约定

n0 只从当前 split 允许的输入提取语义，不能向 test 生成器泄漏机器人答案。n1 验证 GT 场景可加载、坐标与模型契约一致，循环中仅诊断场景适配问题，不修改冻结几何、位置或任务目标；输入错误需隔离样本并版本化修正后统一重跑比较。

train 可在 n1/n2 间诊断并迭代动作与方法；test 仅执行预先冻结的算法和迭代预算，不根据评估结果继续修动作、改语义或调参。n2 默认交付运动学及同步审阅；是否启用动力学、所用控制器和资源预算在比较前固定，baseline 与候选一致，未运行则明确标记未验证。复用已通过范围，无需为调用而重跑完整流程或归档 ScaleTrack。

## 固定边界

1. **数据**：先按源机器人 HOI 接触可信度分 bad/good，再在 good 内随机划分 test/train。正式独立样本 `bad:test:train=1:1:2`；真实坏数据不足就缩小样本池或报告缺口，不把低排名好数据标坏。同源录制、重叠、镜像和跨机器人 retarget 变体不得跨 split。
2. **训练**：只在 train 修改知识、算法、阈值和跨机器人提取/映射规则。可使用源 URDF 和对应 GT 学习，但必须区分源模型实现与可迁移原则。
3. **验证**：test 的重建、回放及评分仅加载 G1 模型。源 test motion 的交互描述在数据整理阶段用冻结提取器预计算，作为评估器私有参考；生成器不能读取它或任何 test 机器人答案。无 G1 GT 时使用语义约束和 G1 可行性，不伪造 G1 GT。
4. **场景**：保持 GT 场景、物体几何和目标不变；允许 G1 在自身能力内改变姿态、站位和关节实现。不能缩放场景或改变接触任务来凑成功。
5. **知识**：原 `reference-knowledge/humanoid-real2sim/` 只读。`reference-knowledge/rsi-learned/` 保存 train 所得；`reference-knowledge/rsi-validated/` 只保存通过 G1 test 的冻结版本。正文以适用条件、判断依据、取舍和参考资料为主；代码、GT/结果/artifact为可选追溯，不是条目门槛。晋升仍需真实隔离test依据。
6. **测试反馈**：只能决定晋升、保留待定、拒绝或撤销；不把 test 失败细节用于新知识。测试挑选也是选择过程，不声称是无偏最终泛化成绩。
7. **结论范围**：URDF 几何/FK、接触推断、动力学实测分别标注。缺质量惯性不能声称测得真实重心；姿态贴合不证明承重或抓取成功。

## 迭代闭环

核验输入和源/G1 模型 → 冻结独立组及质量划分 → 建立 G1 baseline 与同预算比较 → train 学习可迁移规则并做 G1 适配 → 冻结知识、代码、特征与评估协议 → test 只运行 G1 并审查 → 晋升或拒绝 → 保存断点与每日报告。

每次 test 前，agent 根据场景、human 和允许观测选择知识，记录选择理由；不能看到答案或试跑收益后再选。成绩必须同时反映被选用案例和整体策略，包含失败、超时、未选用及预先定义的不可行任务比例。使用配对消融判断知识的增益，绝对合格与相对改善分开报告。

基本错误的知识从两个活动库、索引和检索缓存删除，撤销依赖它的组合知识状态；历史和审计证据留在 outputs，不继续作为可用知识。尚无证据、不适用、G1 能力不足与基本错误不能混为一类。

## 长期执行与落盘

完整运行放 `${REPO_ROOT}/outputs/humanoid-real2sim/<date>/rsi_<dataset-id>/runs/<run-id>/`。使用有限、可恢复的批次，不急于完成、不追求知识条数。每天向用户反馈数据、train 进展、知识增删、test 消耗、效果与回归、资源和次日计划；没有进展也要说明。执行前确认持久 worker/每日唤醒与交付机制实际可用，不能把写本地报告当作已发送。

代码进入既有 `robosimstudio/humanoid_real2sim/` 与 `scripts/humanoid-real2sim/`，不在 skill 中复制求解器。精选二进制走 Git LFS，全量结果留 outputs。原动作知识库没有被本实验修改、test 没有回流训练，是每轮交付必须核对的边界。
