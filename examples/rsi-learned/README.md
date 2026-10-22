# RSI 训练所得知识

本目录存放仅从 train 学习的 whole body HOI 方法和失败机制，独立于 `../humanoid-real2sim/`。执行规则见 [RSI skill](../../codex_skills/evolve-humanoid-hoi-knowledge/SKILL.md)。当前没有实验知识；空目录不代表任何方法已经有效。

正文是主要知识载体。每条记录必须包含适用条件、可迁移方法、train 正反例、未验证范围、实现和复现入口。候选证据放 `artifacts/`，通过所声明检查的生成结果放 `results/`；GT 不得冒充生成结果。完整运行和测试详情留在 outputs。

`index.json` 仅登记已有直接训练证据的版本。冻结版本通过 test 后原样晋升到 `../rsi-validated/`；test 不得用于修订正文或学习新规则。基本错误的知识从当前正文、索引和检索缓存中删除，审计记录留在 outputs。

源机器人可以不同，但必须记录模型与手型、可迁移原则和G1适配方式；只有G1上的test通过才可进入已验证库。跨机器人测试不加载源模型，使用预计算的隔离交互参考与G1自身约束，规则见skill的跨机器人引用。
