# 动作与任务 examples

这里以**动作和任务**组织可复用经验，不按聊天日期堆失败清单。每个 example 先用语言讲清：人在做什么、环境和支撑如何变化、机器人应保持什么、为什么旧方法失败、成功候选做对了什么、可迁移的条件与不能照搬的量。随后链接实现代码、运动文件和审查证据。`index.json` 是检索索引；自然语言正文是主要知识载体。

这里只收录经过用户反馈、修复和审查后有迁移价值的样本。任务代码位于 `scripts/humanoid-real2sim/tasks/<task>/`，必要 motion、资产与审查结果位于 `examples/humanoid-real2sim/<task>/`，本目录用语言和链接组织它们；通用算法放 `robosimstudio/humanoid_real2sim/`，检索程序放 `scripts/humanoid-real2sim/`。下文仅写文件名的任务脚本也按此路径查找。普通尝试、与已有样本重复的 case、无人反馈的候选和运行日志放 `outputs/humanoid-real2sim/<task>/`；若后来出现可迁移的反馈，再从 outputs 提炼样本，不复制整个试验目录。

开始新任务时先按动作和任务检索，再从多个 example 组合：例如先检索“抓长柄物体”的上身接触约束，再检索“单膝下跪”的下身接触切换，最后在同一时间轴组合两者，并重新核对手/物体世界轨迹和接缝。镜像左右腿只迁移约束结构，人体落点、脚尖路径、资产尺寸和时间窗口必须重新测量。

```bash
python scripts/humanoid-real2sim/feedback.py search --query '单膝跪地脚跟抬起脚尖滑动'
python scripts/humanoid-real2sim/feedback.py search --task brush_sweep_1019
python scripts/humanoid-real2sim/feedback.py search --action 下跪
python scripts/humanoid-real2sim/feedback.py show right-knee-toe-slide
```

用户指出新失败时，在 `feedback/` 建有完整叙述的记录，并链接对应 example。修好后更新**同一个** example 的原因、方案、结果、代码、motion 和边界。如果是新动作则新增 example。任务反馈不自动改写主 skill；仅 harness 自身的路由、检索或维护机制需要修改主 skill。
新 example 使用 [语言优先模板](TEMPLATE.md)，并在 `index.json` 加入动作、任务、关键词和状态。
