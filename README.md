# Humanoid Real2Sim + RSI

基于 **RoboSimStudio、ScaleTrack 和 ScaleBridge**，本仓库围绕人形机器人的全身人–物交互（Whole-body Human–Object Interaction，HOI），构建从真实动作重建到可复用知识积累的工作流。

我们关注的不只是姿态相似，还包括机器人在场景中如何抓取、推拉、搬运、下跪和支撑：哪些部位与物体接触，接触如何持续，以及动作能否在物理执行中成立。

- **Humanoid Real2Sim**：把视频、人体动作和任务描述转化为可审阅的人形机器人动作、物体轨迹与仿真场景，并通过跟踪和局部修复检查实际交互。
- **RSI 知识进化**：从配对的人体与机器人交互数据中学习可迁移的方法，在隔离测试中验证知识是否改善 G1 的动作重建，让后续任务能够复用经过检验的经验。

## Humanoid Real2Sim：从真实交互到仿真数据

流程由 [RSS × ScaleTrack real2sim skill](codex_skills/rss-scaletrack-real2sim/SKILL.md) 组织，代码、任务入口和审查证据在仓库中独立维护。

| 阶段 | 工作内容 | 主要产物 |
| --- | --- | --- |
| 1. 输入与对齐 | 对齐视频、人体动作、物体、相机、坐标系与时间轴 | 输入清单、尺度与源帧映射 |
| 2. 场景重建 | 在 RSS live 页面重建场景，检查外观、碰撞几何与自由静置 | 可交互的 `scene.py`、资产与场景审查 |
| 3. 动作重定向 | 先建立全身 body prior，再优化手、脚、膝盖、前臂等与环境的交互 | body prior、HOI motion、接触计划 |
| 3.5. 运动学审阅 | 同步显示 human、humanoid 与物体，沿源时间轴播放和跳帧 | 可复核的动画、几何与运动学检查 |
| 4. 物理跟踪与修复 | 用 tracker 执行动作，回滚并修复局部失败，再恢复跟踪 | 实际物理轨迹、修复候选与回放证据 |
| 5. 数据归档 | 将同一已审查分支的动作、物体轨迹和物理参数打包 | 版本化的 ScaleTrack 数据包 |

其中，**ScaleTrack** 提供 retarget 能力，**ScaleBridge** 执行 tracker 策略推理，**RoboSimStudio / Newton** 执行本流程的物理仿真。共享的 Viser 审阅页用于同步查看场景、人体、机器人和物体。

运动学贴合、物理执行成功和训练数据验收分别记录。每个候选保留版本、证据与尚未验证的范围，避免把“看起来接触”当作已经稳定抓取或承重。

## RSI：从动作案例积累可迁移知识

RSI 在已有场景、human motion 和源机器人 HOI 数据上开展实验，重点改进 body prior 与交互动作重建。学习对象包括接触顺序、支撑关系、手与物体的相对关系，以及失败产生的机制。

实验闭环为：

```text
输入核验与质量筛选 → 独立数据划分 → 建立 G1 baseline
    → train 学习与适配 → 冻结知识、代码和评估协议
    → G1 test 配对比较与回归检查 → 晋升、保留或拒绝知识
```

- **跨机器人学习，固定 G1 验证**：源数据可来自具有可用 URDF 的其他机器人；测试只运行 Unitree G1。没有 G1 GT 时，按冻结的交互语义约束和 G1 自身可行性评价。
- **严格隔离训练与测试**：按独立录制组划分，正式样本采用 `bad:test:train = 1:1:2`；只在 train 修改知识和算法，test 用于判断冻结版本是否有效，失败细节不回流学习。
- **分开维护候选与已验证知识**：[rsi-learned](examples/rsi-learned/README.md) 保存训练所得，[rsi-validated](examples/rsi-validated/README.md) 保存通过测试的冻结版本；实验期间原动作知识库只读。
- **以证据决定保留**：同时检查实际改善、绝对质量和回归，记录适用条件、失败反例与迁移边界。完整运行保留断点和审计材料，支持持续迭代。

执行契约见 [RSI skill](codex_skills/evolve-humanoid-hoi-knowledge/SKILL.md) 与[跨机器人验证说明](codex_skills/evolve-humanoid-hoi-knowledge/references/cross-embodiment.md)。目前仓库已建立实验协议和独立知识库结构，两个 RSI 库尚无实验知识或已验证结果；特征提取、评分等缺失接口需按数据实现并验证，skill 本身不包含后台实验服务。

## 代码、知识与实验结果

| 入口 | 内容 |
| --- | --- |
| [共享工具库](robosimstudio/humanoid_real2sim/README.md) | motion 数据处理、tracker、交互修复与可视化 |
| [任务脚本与启动入口](scripts/humanoid-real2sim/README.md) | 具体场景、任务约束、运行和审查程序 |
| [动作知识库](examples/humanoid-real2sim/README.md) | 按抓取、手推、脚推、下跪、工具使用等机制整理的案例与证据 |
| [反馈库](examples/humanoid-real2sim/feedback/README.md) | 用户反馈、失败原因、修复方法和验证记录 |
| [RSI 训练知识](examples/rsi-learned/README.md) / [RSI 已验证知识](examples/rsi-validated/README.md) | 独立维护的实验知识与晋升结果 |
| [Live 场景页面](examples/live/README.md) | 编辑 `scene.py` 后即时重建，进行场景对照和交互检查 |
| [AGENTS.md](AGENTS.md) | 仓库级协作、反馈与产物管理约定 |

一次运行的全部中间和最终文件，包括失败候选、日志和审查材料，保存到：

```text
outputs/humanoid-real2sim/<date>/<task>/runs/<run-id>/
```

全量实验数据留在 Git 忽略的 `outputs/`。经过审查后，才将可迁移的方法和最小证据整理进 `examples/`；精选二进制使用 Git LFS。动作案例分别标注运动学通过、物理失败或待审等状态，收录不代表所有阶段已成功。

## 开始使用

基础 RSS 环境使用 Python 3.12 和 `uv`。在仓库根目录执行：

```bash
uv sync
git lfs pull

# 按任务、动作和症状检索已有案例
uv run python scripts/humanoid-real2sim/feedback.py search --query '抓取 下跪 接触'

# 查看运动学审阅与物理跟踪入口
uv run python scripts/humanoid-real2sim/kinematic_viewer.py --help
uv run python scripts/humanoid-real2sim/run.py --help
```

完整 humanoid 流程还需要对应的机器人与场景资产、ScaleTrack retarget 环境，以及物理跟踪所需的独立 ScaleBridge 环境、checkpoint 和 CUDA GPU。具体配置与播放示例见[安装与运行说明](examples/humanoid-real2sim/README.md#install-and-run)。全量任务数据不随 Git 克隆分发。

通过 coding agent 执行任务时，使用 `$rss-scaletrack-real2sim` 进入重建流程，使用 `$evolve-humanoid-hoi-knowledge` 进入 RSI 实验流程，并提供相应的输入数据、目标与验证范围。

## 基础项目与许可

本项目建立在 RoboSimStudio 的 Newton-native 仿真工具包之上，保留其场景构建、Studio 和物理能力。原始项目介绍、技术报告链接、致谢与引用完整保存在 [README.original.md](README.original.md)。基础 API 示例见 [Getting started](examples/00_getting_started/README.md)。

许可证：[Apache License 2.0](LICENSE)。
