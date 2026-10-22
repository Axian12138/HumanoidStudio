# 可迁移动作与失败机制 examples

## Example 的正式定义

一个 example 是**可复核、可迁移的单个方法或失败机制**，而不是某个任务目录、一次运行或“最终 motion”的别名。它必须同时满足下表；缺任一项的资料先留在 `outputs`，不能在 `index.json` 登记为完整 example。

| 必备部分 | 内容与验收 |
| --- | --- |
| 具体情境 | 说明人、物体、支撑、接触肢体和源时窗；允许一个动作机制引用多个任务，但不能混淆对象身份和时间轴。 |
| 可迁移结论 | 用一句可操作的规则描述约束/修复方法及适用条件；指出新场景必须重新测量的几何、时间或物理量。 |
| 正反与状态 | 对每个版本明确标出正例、反例或待验证候选。正例只对已经检查的范围成立，区分运动学几何、tracker 物理、物体响应和用户审阅；反例要给症状和原因。没有正例也可作为**失败机制 example**，但不能宣称任务已通过。 |
| 本仓库直接证据 | `evidence_ids` 至少引用一项仓库内精选 `artifacts/` 文件，记录失败、对照、待验证候选或数值审查。图片单帧不能替代整段运动，motion 也不能单独证明动力学抓取。证据须能直接支持正文的一条明确结论。 |
| 已通过范围内的结果 | **有成功结论时**，`result_ids` 链接该动作类别 `results/` 中的 motion NPZ，可附估计物体轨迹或结果图片；[结果清单](results_manifest.json)写明具体通过范围、源帧和支撑审查。结果可以只通过运动学，但必须明确“未验证物理”；没有通过的案例保持 `result_ids: []`，不能把待验证候选升格为结果。 |
| 来源与复现 | [证据清单](evidence_manifest.json)记下原始位置、SHA-256、字节数、状态和时窗/含义；正文链接任务代码、scene/sidecar 或审查入口，并写明重放所需的本地数据。 |
| 边界 | 写清未验证项和不能照搬的值。阶段 ledger 记录单次 take 的批准状态，知识 example 不替代 ledger。 |

精选结果与 artifact 跟随这个仓库，NPZ、图片等由 Git LFS 管理；全量运行、未精选候选与大场景资产仍留在 Git 忽略的 `outputs/humanoid-real2sim/<date>/<task>/`。同一份文件可以被多个相关 example 引用，不复制多份。`index.json` 提供检索字段、`evidence_ids` 和 `result_ids`，正文是主要解释，[审计脚本](../../scripts/humanoid-real2sim/audit_knowledge.py)核对索引、链接、结果/证据文件与哈希。这里的“完整 example”表示知识可复核，**不表示原任务已经物理成功**。

这里按**抓取、手推、脚推、下跪、工具操作、跟踪修复、可视化、场景物理和任务理解**组织可复用经验，不按任务名称或聊天日期分类。每个 example 先用语言讲清人在做什么、环境和支撑如何变化、机器人应保持什么；随后分别写明**反例**（哪个候选错、观察到什么、为何错及证据）和**正例**（哪个候选在什么审查范围内通过、做对了什么及证据）。尚未通过的候选明确标为待验证，不能称为正例。最后说明可迁移的条件与不能照搬的量，链接实现代码、精选运动文件和审查证据。`index.json` 是检索索引；自然语言正文是主要知识载体。

一次 skill 使用的完整结果，无论中间还是最终、成功还是失败，都放 `outputs/humanoid-real2sim/<date>/<task>/runs/<run-id>/`。这里收录对以后任务有迁移价值的**提炼案例**；建立知识库时已录入的所有任务都值得留一条任务理解或动作经验，但它们的 HOI/物理状态必须按证据分别标记。一个案例可以同时包含正例和反例，也可以只记录失败及待验证方案。任务代码位于 `scripts/humanoid-real2sim/tasks/`，motion、资产与审查结果位于 outputs 中的 task 数据目录；通用算法放 `robosimstudio/humanoid_real2sim/`。不要把整次运行结果复制进 examples。

| 知识类别 | 主要问题 |
| --- | --- |
| `grasp/` | 掌侧进入、手闭合、双手搬运与释放 |
| `hand-push/` | 人体目标轨迹与机器人可达性、自由物体接近位姿、手指主推面与卡位分工、机构关节与持续接触 |
| `foot-push/` | 踢击和足面/小腿持续勾推的区别 |
| `kneeling/` | 膝、脚尖、脚跟的接触切换 |
| `tool-use/` 与 `composed-actions/` | 持物扫掠和交互与支撑的时空组合 |
| `tracking/` 与 `visualization/` | 实际执行修复、rigid 对照、3.5 播放 |
| `scene-physics/` 与 `task-analysis/` | 静置和摩擦、交互分类及完整源时窗 |

开始新任务时先按动作和任务检索，再从多个 example 组合：例如先检索“抓长柄物体”的上身接触约束，再检索“单膝下跪”的下身接触切换，最后在同一时间轴组合两者，并重新核对手/物体世界轨迹和接缝。镜像左右腿只迁移约束结构，人体落点、脚尖路径、资产尺寸和时间窗口必须重新测量。

遇到人机尺寸不一致的推动任务，先从 human motion 确定物体最终行程与接触时序，再让机器人根部、手臂和手型适配；不能因为机器人手短就缩短任务目标。自由物体的接近位姿可在不改变最终环境关系的前提下小幅调整，同时检查手、脚扫掠碰撞。Dex3 手型按真实接触面选择：开放双指推、食中指主推配拇指卡位、或拇指宽面主推配食中指卡位；抓取和拳／前臂支撑是其他接触类别。接近阶段先避障、再形成卡位手型。[推椅案例](hand-push/articulated-drawer-and-bimanual-chair.md)给出两种手指分工、人体轨迹与物体偏移的直接证据。

```bash
python scripts/humanoid-real2sim/feedback.py search --query '单膝跪地脚跟抬起脚尖滑动'
python scripts/humanoid-real2sim/feedback.py search --task brush_sweep_1019
python scripts/humanoid-real2sim/feedback.py search --action 下跪
python scripts/humanoid-real2sim/feedback.py show right-knee-toe-slide
```

用户指出新失败时，在 `feedback/` 建有完整叙述的记录，并链接对应 example。修好后更新**同一个** example 的原因、方案、结果、代码、motion 和边界。如果是新动作则新增 example。任务反馈不自动改写主 skill；仅 harness 自身的路由、检索或维护机制需要修改主 skill。
新 example 使用 [语言优先模板](TEMPLATE.md)，并在 `index.json` 加入动作、任务、关键词和状态。
