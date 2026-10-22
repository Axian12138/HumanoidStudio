# Humanoid real2sim harness 复盘（2026-10-20）

本次把所有现有任务都纳入动作知识库，同时将历史全量结果迁到 `outputs/humanoid-real2sim/2026-10-20/<task>/`。一级目录只精确到迁移日期；五个原任务目录也已移到各任务的 `curated_snapshot/`。`examples` 直接按动作类别存放文字案例，并精选仓库内直接证据。迁移前五个任务共 867 个普通文件；逐文件 SHA-256 校验，输出中的精选快照共 330 个文件。任务清单见 [curation_manifest.json](curation_manifest.json)，案例结果见 [results_manifest.json](results_manifest.json)，审查证据见 [evidence_manifest.json](evidence_manifest.json)。全量 `outputs` 被 Git 忽略；小型证据由 Git LFS 管理。

## 已修的 harness 问题

| 问题 | 处理与证据 |
| --- | --- |
| 知识按任务名散落，难以迁移到新动作。 | [index.json](index.json) 现按抓取、手推、脚推、下跪等类别索引 16 条案例；[feedback.py](../../scripts/humanoid-real2sim/feedback.py) 支持 `--knowledge-category` 筛选。全部四个顶层任务及长录制十个子任务至少有一条知识案例。 |
| `examples` 混有完整 rollout、多次拟合和中间文件。 | 全量历史结果和原任务目录均移入按日期归档的本地结果；精选快照与归档位置在 manifest 中逐任务列出。必要资产、motion、审查结果和 stage ledger 也在 outputs。 |
| 移动案例可能使交叉链接失效。 | [audit_knowledge.py](../../scripts/humanoid-real2sim/audit_knowledge.py) 检查索引、任务覆盖、相对 Markdown 链接、精选文件、输出目录形状和 examples 内的 rollout 目录。 |
| 案例容易把局部几何改善说成整个任务通过。 | 案例分别注明正例的通过范围、反例、未验收项；圆篮完整接触式推动仍是候选，约 28% 的篮子位移方向满足双接触受力区检查，右臂残差约 21 cm。 |
| 案例只有文字与本地输出路径，克隆后没有可复核样例。 | [正式定义](KNOWLEDGE.md)要求每例绑定仓库内直接证据；现有 16 例引用 5 份限定范围内通过的 motion 结果（其中 3 份带估计物体轨迹）和 24 份精选审查证据，带来源与 SHA-256。审计会核对文件、哈希和正文链接。 |

## 下一步能力缺口

| 优先级 | 缺口 | 可验收的改进 |
| --- | --- | --- |
| 高 | 很多历史任务脚本仍把 `example_dir(__file__)` 当作读写根目录。这个兼容入口现指向 outputs 下的 `curated_snapshot`，因此旧脚本不会再写进 examples，但仍可能覆盖精选快照，未实现单次运行隔离。 | 将任务脚本的快照输入根与运行输出根分离；默认运行创建 `outputs/<date>/<task>/runs/<run-id>/`，显式 `--out` 仅允许指定运行目录；静态检查阻止脚本写入 examples。对每个活跃入口做一次干净目录烟测。 |
| 高 | 本地 `outputs` 不进 Git。克隆仓库的人可看精选证据，但没有完整 scene/资产时仍不能重放原任务。 | 若需跨机器重放，另建显式的数据包导出/下载流程，记录哈希与案例链接；不要将完整归档误提交。 |
| 高 | 任务理解、源时窗和接触类别曾出错：圆篮被当成抓取，59–69 秒脚推漏切，人体 sidecar 错接其他时间段。 | 在 3.2 前建立机器可读的事件清单，记录源帧范围、接触肢体、物体、抓取/非抓取、支撑和相邻任务重叠；播放器启动时核对 motion、human、rigid tracker 的源帧映射。见[圆篮复盘](../../outputs/humanoid-real2sim/2026-10-20/sequence_1019/curated_snapshot/tasks/06_round_basket/harness_review.md)。 |
| 高 | 接触距离优化不能证明持续推动的受力方向，也不能证明物理运动。 | 把双接触速度方向、连续接触、穿透、摩擦及释放后的静置做成共享审查；真实通过需物理回放及用户在 3.5 审阅。 |
| 中 | 目前的知识检索是文字重合评分。类别筛选已加入，但来源、证据状态、接触肢体和动作阶段尚未作为结构字段统一排序。 | 索引记录类别、肢体、接触类型、阶段、验收范围和证据；检索结果显示采用与拒用理由，并保存拟合前检索记录。 |
| 中 | 精选与归档是一次性人工策展；task README 中历史版本路径有时仍指向早期归档子目录。 | 增加 `promote-example` 命令：从运行目录选证据、生成 manifest、检查链接与 stage ledger、核对归档哈希，再更新索引。历史 README 保留归档说明，不能当作当前可运行命令。 |
| 中 | 不同脚本对输出、候选状态和播放器端口的处理不统一。 | 统一运行 manifest，记录输入哈希、候选状态、源帧范围、URL 和进程；默认只播放未废弃候选，页内可看到确切版本。 |

历史实验是否为好 example 与其任务是否已经通过，是两件事。这里保留所有已建立任务的知识条目；运动学、接触和物理通过状态仍按各自证据判断。优先解决输入/输出分离和时窗/事件校验，再自动化案例提炼。
