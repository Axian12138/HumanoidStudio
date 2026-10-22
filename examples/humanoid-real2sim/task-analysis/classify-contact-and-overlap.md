# 先判定交互类别，再切完整动作窗口

## 反例：看见把手就误判抓取

圆篮有把手，但 1019 原人体在 59–76 秒没有手指合拢提篮：59 秒起左脚/小腿移篮，67 秒左手放入草莓，69 秒后左膝跪地、右前臂推回。早期 `hoi_v1/v2` 误做右手提把手；`hoi_v3` 又从 69 秒切起，把前半段脚推动作丢掉。人体 sidecar 甚至取成 27–34 秒，帧数和 FPS 看似正确，语义和录制时间却错误。失败机制及独立手腕 rigid 对照见[圆篮案例](../foot-push/foot-hook-forearm-return-basket.md)。

## 正确的拆解方法

在切片前同时检查完整视频、人体世界轨迹、物体运动、接触部位、手指开合及支撑变化。物体的名称或几何把手只能提示可能的交互，不能决定抓取。并行动作允许长于一秒重叠：1019 的 03 地面草莓在 27–70 秒持续，04 帽子与 05 抽屉发生在其间；06 圆篮在 59–76 秒，又与 03 的左手放入物品同时发生。不能用上一任务的结束时刻当下一任务的起点。每段保留录制时间、Motive 帧号、人体/机器人对应帧，并在 3.5 同屏验证。[切片清单](../../../outputs/humanoid-real2sim/2026-10-20/sequence_1019/curated_snapshot/segments.json)覆盖本录制十个任务。

对目前仅有 body prior 的任务，以下是**值得保留的任务理解样例**，尚非 HOI 正例：03 左手从茶几下捡地面草莓再放入圆篮；04 右手从沙发拿帽子并放到架子；05 右手关闭原本已打开的抽屉（当前源窗未观察到主动拉开）；08 右手从餐桌拿另一颗草莓放入长方篮；09 双手搬长方篮；10 右手抓花并持握，左手移出藤篮，右手放花释放，最后左手推篮。后续重建要分别补物体身份、接触时窗和放置终点，不得从 body prior 推断抓取已成立。

## 代码和边界

- [长录制切分](../../../scripts/humanoid-real2sim/tasks/sequence_1019/split_body.py)、[body prior](../../../scripts/humanoid-real2sim/tasks/sequence_1019/prepare_body.py)、[59–76 秒人体与 rigid 对时](../../../scripts/humanoid-real2sim/tasks/sequence_1019/tasks/06_round_basket/build_review_59_76.py)。
- [任务级状态](../../../outputs/humanoid-real2sim/2026-10-20/sequence_1019/curated_snapshot/stage_ledger.json)区分运动学候选、审阅与训练验收；新录制的接触时间和目标物不能照搬本例。

## 本仓库结果与审查证据

- **审查证据** [1019_segments.json](artifacts/1019_segments.json) — `reviewed_task_split`；1019 long demo task windows and overlap。
- **审查证据** [round_basket_61s_contact_failure.png](../foot-push/artifacts/round_basket_61s_contact_failure.png) — `negative`；User screenshot at about 61 s: foot pose/contact mismatch。
- **审查证据** [round_basket_67s_context.png](../foot-push/artifacts/round_basket_67s_context.png) — `source_context`；User screenshot at about 67 s: basket final pulled-out position and left-hand deposit。

**可证明的范围与限制：**切分 JSON 和两张用户截图支持 59–76 秒完整时窗与 67 秒并行放物的任务理解；不单独证明 HOI 优化成功。

2026-10-20 的[源动作复核及边界](../grasp/bimanual-carry-and-placement.md#源动作语义纠正2026-10-20)修正了旧帽子终点、抽屉开关假设和花束漏段。该修正来自源视频观察，不把 body prior 或尚在优化的候选升级为 HOI 正例。
