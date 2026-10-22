# 双手搬容器与单手放置：先保留物体身份和终点

1019 长录制任务 09（98–109 秒）是双手搬长方篮，从餐桌移往茶几。目标是两个手与同一篮子的相对位姿在搬运段保持一致、篮子不穿桌沿，终点获得支撑；不能让左右手分别追逐独立点而造成相互拉扯。任务 03、08、10 则分别是地面草莓入圆篮、餐桌另一颗草莓入长方篮、花束入藤篮：同样需要先判明**是哪一个物体、哪一个容器、何时释放**。[完整切分](../../../outputs/humanoid-real2sim/2026-10-20/sequence_1019/curated_snapshot/segments.json)保留并行动作及一秒以上重叠。

这些是当前知识库的**任务语义样例**；本条所存证据仍只有 body prior，没有经过释放后重力静置或 tracker 物理回放的 HOI 正例。Task 09 后来产生了腕部和小臂沿侧壁浅预压的[待审运动学候选](sidewall-arm-preload.md)，其具体接触机制另见该例，不把它写成已通过的物理搬篮。尤其两颗草莓不能合并成同一 scene instance；03 的果实在约 67 秒进入圆篮时，08 的另一颗仍在餐桌。04 帽子终点的旧记录已经过期：用户已明确为架子，2026-10-20 的源视频复核在 47–48 秒看到帽子挂上架、49 秒右手退出。此结论仅修正动作语义，不表示机器人帽子上架 HOI 或支撑检查已通过。[场景与物体审查](../../../outputs/humanoid-real2sim/2026-10-20/sequence_1019/curated_snapshot/scene_audit.json)记录了这些身份和位置问题。

迁移到新任务时先列出物体实例、初始支撑、接触手数、交接/释放、目标容器和最终支撑，再做连续的物手约束。这里的时间、坐标、篮子尺寸和两颗草莓数量仅属于该录制。[切片生成与源帧映射](../../../scripts/humanoid-real2sim/tasks/sequence_1019/split_body.py)可供复现。

## 本仓库结果与审查证据

- **审查证据** [rect_basket_bimanual_body_prior.npz](artifacts/rect_basket_bimanual_body_prior.npz) — `body_prior_only`；Task 09 bimanual basket carry 98–109 s; HOI not fitted。
- **审查证据** [1019_segments.json](../task-analysis/artifacts/1019_segments.json) — `reviewed_task_split`；1019 long demo task windows and overlap。

**可证明的范围与限制：**长方篮 body prior 与任务切分证明双手搬运动作及窗口，尚无物手约束或释放静置通过结论。

## 源动作语义纠正（2026-10-20）

本轮[47–66 秒头部源帧](../../../outputs/humanoid-real2sim/2026-10-20/sequence_1019/runs/21-08-52.042738+0800/analysis/03_04_05/head_1.jpg)显示帽子上架后，右手关闭原本已打开的抽屉，同时左手持续持草莓。10 的[116–123 秒源帧](../../../outputs/humanoid-real2sim/2026-10-20/sequence_1019/runs/21-08-52.042738+0800/analysis/10_flower_bouquet/head_1.jpg)与[124–129 秒源帧](../../../outputs/humanoid-real2sim/2026-10-20/sequence_1019/runs/21-08-52.042738+0800/analysis/10_flower_bouquet/head_2.jpg)补全了右手持花、左手移篮、右手放花并释放、左手再推篮的顺序；末态花束斜靠柜板。以上是对旧叙述的纠正，新候选仍保留在 outputs，未增加成功 motion 示例。
