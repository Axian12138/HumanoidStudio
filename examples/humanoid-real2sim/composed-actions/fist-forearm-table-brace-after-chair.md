# 推椅后握拳撑桌：把物体操纵与身体支撑分成独立片段

## 人和环境在做什么

1019 录制中，人先在 79–83.4 秒双手推灰色扶手椅进入餐桌下，随后松开椅子，在约 84.5–88.8 秒用**右手握拳、手心朝下，连同小臂撑住桌面**。这两段的接触对象和受力角色不同。可独立训练的撑桌片段取录制时间 `[83.5, 91.0)` 秒、50 Hz 共 375 帧；开头与推椅片段重叠 1 秒，便于之后对齐。椅子在本片段保持推完后的估计位姿，桌面属于客厅场景。

## 反例：原有姿态未形成桌面支撑

原 `hoi_v3_palm_push` 的后半段仍沿用推椅轨迹。右掌朝下程度在撑桌窗口内平均只有 0.135（1 表示完全向下），右手离桌沿较远；左手网格在桌面范围内有 **297 帧穿透**，最低点为 0.616 m，而桌面为 0.754 m。[改动前几何审查](artifacts/table_brace_1019_before_review.json)由同一 MuJoCo 模型对原 motion 的 `[83.5,91.0)` 窗口复算。仅检查腕部位置或单个手掌碰撞体，会遗漏左手和指节穿模。

## 解决方案与可迁移约束

先把推椅和撑桌切成两条 motion，并保留 1 秒重叠。对撑桌段，在 86 秒找到**机器人可达且拳头、小臂靠近桌面**的腕/肘姿态。右手 OptiTrack rigid 的相对位移和相对旋转引导这一锚点之后的运动；由于手背 marker 安装和人机尺寸不同，不把经相机外参换算的绝对机器人手腕 pose 强制设为目标。右掌法向向下，拳头接近桌面，右小臂底部靠近桌面；同时把左手抬离桌面。用完整手部碰撞网格逐帧扫桌面覆盖区，并限制关节范围与轨迹速度。

迁移到其他桌面时，需要重新测量桌面高度、边沿、拳头与小臂的碰撞几何、人体 marker 到机器人可行锚点的关系及接触时窗。相机对齐腕 pose 可作为另一个弱先验候选，但这个已通过的版本没有将它用于拟合。来源姿态对接触不利时，应保留多种接触候选供审阅，不能把 rigid 旋转当成必须逐帧满足的真值。

## 正例：运动学视觉范围内通过

[撑桌 motion](results/table_brace_1019_motion.npz)在 85.3–87.8 秒的右掌向下程度平均为 **0.9997**；右小臂碰撞体底部平均为 **0.758 m**，桌面为 0.754 m。83.5–90 秒的完整手部网格扫掠中，左右手在桌面范围内均为 **0 帧穿透**；右手最低点 0.7578 m，左手最低点 0.8182 m。关节无越限，最大关节速度约 6 rad/s。[改动后审查](artifacts/table_brace_1019_after_review.json)记录这些数值。用户于 2026-10-20 观看后反馈“握拳撑桌感觉没问题”。这证明了运动学姿态与视觉可接受性，**尚未验证实际支撑力、tracker 跟随或桌面动力学接触**。

## 代码与 motion

- [撑桌拟合与全网格审查](../../../scripts/humanoid-real2sim/tasks/sequence_1019/tasks/07_armchair/refine_split_feedback.py)：右手 rigid 相对运动、桌面接触、左手避让和关节约束。
- [相机对齐腕参考的提取](../../../scripts/humanoid-real2sim/tasks/sequence_1019/tasks/07_armchair/extract_hand_camera_reference.py)：参考量已保存，但本片段未将绝对 pose 加入优化。
- [独立运动学包归档脚本](../../../scripts/humanoid-real2sim/tasks/sequence_1019/tasks/07_armchair/archive_table_brace.py)：七个机器人字段、椅子估计轨迹、scene YAML、源时间与 rigid 诊断。
- [任务 stage ledger](../../../outputs/humanoid-real2sim/2026-10-20/sequence_1019/curated_snapshot/stage_ledger.json)记录撑桌视觉认可和未开始的动力学阶段。完整运行在 `outputs/humanoid-real2sim/2026-10-20/sequence_1019__armchair_split_feedback/runs/16-57-32.904219+0800/`；ScaleTrack 包在 `data/1019test/brace-dining-table-kinematic-v1/`。

## 本仓库结果与审查证据

- **运动学视觉正例：**[table_brace_1019_motion.npz](results/table_brace_1019_motion.npz)，对应的[椅子估计轨迹](results/table_brace_1019_chair_estimate.npz)仅是静止位姿估计，不是物理 rollout。
- **反例证据：**[table_brace_1019_before_review.json](artifacts/table_brace_1019_before_review.json)，证明旧轨迹左手网格穿桌。
- **正例审查：**[table_brace_1019_after_review.json](artifacts/table_brace_1019_after_review.json)，证明新轨迹在所测桌面区域内没有手部网格穿透，不能证明真实接触力。
