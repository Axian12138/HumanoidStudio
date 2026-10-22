# 圆篮勾出与推回：左脚／小腿、左膝和右前臂的接触切换

**反例：**`hoi_v1/v2` 把非抓取动作误当抓把手，`hoi_v3` 漏掉 59–69 秒主要足/篮交互；旧人体 sidecar 也错配时窗。**待验证候选：**`hoi_v4_59_76` 改善脚姿态和局部几何，但篮子位移方向、右臂目标仍有明显未通过项。**当前没有通过的完整足/篮正例。**仍待用户在 3.5 Viser 审阅，后续还需物理验证。

## 人和环境在做什么

1019 长录制的完整圆篮交互是约 **59–76 秒**：左脚和小腿先移动落地圆篮，约 69 秒后左膝着地，右前臂将篮子推回。59–70 秒与左手放草莓任务并行。人体与源视频呈现的是非抓取操纵，手指没有合拢把篮子提起来。辨认动作要同时看相应肢体、篮子运动方向、膝盖支撑变化和接触时窗；篮子有把手不等于在抓把手。

## 失败及原因

旧 `hoi_v1`/`hoi_v2` 误建模为右手抓提手。首次非抓取修复 `hoi_v3` 又把任务错误切成 **69–76 秒**，遗漏了 59 秒开始的主要脚推篮动作，并在 69.7–70.5 秒另拟合了不可信的足／小腿勾篮。旧播放器的人体 sidecar 更误用全长轨迹的前 350 帧，即约 27–34 秒；仅靠帧数和 FPS 检查无法发现。正确的完整任务窗口是全长 27 秒起始轨迹的索引 1600–2449，共 850 帧。

修复时参考了用户指出的[推垃圾桶任务](../../../outputs/humanoid-real2sim/2026-10-20/push_bin_1019/curated_snapshot/README.md)及其 `fit_hoi.py`：从物体外侧选接触点，并审查物体位移方向是否落在两个接触点的受力方向之间。推桶用手指，接触点不能直接复制给脚勾；这里改用足面和小腿两处接触，使篮子位移方向大致落在二者形成的凹陷受力区域。左膝接地的既有结果与[左膝示例](../kneeling/left-knee-ground-1019.md)可作审查参考。**这次没有留下在拟合前运行 `feedback.py search` 并阅读匹配 example 的记录**；这些链接记录了实际借鉴和事后归档，不构成完整的事前检索证据。

## 解决方案与可迁移约束

已建立覆盖 59–76 秒的 `hoi_v4_59_76` 和下跪入口修订 `hoi_v5_kneel_59_76`，但接触受力方向和右臂目标仍未通过审查。下一轮 3.2 应从已审查 body prior 继续拟合 59–69 秒足面／小腿贴篮的持续移动，并与 69 秒后的左膝接地、右前臂推回组合。篮子应受地面支撑并在水平面移动；Dex3 指关节维持 body prior，不引入开合或握持目标。按 50 Hz 加半帧采样审查足、小腿、膝盖、前臂的网格对篮子凸包和地面的距离；检查多接触受力方向、穿透、速度与时窗接缝。右手动捕疑点要先用独立 rigidbody tracker 与骨架腕点在共同世界坐标中比较。

迁移到其他容器时要重新测量源窗口、接触肢体、篮壁形状、支撑面、摩擦和路径。脚与小腿形成的局部凹陷是这个几何条件下的候选约束，不是所有篮子或鞋型的固定抓点。用于 3.5 的人体 sidecar 应以任务 `human.npz` 和 `source_sync.npz` 作启动校验；场景物体初始位姿与源视频还要复核。

## 结果与证据

用户在 59–76 秒诊断页指出：61 秒左脚没有跟随人体翘起，篮子路径缺乏接触，且无桌子作位置参照。原因是全身支撑 prior 将左脚保持近水平，而诊断页的篮子在 59–69 秒只是手工时相插值。新的 [`hoi_v4_59_76`](../../../outputs/humanoid-real2sim/2026-10-20/sequence_1019/curated_snapshot/tasks/06_round_basket/hoi_v4_59_76/review.json) 从人体左脚掌相对旋转恢复踝姿态，并用足面和小腿网格共同拟合移动篮子的中心；播放器加入同坐标的 MORBYLANGA 桌子。61–64 秒脚掌法向与人体中位误差约 0.35°，足底最低约入地 1.9 mm。61–63.7 秒两接触面 50 Hz 间隙绝对值第 95 百分位约 6.5 mm，最大径向交叠约 4 mm。

用户进一步指出 67 秒篮子已到最终移出位置，左手向篮中放入一个东西。因而篮子在脚退出后至 69 秒应保持这一位置，且并行的左手动作必须与同一篮口位置核对。候选在 66.4–67.1 秒局部拟合左臂，使机器人左食指位于篮口投影内；67 秒时距篮口中心约 9.9 cm、高于篮沿约 6.9 cm。该几何关系不代表物品已落入，物品轨迹属于重叠的草莓任务。先前把桌子作为 RSS 静态网格加入场景，但用户截图中仍看不到；3.5 现用同一世界坐标的 Viser 上下文网格明确显示它。

后续用户指出 70 秒左膝下跪入口仍不自然。数值诊断发现旧候选虽然 71 秒后膝盖和脚尖已接地、脚跟约抬高 16 cm，但 69.5–70.5 秒脚尖曾离地最高约 7 cm，且旧 69–76 秒子段拼接出的左腿在 70 秒附近偏离 body prior。修复参考[右膝脚尖滑动](../kneeling/right-knee-toe-slide.md)及[同一录制左膝接地](../kneeling/left-knee-ground-1019.md)，在 69–71 秒单独重求左腿：以原人体左脚掌的水平轨迹为脚尖目标，显式约束脚尖网格离地、脚跟逐渐抬起、膝盖网格下降及髋轴扭转；保留右支撑腿、根和 71 秒后的已接地动作。新 [`hoi_v5_kneel_59_76`](../../../outputs/humanoid-real2sim/2026-10-20/sequence_1019/curated_snapshot/tasks/06_round_basket/hoi_v5_kneel_59_76/kneel_review.json) 中，69.6–70.9 秒脚尖约后移 3.0 cm，网格离地 0.4–1.5 mm；70.9 秒脚跟离地 16.5 cm、膝网格离地约 1.1 mm，最低穿地约 0.5 mm。仍须由用户在 3.5 页面审阅自然性。

**尚未完成：**篮子位移方向仅约 28% 有效采样落在足面／小腿的受力方向之间；新篮子位置下右臂固定腕／肘目标最大残差约 21 cm。这个候选用于用户观察脚姿态、桌子和物体位置，不能作为已通过的接触式推篮 example。迁移经验是不能只最小化两处接触距离，还须把连续物体速度方向纳入求解并在输出后单独审查；场景中的桌子等参照物应与物体同框显示。

[`review_59_76`](../../../outputs/humanoid-real2sim/2026-10-20/sequence_1019/curated_snapshot/tasks/06_round_basket/review_59_76/review.json) 是诊断页，59–69 秒仍是 body prior 加暂定篮子路径；不能将此段称为完成的足／篮 retarget。独立 rigid tracker 的固定安装偏移用 59–64 秒拟合后，右腕与人体骨架的中位残差在拟合段约 2.5 mm，在 69–76 秒约 16.6 mm；左腕对应约 3.2 / 13.1 mm。该差异提示检查动捕，但尚不能判定哪路有误。

旧 `hoi_v3_nonprehensile` 只覆盖 69–76 秒。其原审查中，69.8–70.2 秒左足／小腿距篮壁不超过 10 mm 的比例约 93%／100%，右臂推篮时相关表面距篮壁不超过 10 mm 的比例为 100%，且无关节越限。但因为真正脚推篮从 59 秒开始，这些指标不能证明完整任务合理。

篮子凸包与机器人网格仍有最高约 3.4 mm 局部交叠，足底约 1.4 mm 入地。篮子初始位置和地面轨迹是待复核估计。以上数值不等于物理持续接触或实际推篮成功；还需要用户在 3.5 审阅，并在后续阶段测物体响应、接触力和摩擦。

## 代码与 motion

2026-10-20 按用户要求将 `hoi_v5_kneel_59_76` 作为**待审运动学候选**归档到
`/home/galbot/workspace/ScaleTrack/data/1019test/round-basket-foot-push-kinematic-v1/`。
包内有 850 帧、50 Hz 的机器人 motion、同帧圆篮估计轨迹、源帧映射、人体与手腕 rigid
参考、场景文件和审查记录。[归档脚本](../../../scripts/humanoid-real2sim/tasks/sequence_1019/tasks/06_round_basket/archive_kinematic.py)
从原候选生成该包，目标目录重载校验通过。归档没有改变前述审查结论：脚/小腿受力方向、
右前臂目标和用户视觉审阅仍未通过；`accepted` 与 `training_data_ready` 均为 false。

随后用归档 motion 运行了 [Stage 4 RSS/ScaleBridge 物理基线](../../../outputs/humanoid-real2sim/2026-10-20/sequence_1019/runs/stage4_round_basket_baseline_01/review/stage4_baseline.json)。
850 帧、50 Hz 的 rollout 已保存；页面默认叠加青色半透明 refmotion 机器人。
这次**物理反例**中，篮子约在源录制 63 秒侧翻，倾角约 83.5°，此帧物理篮子与估计参考位置相差约 0.58 m；
全段中位相差约 0.55 m。机器人 14 个 tracker 身体点位置误差中位约 7.5 cm。
这说明当前 Stage 3 接触条件与暂定物理参数尚不足以实现篮子目标轨迹，不能将此 rollout 作为已修复动作。
此分支只播放原参考，没有接管或恢复；桌子仅供视觉参照而无碰撞。

用户随后指出篮子应更重、重心更低。独立 [Stage 4 参数候选](../../../outputs/humanoid-real2sim/2026-10-20/sequence_1019/runs/stage4_round_basket_heavy_low_com_01/review/parameter_comparison.json)
将质量设为 1.6 kg、局部重心高度设为 0.08 m，摩擦仍为 0.8；这些数值是待测量的假设。3 秒自由静置没有水平漂移。
同一 850 帧物理回放中，源帧 200 的倾角从基线 83.49° 降至近 0°，全程最大倾角 4.81°；
篮子与估计参考轨迹的中位距离从 0.546 m 降到 0.333 m。然而篮子实际水平位移仅 0.063 m，
第 200 帧仍距参考位置 0.212 m；机器人身体误差中位约 7.5 cm。该候选证明较低重心消除了基线的侧翻，
没有证明足／篮动力学接触能推出目标路径，因此仍未接受为训练数据。新回放保持青色 refmotion 虚影。

用户又指出篮子与桌脚之间需要更宽的入脚间隙，要求**固定 humanoid motion**，先把运动学篮子起点沿世界 −X 移动 5 cm，再按原接触拟合方法重求篮子轨迹。第一次只在 Stage 4 物理场景中平移篮子，仍读取旧运动学轨迹，是错误范围的候选，已标为反例。
修正后的[篮子运动学轨迹与接触结点](../../../outputs/humanoid-real2sim/2026-10-20/sequence_1019/runs/stage4_round_basket_refit_minus_x_5cm_01/retarget/kinematic_review.json)由固定的机器人足面／小腿网格、篮壁半径和踝部位移先验重新拟合。初始 X 从 0.12475 m 改为 0.07475 m；篮子保持静止到源帧 132（录制约 61.64 秒）左脚接近篮壁，然后才沿新接触路径移动。桌脚与篮子网格的抽样顶点最近距从约 15.6 cm 增至 20.4 cm，这只是几何近似。机器人输入 motion 文件及哈希保持相同；69 秒后沿用原篮子回程形状，仅按新拉出终点对齐。

新[Stage 4 动力学回放对比](../../../outputs/humanoid-real2sim/2026-10-20/sequence_1019/runs/stage4_round_basket_refit_minus_x_5cm_01/review/trajectory_comparison.json)包含 850 帧实际物体轨迹。运动学篮子全段水平位移约 18.4 cm，物理篮子仅约 0.39 cm，源帧 200 时两者相距约 15.2 cm；重篮与低重心保持直立，但固定的人形动作仍未推动它。该结果只验证了运动学物体位置重拟合与物理反应，尚未通过右前臂／左手接触、用户视觉审阅或训练验收。
Stage 4 页面用黄色线框标出逐帧运动学篮子、棕色实体标出物理篮子，并保留青色机器人 retarget 虚影；线框根据篮子网格尺寸构造，只用于区分两条轨迹。

用户选择这条较优的运动学篮子轨迹后，已在原路径替换 [ScaleTrack 数据包](/home/galbot/workspace/ScaleTrack/data/1019test/round-basket-foot-push-kinematic-v1/README.md)。[安装后校验](/home/galbot/workspace/ScaleTrack/data/1019test/round-basket-foot-push-kinematic-v1/metadata/validation.json)核对了 850 帧、机器人七个 motion 字段与原 Stage 3 完全相同、主 NPZ 与新物体 sidecar 逐帧一致。包内新 USD 写入质量 1.6 kg、局部重心高度 0.08 m、静／动摩擦 0.8，并由 `motions.yaml` 引用。Stage 4 物理回放仅用作物理参数诊断，未进入归档 motion。旧 RGB、mask、DINO 数组对应旧篮子路径，已从新包移除并将 visual GT 标为待重生成；先前完整包保存在本次运行的 `prior_archive/`。用户选定此归档版本不等于足／篮物理接触已通过或数据可直接训练。

- [任务记录与播放器命令](../../../outputs/humanoid-real2sim/2026-10-20/sequence_1019/curated_snapshot/tasks/06_round_basket/README.md)
- [本次 harness 复盘](../../../outputs/humanoid-real2sim/2026-10-20/sequence_1019/curated_snapshot/tasks/06_round_basket/harness_review.md)
- [59–76 秒诊断与 rigid 提取](../../../scripts/humanoid-real2sim/tasks/sequence_1019/tasks/06_round_basket/build_review_59_76.py)、[新脚姿态／篮子拟合](../../../scripts/humanoid-real2sim/tasks/sequence_1019/tasks/06_round_basket/fit_foot_pull_59_76.py)、[下跪入口修复](../../../scripts/humanoid-real2sim/tasks/sequence_1019/tasks/06_round_basket/refine_kneel_entry_69_71.py)、[旧子段拟合](../../../scripts/humanoid-real2sim/tasks/sequence_1019/tasks/06_round_basket/fit_nonprehensile.py)、[旧子段几何审查](../../../scripts/humanoid-real2sim/tasks/sequence_1019/tasks/06_round_basket/audit_nonprehensile.py)
- [motion](../../../outputs/humanoid-real2sim/2026-10-20/sequence_1019/curated_snapshot/tasks/06_round_basket/hoi_v3_nonprehensile/motion.npz)、[物体估计](../../../outputs/humanoid-real2sim/2026-10-20/sequence_1019/curated_snapshot/tasks/06_round_basket/hoi_v3_nonprehensile/object_estimate.npz)、[人体](../../../outputs/humanoid-real2sim/2026-10-20/sequence_1019/curated_snapshot/tasks/06_round_basket/hoi_v3_nonprehensile/human_motion_world.npz)、[审查结果](../../../outputs/humanoid-real2sim/2026-10-20/sequence_1019/curated_snapshot/tasks/06_round_basket/hoi_v3_nonprehensile/audit.json)、[stage ledger](../../../outputs/humanoid-real2sim/2026-10-20/sequence_1019/curated_snapshot/stage_ledger.json)

## 本仓库结果与审查证据

![用户指出的约 61 秒脚面与篮壁接触失败画面](artifacts/round_basket_61s_contact_failure.png)

这张 61 秒画面是反例；67 秒画面提供篮子最终拉出位置和并行左手动作的场景参照。v5 motion 仍是待验证候选。

- **审查证据** [round_basket_61s_contact_failure.png](artifacts/round_basket_61s_contact_failure.png) — `negative`；User screenshot at about 61 s: foot pose/contact mismatch。
- **审查证据** [round_basket_67s_context.png](artifacts/round_basket_67s_context.png) — `source_context`；User screenshot at about 67 s: basket final pulled-out position and left-hand deposit。
- **审查证据** [round_basket_69_76_only_motion.npz](artifacts/round_basket_69_76_only_motion.npz) — `negative`；Wrong 69–76 s crop; omits 59–69 s foot manipulation。
- **审查证据** [round_basket_59_76_v5_motion.npz](artifacts/round_basket_59_76_v5_motion.npz) — `pending_kinematic`；Full 59–76 s candidate; force direction and arm residual not passed。
- **审查证据** [round_basket_v4_review.json](artifacts/round_basket_v4_review.json) — `negative_diagnostic`；Foot/basket contact proximity does not establish force direction。
- **审查证据** [left_knee_entry_v5_review.json](../kneeling/artifacts/left_knee_entry_v5_review.json) — `kinematic_geometry_checked`；Left knee entry 69–71 s; full task pending。

**可证明的范围与限制：**61 秒截图是用户指出的旧错误；v3 motion 错漏 59–69 秒。v5 是完整时窗的运动学候选，篮子受力方向与右臂目标仍未通过，更不能证明物理推动。
