# 左膝两次接地：自动检测漏掉第二次

**反例：**旧 body prior 让机器人两次左膝都悬空，7 cm 单阈值漏掉第二次。**正例（运动学几何）：**`ground_support_27_end` 补标两次接地并通过膝盖间隙与站立脚滑移审查；尚无物理接触结论。

同一 1019 录制的后续动作在约 `28–33 s` 和 `70–76 s` 两次以左膝下跪。旧 body prior 两次都让机器人膝盖离地约 `21–23 cm`。第一次人体膝高落入自动检测阈值；第二次人体膝高约 `8–10 cm`，高于原 `7 cm` 阈值，若只靠阈值就漏掉接触。结合视频、人体世界轨迹与动作持续时间后，第二次需要显式补标。

这条 example 说明：膝盖接触从**动作语义、连续时间和支撑几何**确定，单一人体关节高度阈值只是候选生成器。接地侧脚跟可抬起，脚尖贴地水平移动；另一侧足底继续稳固支撑。接触终点用机器人的膝/脚网格，而不是人体关节点高度。入口/退出要重新检查脚锚和接缝。

现有运动学候选 `outputs/humanoid-real2sim/2026-10-20/sequence_1019/curated_snapshot/retarget/ground_support_27_end/supported_motion.npz` 的两次左膝网格间隙约 `0.98 mm`，站立脚滑移低于 `0.001 mm`。代码在 `scripts/humanoid-real2sim/tasks/sequence_1019/refine_support.py` 与同目录的 `audit_body.py`，审查数据在 `retarget/ground_support_27_end/review/body_prior_quality.json`。这是可复用的接触切换方法，不可直接复制左膝轨迹到右膝或把两次时间窗口套到新 take。

## 本仓库结果与审查证据

- **结果（仅左膝入口）** [left_knee_entry_69_71_motion.npz](results/left_knee_entry_69_71_motion.npz) — 从 59–76 秒候选截取的 69–71 秒、100 帧左膝入口；只用于审查脚尖、脚跟和膝盖的运动学几何。
- **审查证据** [round_basket_59_76_v5_motion.npz](../foot-push/artifacts/round_basket_59_76_v5_motion.npz) — `pending_kinematic`；Full 59–76 s candidate; force direction and arm residual not passed。
- **审查证据** [left_knee_entry_v5_review.json](artifacts/left_knee_entry_v5_review.json) — `kinematic_geometry_checked`；Left knee entry 69–71 s; full task pending。

**可证明的范围与限制：**v5 motion 与入口审查只支持 69–71 秒左脚尖贴地及膝盖下降的运动学几何；整段脚推篮仍未通过。
