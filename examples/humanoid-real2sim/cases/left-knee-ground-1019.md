# 左膝两次接地：自动检测漏掉第二次

同一 1019 录制的后续动作在约 `28–33 s` 和 `70–76 s` 两次以左膝下跪。旧 body prior 两次都让机器人膝盖离地约 `21–23 cm`。第一次人体膝高落入自动检测阈值；第二次人体膝高约 `8–10 cm`，高于原 `7 cm` 阈值，若只靠阈值就漏掉接触。结合视频、人体世界轨迹与动作持续时间后，第二次需要显式补标。

这条 example 说明：膝盖接触从**动作语义、连续时间和支撑几何**确定，单一人体关节高度阈值只是候选生成器。接地侧脚跟可抬起，脚尖贴地水平移动；另一侧足底继续稳固支撑。接触终点用机器人的膝/脚网格，而不是人体关节点高度。入口/退出要重新检查脚锚和接缝。

现有运动学候选 `examples/humanoid-real2sim/sequence_1019/retarget/ground_support_27_end/supported_motion.npz` 的两次左膝网格间隙约 `0.98 mm`，站立脚滑移低于 `0.001 mm`。代码在 `scripts/humanoid-real2sim/tasks/sequence_1019/refine_support.py` 与同目录的 `audit_body.py`，审查数据在 `retarget/ground_support_27_end/review/body_prior_quality.json`。这是可复用的接触切换方法，不可直接复制左膝轨迹到右膝或把两次时间窗口套到新 take。
