# 脚推物体：持续勾推与一次踢击不能混用

## 动作判断

`kick_basket` 短片原参考把左脚置于圆篮外壁接触窗口；物理 tracker 回放中，左踝平均位置误差约 13.9 cm，篮子只移约 1.4 cm，而参考约 21.2 cm。把质量、摩擦或重心单独调低后，篮子只增加到约 2.3 cm；主要缺口仍是脚没有到达参考接触位置。一次 5 cm 预算的局部修复不能掩盖这个大范围跟随误差。这是**失败反例**，不是可复用的成功踢篮策略。[任务说明](../../../outputs/humanoid-real2sim/2026-10-20/kick_basket/curated_snapshot/README.md)区分了 RSS 和 Isaac 上同样存在的接触误差。

长录制的圆篮操作则是另一种模式：59 秒起左脚面与小腿形成凹陷，将篮子连续移出，67 秒停在左手可放入物品的位置；其后左脚尖贴地后移、脚跟抬起、左膝落地，右前臂推回。仅让脚在某一帧碰到篮壁，或用一脚踢出的自由飞行轨迹代替连续接触，都不能解释这个动作。[连续脚推与下跪案例](foot-hook-forearm-return-basket.md)是**待审运动学候选**，不是物理成功记录。

## 可迁移检查与代码

先用完整源视频和人体判断“短时踢击”还是“持续勾推”，标出篮子位移、脚尖/小腿接触和地面支撑的共同时间窗。持续勾推要同时审查两处表面间隙、篮子运动方向是否在可施力范围内、接触丢失时刻以及物理位移；不能把路径距离小误当作推力成立。

- [短片 RSS tracker 基线](../../../scripts/humanoid-real2sim/tasks/kick_basket/baseline.py)、[差异分析](../../../scripts/humanoid-real2sim/tasks/kick_basket/analyze.py)。
- [长录制足面/小腿拟合](../../../scripts/humanoid-real2sim/tasks/sequence_1019/tasks/06_round_basket/fit_foot_pull_59_76.py)、[下跪入口修复](../../../scripts/humanoid-real2sim/tasks/sequence_1019/tasks/06_round_basket/refine_kneel_entry_69_71.py)。
- 物理接触是否成立仍需实际 tracker/RSS 回放；运动学篮子路径不能作为已测轨迹。

## 本仓库结果与审查证据

- **审查证据** [kick_tracker_failure.json](artifacts/kick_tracker_failure.json) — `negative_physics`；Short kick task tracker replay。
- **审查证据** [round_basket_61s_contact_failure.png](artifacts/round_basket_61s_contact_failure.png) — `negative`；User screenshot at about 61 s: foot pose/contact mismatch。
- **审查证据** [round_basket_59_76_v5_motion.npz](artifacts/round_basket_59_76_v5_motion.npz) — `pending_kinematic`；Full 59–76 s candidate; force direction and arm residual not passed。

**可证明的范围与限制：**短踢任务分析 JSON 是失败物理证据；61 秒截图显示持续接触缺失，v5 motion 仍是待审候选。两段动作不可混作已通过脚推正例。
