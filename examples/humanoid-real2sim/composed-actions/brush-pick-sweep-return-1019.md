# 1019 推桶后取刷、右膝下跪扫桌、放回刷子

**反例：**旧取刷候选手背穿入物体，旧下身候选右膝悬空，旧扫桌候选漏掉远端行程。**正例（运动学参考）：**v20 取刷、v21 接地和 v22 扫桌的修复分别有几何审查与任务归档；Stage 4/5 未做，不能据此判定物理抓取或训练可用。

## 任务叙述

同一录制先推垃圾桶，起身后用右手从托盒外侧拿起绿手刷，接着以右膝跪地姿态扫桌，最后把刷子放回白色托盒。这个任务不是把若干单帧 IK 串起来：推桶与取刷之间有时间接缝；取刷需要掌侧进入；持刷扫桌时右膝、右脚尖、左支撑脚、躯干和右臂同时受不同接触约束；放回时刷子与托盒的几何关系再次重要。

## 哪些部分做得好，哪些曾失败

已审查的前缀保留垃圾桶推的 0–9.46 秒。取刷 v20 修正了拳眼对准但手背穿入、手腕过弯的问题；它提供了可复用的[掌侧接近 example](../grasp/palm-first-long-handle-grasp.md)。后续用户发现扫桌下跪时机器人右膝悬空；原 body prior 虽然脚不滑，却错误沿用站立脚接触模式。修正见[膝盖/脚尖 example](../kneeling/right-knee-toe-slide.md)：依据人体世界轨迹判定右膝落点与接地段，允许右脚尖贴地水平移动、脚跟抬起，同时左脚保持支撑。

## 动作组合规则

先分别保存上身 HOI 和下身接地候选，再按**共同源帧**组合。16.4–22.18 秒的右膝核心段以接地求解控制根和腿；刷子的世界扫桌路径是已审查的上身目标，用限速右臂 IK 补偿下身变化，物体轨迹随实际更新的手腕保持原局部变换。接地入口与退出仍可能出现手腕世界轨迹残差，必须单独报告，不能只报核心段膝盖达标。任何新下身动作都要重新检查桌面刷毛净空、手腕各轴、另一侧脚的滑移、全身接缝和物体相对手的轨迹。

当前组合候选是 8–28 秒的 `retarget/brush_sweep_kinematic_v21_knee/motion.npz`；完整推桶到放刷序列为 `retarget/push_to_brush_kinematic_v4_knee/motion.npz`。生成代码为 `refine_knee_v21.py` 和 `splice_push_v3.py --new ... --out ...`，核验为 `audit_knee_v21.py`。`review/knee_audit_v21.json` 给出接地、目标落点、脚尖、站立脚和刷毛数值；`retarget/brush_sweep_kinematic_v21_knee/review.json` 给出腕部与旧世界轨迹的残差。

## 可视化复用

这个 take 已接受的审阅交互是 `scripts/humanoid-real2sim/kinematic_viewer.py` 的 RSS/Viser 页面：可拖动进度条、同步原始 human 骨架与实体 humanoid、叠加估计刷子。右膝 v21 只是 motion 更新，因此继续传 `--motion retarget/push_to_brush_kinematic_v4_knee/motion.npz`、同源 `human_motion_world.npz`、原 `scene.py` 和 `green_hand_brush_viewer.npz`，保持帧范围 470–1380；不要为突出膝盖另写页面。新版实际审阅页为 `http://localhost:8421/`，页状态已验证可跳到帧 900 且 `sim_steps=0`。若服务重启，以实际打印端口为准。

## 20 秒扫桌循环反馈

用户继续指出 v21 虽然修好了抓取和膝盖，却没有表现原人体的抬刷、远端落刷、贴桌扫回、再抬刷。根因是旧扫桌拟合把刷毛 X 锁在 0.85 m，丢掉人体约 26 cm 的水平行程。新 v22 只修右臂扫桌时窗，完整的原因、源帧轨迹、求解方法、结果和边界见 [扫刷循环样本](../tool-use/brush-lift-far-place-table-stroke-lift.md)。局部 motion 是 `retarget/brush_sweep_kinematic_v22_cycle/motion.npz`，完整动作是 `retarget/push_to_brush_kinematic_v5_cycle/motion.npz`。上面的 v21 页面为历史审查记录；相关旧 Viser 服务已关闭。当前 v5 审阅页为 `http://localhost:8461/`，已验证进度条跳到帧 1000，human 与刷子轨迹正常加载。

## 归档与边界

本任务按用户要求在 Stage 3 后直接归档，Stage 4 tracker 和 Stage 5 物理抓取/扫桌/释放检查跳过。`object_estimate.npz` 与完整序列的 `objects_estimated.npz` 是运动学估计；桌面最低刷毛间隙约 `5.7 mm`，所以没有物理扫桌接触的证据。训练验收保持 false。以后检索类似任务时，可分别迁移“掌侧抓长柄”和“单膝接地脚尖滑动”的约束，再按新的接触时间轴组合；不要复制本录制的源帧、坐标、刷柄握点或几何补偿量。

## 本仓库结果与审查证据

- **结果（见限定范围）** [brush_sweep_v22_motion.npz](../tool-use/results/brush_sweep_v22_motion.npz) — `kinematic_geometry_checked`；1019 brush sweep 8–28 s segment。
- **结果（见限定范围）** [right_knee_v21_motion.npz](../kneeling/results/right_knee_v21_motion.npz) — `kinematic_geometry_checked`；1019 brush sweep 8–28 s segment, right knee。
- **审查证据** [brush_source_body_candidate_20_5s.jpg](../tool-use/artifacts/brush_source_body_candidate_20_5s.jpg) — `comparison`；Source head camera, body prior, brush candidate at 20.5 s。

**可证明的范围与限制：**两份 motion 展示右膝支撑与刷子轨迹的组合，图片只对照 20.5 秒单帧。尚不证明刷子被动力学稳定握持。
