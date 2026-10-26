# 本地仓库、命令与数据接口

此引用用于当前 `/home/galbot/workspace` 布局；迁移机器时先重新定位仓库与 checkpoint。命令中的 `/path/to/...` 是任务输入，不是现成产物。

## 仓库入口

| 用途 | 路径 |
| --- | --- |
| RSS 场景构建、物理与 Studio | `/home/galbot/workspace/robosimstudio` |
| RSS live 场景迭代 | `robosimstudio/reference-knowledge/live/scene.py`、`ctl`、`REFERENCE.md` |
| RSS 柔性示例 | `robosimstudio/reference-knowledge/01_cloth/`、`02_soft/`、`03_rope/` |
| 当前人形修复编辑器 | `robosimstudio/scripts/humanoid-real2sim/run.py`；实现见 `robosimstudio/robosimstudio/humanoid_real2sim/track/` |
| 运动学审阅实现与兼容入口 | `robosimstudio/robosimstudio/humanoid_real2sim/visualization/kinematic.py`；`robosimstudio/scripts/humanoid-real2sim/kinematic_viewer.py` |
| RSS tracker/时间线/导出 | `robosimstudio/robosimstudio/humanoid_real2sim/track/` |
| 示例障碍物场景 | `robosimstudio/scripts/humanoid-real2sim/hurdle_scene.py` |
| ScaleTrack retarget skill | `ScaleTrack/codex_skills/real2sim-human-object-retarget/SKILL.md` |
| ScaleTrack 算法与字段 | `ScaleTrack/retarget3/docs/METHOD.md`、`USAGE.md`、`retarget3/src/retarget3/motion_io.py` |
| ScaleTrack 参考 tracker launcher | `ScaleTrack/logs/debug/20261013/play_1012_tracking_only.sh`；仅参考输入顺序、checkpoint 与对比方法，不执行 Isaac 物理 |
| ScaleBridge BFM agent | `ScaleBridge/scalebridge/agent/bfm_agent.py` |

## n1：RSS live 场景

在 RSS 仓库的 `reference-knowledge/live` 目录运行 `./ctl status`，需要时运行 `./ctl up`。用 `./ctl ref frames <clip> --at ...`、`./ctl snapshot --camera <cam> --beside <still>`、`./ctl compare --camera <cam> --key ...` 和 `./ctl check --camera <cam> --object <name>` 检查场景。详见 `reference-knowledge/live/REFERENCE.md`；不要把 `check` 的几何通过解释为 tracker 或柔性接触验证。

供人形编辑器加载的场景文件必须提供 `build_sim(asset, motion) -> rss.Sim`。`asset` 是已转换为位置驱动的 G1 Dex3 MJCF；场景应使用此 asset，保留浮动根和参考动作所需名称，返回 `scene.build(control_mode="joint_position")` 的结果。`hurdle_scene.py` 演示此契约，障碍物尺寸和位置是 GIF 的估计，不代表精确重建。

## n2 运动学：retarget 输出

n2 运动学使用运动学 retarget。运动学参考在 n2 可选动力学通过 WholeBody-14 和 RSS 物理播放检查抓取、支撑及物体响应。

当前 RSS 编辑器接受 50 Hz ScaleTrack 七字段 NPZ：`joint_pos`、`joint_vel`、`body_pos_w`、`body_quat_w`、`body_lin_vel_w`、`body_ang_vel_w`、`fps`。若 NPZ 没有名称，用 `ScaleTrack/retarget3/assets/robot/motion_order.json` 映射 43 关节和 50 身体；不要凭数组列号猜测关节。另保留物体 sidecar、接触时间表与源帧映射。ScaleTrack retarget skill 有详细的身体、手掌/物体接触与局部时窗审查步骤；在本组合流程中替换其 Isaac 物理验证步骤。

## 运动学审阅播放器

从 RSS 仓库根目录启动；运动学页复用 `--scene-file` 所构建的 RSS 场景和 n2 动力学 Studio 三维视图，但只按源帧设置 `qpos` 并做 FK：

```bash
.venv/bin/python scripts/humanoid-real2sim/kinematic_viewer.py \
  --motion /path/to/retargeted_motion.npz \
  --human-motion /path/to/human_motion_world.npz \
  --scene-file /path/to/reconstructed_rss_scene.py \
  --start-frame 300 --stop-frame 600 --port 8090
```

`--human-motion` 是真实人体骨架虚影的逐帧世界坐标 sidecar，须含 `positions_w`（N×J×3）、`joint_names`（J）、`edges`（E×2）和 `fps`，并与 retarget motion 帧数/FPS 对齐。若路径未传，播放器会尝试 motion 上两级目录的 `sidecars/human_motion_world.npz`；仍不存在就报错，防止把 retarget 机器人当成人体。原始 BVH 到场景的旋转、平移、时间采样要单独记录；不要缩放人体肢体来掩盖对齐误差。`--start-frame` 含端点，`--stop-frame` 不含端点；50 Hz 的 6–12 秒对应 `[300, 600)`。可选 `--object-track /path/to/object_estimate.npz --object-name <scene_free_joint>`，sidecar 须有逐帧 `position_w` 和 `quaternion_xyzw`。访问 `http://localhost:8091/`，进度条允许直接跳到范围内任意帧；播放结束停在末帧。页面不加载 checkpoint、不启动 ScaleBridge、不推进物理，也无修复或导出操作；物体随估计轨迹显示不等于物理抓取成功。

## n2 可选动力学：修复编辑器

RSS 使用仓库的 `.venv`（Python 3.12）；BFM worker 使用 `ScaleBridge/.venv`（Python 3.11），所给 Torch-TensorRT checkpoint 需要 CUDA。已有环境可从 RSS 根目录运行：

```bash
uv run --no-sync scripts/humanoid-real2sim/run.py \
  --motion /path/to/retargeted_motion.npz \
  --scene-file /path/to/reconstructed_rss_scene.py \
  --start-frame 300 --stop-frame 600 \
  --hand-open-frame 490 --hand-close-frame 560 \
  --out /path/to/review_output
```

`--start-frame` 从该源帧的机器人姿态及场景物体初始位姿建立新物理分支，`--stop-frame` 是暂停的源帧边界；录制内容不会在此前自动经历物理演化。手部帧数仅为命令写法示例，必须从当前 clip 选取。无手的 29 关节样例省略手部参数。必要时显式提供 `--motion-order`、`--metadata`、`--checkpoint`、`--robot`、`--fk-model` 和 `--scalebridge-python`；`--soft` 加入默认 RSS 海绵，`--soft-solver superdex` 需要相应 RSS extra。无 CUDA 时可进行场景、时间线、回滚 smoke test，但无法完成真实 checkpoint 执行。

n2 动力学 三维页面默认显示青色半透明 **retarget motion 机器人**，按当前物理 slot 的源帧同步，与实体物理机器人直接对比；`--hide-reference` 仅供需要临时清屏时使用。它不是原始 human 骨架。可选 `--reference-video /path/to/source_clip.mp4` 在网页角落同步显示原始人类视频；视频不能代替 n2 运动学 的三维人体虚影。

当 tracker 的手腕执行不适合抓取时，可加 `--reference-pid-wrists right`（也支持 `left`、`both`）：选定手腕的 roll/pitch/yaw 三个位置执行器直接接收当前 refmotion 关节目标，恢复段接收插值目标；其他关节仍由 tracker 控制。此设置只改变物理控制命令，不直接设置实际 qpos，也不保证物理关节零误差跟随。rollout 保存该配置，旧配置的缓存不能混入新分支。

命令启动后保持进程运行，把**控制页面** URL（`--port` 加 1，例如 `--port 8093` 对应 `http://localhost:8094/`）交给操作者。GUI 顺序：安全初始帧暂停 → 单一播放/暂停键推进动力学并缓存逐帧状态 → 点击或拖动一条时间轴查看白色有效缓存，也可直接点橙色修复标记回退 → 双击五点圆点或文字开启 6D 目标控件，拖动位移轴与旋转环并观察半透明目标手脚，按各 link 独立滑杆设线速度及角速度 → 同一播放键从当前 slot 提交改变并继续仿真 → 点击“进入恢复”，机器人停在实际物理状态，同一时间轴改为参考衔接帧选择并显示半透明全身机器人，设插值时长后按播放进行物理 track；必要时点“回到修复”恢复进入恢复前的缓存 → 保存候选。左右手各有一个连续开合滑杆，100% 对应各手实际关节的接近限位闭合目标。恢复段双手保持进入恢复时的 PD 目标，不随身体 qpos 插值松开；已有手动开合覆盖在恢复结束后继续生效，直至操作者主动修改。移动中的五点目标从第一次 mode-4 inference 起就按当前实测位姿判断方向、让命令位姿累积推进，并从命令位姿展开 offsets 0、1、2、3、4、32 的 future ramp；接近目标时线速度和角速度分别归零，远期点会在总修复位移边界处截断。回退后没有改变命令就直接播放已有缓存，不新增标记；新修复令后缀变灰并保留连续前缀。接管期未修复的 link 继续沿源动作前进，已修复的 link 按修复目标行动，因此输出帧数仍可能改变。修复模式时间轴不允许跳到尚未模拟的物理状态；恢复模式时间轴仅预览待衔接参考，不改当前物理状态。编辑器保存控制命令和机器人、物体、柔性粒子物理状态；不要在同一分支偷偷用其他 Studio 操作改变物体状态，除非也记录该操作。

输出目录含 `rollout.npz` 和 `manifest.json`。完成恢复、使用 43/50 参考且模型顺序匹配时，还产生 `candidate_motion.npz` 和 `candidate_mapping.npz`。`rollout.npz` 的 `source_frame`、`phase`、`mode`、`commanded_joint_target_q`、`tracker_action`、`achieved_joint_q/qd`、`achieved_body_q/qd`、`achieved_particle_q/qd` 供审查。`candidate_mapping.npz` 提供输出帧到源帧与接缝权重。导出用 MuJoCo 做 FK，只依据 RSS rollout 判断动力学；manifest 的 `candidate_unreviewed` 不是训练数据批准标记。

## n1 场景修正：物体和最终复核

当前没有一条自动命令能把 Blender 物体修改、RSS 柔性状态回滚和最终数据发布连成完整闭环。保存场景资产版本和差异；启用动力学时，在 RSS 内重新播放并记录实际物体/柔性状态；按相同源帧和相同相机对比。场景修改后重验受影响的运动学；启用动力学时才重跑其影响的物理片段与候选数据检查。记录用户/AI 所修正的几何、物理参数、手部时序或 motion，以及改动前后的量化结果。

## 最终归档接口

先按验证范围选择输入。**纯运动学归档**使用已审查 motion、对应场景资产、源帧映射及可用的同步物体估计，标记 `dynamics: not_run`；不要求 rollout、恢复 manifest 或 candidate mapping，不能把估计物体轨迹标成实测。仍执行下述字段、时间轴、USD/YAML、路径及哈希校验；没有物体观测时明确缺失，不伪造数组。

**以下 rollout/恢复分支规则仅适用于动力学归档。**两种范围都使用同一已审查版本组合；公共资产与字段检查同样适用。

以目标任务的现有数据包作为 YAML/NPZ schema 参考，但从用户最后保存的修复分支重新生成新版本，不复制旧包的 `motion.npz`。先查 `manifest.json` 的 `output_frames`、`segments`、`recovery_endpoint_source_frame`、`recovery_duration_frames`，核对 `candidate_motion.npz`、`candidate_mapping.npz` 和 `rollout.npz` 的长度与阶段一致。`candidate_motion.npz` 的七字段是 `joint_pos`、`joint_vel`、`body_pos_w`、`body_quat_w`、`body_lin_vel_w`、`body_ang_vel_w`、`fps`；保留 43 关节 / 50 身体的原顺序和 `wxyz` 姿态。`candidate_mapping.npz` 的 `source_frame` 和 `phase` 是每个**输出帧**的映射，不假定源帧单调。新 `timestamps=np.arange(N)/fps`；原始录制帧可由源片段起点和 `source_frame` 单独记录。

目标物体从 `rollout.npz` 的 `achieved_body_q/qd` 取位置、姿态、线速度、角速度；先按 RSS 场景 `body_label` 确认索引，不能从数组末尾或旧对象顺序猜。`achieved_body_q` 姿态 `xyzw` 要转成物体轨迹 `quaternion_wxyz`。如果沿用旧包的其他物体数组，应按 `source_frame` 逐输出帧索引，并把目标物体列替换成该 rollout；目标物体 sidecar 与主 `motion.npz` 的四项物体数据必须逐帧相同。复制或重算的其他物体轨迹应分别标记来源，不把仿真值记为真实拍摄测量。

USD 直接引用 ScaleTrack `source/scaletrack/scaletrack/assets` 中的初始公共资产，不默认派生、烘焙或复制新 USD，也不改写公共资产。YAML 的 `scene_setup.object` 与 `scene_setup.objects` 同一对象须使用相同初始路径、位姿和尺度；能由配置表达的参数放在配置。重新打开引用的 USD，核对默认 prim、单位、轴、依赖和哈希。初始资产与审阅代理若存在形状、碰撞或物理参数差异，保存差异及原证据，不能把代理检查冒充初始 USD 的验收；另有明确授权要求修改资产时才执行该修改范围。

归档包包括 motion、可用的目标物体 sidecar、公共 USD 引用及哈希、`motions.yaml`、逐输出帧映射、provenance 和 validation；动力学包另含原始 RSS rollout 与分支 manifest。YAML 更新新版本路径、`source_frame_count`、`duration_seconds`、物体轨迹与物体 USD；保留源录制区间以供追溯。旧接触时刻若仍引用，说明它属于源动作，不是扩展后输出时间轴的事件。没有同步生成的新视觉数据时，移除旧视觉字段和引用，明确写 `visual_gt: not_generated` 或等效状态。最后从目标安装目录重新加载 NPZ/YAML/USD，核对字段形状、有限值、四元数范数、路径、资产参数、哈希及目标物体轨迹一致性；写明是否完成训练验收。
