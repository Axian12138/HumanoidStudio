# NX1 WBC HOI → G1：交给 Codex Dot 的长期 RSI 实验计划

状态：仅计划。尚未创建控制程序、启动实验、划分正式数据或配置每日任务。

执行依据：[evolve-humanoid-hoi-knowledge](../../codex_skills/evolve-humanoid-hoi-knowledge/SKILL.md)、其 [protocol](../../codex_skills/evolve-humanoid-hoi-knowledge/references/protocol.md) 与 [cross-embodiment](../../codex_skills/evolve-humanoid-hoi-knowledge/references/cross-embodiment.md)。本文只补充本数据集接入和远端管理方案；通用实验协议以 skill 为准。执行前重新读取服务器 checkout 的 AGENTS.md 和 skill。

## 1. 目标、职责和边界

用户将本计划交给本地 Codex Dot 长期执行。第一项工作是在 Dot 沙箱中创建一个小型**服务器实验管理代码库**，通过 SSH 管理 DSW 上的有限、可恢复批次。本地模型负责阅读允许的证据、提出训练假设、选择冻结知识和汇报；Python 求解器、数据适配、FK、场景检查、retarget、评估及全量产物都在服务器执行。不要把“本地模型”理解为需要在服务器部署一个大语言模型，也不要把实验迁回沙箱计算。

最终有两类独立产物：

1. 有 train 因果证据、经过隔离 G1 test 筛选的 Stage3 知识：适用条件、动作机制、失败边界、实现与最小证据。
2. 本 NX1 数据集对应的 G1 body-prior 和 HOI retarget motion，以及每个 episode 的来源、质量状态、失败原因和验证范围。不存在 G1 GT 不是阻塞；禁止把生成结果改名为 GT。

先完成可信的小规模闭环，再扩大覆盖，不预设必须每天新增知识。Stage1/2沿用给定场景，只校验输入；以Stage3.1/3.2/3.5为主。Stage4不是本轮成功的必要条件，运动学合格不代表物理抓取成功。

## 2. 已确认的数据与位置

SSH：

```bash
ssh -p 22 root@dsw-notebook-dsw-075cjah521ezb1dztc-22.vpc-2ze1gdbfg5bklsvv01624.instance-forward.dsw.cn-beijing.aliyuncs.com
```

| 项目 | 位置或事实 |
| --- | --- |
| 服务器仓库 | `/mnt/home/xuehan/robosimstudio`，启动时核实 commit 和工作区 |
| 服务器环境 | `${REPO_ROOT}/.venv/bin/python`；可使用 `/mnt/home/xuehan/deployment/rss/activate.sh`，先核实路径 |
| 原始数据 | `/mnt/home/xuehan/huangjy/datasets/01_1014.zip`、`02_1014.zip`，约18.3和19.1GB |
| 原始索引 | 上次分别发现1360、1297个 `traj.npz`；这是文件数，不是有效独立组数 |
| 场景 | `/mnt/home/xuehan/huangjy/datasets/Room_assets/` |
| 已有G1结果 | `/mnt/home/xuehan/liusy/PerceptiveLoco_project/storage/retarget_nx1_g1_*` 及 `retarget_nx1_ik_*`，含validated/pending等目录；名称不代表本实验验收 |
| 既有转换代码 | `/mnt/home/xuehan/liusy/PerceptiveLoco_project/src/mjlab/scripts/tracking/retarget_nx1_archive_to_g1.py` 等；它们属于外部项目，只读审计 |
| NX1仓库 | `http://git.galbot.com/galbot_description/robots/galbot_nx1_description.git` |
| 已核实模型版本 | 录制元数据：NX1 v1.0.1，`direct_drive_knees_movable_head`，32个关节，根加关节共39维 |

未确认原始数据已完整解压到liusy下面。其已确认的01_1014/02_1014目录是NX1→G1派生结果。执行agent应先查现有目录和来源，不重新下载整套数据、不覆盖外部项目。压缩包可以作为只读输入；按需解出有限批次并缓存，避免每个样本反复启动SSH扫描ZIP。

上次本地审阅资料：

- `outputs/humanoid-real2sim/2026-10-22/wbc-hoi-1014/runs/14-02-09.180626+0800/` 下的 `review_report.json`、`robot_asset_audit.json`、`scene_alignment_audit.json`、`source_index.json`（各样本子目录）和原样本。
- 任务适配代码：`scripts/humanoid-real2sim/tasks/wbc_hoi_1014/{prepare.py,export_scene.py,view.py}`；这是可参考的审阅适配器，**不是已经完成的RSI adapter或评分器**。
- 本地审阅时机器人仓库HEAD为 `841e1460157762231222d9086b76977e1a699605`；执行时固定该版本或经过审计的新版本，不能静默使用最新型号。
- 两个已看过的episode：`20261008_172423_003`、`20261008_164701_001`。它们及同源/近重复/派生版本只能作为校准或train候选，不能进入test。

只把必要代码、计划和小型来源回执同步到服务器；本地模型资产若服务器缺失，先确认已授权接入范围和已有副本，再按哈希部署必要闭包，不复制整个本地工作区。后续所有数据路径以远端配置为准，不把本地绝对路径写死进求解器。

## 3. 先在沙箱实现实验管理程序

以下为**待实现接口**，不是现成命令。建议建立 `rsi-remote-control/`，只依赖Python标准库与系统SSH/rsync，不在控制端安装RSS、CUDA或数据集。

```text
rsi-remote-control/
  remote_ctl.py            # CLI与SSH调用，结构化JSON请求/响应
  config.json              # 主机、端口、远端repo/run根、资源预算
  state.json               # 当前experiment/run/batch、版本、最后回执
  jobs/                    # 小型提交清单；不含GT或测试答案
  receipts/                # 状态/哈希/退出码/受限摘要
  reports/                 # 每日用户汇报及投递回执
```

建议命令合同：

| 命令 | 行为与成功证据 |
| --- | --- |
| `preflight` | 核验SSH、仓库commit、环境、GPU/磁盘、retarget依赖、只读输入、知识库基线哈希；输出具体缺口 |
| `sync-code --revision …` | 只同步本实验代码或固定Git revision，检测远端脏工作区并用隔离checkout；记录文件/commit哈希 |
| `submit --manifest …` | 提交有run ID、预算、输入白名单和版本哈希的有限批次；返回job ID及持久状态路径 |
| `status --job …` | 读心跳、PID/进程起始时间、队列、GPU占用、已完成量和退出码，不靠SSH连接存活推断任务状态 |
| `collect --job …` | 只拉取摘要、审计ID和允许证据；按train/test角色过滤，不递归下载outputs |
| `pause/resume --job …` | 当前样本完成后落断点；恢复时核对完整指纹，不重复消费test |
| `stop --job …` | 只终止登记在该run中的进程组；不得使用宽泛pkill或停止他人作业 |
| `daily-report --date …` | 汇总允许的进度、晋升状态、资源、阻塞与下一批计划；交给Dot发送给当前用户 |

远端薄worker放 `scripts/humanoid-real2sim/`，可复用逻辑放 `robosimstudio/humanoid_real2sim/`；不把求解器放进skill。worker使用独立进程/持久会话执行，持久化stdout/stderr、退出码、心跳与checkpoint；SSH断线不应终止计算。采用任务锁、原子写状态、幂等job ID。崩溃后检查已有产物和哈希再续跑，不因为控制端回执丢失重新提交同一个测试。

使用 `subprocess` 参数数组传SSH参数，结构化JSON通过stdin送远端；不要把样本名、路径或模型输出直接插入shell。凭据使用既有SSH配置，不能写进config、日志或报告。支持命令超时、有限重试和取消；输出轮询间隔有上限，不无限忙等。

服务器完整目录：

```text
${REPO_ROOT}/outputs/humanoid-real2sim/<date>/rsi_wbc_nx1_1014/runs/<run-id>/
  experiment_config.json
  inventory/  calibration/  quality/  splits/
  source_analysis/  train/  frozen/  production/
  evaluator_private/  reviews/  logs/  checkpoints/
  ledger.json  status.json
```

沙箱只保留能够恢复控制的状态。服务器是任务状态和产物的权威来源。控制程序必须经过断线、重复submit、worker失败、预算超限及resume的干运行验证，之后才允许提交正式批次。不要提交DLC作业。

## 4. 把原始NX1格式转为显式数据合同

每个episode通常包含 `motion.bvh`、`traj.npz`、`scene_config.json`、`scene.xml`、`qc.json`、`video.mp4`。先生成全库索引和资产依赖清单，不假定两个已审阅样本代表全库。

| 原字段 | 已知语义 | 适配要求 |
| --- | --- | --- |
| `meta` | JSON字符串，含robot/model/joint_order/来源/场景配置 | 逐条验证robot=nx1、版本、关节列表、单位，保存原文哈希 |
| `qpos [N,57]` | 已观察布局：world root7 + joints32 + padding11 + object7 | 按meta解析；32关节为7:39，不能把padding当手指，也不能只凭57猜契约 |
| `qpos_ref [N,39]` | 参考根7+32关节，根在参考坐标系 | 与qpos根的spawn变换不同；不是第二个机器人、不是G1答案 |
| `axis_skel_pos [N,20,3]` | 原始人体关节位置 | 使用命名关节与valid标记；不要用NX1 FK替代人体 |
| `axis_skel_joints` | object dtype命名数组 | 受控读取可信源后转成纯Unicode数组；标准产物使用allow_pickle=False |
| `axis_skel_valid [N]` | 人体有效性 | 缺失段不能静默插值后当完整GT，保留mask和处置规则 |
| `object_pose [N,K,7]`、`object_ids` | 物体世界位姿，已观察wxyz | 逐对象按ID绑定，支持K不为1；由元数据核实四元数顺序 |
| `sample_hz`、`t_sim`、`t_wall` | 采样率及两个时钟 | 两个样本50Hz，不等于所有episode都均匀；保留原时间和源帧映射 |
| `dex3_left_q/right_q` | 额外记录通道 | 不能假定是NX1实际手型或直接拿来控制G1 Dex3 |
| `phase/wrist_dist/foot_ground_dist/scene_collision` | 录制器产生的状态/诊断 | 未核实生成逻辑前只作辅助；had_collision=false不等于接触可信 |
| `scene_config_json/object_config_json` | 逐对象几何、位姿、资产、轨迹绑定等 | 解析成显式清单，不能仅根据旧scene.xml标题选择机器人 |
| `motion.bvh` | 原人体动作 | 与NPZ交叉验证帧、轴和长度单位；BVH单位不能直接当米 |

标准样本建议分离为：`human.npz`、`source_nx1.npz`、`objects.npz`、`scene_manifest.json`、`source_sync.npz`、`provenance.json`。字段包含命名顺序、meters/Z-up声明、wxyz/xyzw标记、原始时间、有效掩码和资产哈希；源GT不得和生成器输入混在同一NPZ。不要直接把原始traj.npz交给test生成器。

### 坐标与时钟最容易出错的地方

上次审阅用 `T_world_from_ref = T_qpos(0) × inverse(T_qpos_ref(0))` 对齐人体。两样本全段根位置RMS小于5e-8m，32关节逐帧相同，说明它们是固定spawn映射；这只证明这两个样本，不是全库保证。

正式实验优先从录制器坐标合同、场景start_pose/spawn参数和输入标定元数据重建同一个映射，在校准/train上验证并冻结。test生成器只能读已标准化human和显式标定，不能读取源qpos后自己拟合对齐。若必须从源GT恢复缺失标定，先区分“录制数据预处理的输入标定”与“为降低输出误差而拟合答案”；将其留在隔离整理流程，核验独立记录证据并披露来源。无法独立确认、只能逐样本猜变换的样本留uncertain，不作为正式test。禁止逐帧平移、Procrustes或时间拟合掩盖误差。

校验 `t_sim` 的间隔、重复帧、停顿和丢帧；`t_wall`及视频时长可能包含等待，不能用视频总时长平均摊配motion。若需重采样，位置/关节与四元数分别处理，所有human、source、objects使用同一时间映射，保留原始版本及源帧索引；评估接触时序容忍度在校准/train确定。

上次双样本页面为展示把第二段全体平移了+12m X并拼接时间轴。**这些显示变换绝不能进入RSI标准数据、评分或G1导出。**从原traj.npz重新适配，每个episode保持自己的原始世界与时钟。

## 5. 模型与场景门槛：先补闭包，再谈HOI

源NX1固定v1.0.1真实模型。33个原始视觉网格在两个样本中与指定仓库哈希一致，但捕获时的 `galbot_nx1_v1_0_1_simplify.xml` 还未恢复；版本匹配URDF不自动证明惯性、collision、根和工具端完全相同。根据metadata核验FK、关节轴/限位、末端和实际手型，记录model hash及inertial_quality。骨盆不能冒充CoM；虚拟抓取/焊接也不能冒充真实手接触。

G1使用仓库已有、固定版本的真实模型及手型。优先审计已有Stage3所用G1+Dex3契约；若只有29DOF无手模型，先做body prior/适用的非抓取动作，抓取任务标能力或资产不足。43DOF/50身体是已有编辑器的一种合同，必须实际核验，不能为凑字段增加不存在的手指。记录各关节和body名字，导出速度用真实时间求导并作G1 FK。

Room_assets中 `IKEA_0907_100Scenes_TrainingAssets/IKEA_0907_100Scenes_IsaacSim/scenes/scene_25.usda` 和 `scene_48.usda` 已分别与录制146/149个对象世界变换逐项一致。后续每个scene照此验证，不能把它推广为全库资产都完整。保持GT几何/目标/支撑面，不为G1改尺寸或移动目标。

精确资产恢复包 `dataset_restore/assets/` 不完整，上次请求248个视觉/贴图资产只有13个存在。水果原网格未找到。可以用原USD恢复已匹配的场景，但橙色标记仅作显示，绝不能用于抓取距离、穿透、质量筛分或知识晋升。

按能力范围处置缺失：

- 可确认场景、human、源模型的episode继续做Stage3.1、导航/支撑等**证据完整的子任务**，报告覆盖范围。
- 依赖缺失物体几何/源手几何的完整HOI质量属于uncertain；先找资产、校验绑定，不能标good或bad来凑比例。
- 若全部水果交互都缺几何，正式完整HOI RSI暂停在输入补全；可交付body-prior技术预研，但不得声称完成skill的完整HOI闭环。
- 有明确几何接触也只有运动学证据；teleop_replay、attachment或object跟随不能证明抓取力/承重。

## 6. 质量划分与泄漏隔离

先审计原知识库和外部派生结果的消费历史。已被本地查看并纳入基础知识的两个episode及其组禁止进test；已有G1转换/修复若曾针对同源样本调参，也记录污染范围。无法证明未用于知识或算法调参的组不应宣传为独立test。没有独立剩余数据时停止test晋升，不用换随机种子消除污染。

以完整连续录制/近重复组为独立单位，而非帧。跨ZIP、同actor录制、同任务重复采集、镜像和G1变体检查重复与重叠。重复相同场景不自动等于同录制；分组采用实际来源和相似性证据，并报告actor/scene覆盖与泛化边界。

先留出校准组冻结源NX1 HOI质量规则，再分invalid、uncertain、真实bad与good。qc.json和原目录的validated只是参考，需在正确模型、真实几何上审查接近/接触/操作/释放、支撑、滑移、穿透及连续性。

正式池严格遵循skill：`k <= min(B, floor(G/3))`，选k个真实bad、3k个good；good固定种子分k个test、2k个train。bad只作质量审计，不能用于学习、调参或晋升；其余reserve不偷偷使用。没有足够真实bad/good就缩小k或报告阻塞。冻结组映射、规则、种子和清单哈希，不按结果重洗。

权限边界应以进程/受限工作区实现，而非同一个agent看完答案后承诺忘记：

| 角色 | 允许读取 |
| --- | --- |
| 数据整理 | 原始数据与源NX1；冻结提取器生成test私有参考，不做test驱动规则修改 |
| train研究 | train的human、场景、源NX1 GT、训练诊断；原库只读与rsi-learned |
| test生成 | 标准human、场景、任务、已声明物体输入、冻结知识；只加载G1 |
| test评估 | G1输出与冻结的私有交互描述；只加载G1，不加载NX1模型/GT数组 |
| 本地总控/训练上下文 | 调度回执、资源与晋升/待定/拒绝/撤销状态及审计ID；不拉取逐test失败轨迹、截图或诊断 |

test参考提取器在train冻结后，于隔离整理流程预计算。若要人工看详细test证据，使用独立评估上下文，不把该上下文续作train研究agent。评价字段不可提取时标unknown并使用冻结处置规则，不事后挑掉难例。

## 7. G1 baseline、知识学习与配对验证

主实验固定为 **human+场景+任务→G1**。建议本轮把记录物体轨迹列为允许输入约束，baseline和候选得到完全相同的轨迹；必须在首次正式运行前冻结。物体路径复制准确不算知识收益，关键是G1是否在真实几何下实现相应接触和支撑。

训练可用NX1 GT解释人机接触，但主生成器不能逐帧抄NX1关节。原PerceptiveLoco的按关节同名映射、限位裁剪与根对齐可作为**train/生产转换诊断**，不是test主baseline，也不是G1 GT；它读取源机器人答案，不能与只读human的方案混算增益。任何需要NX1输入的直接转换另标为source-assisted，禁止进入独立test生成路径。

执行顺序：

1. 冻结基础RSS/ScaleTrack harness、原只读知识库、G1与手模型，跑human→G1 baseline。确认retarget3实际可用，缺少接口先实现并在校准/train验证，不假定skill自带solver。
2. Stage3.1求全身prior，保持世界轨迹和支撑面；人体比例只用于姿态先验，不缩放场景/根XY/落脚点。保留单独body-prior产物及源帧映射。
3. 在train中分析NX1与human、物体关系，提取物体局部接触区域/法向、接近方向、支撑切换、时序和可观测物体效果，标observed/inferred/unknown。语义角色不等于同名腕关节原点。
4. Stage3.2在G1上求可达接触区域与连续轨迹；允许符合冻结规则的站位、肩肘/躯干和摆动期落脚调整，保护已着地支撑锚点；不靠滑脚或改物体目标凑成功。按真实手型安排开闭和全手扫掠。
5. 每轮只研究一个train中重复出现的机制，保留无知识/有知识、失败反例和同预算消融。候选知识写完整因果假设和边界，不把take坐标、帧号、NX1习惯姿态写成通用规则。
6. 在多个独立train组复现后冻结知识正文、实现、角色映射、选择策略、参数和评分协议。test前只根据允许输入选择知识，保存理由与哈希，再运行配对G1 baseline/候选。
7. test只作晋升判断，不迭代修当前test动作；把学习上下文可见反馈限制为规定状态。每轮记录消费组和查询次数，优先新组，test耗尽后继续train但停止晋升。

可探索但尚未成立的知识方向：狭窄家具间的接近与末端可达性协同、稳定支撑下的弯腰伸手、肩肘优先的接近路径、物体局部接触区域向不同手型的映射、操作前后支撑切换。只能从数据实际包含的动作提炼，不假定这两包覆盖脚推、下跪或双手搬运。

冻结指标分别报告：语义角色/事件顺序、接触区域与法向偏差、持续接触/滑移、场景与自碰撞、足底支撑/脚滑、限位/速度/加速度/接缝、失败超时和资源。CoM只在可信惯性下计算，准静态指标不能冒充动态稳定性。无匹配G1 GT不计算跨模型joint RMSE作为目标。门槛、最小收益及可接受回归在校准/train确定，不在本计划里虚构通用毫米阈值。

知识晋升须同时满足绝对G1合格、可归因的配对增益和冻结回归标准；略有改善但仍穿模不算通过。分别报告选用子集与含未选用、失败、超时、预判不可行的整体策略，test筛选成绩不称无偏最终泛化成绩。

## 8. G1数据产出与知识产出分开管理

每个episode产物至少包含：`body_prior.npz`、`hoi_motion.npz`（若完成）、命名joint/body顺序、fps/速度、`human.npz`、对象轨迹、原始源帧/时钟、model/scene/input/knowledge/code哈希、接触计划、质量报告及Stage3 ledger。输出结构按实际G1模型验证，不复制NX1 FK body数组；四元数边界明确wxyz或xyzw。

状态至少区分 `body_prior_only`、`hoi_candidate`、`kinematic_qualified`、`input_blocked`、`target_infeasible`、`solver_failed`、`physical_verified`（仅有实际证据时）。全库覆盖表保留失败和缺失；不把坏源动作机械转换后作为高质量结果交付。

RSI冻结版本得到验证后，可另起 `production` 批次扩大G1 retarget覆盖。仍保持split来源标签：train可继续诊断；test转换结果不回流学习；bad如做转换只作隔离审计；reserve若转为正式训练/评估必须先有新协议版本。生产若读取NX1 motion来辅助初始化，显式标source-assisted且不计入原test成绩。

新知识只进入 `reference-knowledge/rsi-learned/`，test通过的冻结副本原样进入 `reference-knowledge/rsi-validated/`。原 `reference-knowledge/humanoid-real2sim/` 在实验期间只读，启动与每轮结束核对哈希。实验发现、反例和测试结论也不能写进原库；本计划之前的仓库反馈属于启动基线。代码普通Git，精选二进制LFS，全量结果outputs。错误知识从两个活动索引及检索缓存撤销，历史保留outputs。

审阅复用Stage3.5播放器：同源human+实际G1+真实场景/物体，支持任意源帧；每个活动审阅只交付一个URL。只有缺失资产时才能明确显示诊断标记，其结论范围受限。不要为RSI另建一套播放器，不把运动学审阅写成物理通过。

## 9. 长期执行与每天汇报

Dot首次执行须确认自己的持久状态、每日唤醒及向当前用户交付能力。默认北京时间20:00，每日都汇报，包括无进展、暂停和阻塞；只有实际投递成功才写 `delivered_at`/回执。不要把服务器上生成Markdown当作已经通知用户，也不要自行给其他人发消息。若Dot运行平台没有持久唤醒或投递能力，明确报告该能力缺口并保留恢复入口，不承诺已经自动化。

每次唤醒按顺序：读取服务器ledger与本地回执→核对版本和任务锁→收集完成/失败→只消费允许的train证据或test晋升状态→更新待办→在资源预算内提交下一有限批次→保存断点→每日汇报。不能让已结束的SSH会话承载长期状态。

建议从1个GPU worker、每批最多4个独立组的试运行起步；这只是初始上限，preflight根据GPU空闲/显存、求解实测和其他用户进程确定实际预算，禁止抢占。各样本有求解次数/墙钟上限，基础设施错误连续重复时停止该批而非无限重试。暂停/停止请求优先，落盘后不再提交新任务。

日报模板：数据库存与quality/split进度；train完成/失败/超时；知识新增/晋升/待定/撤销；G1输出覆盖与合格比例；test已用/剩余及泄漏审计；对固定baseline的变化与回归（由独立汇报角色汇总，详细test诊断不回训练）；资源/阻塞；下一批train假设。始终区分“没有可靠新增”和“没有执行”。

交接至少保存experiment ID、远端根目录、当前run与worker、冻结配置/知识版本、split/测试消费账本、最后日报投递回执、未完成任务和恢复命令。新agent先读账本，不重新划分、不重复提交、不默认继承旧test通过状态。

## 10. 分阶段验收与第一条执行指令

| 里程碑 | 必须交付 | 未通过时 |
| --- | --- | --- |
| M0 控制端可恢复 | 沙箱管理代码、远端worker、断线/重复提交/退出码/恢复演练、每日交付能力确认 | 不启动长期正式批次 |
| M1 数据合同可信 | 全库索引、模型/资产闭包、坐标时钟报告、污染清单、质量分类 | 缺水果几何者不进入完整HOI正式池 |
| M2 协议冻结 | 真正独立的1:1:2、生成器/评估器隔离、G1 baseline、预算与指标冻结 | 不消费test |
| M3 首个闭环 | train证据→冻结→G1配对test→晋升/拒绝，原库哈希未变 | 可以如实得到“无有效新增”，不降低门槛 |
| M4 长期双产出 | 累积有证据的新Stage3知识；可追溯G1 retarget数据、覆盖/失败清单；日报与恢复记录 | 仅报告已完成范围，不宣布全库成功 |

给执行Dot的第一条指令：**先在沙箱创建上述SSH实验管理程序，读取指定skill，在服务器做只读preflight和数据/资产/污染审计。将已看过的两例作为校准候选排除test，先验证一个无显示平移的完整episode标准化和G1 body-prior链路。具备真实物体几何及可信质量规则后，再冻结正式划分并启动长期RSI；每天向当前用户汇报。**
