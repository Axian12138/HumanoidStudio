# 手推交互：抽屉关节、体型差异与双手移椅

1019 长录制任务 05 在约 48–55 秒用右手开关柜子抽屉。知识点是先确认物体有受限平移关节和开、关两个方向，再从手/把手接触、抽屉行程及箱体固定性构造 HOI；当前只有全身 body prior，**没有已通过的抽屉交互轨迹**。[任务切片](../../../outputs/humanoid-real2sim/2026-10-20/sequence_1019/curated_snapshot/segments.json)提供源时间和 Motive 帧范围。不能用手腕靠近柜体的画面冒充抽屉被拉开。

任务 07 在 75–84.5 秒用双手推灰色扶手椅进入餐桌下，随后另起一段右手撑桌。推椅有食指／中指作为主推面和拇指作为主推面两种手型，均需检查接触、物体位移方向和完整手部网格。最终拇指方案已获用户运动学视觉认可；尚未验证 tracker 跟随和物理推动。[任务页](../../../outputs/humanoid-real2sim/2026-10-20/sequence_1019/curated_snapshot/tasks/07_armchair/README.md)给出源时窗和任务资产。

这两个任务都属于手推/接触式操纵，但约束不同：抽屉保留机构关节和行程，椅子保留自由刚体与地面支撑。分类应由物体自由度和接触模式决定，不能因为都由手操作就复用同一 IK 目标。

推椅后续的右手握拳撑桌已单独切出并完成运动学视觉审阅；它是[独立的桌面支撑案例](../composed-actions/fist-forearm-table-brace-after-chair.md)，不应与待审的推椅手部候选合称为一个已通过动作。

## 体型不一致时的轨迹优先级

**人体动作决定任务要完成的物体位移和接触时序；机器人尺寸决定怎样够到接触面。**早期推椅候选因机器人手臂短，椅子走不到人体动作中靠近桌边的位置。用户指出这改变了任务本身，于是把推动末段及终点重新对齐人体动作，改由机器人根部、肩肘、手腕和局部接触姿态适配。人手与机器人手腕的绝对位置不必逐点重合，但不能仅因机器人手短就缩短椅子的目标行程。每次修改后并排看 human 虚影、机器人实际接触和椅子相对桌沿的终点。

**可移动自由物体的接近位姿可以在有限范围内调整。**最初椅子的位置对人体可用，却让机器人接近时手指过早扫入椅背。此例在接近阶段把椅子最多沿后续目标方向移约 6 cm，为机器人手部留空间；79.2–80 秒保持偏移，83.4 秒前回到人体对应的物体轨迹，末段和终点不变。[逐时刻物体偏移审查](artifacts/armchair_approach_object_offset.json)记录了相对原估计轨迹的差值。迁移到新任务前，先核对物体确实可自由移动、初始摆放没有严格测量约束，并确认偏移不会改变最后与桌子等环境的接触关系。脚、手和物体扫掠都要重新查穿模；不能通过偏移物体来掩盖接触或根轨迹错误。

## Dex3 手推接触的选择

Dex3 在这些任务中可把食指和中指当作一个协同接触单元，再决定拇指承担主推还是卡位。拇指根部的轴向拧转不应拿来补偿整掌方向错误；本次拇指主推候选将 `thumb_0` 保持在零附近，翻掌由腕和手臂承担，拇指两节伸直。它是本任务的接触先验，并非 Dex3 各关节的通用禁令。选手型前要标出**机器人上真正参与施力的面**与**物体上的对应面**，同时核对推力方向、边缘方向、可持续接触长度和接近路径。

| 接触任务与已见方案 | 主接触与手指安排 | 需要特别检查 |
| --- | --- | --- |
| [刷子抓取](../grasp/hand-closure-and-held-object.md) | 掌侧进入后合手，形成握拳式包络抓取。 | 闭手姿态之外还须验证指垫受力、物手相对滑移和提起保持；本例物理抓取仍未通过。 |
| [垃圾桶外侧推](continuous-exterior-bin-push.md) | 手先张开，拇指与食指在桶外侧后缘形成开放的双接触推面。 | 桶位移方向应落在可施力区内，推到桌沿前不能提前脱手；物理推动仍需复核。 |
| [握拳撑桌](../composed-actions/fist-forearm-table-brace-after-chair.md)及类似靠／怼动作 | 握拳或近握拳，掌面和小臂共同形成较大的支撑区。 | 区分支撑与抓取；检查拳、小臂以及另一只手是否穿入环境。 |
| 椅背方案 A：食指／中指主推 | 两指并列贴椅背形成主推面，拇指在边缘辅助卡位。 | 两指的接触面方向、椅背上沿与全程避障；该方案获用户“做得很好”的视觉反馈。 |
| 椅背方案 B：拇指主推 | 拇指宽面贴椅背后侧竖直面、长轴向下；食指／中指接触后弯曲卡位。 | 掌面须绕拇指轴翻到正确一侧；接近时两指先张开，不能把卡位手型提前带进椅背。 |

这些是候选接触**类别**，不对应固定关节角。刷柄、桶沿、椅背和桌面形状不同，弯指量、接触面和进入方向必须重新由物体几何与人体动作确定。

### 宽椅背的接近与手指形状

用户指出旧版只在手碰椅背后优化指尖，接近过程中手掌低于上沿、掌面没有对准外侧接触区，拇指第二节又弯进椅背。应先判定**整个椅背外表面及厚度**，从人体手腕的外侧、上方绕行轨迹建立接近目标，让掌侧面向交互区域；连续审查整只手的碰撞网格，而不只检查指尖或手腕。

**接触方式必须由真正相贴的两个面决定。**本任务中，a 是拇指的宽接触面，b 是椅背**朝后的竖直面**。伸直拇指的长轴在 b 面内基本竖直向下，a 面法线朝向 b 面；椅背上沿水平，拇指方向与边沿近乎垂直，但**不是与 b 面垂直**。旧 `thumb_surface` 把近节留在距椅背约 55 mm 处，只剩指尖点接触；下一版 `thumb_strip` 虽形成较长接触，却错误地贴到椅背斜上沿。`thumb_backface` 虽把拇指长轴和接触位置摆对，却把手掌绕拇指长轴翻反了：拇指网格的 -X 面并非本例用户指定的 a 面。新 `thumb_backface_flip` 保持长轴向下，改以 +X 面朝向 b，并通过整条手臂与手腕解出翻掌姿态，拇指两节保持伸直。小瓶等曲面物体可能需要弯曲手指包络接触，不能照搬直拇指。

[局部修复脚本](../../../scripts/humanoid-real2sim/tasks/sequence_1019/tasks/07_armchair/refine_split_feedback.py)在 78.0–79.1 秒增加掌侧接近目标和外侧避障；接近时椅子朝桌侧最多移 6 cm，推椅终点不变。[食指／中指候选](../../../outputs/humanoid-real2sim/2026-10-20/sequence_1019__armchair_split_feedback/runs/17-54-03.563689+0800/README.md)已获用户“做得很好”的视觉反馈。[拇指翻掌候选](../../../outputs/humanoid-real2sim/2026-10-20/sequence_1019__armchair_split_feedback/runs/19-12-20.289127+0800/README.md)在 81 秒的向下夹角约 3°／2°，指定 a 面朝 b 面的点积约 1.00／1.00，后侧面内末节近表面竖直跨度约 44／57 mm。先前 `thumb_backface` 的错误 -X 面虽有接触，但掌面绕拇指轴反向。[后侧面审查脚本](../../../scripts/humanoid-real2sim/tasks/sequence_1019/tasks/07_armchair/audit_thumb_backface.py)及[本次反馈记录](../feedback/2026-10-20-chair-approach-straight-thumb.json)保存几何证据与边界。所有这些均为运动学候选，未验证物理推动。

在拇指和整掌姿态已对齐之后，仍须检查**接近过程中的手指时间形状**。78.88 秒的旧候选在食指／中指过早弯曲时产生明显穿模；不能只看接触帧的正确手型。用户观看新的[手指接近结果](results/armchair_thumb_push_finger_entry_motion.npz)后评价“很好”：保留原手腕、拇指、手臂、椅子轨迹，让食指／中指先张开绕过椅背上沿，到接触时再弯曲卡住椅背。[几何抽查](artifacts/armchair_finger_entry_clearance.json)显示 78.88 秒最严重的手指穿入从旧版约 23 mm 降至局部约 2.1 mm。手指形状应按接近、接触、持续推动分别审查，而不是把接触手型提前套在整段动作上。

- [扶手椅双掌拟合](../../../scripts/humanoid-real2sim/tasks/sequence_1019/tasks/07_armchair/fit_hoi_v2.py)、[椅子几何审查](../../../scripts/humanoid-real2sim/tasks/sequence_1019/tasks/07_armchair/audit_hoi_v2.py)。
- 抽屉尚无 HOI 专用求解代码；源切片和 body prior 的证据在 `sequence_1019/tasks/05_drawer`，后续实现应另建候选。

## 本仓库结果与审查证据

- **审查证据** [armchair_backrest_hook_motion.npz](artifacts/armchair_backrest_hook_motion.npz) — `pending_hoi`；1019 armchair HOI candidate。
- **审查证据** [1019_segments.json](../task-analysis/artifacts/1019_segments.json) — `reviewed_task_split`；1019 long demo task windows and overlap。
- **运动学视觉结果** [armchair_thumb_push_finger_entry_motion.npz](results/armchair_thumb_push_finger_entry_motion.npz)及[椅子估计轨迹](results/armchair_thumb_push_finger_entry_object_estimate.npz) — 75–84.5 秒、475 帧、50 Hz；用户认可手指接近与拇指主推姿态。
- **直接审查证据** [armchair_finger_entry_clearance.json](artifacts/armchair_finger_entry_clearance.json)与[armchair_approach_object_offset.json](artifacts/armchair_approach_object_offset.json) — 分别记录手指穿模改善及椅子接近阶段偏移、末端回归原轨迹。

**可证明的范围与限制：**最新推椅 motion 获用户运动学视觉认可，网格间隙是离散抽查；真实施力、tracker 跟随及物理移动未验证。抽屉仍只有 body prior，没有 HOI 通过结论。
