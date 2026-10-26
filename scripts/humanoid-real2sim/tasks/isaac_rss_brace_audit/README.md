# 撑桌动作的 Isaac / RSS checkpoint 对照

目的：区分初始化状态、checkpoint 和仿真/控制差异。只记录，不修改源动作、源 USD 或运行中的修复页。

`run_rss.py --checkpoint <pt> --out <absolute-or-relative-directory> --steps 200`
记录 RSS 修复流程实际初态及逐步状态。使用现有元数据与控制设置，仅替换 checkpoint。

Isaac 使用用户指定的 ScaleTrack `.venv`，从 `play_tracking_manifest.py` 生成只增加数据记录的副本，
保留原播放配置、控制与 reset 逻辑。副本和四组完整状态均保存于：

`outputs/humanoid-real2sim/2026-10-25/isaac_rss_brace_comparison/runs/21-36-25.582586+0800/`

`comparison.json` 保存指标。四组均运行 200 个 50 Hz 策略步，没有中途 reset；不是完整动作成功验收。

| 引擎 | checkpoint | 第 200 步左腕参考误差 | 右腕参考误差 |
| --- | --- | --- | --- |
| RSS | 22200 | 74.03 cm | 63.45 cm |
| RSS | 010000 | 69.57 cm | 55.29 cm |
| Isaac | 22200 | 15.97 cm | 19.22 cm |
| Isaac | 010000 | 54.59 cm | 56.97 cm |

同一引擎切 checkpoint 时，未推进的初始关节角、关节速度与选定身体位置完全一致。
跨引擎初始关节角/速度也完全一致，选定身体位置分量最大差小于 0.7 微米。
这排除了初态姿态/关节速度不同的解释，但没有证明碰撞几何和接触响应等价。

29 个身体关节的 Kp/Kd 在 1e-4 内一致。RSS 物理步长 0.002 s，Isaac 0.005 s；
RSS 手指 Kp 全为 2.0，Isaac 两个 thumb_0 为 2.0、其余 12 个为 0.5。
这些是已确认的差异，尚未通过独立消融确定各自的因果贡献。

结论：checkpoint 影响后续跟踪，但相同 22200 下仍有显著跨引擎差异。
应以目标 ScaleTrack/Isaac 配置作为该数据的验收基线，先校准 RSS，不能只因 RSS 卡桌就修改原动作。
