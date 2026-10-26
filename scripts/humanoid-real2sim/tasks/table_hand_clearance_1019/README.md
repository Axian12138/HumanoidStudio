# 三组 motion 的局部桌板避让

本次用户范围：只改 motion；两种推椅尽量相同，后续撑桌各有对应版本。保留源 USD、YAML、根、腿、物体轨迹和非运动字段。推椅手指轨迹不改；撑桌开头允许保留与推椅对应的手型过渡。

`repair.py` 对当前房间 MORBYLANGA 桌板包络做局部手部避让，只优化肩肘腕。全手碰撞网格顶点/凸包采样及手指盒体参与约束，100 Hz 插值复查。它不证明任意 tracker 的动力学执行无穿透。`pair.py` 将两种修正后推椅在源83.5秒（第425帧）的关节姿态分别接入撑桌；首秒衰减交互差异，第52帧起两份撑桌的六个运动字段完全相同。推椅和撑桌有源时窗重叠，不应将两份完整文件首尾直接拼接。

`package.py` 只替换 NPZ 内六个运动数组（fps保持相同），其他压缩成员保持原值。`install.py` 核对源文件仍与备份相同后原子替换三份 motion.npz，并在撑桌目录新增 motion_after_thumb.npz、motion_after_index_middle.npz；默认 motion.npz 是拇指衔接版本。

原件、候选、检查和安装哈希：
`outputs/humanoid-real2sim/2026-10-25/table_hand_clearance/runs/`

- `original_backup/`：三份原始 NPZ。
- `validation.json`：字段范围、根腿精确保留、全手100 Hz几何检查、两版本共同后缀和首帧衔接。
- `installed.json`：实际安装路径和 SHA256。

已知范围：旧视觉 GT 原样保留（用户要求只改 motion），因此不再与修改后的手臂逐帧匹配。两份推椅原有少数手指超出此 MJCF 的限位，未扩大、未改动；这是源数据已有问题，未混入本次局部避桌修正。
