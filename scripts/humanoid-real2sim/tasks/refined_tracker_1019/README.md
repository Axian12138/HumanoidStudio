# 1019 tracker 场景适配

`scene.py` 为已有的九个 `1019test` 动作加载对应 RSS 物理场景，供
`run.py --scene-file` 使用。`RSS_REVIEW_PACKAGE` 指定归档动作包，
`RSS_REVIEW_SCENE_OUT` 指定独立运行目录。tracker checkpoint 仍由 `run.py` 传入。

帽子和抽屉需额外传 `--robot <package>/scene/robot_props.xml`，保证组合模型中的
机器人执行器先转换为 tracker 的位置控制；工厂使用转换后的 asset。
物体 sidecar 和抽屉关节只初始化首帧，播放时不按参考轨迹覆盖物理状态。
推椅和撑桌复用独立椅子/桌子凸包；圆篮保留 65 个凸包和归档质量/重心，并加载餐桌碰撞。

每个场景输出 `scene_audit.json`，记录资产入口、初始物体位姿、关节和 shape 数。
运行目录的 `playback_check.json` 记录短段推进与重置检查。这些检查证明页面可播放，
不代表整个动作、接触稳定性或训练数据验收通过。部分继承场景仍使用近似碰撞代理。
