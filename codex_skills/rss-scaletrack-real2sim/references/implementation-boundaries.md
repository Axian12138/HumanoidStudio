# 当前实现边界

n2 运动学沿用现有 retarget 求解器和直接设置 qpos 的播放器。动力学是独立可选分支，RSS 编辑器已有五点接管、播放/暂停、回滚、手部开关和候选导出；不能把运动学回放当成物理执行。

n1 通过 RSS live page 和 `--scene-file` 接入，几何来自测量、Blender 或可靠资产。当前尚无通用的“拖动物体并连同柔性求解器状态一起分支回滚”编辑器；修改场景后需重建，重验受影响的运动学；启用动力学时还需重跑受影响物理段。没有 CUDA 不能完成真实 BFM TensorRT rollout，假 tracker smoke test 不算动力学验收。命令和产物见[本地接口](local-interfaces.md)。
