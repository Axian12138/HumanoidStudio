# 1019 动力学可视化与修复

```bash
# 默认打开湿巾撑桌，暂停启动；页面地址见终端输出
./launch_scripts/visualize_1019_dynamics.sh

# 列出当前数据包；指定一个包启动，默认端口 8200（界面也使用相邻端口）
./launch_scripts/visualize_1019_dynamics.sh --list
./launch_scripts/visualize_1019_dynamics.sh brush-sweep-kinematic-v1 --port 8200

# 构建模型、检查属性、推进五个物理帧；不启动页面或策略推理
./launch_scripts/visualize_1019_dynamics.sh brace-dining-table-wipes-kinematic-v1 --check-scene
```

使用已有 RSS 修复界面、青色参考机器人和默认 refine_eef `ckpt_010000.pt`。
Ctrl-C 关闭页面与推理 worker。一次启动一个包；需要同时查看时指定不同的端口，
每个页面保留两个相邻端口。`CHECKPOINT`、`METADATA`、`RSS_PYTHON`、
`SCALETRACK_ROOT`、`DATA_ROOT` 可以通过环境变量覆盖；其他参数透传到修复入口。
USD 导入依赖可用 `uv sync --extra usd` 安装；运行策略仍需要原有 CUDA/ScaleBridge 环境。

场景从当前 `motions.yaml` 引用的 USD 直接组合，源文件只读。
显式 `physics_mode`、`collision_enabled` 覆盖 USD；未指定时保留 USD 有效属性。
每个物体的初始位姿和尺度来自 YAML，房间抽屉的初始关节位置也来自 YAML。
Kinematic 物体保持初始化位姿，不受力推动；不会逐帧播放物体参考轨迹。
Dynamic 物体只初始化，之后由物理推进。

启动前检查导入后的 body flags 和 collision flags；丢失启用碰撞体、错误启用碰撞、
物理模式不匹配或不支持的资产会报错退出。不会退回旧历史场景或把失败当作通过。
结果位于 `outputs/humanoid-real2sim/<date>/package_dynamics_review/runs/<run-id>/scene/`：

- `scene.usda`：引用源 USD 的运行组合层。
- `source_contract.json`：YAML 覆盖后应生效的物体与碰撞体属性。
- `physics_audit.json`：最终 Newton 模型与来源属性的比较。
- `step_check.json`：`--check-scene` 的短段步进结果。

物理由 Newton 生成接触、MuJoCo 求解，支持源场景的平面网格地板。
这里保证的范围是经检查的 kinematic/collision 开关，不是 Isaac/PhysX 物理等价。
摩擦、恢复、惯量推导、碰撞近似和完整动作成功仍须单独验证；检查模式不执行 tracker。
旧 `--scene-file` 入口保持原有行为，新脚本使用独立的 `--package` 入口。
