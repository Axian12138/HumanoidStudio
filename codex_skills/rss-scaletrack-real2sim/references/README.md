# n0 / n1 / n2 子任务索引

从 n0 理解输入与语义，再按问题在 n1 和 n2 间循环；只读取当前所需的引用。

| 子任务 | 说明 | 查看方式 |
| --- | --- | --- |
| n0 数据、坐标和语义 | [scene-objects.md](scene-objects.md) | 输入与源帧映射 |
| n1 场景初建及修正 | [scene-objects.md](scene-objects.md) | RSS live、同相机对照 |
| n2 运动学重建 | [retarget.md](retarget.md) | [运动学审阅](visualization.md) |
| n2 可选动力学重建/修复 | [track.md](track.md) | RSS 物理修复页 |
| 按需归档 | [scene-objects.md](scene-objects.md#按需归档到-scaletrack-数据包) | 按运动学/动力学范围校验 |

工具见 [tool-map.md](tool-map.md)，命令和数据契约见 [local-interfaces.md](local-interfaces.md)。用户反馈进入[反馈库](../../../reference-knowledge/humanoid-real2sim/feedback/README.md)；历史 take 的版本和批准记录保留在其 `stage_ledger.json`。
