# 反馈与失败案例库

**动作/任务优先入口：**每次先用 `feedback.py search --query '下跪膝盖没有碰地 脚尖滑动'` 检索，再用 `show right-knee-toe-slide` 阅读 [动作 examples](../README.md) 的完整语言叙述、代码、motion 和边界。反馈 JSON 保存用户纠正；示例 Markdown 是以后执行类似任务的主要知识。

凡在本仓库会话中，用户指出已有结果的失败、遗漏、新要求或尚待调查的问题，先记录一条案例并标明 `kind`。错误要修复实际产物；新要求要写可观察验收条件；原因未知的问题先保留观察与不确定性。调查后分别决定是否更新 skill、共享工具和动作 example，三项都写理由；处理方法见 [反馈处理与知识落点](../../../codex_skills/rss-scaletrack-real2sim/references/feedback-routing.md)。这里记录**跨任务可复用的教训**；某个 take 的帧号、版本和批准状态继续放在其 `stage_ledger.json`。不要为普通候选迭代制造反馈案例。非 humanoid 任务也记录在这里，用 `repository` 阶段。

文件是一例一个 JSON，便于 `rg` 和脚本搜索：

```bash
python scripts/humanoid-real2sim/feedback.py search --query wrist
python scripts/humanoid-real2sim/feedback.py search --knowledge-category foot-push --query '持续接触'
python scripts/humanoid-real2sim/feedback.py search --stage retarget --category interaction
python scripts/humanoid-real2sim/feedback.py add --id 2026-10-20-short-name --kind error --stage retarget --category motion --impact high --task '任务名' --actions '下跪,抓取' --title '简短标题' --narrative '动作、环境和症状' --expected '用户期望' --observed '实际问题' --source '会话/截图/任务路径' --example-id right-knee-toe-slide
python scripts/humanoid-real2sim/feedback.py update 2026-10-20-short-name --cause '经证据确认的原因' --fix '代码/数据改动' --validation '可复核证据' --route 'skill=none: 属于单次动作约束' --route 'tool=update: 修复可复用求解器' --route 'example=update: 补全下跪样本' --status verified
```

`kind` 为 `error`、`new_requirement` 或 `new_problem`。`stage` 为 `scene`、`retarget`、`track`、`visualization`、`archive`、`cross-stage` 或 `repository`；`category` 为 `input`、`geometry`、`interaction`、`motion`、`physics`、`tooling`、`delivery`、`data`、`code`、`test`。`impact` 为 `high`、`medium`、`low`。状态 `open`、`fixed`、`verified`；新记录只有填写实际改动、验证和三项路由决定后才可标成 `verified`，错误还须填写原因。路由决定格式为 `skill|tool|example=update|none|defer: 理由`。重复事件更新相关案例的 `recurrences` 并补来源，不建立多个内容相同的规则。旧记录无需迁移即可继续搜索和阅读。

每次开始相同任务、动作或症状时搜索库。无论频率和影响如何，任务反馈都先扩充对应 example 的语言叙述、代码、motion 和审查证据；跨任务可复用性通过关键词、相关样本和自然语言描述体现，不将动作规则提升到主 skill。只有用户反馈 harness 的路由、检索、样本维护等机制时才更新主 skill。

新反馈立即记录任务、动作和自然语言情境；修好后在同一案例中补原因、修复与验证，并按路由决定更新关联示例、工具或 skill。新动作按 [模板](../TEMPLATE.md) 建示例并加入 `index.json`。运动学几何验证不可标成物理接触验证；单次 take 的批准状态仍在其 stage ledger。
