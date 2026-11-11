# HPO v1 实现验证记录（2026-11-11）

实现位于 `codex/hpo-v1`，基线为 `771d68d7`。最终源码摘要为 `456066fc215847ebd81f7ffa5cbdfc592ad0a728782d2dfd50957eb65e7eaef5`。本记录验证软件执行及预算契约，不代表真实 GlassNet 材料目标达成。

## 自动化与安装验证

- 相关回归：468 passed、1 skipped，179.39 秒。覆盖 HPO、配置、训练预算、重复采样、训练样本、精确恢复、批量执行、旧 HPO 兼容及 GlassNet 适配契约；跳过项要求准备好的真实 GlassPy 资产。
- 最终冻结源码上的 HPO 专项：164 passed，26.22 秒。覆盖方向无关评分、严格比率、双向预算、范围边界更新、追加 seed 的最新证据、学习预算预检、控制器、CLI、报告及训练/生成恢复。此集合与上述回归部分重叠，不能相加当作独立测试总数。
- 非 editable wheel 在临时目录构建并安装，从仓库外工作目录、仅安装目录的 `PYTHONPATH` 验证 console CLI、模块入口和 A2C/bootstrap 配置校验，均退出码 0。
- 全部新增/修改 Python 编译检查、变更 YAML 解析和空白检查通过。Ruff 在本解释器中未安装，未声称运行该检查。

安装验证命令和证据保存在本机 `/private/tmp/hpo-wheel-check-ithf03el/verification.json`。

## 三算法完整 CLI 实验

使用 `configs/hpo/v1/dummy_*.yaml`，无旧 RL 权重或旧 HPO 结果。每条均完整执行 `ranges → train-budget → train-search → generation`，退出码 0。训练初始预算 64，双向扫描 48、64、96；生成初始预算 16，双向扫描 8、16、32。温度集合 0、1、100，训练和生成账本独立。

| 方法 | 训练实际任务 | 训练尝试 A／有效 U | 生成实际任务 | 生成尝试 A／有效 U |
|---|---:|---:|---:|---:|
| A2C | 26 | 2312／1792 | 18 | 286／230 |
| DQN-bootstrap | 24 | 1669／1664 | 18 | 286／230 |
| DQN-MC | 24 | 1666／1664 | 18 | 286／230 |

这些是不同实际作业的累计成本，不是将材料在整个 campaign 去重后的新颖性数量。失败作业的费用也已计入。A2C 的额外准备采样如实增加 A，不增加 U。

独立核对确认：每个归档候选身份唯一，记录数等于账本 U；完成作业达到目标 B 且满足严格 `A/B < 1.5`；每条 campaign 的六个温度 0 生成作业均仅找到一个材料并按不可逆效率失败提前停止；模型、scaler、检查点、归档和已记录文件的摘要保持一致；`selection.json` 成本与实际作业独立求和相符。

| 方法 | 选择训练 B | 训练收敛证据 | 选择生成 B／温度 | 生成收敛证据 |
|---|---:|---|---|---|
| A2C | 48 | `OBSERVED_PLATEAU` | 8／100 | `BUDGET_LIMIT_REACHED` |
| DQN-bootstrap | 48 | `OBSERVED_PLATEAU` | 8／100 | `BUDGET_LIMIT_REACHED` |
| DQN-MC | 48 | `BUDGET_DEGRADATION` | 8／100 | `BUDGET_LIMIT_REACHED` |

上述数值仅为软件 smoke，不是 GlassNet 推荐参数。选择预算与收敛证据彼此独立：某配对可以按加权评分及容差被选中，但完整更大预算扫描发现原始质量仍改善时，不能报告平台。MC 复核发现退化也被明确保留。

完成后的 A2C campaign 再依次运行 `resume`、`run --stage ranges`、`run --stage search`，均退出码 0；所有 `result.json`、候选账本和原始归档的集合及摘要完全不变，没有启动新采样或重复计费。同步检查点的训练/生成中断恢复及失配拒绝由自动化测试覆盖。

本机原始核验摘要：

- `/private/tmp/rl_matdesign_hpo_v1_a2c_final_verification.json`
- `/private/tmp/rl_matdesign_hpo_v1_final_dqn_verification.json`

最终 campaign 目录：

- `/private/tmp/rl_matdesign_hpo_v1_a2c_final_20261111`
- `/private/tmp/rl_matdesign_hpo_v1_final_dqn_bootstrap_1794389591537`
- `/private/tmp/rl_matdesign_hpo_v1_final_dqn_mc_1794389591537`

## 已处理问题与验证边界

审查修复了跨配置更大预算仍改善时误报平台、追加 seed 后旧成功证据覆盖新失败、范围外扩后预算不足或 replay 太小却被当作正常学习、报告中未知恢复耗时相加错误，以及终态作业缺少父进程结果文件时重复启动的问题。

开发期间的首轮集成实验遇到源码摘要变化，生成阶段正确拒绝加载旧执行契约的训练模型。未绕过校验；最终实验在源码冻结后新建空目录重跑。最终运行中存储等待增加墙钟时间，已记录约 753–779 秒训练作业累计时间及约 71 秒生成时间；这不是模型性能基准，也不是跨机器耗时承诺。

真实 GlassNet HPO 按用户要求交由用户在 benchmark 场景及模型资产主机执行。三份独立 HPO 模板引用用户给出的 `coarse` 配置路径，可切换 `medium`／`large`；T3 阈值仍为 `null`。未测量或推断该训练集的中位数，没有填写假定的工业 CTE/Tg 条件，没有导入历史 HPO 参数。
