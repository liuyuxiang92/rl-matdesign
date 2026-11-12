# 精简 HPO 接口验证记录（2026-11-12）

实现与验证位于 `codex/hpo-v1`。当前使用方法见 [HPO 使用说明](hpo_v1.md)。验证期间固定的包源码摘要为 `10ea60bf2fbf014e9ba551c0bd13537cd5f0b36b1dd1595eddee9499ad4fe392`。以下结果验证软件行为，不代表真实 GlassNet 科学目标已达成。

## 完整回归

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/bin/python \
  -m pytest tests -p no:cacheprovider -o addopts='' -q
```

最终结果：**2208 passed、5 skipped、93 warnings，686.41 秒**。跳过项需要准备好的 GlassPy、性质模型或结构模板。警告包含旧版 HPO 配置兼容提示与原有环境接口弃用提示。

新增回归覆盖：场景目标继承、性质名称与模型输出名称不同、阈值含等号、正零负性质映射、校准尺度冻结及篡改检测、统一总分容差、阶段独立预算模式、固定训练重复组、生成模型与重复次数的组合、重复 TPE 提案去重、预算预检输出、生成策略顺序及配置排序后重新加载。

真实进程回归发现并修复了 `generation_selection` 元数据误入科学评价契约的问题：冻结网络可以切换生成方法；生成恢复仍要求选中策略及有效采样参数一致，配置来源变化不改变科学模型契约。

YAML 解析与 `git diff --check` 通过。当前解释器未安装 Ruff，因此未将静态 lint 列为通过项。

## 三算法完整流程

使用命名性质 dummy 场景，无外部预测模型、历史 RL 权重或历史 HPO 数据。以仓库内三份 `dummy_*.yaml` 为基础，在临时独立 YAML 中切换生成策略；所有运行均完成 `ranges → train-budget → train-search → generation`。训练扫描 48、64、96，生成扫描 8、16、32，训练重复次数均为 2。

| 算法 | 生成方式与候选值 | 每模型生成重复 | 训练任务；尝试／有效候选 | 生成任务；尝试／有效候选 |
|---|---|---:|---:|---:|
| A2C | temperature：0、1、100 | 2 | 14；1242／960 | 36；572／460 |
| DQN bootstrap | epsilon：0、0.5、1 | 1 | 14；961／960 | 18；291／230 |
| DQN MC | top fraction：0、0.5、1 | 1 | 14；962／960 | 18；287／230 |

A2C 每个生成设置实际使用 2 个冻结模型 × 2 次采样，共 4 次。两个 DQN 各为 2 个冻结模型 × 1 次。A2C 有 12 个、每个 DQN 有 6 个贪心生成任务触发效率淘汰；这些任务的成本全部保留在上表中。

最终训练推荐预算分别为 48、64、48，生成推荐预算均为 8。A2C 与 DQN MC 的最终训练选择具有本次网格内的 `OBSERVED_PLATEAU` 证据；bootstrap 为 `BUDGET_LIMIT_REACHED`，没有被报告为已收敛。三者生成选择均为 `OBSERVED_PLATEAU`。这些预算和参数只属于小型软件验证，不是 GlassNet 推荐值。

逐项核对了模型、记录、账本与校准文件的 SHA-256；所有可评分任务使用同一 campaign 内冻结的质量尺度；报告中的物理任务数和候选成本与作业账本之和一致。三个已完成 campaign 再次 `resume` 均无新采样，作业结果、账本和候选记录字节保持不变。A2C 额外验证了 `--stage ranges`、`--stage search` 的已完成结果复用。

本机详细证据：

- `/private/tmp/rlmat-hpo-v2-regression-final.log`
- `/private/tmp/rlmat-hpo-v2-smoke-qzamugy6/verification.json`
- `/private/tmp/rlmat-hpo-v2-smoke-qzamugy6/{a2c,dqn_bootstrap,dqn_mc}/`

## GlassNet 验证边界

三份独立 GlassNet HPO 模板继续引用用户提供的 benchmark 场景路径。Tg/CTE、性质名称、优化方向及模型资产由场景提供；`evaluation.threshold: null` 仍待用户填写。这里未运行真实 GlassNet 模型，也未证明模板中的范围、权重、预算比例或试验次数在该材料任务上最优。

## 后续配置精简验证

同日按用户要求，将训练范围每侧额外补测次数固定为 2，将训练与生成预算的倍增／折半因子固定为 2，并从六份模板中删除对应字段。当前包源码摘要为 `d8b3f5748e3a0c36413bd85a92e5ddae6c0b7cf71f11d1973971813639a7a8e6`。

本次运行全部 HPO 测试：**310 passed、69 warnings，43.73 秒**，日志为 `/private/tmp/rlmat-hpo-internal-controls-regression.log`。回归包含内部固定值、拒绝已移除的公开字段、两种预算模式及主种子 42 派生的精确种子序列。六份模板解析、三算法 dummy 计划解析及 `git diff --check` 均通过。上文完整测试与实际 campaign 结果保留为前一源码版本的证据，本次没有重跑这些实验。

随后将公开字段统一命名为 `generation.repeats`，含义仍是每个冻结网络的生成重复次数。该版本源码摘要为 `d24ed70166475065f63279ef2663434e79629f1d5657a6a8315b8b6682f429a1`。全部简化接口测试 `tests/test_hpo_v2*.py` 通过：**108 passed、9 warnings，9.69 秒**，包括新字段解析、旧名称拒绝及三个网络各生成两次的完整控制器流程。六份模板同步解析通过。日志为 `/private/tmp/rlmat-hpo-generation-repeats-regression.log`；本次未重跑真实模型实验。

再按用户要求，将每轮选入复核的候选组合数内部固定为 3，并从六份模板移除 `training.validation`。该版本源码摘要为 `81dd419e6177da9b87d716f3138d110677da21bec8d36effa85c2010bd69b27b`。全部简化接口测试通过：**110 passed、8 warnings，7.83 秒**，包括拒绝公开复核数量字段、重复提案不占名额、前三个不同有效组合进入复核而第四个不进入，以及候选不足三组时的完整流程。原有对照组合继续保留。日志为 `/private/tmp/rlmat-hpo-fixed-top-three-regression.log`。

## DQN 一致性与提交前验证

确认 A2C、DQN MC 与 DQN bootstrap 共用简化配置和内部规则，并将默认规则、搜索轮次与六份模板的测试扩展到三种算法。GlassNet 模板解析测试临时引用本地 dummy 场景并填写测试阈值，只验证 HPO 配置，不加载真实 GlassNet 模型。bootstrap 保留独有的目标网络更新间隔搜索；MC 不引入该参数；两者均不优化 `dqn_grad_steps_per_ep`。

冻结源码摘要仍为 `81dd419e6177da9b87d716f3138d110677da21bec8d36effa85c2010bd69b27b`。提交前运行 `tests/test_hpo*.py`、`tests/test_generation_selection.py` 和 `tests/test_numerical_runtime_contract.py`：**396 passed、68 warnings，35.68 秒**。日志为 `/private/tmp/rlmat-hpo-prepush-regression.log`。独立一致性审查未发现两种 DQN 与 A2C 的规则差异；`git diff --check` 通过。
