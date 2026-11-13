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

## 每模型三次生成的默认值

按用户确认，将六份模板及解析器的 `generation.repeats` 默认值统一为 3。源码摘要为 `6b4f35097748c2f71d560faf874ed8fb7f3146fcfc60c37a3fb3ea351ac1e45b`。全部简化接口测试通过：**121 passed、8 warnings，7.73 秒**，确认三个模型在省略该字段或显式设为 3 时均得到九次生成，显式设为 2 时仍为六次。日志为 `/private/tmp/rlmat-hpo-three-generation-repeats-regression.log`。两份交付的 DQN HPO YAML 与用户最新 coarse 场景进行了临时阈值解析检查，确认继承新 CTE/Tg 约束及温度生成方式；交付文件的科学阈值仍保留 `null`，没有运行真实预测模型。

随后将生成范围每侧额外补测上限也内部固定为 2，从六份模板和两份已交付的 DQN HPO YAML 移除对应字段。源码摘要为 `d36cb3421288539990253b5954d0c6a2dda8cbb231008ab408fc647f17a29981`。全部简化接口测试通过：**123 passed、8 warnings，9.04 秒**，包括三算法固定规则、拒绝旧公开字段及生成模型重复流程。交付文件再次完成临时阈值解析检查，仍保留 `generation.repeats: 3`。日志为 `/private/tmp/rlmat-hpo-internal-generation-probes-regression.log`。

## 可选并行 HPO（2026-11-12）

新增顶层 `parallel_jobs`，默认1；独立参数、预算、种子与生成模型配对共用一个 worker 上限。随机 startup 提案可以批量执行，自适应 TPE 等待前一提案反馈。质量校准及评分按原始请求顺序处理；异常清理只保留待评分结果，恢复时不允许后完成的实验替换参考，也不允许已知失败退出被完成账本覆盖。

冻结包源码摘要：`de0877ab451bccee168e8c8dee02c92eca9a64ba246bc0589a3d766d762286d8`。全部 HPO 测试及生成策略、数值运行契约回归：**454 passed、77 warnings，30.92 秒**。日志：`/private/tmp/rlmat-hpo-parallel-regression.log`。覆盖全局并发上限、组合去重、固定 seed/repeat、Optuna 随机与自适应提案顺序、逐阶段串并行结果一致、失败 seed 淘汰、进程组终止、待评分结果恢复与计时保留。

额外运行三算法真实 worker 子进程，使用小型 dummy 场景：每个算法分别以 parallel_jobs1和3执行3个训练种子、每个冻结模型3次生成，共72个作业。单次训练预算8、生成预算4、A2C准备采样2，仅用于软件验证。三算法的训练与生成分数、原始指标、参数及种子逐项一致；冻结模型哈希未变化；重新读取全部训练及生成任务未新增采样或改变候选档案。

| 算法 | 串行耗时（秒） | 3路并行耗时（秒） | 科学输出对比 |
|---|---:|---:|---|
| a2c | 37.98 | 12.94 | 一致 |
| dqn_mc | 34.22 | 13.05 | 一致 |
| dqn_bootstrap | 33.91 | 13.25 | 一致 |

以上耗时来自本地 dummy 运行，不代表 GlassNet 或 GPU 加速效果。结果与逐作业日志：`/private/tmp/rlmat-parallel-runtime-sg8ncnnw/verification.json`。第一次临时 smoke 将 A2C warmup设为0，scaler拟合没有样本而失败；修正临时场景为2次准备采样后在全新目录完成验证，未修改生产训练逻辑。

## 原生停止与恢复的回归范围

当前正常停止与恢复由原生 HPO 控制器、worker 和 `restart` 负责。回归用例使用新建的 CPU dummy campaign，覆盖 A2C、DQN MC 和 DQN bootstrap 的采样中断、完整训练单元保存、同步断点续跑及生成结果收尾：

- `tests/test_hpo_cooperative_worker_stop.py`：候选请求已开始时收到信号，完成当前训练单元后保存；再次 Ctrl+C 强制停止；目标预算已完成时正常导出。
- `tests/test_hpo_cooperative_restart.py`：真实 worker 停止后通过原生作业调度恢复，比较网络、优化器、随机状态、候选记录与不中断对照；覆盖生成恢复及最终导出，以及并行控制器 SIGTERM 后的原生 CLI restart。
- `tests/test_hpo_stop_coordinator.py`：控制器停止启动新作业、等待完整断点、清理 worker，并保留已结算终态作业供结果收尾。
- `tests/test_hpo_checkpoint_retention.py`：仅覆盖最新完整 checkpoint，不累积编号快照；原子保存失败时保留上一份完整断点。

只读预检、运行时 `--parallel-jobs` 覆盖、已完成作业复用与预算账本校验仍保留。源码、配置、资产或运行契约变化不自动放行；新的实验使用空输出目录。这里描述软件验证范围，不代表真实 GlassNet/GPU 结果已经达标。

本轮移除旧任务修复与源码迁移代码后，全部 `tests/test_hpo*.py`、严格恢复合约、exact resume、生成策略、数值运行契约和模型合约测试共 **605 passed、77 warnings，244.07 秒**。独立 wheel 构建与安装成功，安装包包含原生 stop/restart，不含旧兼容模块及登记表；在源码目录之外使用安装版 CLI 验证 dummy A2C 配置通过，未调用性质预测器。
