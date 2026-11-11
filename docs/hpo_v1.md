# YAML 驱动的两部分 HPO

新入口使用 `hpo_v1` 协议，训练参数按训练收集结果选择，生成参数按冻结网络的生成结果选择。它保留旧 HPO 脚本的行为，不把旧的生成 top-k 评分悄悄替换为新评分。

已接入 rl-matdesign 的场景通过两份 YAML 使用相同流程：场景文件负责设计空间、预测模型、命名性质和科学约束；HPO 文件负责引用场景、目标方向及阈值、参数范围、预算与采样效率限制。切换已接入场景不需要修改 HPO 代码。尚未接入工程的预测器或环境仍需要先完成普通场景接入。

## 先验证，再执行

安装项目后使用已有的 `rl-matdesign` 命令。源码 checkout 也可以在前面设置 `PYTHONPATH=src`，把 `rl-matdesign hpo` 替换为 `python -m rl_matdesign.hpo`。HPO 运行需要项目的训练依赖以及 Optuna；`validate` 和 `plan` 不调用性质预测器、不训练网络。

```bash
rl-matdesign hpo validate --hpo-config configs/hpo/v1/dummy_a2c.yaml
rl-matdesign hpo plan --hpo-config configs/hpo/v1/dummy_a2c.yaml
rl-matdesign hpo run --hpo-config configs/hpo/v1/dummy_a2c.yaml \
  --out runs/hpo_a2c_smoke
```

`validate` 检查配置、目标字段、预算档位与准备采样的最低预算。`plan` 另外展示生效配置、双向预算顺序与阶段，不创建试验结果。`run` 默认运行完整流程，输出目录应为本次新建的空目录。已完成的同一 campaign 作业只有在配置、代码、资产及结果校验一致时才会复用。

方法可以放在 YAML 中，也可以显式传入：

```bash
rl-matdesign hpo run --method a2c --hpo-config hpo_a2c.yaml --out runs/my_hpo
```

支持 `a2c`、`dqn-bootstrap`、`dqn-mc`。命令行和 YAML 同时指定的方法必须一致；命令行不会静默改写另一份方法的搜索空间。

## 分开运行第一部分和第二部分

第一部分只做训练参数范围诊断，不执行生成：

```bash
rl-matdesign hpo run --hpo-config hpo_a2c.yaml --out runs/my_hpo --stage ranges
```

检查该目录中的 `ranges.json` 和 `range_decisions.jsonl` 后，继续第二部分：

```bash
rl-matdesign hpo run --hpo-config hpo_a2c.yaml --out runs/my_hpo --stage search
```

| `--stage` | 执行内容 | 前置结果 |
|---|---|---|
| `all` | 第一部分及完整第二部分，默认值 | 无 |
| `ranges` | 参考配置、上下界、失稳内收和有证据的外扩 | 无 |
| `search` | 训练预算、训练组合搜索与复核、生成温度和数量 | `ranges` |
| `train-budget` | 代表配置在完整双向预算网格上的校准 | `ranges` |
| `train-search` | 固定预算的 TPE 搜索、全网格复核及有限次预算修订 | `train-budget` |
| `generation` | 冻结验证网络的温度和生成数量联合选择 | `train-search` |

缺少前置结果时命令会报错，不自动偷偷执行前一阶段。第一部分保存的范围附带条件证据；少数点的正常运行不证明整个连续区间都稳定。正常但低分的边界会保留，效率失败只说明当次完整配置不合格。

追加 seed 复核后，后续决策使用更新后的证据；旧成功记录保留在 `measurement_history` 中，不会覆盖后来的失败。范围外扩或联合组合若与共同预算不兼容，或 DQN 批大小超过固定回放容量，会预检为 `INFEASIBLE_LEARNING_BUDGET`，记录原因且不启动采样，不把没有学习更新的运行当作正常训练，也不单独提高该参数组合的预算。

如果查看范围后希望手动改变范围、目标或预算设置，请保存新的 HPO YAML 并使用新的输出目录。不要直接改已有 campaign 的 manifest 或结果文件以绕过一致性检查。

## 场景与评分配置

`base_config` 可以使用绝对路径；相对路径以 HPO YAML 所在目录解析。场景内部资产路径保持相对场景 YAML 的含义，不会因作业输出到不同目录而改变。HPO 文件无需再复制模型路径或材料约束。

```yaml
protocol: hpo_v1
base_config: ../../my_scenario/config.yaml
method: a2c
parameter_profile: builtin:a2c

objective:
  field: my_property
  direction: maximize
  threshold: 0.8
  threshold_inclusive: false
  normalization: {kind: sigmoid, scale: 0.1}

metrics:
  best_fraction: 0.10
```

这里的性质、阈值及尺度仅演示字段格式。HPO 的目标阈值在整个 campaign 固定，不能对每个 RL 试验重新求中位数。

| 指标 | `minimize` | `maximize` |
|---|---|---|
| `qualified_fraction` | 性质低于阈值且通过场景约束的比例 | 性质高于阈值且通过场景约束的比例 |
| `best_fraction_mean` | 通过场景约束的最小一部分值的均值 | 通过场景约束的最大一部分值的均值 |
| `best_value` | 通过场景约束的最小值 | 通过场景约束的最大值 |
| `attempts_per_unique` | 尝试次数除以成功评价的不同候选数 | 相同 |

`threshold_inclusive: true` 分别改为 `<=` 或 `>=`。达标比例分母包括所有成功评价的不同候选，不能先删除约束不合格材料再计算。最优子集槽位数为 `ceil(best_fraction * budget)`；约束可行材料不足时，归一化子集质量按槽位填充率折减。没有可行材料时原始均值、极值为 `null`，不伪造数值或宣称质量平台。

默认训练分权重依次为达标比例 0.45、子集质量 0.30、最优质量 0.10、采样效率 `unique / attempts` 0.15；生成分为前三项 0.50、0.40、0.10。可通过 `metrics.training_weights` 和 `metrics.generation_weights` 修改，所有权重和归一化规则会被固定记录。HPO 评分不会自动替换场景的 RL reward。

正阈值和正目标可以使用 `positive_threshold_ratio`；零、负数或其他尺度的目标使用带固定正 `scale` 的 `sigmoid`。方向同步控制达标比较、子集排序、极值和归一化，不需要修改代码。

## 预算向小、向大都测试

```yaml
budget:
  mode: fraction_of_space
  strategy: bidirectional_grid
  train:
    min_fraction: 0.000001
    initial_fraction: 0.0001
    max_fraction: 0.01
    factor: 2
    min_unique: 100
  generation:
    min_fraction: 0.000001
    initial_fraction: 0.0001
    max_fraction: 0.01
    factor: 2
    min_unique: 100
  tolerances:
    score: 0.01
    qualified_fraction: 0.02
    best_fraction_mean: {kind: absolute, value: 1.0}
    best_value: {kind: absolute, value: 1.0}
    attempts_per_unique: 0.10
  upper_confirmation_levels: 2
```

`0.0001` 表示 0.01%，只是起点。程序从起点按 `factor` 向下和向上生成档位，显式包含上下边界，再按精确不同候选数取整、合并重复整数预算。训练和生成各有一份独立网格。物理指标的绝对容差必须使用目标的原始单位；也可指定 `threshold_fraction`，但零阈值必须使用绝对容差。

完整网格在少量代表配置上校准。联合搜索后，入围配置再次在所有预算档位上复核，允许预算向任意方向改变；预算改变后建立独立 study 补充搜索，不把不同预算的测量混为同一 TPE 保真度。

百分比要求有限且能精确计数的规范候选空间，不能把动作路径数或训练数据行数当作分母。不能精确计数的场景可明确选择绝对预算：

```yaml
budget:
  mode: absolute_unique
  train: {min_unique: 100, initial_unique: 400, max_unique: 1600, factor: 2}
  generation: {min_unique: 50, initial_unique: 200, max_unique: 800, factor: 2}
```

预算选择返回两个不同结论：`selected_budget` 是本轮已测范围内质量足够接近参照结果的最小预算；`convergence_status` 是关于平台的证据。`OBSERVED_PLATEAU` 要求所选配置在至少指定数量的更大档位上也合格，指标变化都在容差内，且整个计划网格已经测完。更大预算明显变差、效率失败、缺失测量或达到预算上限分别报告，不冒充平台。

所有实际准备采样也计入尝试。A2C 的 scaler-only warmup 不产生性质评价，因此会消耗尝试而不增加有效 unique。例：20 次准备采样、严格 1.5 上限，至少需要 41 个 unique；程序会从共同网格选合法档位，不自动降低 warmup 或放宽上限。计时只更新 ETA，不按期望时长倒推或降低预算。

## 尝试上限及生成温度

```yaml
efficiency:
  max_attempts_per_unique: 1.5
  comparison: lt
  train: {}
  generation: {}  # 可覆盖，例如 {max_attempts_per_unique: 1.8}

generation_search:
  parameter: gen_temperature
  initial_values: [0, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
  max_extra_values_per_direction: 2
  fixed: {gen_epsilon: 0, gen_top_frac: 0, exploration_gen_eps: 0}
  paired_seeds: [401, 503, 607]
```

用户所说的 `gen_temp` 对应规范字段 `gen_temperature`。温度 0 是贪心诊断点，正温度对网络分数使用 softmax 温度采样。DQN 的 Q 值与 A2C 的 logits 尺度不同，温度必须按各自训练网络测量。每个预算比较同一温度集合，不能在初始数量就淘汰温度后声称所有数量都已检验。生成 seed 数量必须与验证网络数量一一对应。

训练和生成分别去重。训练中出现过的候选在生成阶段首次出现，仍计入生成预算；同一阶段内部再次重复只增加尝试和时间。失败、无效候选及重新采样的重试同样计入尝试。

上限默认是严格小于 1.5：目标 1000 个 unique 最多允许 1499 次尝试。不会仅因运行前缀暂时超过 1.5 就停止；只有 `已开始尝试 + 尚缺 unique` 已超过允许总尝试次数时，才确认即使余下每次都成功也无法达标，提前停止。批量结果结算后再做此判断，不能把 pending 当永久失败。

效率不合格保存候选、账本和可用检查点，状态为 `INFEASIBLE_EFFICIENCY`。该配置不进入完整预算的优胜排序，任一必需 seed 失败都不能靠其他 seed 的平均分掩盖。控制器不会在原试验内提高温度、降低预算、放宽上限或重置已经消耗的尝试。

## 输出、错误与恢复

主要输出：

| 路径 | 内容 |
|---|---|
| `manifest.json` | 生效配置、代码和资产契约 |
| `state.json`、`progress.json` | campaign 阶段、状态与进度 |
| `ranges.json`、`range_decisions.jsonl` | 参数范围和每项边界修改依据 |
| `stages/*.json` | 各阶段完整汇总与预算证据 |
| `best_training.yaml`、`frozen_models.json` | 训练参数选择摘要与冻结验证网络 |
| `best_generation.yaml` | 生成温度与数量选择 |
| `selection.json` | 选择结果与实际成本汇总 |
| `jobs/*/candidate_records.jsonl` | 本作业独立的完整原始候选档案 |
| `jobs/*/candidate_budget.json` | 真实尝试、成功 unique、准备采样及终止状态 |
| `jobs/*/stdout.log`、`stderr.log` | 运行日志及错误详情 |

`trials.csv`、`generation_trials.csv` 按实际作业汇总费用，复用同一个作业不会再次累加；未完成作业的已花尝试也在 `selection.json` 中保留。不能恢复的历史耗时标记为未知，不写成零。已知精确空间大小时，CSV 同时给出整数预算及其实际空间占比。`best_training.yaml` 是包含 HPO 参数与收敛状态的选择摘要，不能直接替代原始场景配置；场景与生效作业配置仍保存在 manifest 和 `jobs/*/config.yaml`。

```bash
rl-matdesign hpo status --out runs/my_hpo
rl-matdesign hpo resume --out runs/my_hpo
```

`resume` 恢复上次请求的阶段范围，并复用契约一致、校验通过的已完成作业。中断作业只有在训练/生成状态、随机状态、候选账本和档案与同步检查点一致时才能继续。硬退出后若账本已经前进但检查点没有同步，命令会拒绝抹掉已花费额度；保留目录用于诊断，不能把重新开始伪装成同一次低成本试验。

本版逐次持久化尝试账本，并保存同步检查点。快速预测器场景中，写盘可能占明显时间；`--out` 适合放在低延迟本地存储。ETA 使用实际作业墙钟时间，包含模型初始化、训练和持久化开销，不能把候选预测时间直接当作完整 HPO 运行时间。后台存储负载变化时估计会随新测量更新。

修改目标、场景、代码、模型资产或 HPO 配置后，应使用新的输出目录。恢复不允许换预算或效率上限，不从其他 campaign 导入旧 RL 优胜权重。生成失败不会解除冻结网络的约束。

`INFEASIBLE_NUMERICAL` 表示配置导致的数值失稳；`SYSTEM_ERROR` 表示运行或资产问题，不能当作参数低质量证据。全部配置不合格时返回 `NO_FEASIBLE_CONFIGURATION`；有评价但没有约束可行材料时报告 `NO_FEASIBLE_MATERIAL_EVIDENCE`；预算修订次数耗尽仍变化时报告 `BUDGET_UNRESOLVED`。这些情况保留证据，不强制导出成功结论。

## 三个软件 smoke 与 GlassNet 首次验证

仓库提供三份可以引用现有 dummy 场景的 HPO 配置：

```bash
rl-matdesign hpo run --hpo-config configs/hpo/v1/dummy_a2c.yaml --out runs/hpo_smoke_a2c
rl-matdesign hpo run --hpo-config configs/hpo/v1/dummy_dqn_bootstrap.yaml --out runs/hpo_smoke_dqn_bootstrap
rl-matdesign hpo run --hpo-config configs/hpo/v1/dummy_dqn_mc.yaml --out runs/hpo_smoke_dqn_mc
```

这三份配置只用于验证软件流程：训练预算为 48、64、96；生成预算为 8、16、32；两个训练 seed、两个全新的验证网络；生成温度为 0、1、100。训练搜索仅两个组合，大多数参数固定为单个候选，范围补点关闭。它们不是参数科学推荐，dummy 的合成预测也不能替代 GlassNet 验证。贪心温度可能因重复而正常被效率门槛拒绝，这不是程序崩溃。

GlassNet 提供 `glassnet_a2c.example.yaml`、`glassnet_dqn_bootstrap.example.yaml`、`glassnet_dqn_mc.example.yaml` 三份独立可编辑的 HPO 模板。它们引用用户给出的 benchmark 场景根目录：

```text
/aisi/yxliu/rl-matdesign/glasspy/glassnet_benchmark_nodes_20261106/configs/
```

| HPO 模板 | 当前 `base_config` 对应的场景 |
|---|---|
| `glassnet_a2c.example.yaml` | `coarse/a2c.yaml` |
| `glassnet_dqn_bootstrap.example.yaml` | `coarse/dqn_boostrap.yaml` |
| `glassnet_dqn_mc.example.yaml` | `coarse/dqn_mc.yaml` |

`dqn_boostrap.yaml` 沿用用户提供的拼写，需在实际运行主机核实文件名；HPO 方法名仍是 `dqn-bootstrap`。模板使用完整绝对路径，便于复制到任意位置并另存为自己的 `hpo.yaml`：

```bash
cp configs/hpo/v1/glassnet_a2c.example.yaml hpo.yaml
# 编辑 hpo.yaml，填写用户确定的 objective.threshold。
rl-matdesign hpo validate --hpo-config hpo.yaml
rl-matdesign hpo plan --hpo-config hpo.yaml
rl-matdesign hpo run --hpo-config hpo.yaml --out runs/glassnet_coarse_a2c_hpo
```

同一算法从 coarse 切换到 medium 或 large 时，只修改 `base_config` 中的目录，例如：

```yaml
base_config: /aisi/yxliu/rl-matdesign/glasspy/glassnet_benchmark_nodes_20261106/configs/medium/a2c.yaml
```

模型位置、实际氧化物设计空间及 CTE/Tg 条件全部沿用被引用的 benchmark 场景配置，不搬进 HPO 文件，也不在 HPO 中重新定义。运行目录则使用新的 `--out`，避免跨设计空间复用 campaign 结果。

三份模板有意保留 `objective.threshold: null`。T3 达标线仍由用户确定，程序不会从运行结果临时生成阈值或猜测工业要求。场景文件存在且其他配置有效时，未填写阈值的 `validate` 会返回退出码 2，并显示 `HPO error: objective.threshold must be a finite number`；这是预期的配置提示，不会启动训练或抛出 traceback。若当前主机没有上述场景文件，先明确报告路径不存在。

模板中的搜索范围、预算比例、试验次数和容差是可改的工程初值，尚未通过这套真实 GlassNet benchmark 验证，不应解读为已确定的科学推荐。尚未确定的性质阈值保持待定，不妨碍程序和软件测试先完成。真实 GlassNet HPO 由用户在场景及模型资产所在主机运行。

GlassNet 的命名 T3 输出、T3/Tg 的 K 单位及 CTE 变换由现有场景适配器确认。若 T3 阈值选用 GlassNet 训练标签中位数，应先核实训练数据版本、对应性质列及单位，并把得到的固定值写入 HPO YAML。软件 smoke 通过不代表材料目标已经实现。
