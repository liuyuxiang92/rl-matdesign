# YAML 驱动的两部分 HPO

当前接口将用户配置分为公共评价、训练、生成三部分：第一部分诊断训练参数范围；第二部分校准训练预算、联合搜索及复核训练参数，再使用冻结网络联合选择生成参数与生成数量。训练参数按训练收集结果评分，生成参数按生成结果评分。HPO 评分不会替换场景的训练 reward。

场景 YAML 负责设计空间、模型资产、命名性质、优化方向及材料约束；HPO YAML 引用场景，负责达标阈值、评分权重、超参数范围和实验规模。已接入本项目的场景无需修改 HPO 代码；新预测器或新环境仍需先完成普通场景接入。

## 从独立 HPO 文件运行

GlassNet 模板位于 `configs/hpo/v1/glassnet_a2c.example.yaml`、`glassnet_dqn_bootstrap.example.yaml`、`glassnet_dqn_mc.example.yaml`。目录名为历史路径；当前模板采用精简接口，运行时记录为 `hpo_v2`，用户不需要填写 `protocol`。

```bash
cp configs/hpo/v1/glassnet_a2c.example.yaml hpo.yaml
# 编辑 base_config，填写 evaluation.threshold，检查场景中的生成方式。
rl-matdesign hpo validate --hpo-config hpo.yaml
rl-matdesign hpo plan --hpo-config hpo.yaml
rl-matdesign hpo run --hpo-config hpo.yaml --out runs/my_hpo
```

安装项目后使用 `rl-matdesign hpo`；源码 checkout 可用 `PYTHONPATH=src python -m rl_matdesign.hpo` 替代。执行实验需要项目训练依赖及 Optuna。`validate`、`plan` 不调用性质预测器、不训练网络；自动质量尺度要等实际参考实验后才能确定。

支持 `a2c`、`dqn-bootstrap`、`dqn-mc`。方法可以写入 YAML，或添加 `--method a2c`；两处同时指定时必须一致。相对 `base_config` 路径按 HPO 文件所在目录解析，场景资产仍按场景文件目录解析。

下面是结构示例。数值仅展示格式，不能视为经过 GlassNet 验证的推荐值；完整算法范围见对应模板。

```yaml
base_config: config.yaml
method: a2c
random_seed: 42
parallel_jobs: 1
evaluation:
  threshold: null
  best_fraction: 0.10
  score_tolerance: 0.01
  max_attempts_per_unique: 1.5
training:
  repeats: 3
  score_weights:
    qualified_fraction: 0.45
    best_fraction_quality: 0.30
    best_value_quality: 0.10
    sampling_efficiency: 0.15
  parameter_ranges:
    pg_lr_actor: {type: float, low: 0.00001, high: 0.01, log: true}
  budget:
    mode: fraction_of_space
    min_fraction: 0.000001
    initial_fraction: 0.0001
    max_fraction: 0.01
  search: {max_rounds: 3, trials_per_round: 30, startup_random_trials: 10}
generation:
  repeats: 3
  score_weights:
    qualified_fraction: 0.50
    best_fraction_quality: 0.40
    best_value_quality: 0.10
  # initial_values 可省略，按场景选中的采样方式提供初始值。
  budget: {mode: absolute_unique, min_unique: 100, initial_unique: 400, max_unique: 1600}
```

`parameter_ranges` 覆盖内置算法范围中的同名项。省略某个参数会保留内置范围，不表示固定该参数；要固定支持的参数，可设置 `type: categorical, choices: [固定值]`。`low`、`high` 是范围诊断的初始边界，`choices` 是初始离散选项；有实验证据时可按限制外扩，数值失稳时可内收。正常但低分的区域不会仅凭低分被删除。`dqn_grad_steps_per_ep` 继续使用场景值，不参与搜索。

训练范围诊断和生成参数范围搜索均内部固定为：每个参数的每一侧最多补测两个新值。用户 YAML 不再包含 `training.range_search` 或 `generation.max_extra_values_per_direction`。训练与生成的预算档位也统一使用内部固定倍数 2，不再填写 `budget.factor`。每轮 Optuna 搜索后，内部固定最多选择三个不同且符合条件的候选参数组合参加复核；用户 YAML 不包含 `training.validation`。这些规则在解析后的执行计划中仍会显示，其中训练和生成每侧补测上限均为 2，复核入选组合数上限为 3，便于核实实际实验安排；后续需要可调时再扩展用户接口。

## 目标继承与评分

程序解析场景 reward，要求它有且只有一个非零优化目标，并通过该目标的 `properties[].name` 获取性质、`direction: min/max` 和单位。约束项不自动成为优化目标。性质名称可以不同于模型的 `output`：例如 `name: working_temperature`、`output: T3` 时，HPO 使用 `working_temperature`。多目标、目标方向不明确或 reward 系数与方向矛盾时直接报告配置问题，不猜测第一个性质。

HPO 只填写 `evaluation.threshold`。最小化使用 `性质值 <= 阈值`，最大化使用 `性质值 >= 阈值`，等于阈值也达标。阈值在整个实验固定，不按每组材料重新求中位数，也不自动加入训练 reward。GlassNet 模板保留 `threshold: null`，执行前必须填写；如采用训练数据中位数，应先核实数据版本、对应列和单位。Tg、CTE 条件只来自场景配置，HPO 不自动添加或复制它们。

| 评分项 | 含义 |
|---|---|
| `qualified_fraction` | 同时满足场景约束及目标阈值的不同材料数，除以全部成功评价的不同材料数 |
| `best_fraction_quality` | 约束可行材料中最好一部分的均值，经固定质量映射转换后的分数 |
| `best_value_quality` | 最好一个约束可行材料的性质，经同一质量映射转换后的分数 |
| `sampling_efficiency` | 成功评价的不同材料数除以尝试次数，只纳入训练加权评分 |

`best_fraction: 0.10` 表示最好 10%，方向随场景变化。名额数为 `ceil(best_fraction × 目标预算)`；约束可行材料不足时，均值使用已有可行材料，质量分数再乘以名额填充比例。没有可行材料时，两项质量分数为零，原始均值和极值保留为 `null`。不满足约束但成功评价的材料仍在达标比例分母中。每阶段的权重非负且和为 1；全部规定重复必须完成并满足效率上限，不能只平均其中成功的实验。

新版固定采用 sigmoid 质量映射，支持正数、零和负数，不开放 `normalization.kind`。最小化时，输入为“阈值减性质值，再除以尺度”；最大化时相反。性质等于阈值时分数为 0.5，性质越好分数越高。

尺度来自按原有计划顺序第一份符合条件的训练参考作业，候选采样成本照常计入。并行完成的先后不会改变尺度来源。程序依次尝试目标性质的四分位距、最大最小值之差、与阈值距离的绝对值中位数，取第一个正的有限尺度，保存到 `quality_calibration.json`。尺度随后冻结，所有参数组合、预算和生成实验共用，不会每次分别归一化。若数据全部等于阈值等情况导致无法得到正尺度，保留成本并报告 `QUALITY_CALIBRATION_REQUIRED`，不能悄悄使用任意物理单位尺度；应检查参考预算及场景，在新目录开始调整后的实验。

`evaluation.score_tolerance` 是统一的**绝对总分差**：0.01 表示总分差 0.01，不是相对改善 1%。范围边界改善、候选比较和预算平台判断共用它。当前只根据总评分判断改善与平台；原始分项仍记录，但不再各设容差或阻止总分平台判断。材料约束及效率上限继续独立执行。

## 两个重复次数与自动种子

`random_seed` 控制可复现的种子派生；用户只设置两个重复次数，不填写阶段种子列表。

| 实验 | 次数控制 | 配对规则 |
|---|---|---|
| 参数范围诊断 | `training.repeats` | 对照与各参数取值使用相同探索种子 |
| 训练预算校准 | `training.repeats` | 不同预算与代表配置使用相同探索种子 |
| 联合参数搜索 | `training.repeats` | 每套参数使用相同探索种子 |
| 候选参数与预算复核 | `training.repeats` | 自动派生另一组验证种子，与前三步分开 |
| 生成参数及数量搜索 | `generation.repeats` | 每个冻结网络重复采样，各设置使用相同模型及种子配对 |

前三步共享探索种子，第四步重复次数相同但使用新种子。新版所有训练比较都使用固定重复组；范围结果存在不一致时保留为未解决证据，不隐式追加种子或增加重复次数。预算修订沿用搜索及复核规则，不新增一套用户种子。

当前实现中，`random_seed: 42` 与 `training.repeats: 3` 派生的探索种子为 `[2746317213, 1181241943, 958682846]`，独立验证种子为 `[3163119785, 1812140441, 127978094]`。因此三次重复不是都使用 42；42 决定整组可复现种子。实际组别会写入解析计划的 `seed_schedule`，修改主种子或重复次数后应重新查看计划。

`generation.repeats` 表示“每个冻结模型的生成重复次数”，默认值为 3，与训练阶段统一使用 `repeats` 这个字段名。生成阶段已有多个训练种子产生的冻结模型，该字段再控制每个模型使用多少个生成随机种子。区分这两个来源，才能同时观察训练随机性和采样随机性，不会把它误解为整个生成阶段的总实验次数。

例如 `training.repeats: 3`、`generation.repeats: 3`：最终选定一套训练参数，保留它的三个验证网络；每个“生成参数值 × 生成预算”设置对每个网络采样三次，即 **3 个网络 × 3 次 = 9 次生成实验**，不重新训练网络。每轮搜索后内部最多选入三套不同候选参数参加复核，符合条件的组合不足三套时按实际数量选入；这与 `training.repeats` 独立，每套参数分别按该重复次数训练。所有实际种子及模型配对都写入结果。

## 可选并行执行

新启动实验通过顶层 `parallel_jobs` 设置并发上限，必须为正整数：省略或设为 `1` 时顺序执行，设为 `3` 时整个 HPO 最多同时运行三个独立 worker 作业。一个作业对应一个参数组合与一个训练种子，或一个冻结模型与一个生成种子；上限不会按参数组合或种子层数相乘。它不要求凑满：例如设为10但当前只有3个独立作业时，就运行这3个；只有1个时运行1个，没有作业则不启动 worker，不会等待或额外生成任务来填满名额。`plan` 的 `resolved_plan.parallel_jobs` 和运行清单记录生效值。

```yaml
parallel_jobs: 3
```

初始训练参数网格中不同取值和种子的作业、已确定的预算网格、候选复核，以及生成参数和预算网格中的独立作业可以并行。依赖实测结果的范围外扩、预算修订和阶段切换仍按先后依赖执行。Optuna 启动阶段互不依赖的随机提案及其种子可以并行；之后的自适应 TPE 提案保持逐组提出和反馈，每个提案内部的训练种子可以并行；不会在上一组尚未反馈时提前猜测下一组自适应参数。

并行只改变作业调度，不改变算法、评分、材料预算、重复次数或派生种子。共享质量尺度按原有计划顺序选取参考结果并冻结，结果汇总也按计划顺序处理，避免把最先完成的作业当作参考。现有作业复用和恢复校验继续生效；不要为恢复直接修改已归档的原 YAML，可使用下文 `restart/resume --parallel-jobs` 运行时覆盖。

每个 worker 独立加载模型。多个作业使用同一 GPU 时，显存和模型加载开销会叠加，CPU、内存与磁盘也可能成为限制，因此增加 `parallel_jobs` 不保证线性提速。根据模型大小和可用资源设置并行数；它不负责自动分配多张 GPU，也不修改场景的设备设置。

## 训练和生成预算分别设置

每个阶段的 `budget.mode` 可独立选择，允许训练按比例、生成按绝对数量，反之亦可。

| 模式 | 字段 | 预算含义 |
|---|---|---|
| `fraction_of_space` | `min_fraction`、`initial_fraction`、`max_fraction` | 设计空间不同候选总数乘比例，向上取整 |
| `absolute_unique` | `min_unique`、`initial_unique`、`max_unique` | 直接指定成功评价的不同候选数量 |

比例 `0.0001` 是 0.01%，不是 0.0001%。两个阶段都从初始值按内部固定倍数 2 向下除、向上乘，纳入边界并合并相同整数预算。例如起点 400、下界 100、上界 1600 对应档位 100、200、400、800、1600。比例模式不接受额外 `min_unique`；程序只因学习更新和准备采样的实际可执行条件剔除过小档位，不再使用隐藏的 100 候选下限。设计空间必须可精确计数，不能使用动作路径数或 GlassNet 训练集行数；不能精确计数时使用绝对模式。

`validate` 和 `plan` 输出中的 `minimum_training_unique` 是当前算法范围与场景准备成本要求的最低训练预算；`omitted_training_budgets` 列出因此剔除的用户预算档位，`budget_grids` 列出实际可执行档位。它们只说明能否执行，不意味着最低预算已足够学习或结果已收敛。

预算指**单个参数组合、单个重复、单个阶段**的有效候选数，不是整个 HPO 的总开销。不同训练预算分别从头训练，不能自动把小预算模型继续训练当作独立大预算比较。准备采样也有成本：A2C 仅拟合 scaler 的采样增加尝试、不增加成功评价数，可能使小预算无法满足效率限制。计时只更新 ETA，不根据期望时长降低预算。

`search.trials_per_round` 是每轮提出的参数组合数；其中前 `startup_random_trials` 组随机探索，已包含在该总数内。后续使用 TPE 按训练评分提案。每轮搜索结束后，程序按评分选出最多三个不同且符合条件的候选参数组合；符合条件的组合不足三个时按实际数量选入。这个上限内部固定，不受 `training.repeats` 影响。程序继续保留原有对照组合，并记录实际复核数量，因此整组复核实验可以包含超过三套参数。验证在完整训练预算网格上复核；推荐预算改变时，在新预算建立下一轮独立搜索。`max_rounds` 包含初始轮，因此 3 轮、每轮 30 组最多是 90 个联合搜索提案，范围、预算、复核和生成实验另计。

`training.search.max_rounds: 3` 不表示固定执行三轮。比如训练预算校准先推荐 1000，第一轮就在 1000 下优化参数；若独立验证仍推荐 1000，则第一轮后结束。如果验证改为推荐 2000，第二轮才在 2000 下重新搜索，因为参数在更大预算下的表现可能不同。第二轮验证仍选 2000 时结束；若改为 4000，则进入第三轮。若第三轮后验证仍推荐另一档预算，就达到轮数上限并报告 `BUDGET_UNRESOLVED`，不会把未稳定的预算写成最终结论。这个上限用于控制“预算改变后重新优化参数”的次数。

预算可向更小或更大改变。结果中的 `selected_budget` 与 `convergence_status` 分开：前者是已测范围内推荐的较小预算，后者说明是否有总分平台证据。平台仍要求足够的更大预算观测，当前内部固定至少两个更大档位。到达上界仍改善、较大预算明显恶化、效率失败或证据缺失均如实报告；不能把推荐某一预算写成已证明全局收敛。预算修订轮数耗尽仍未稳定时报告 `BUDGET_UNRESOLVED`。

## 生成方式从场景继承

场景中 `gen_temperature`、`gen_top_frac`、`gen_epsilon` **最后显式出现的字段**选择一种生成方式；0 也属于显式选择，程序补充的默认值不能覆盖它。例如：

```yaml
gen_temperature: 1.0
gen_epsilon: 0.1  # 最后出现：使用 epsilon，温度在内部关闭
```

HPO 只优化被选中的参数。`generation.initial_values` 是该参数的初始候选值，可省略以使用对应方式的内置值；epsilon 与 top fraction 限于 `[0,1]`，temperature 非负。数量来自 `generation.budget`，不会将 `initial_values` 当作生成个数。

生成参数范围搜索在初始候选值之外，向较小、较大的方向各最多增加两个生成参数取值；这个上限内部固定为 2，与训练范围诊断一致，不向用户开放。它控制测试哪些参数值，生成预算与每个模型的重复次数仍分别由 `generation.budget` 和 `generation.repeats` 控制。程序先测试初始候选值；某个预算下的最佳结果落在当前取值范围的边界时，才尝试向对应方向扩大范围。温度向下测试最小正温度的十分之一，向上测试最大正温度的十倍。例如初始正温度为 `[0.1, 1, 10]`，若下边界持续最优，可依次补测 `0.01`、`0.001`；若上边界持续最优，可依次补测 `100`、`1000`。每个新取值都按相同预算网格和模型重复规则评价。如果当前所有结果都不符合条件，程序也可在上限内向更大值补测。epsilon 和 top fraction 的范围有限，向外补测直接到合法边界 0 或 1，不会越界。解析后的执行计划仍会显示每侧补测上限 2。

温度 0 是贪心，较小正温度集中于高网络分数动作，较大温度使采样更分散；它不是探索百分比，A2C 与 DQN 的分数尺度也可能不同。epsilon 调节随机动作概率，top fraction 调节保留的高分动作比例。HPO 内部关闭未选中的方式及额外随机生成批次，用户不需要填写三个 `fixed: 0`。

普通场景解析器也执行最后显式字段优先。生效配置中的内部 `generation_selection` 快照保存选中的方法和有效值，防止导出时排序、补全默认字段或重新加载改变选择。显式覆盖优先于场景；编辑已解析配置的策略值时会识别变化。普通用户编辑原始场景即可，无需手工维护快照。

训练与生成分别记账：生成第一次遇到训练中出现过的材料，仍计入生成预算；同一阶段再次重复只增加尝试。成功评价但不满足材料约束仍消耗有效预算；失败、无效候选和重新采样也消耗尝试。

效率统一要求 `尝试次数 / 有效不同候选数 < evaluation.max_attempts_per_unique`，默认严格小于 1.5。目标 1000 个不同候选最多允许 1499 次尝试。不会仅因早期比例偏高就停止；待已启动结果结算后，若即使余下每次都成功也无法在上限内完成，标记 `INFEASIBLE_EFFICIENCY` 并停止。已消耗成本保留，不自动放宽上限、提高温度或重置预算。

## 分阶段执行、输出与恢复

```bash
rl-matdesign hpo run --hpo-config hpo.yaml --out runs/my_hpo --stage ranges
rl-matdesign hpo run --hpo-config hpo.yaml --out runs/my_hpo --stage search
```

| `--stage` | 内容 | 前置结果 |
|---|---|---|
| `all` | 完整两部分流程，默认值 | 无 |
| `ranges` | 训练参数范围诊断 | 无 |
| `search` | 训练预算、参数搜索及复核、生成参数与数量 | `ranges` |
| `train-budget` | 训练预算校准 | `ranges` |
| `train-search` | 训练参数联合搜索、复核及预算修订 | `train-budget` |
| `generation` | 冻结网络的生成参数与数量搜索 | `train-search` |

缺少前置结果时明确报错。主要输出包括 `manifest.json`（生效配置与来源）、`quality_calibration.json`、`ranges.json`、`range_decisions.jsonl`、`stages/*.json`、`frozen_models.json`、`best_training.yaml`、`best_generation.yaml`、`selection.json` 及每个 `jobs/*/` 的配置、候选档案、预算账本和日志。选择 YAML 是摘要，不能直接替代完整场景配置。汇总包括失败和未完成实验已经花费的成本，复用同一作业不重复累计。

```bash
rl-matdesign hpo status --out runs/my_hpo
rl-matdesign hpo restart --out runs/my_hpo --check-only
rl-matdesign hpo restart --out runs/my_hpo
# resume 仍可使用，与 restart 采用相同恢复检查
rl-matdesign hpo resume --out runs/my_hpo
# 恢复时只调整独立任务的并发上限，不编辑原 YAML
rl-matdesign hpo restart --out runs/my_hpo --parallel-jobs 10 --check-only
rl-matdesign hpo restart --out runs/my_hpo --parallel-jobs 10
```

`restart` 和 `resume` 支持运行时 `--parallel-jobs N`（正整数）。不传时使用原 YAML 的值，旧串行计划缺失时为 1；上一次覆盖不会自动继承。覆盖只应用于独立作业调度，不修改原 YAML、manifest、实验参数、种子、每项预算或 job ID。完成任务直接复用，中断任务各自恢复；TPE 和自适应边界探测继续遵守依赖顺序。实际独立任务不足 N 个时按可用任务数量运行。`--check-only` 报告原始、请求和实际并发数但不写文件；实际恢复在全部预检通过后把设置变化另存到 `execution_options.json`。并行故障时可能已有更多任务消耗成本，这些成本仍完整记账。

新运行使用空输出目录。恢复仅复用配置、代码、资产与结果校验一致的作业；中断继续还要求检查点、随机状态和候选账本同步，不能抹掉已花费额度。配置、模型资产或未经审核的源码变更仍要求新输出目录；下述已登记的 checkpoint 存储修复可以继续已有 campaign。ETA 随真实完整作业耗时更新，包含模型初始化、训练和账本持久化。

历史 `protocol: hpo_v1` YAML 仍由兼容读取器识别并发出提示，保留旧显式目标、归一化和阶段种子语义；它们不是当前推荐接口。六份模板已全部更新，路径保持不变。升级为精简配置会改变协议及评分契约，不能继续使用旧 campaign 目录。[第一版实现计划](hpo_v1_implementation_plan.md) 和历史验证报告描述此前版本，不代替本页。

## Checkpoint 磁盘占用与旧任务清理

HPO 训练只原子覆盖一份 `jobs/<id>/checkpoint.pt`，不再逐步累积 `checkpoint-epN.pt`（DQN）或 `checkpoint-iterN.pt`（A2C）。保存频率在 HPO 内部固定为 `--save-checkpoint-freq 1`：DQN 两种方法每完成一次候选 episode 及相应训练处理后保存（含重复候选，不是每个 action）；A2C 每完成一个训练 batch 并更新网络后保存，因此一个 batch 含多个候选时不是每个候选保存一次。断点保存优化器、随机状态、已花预算和 DQN replay buffer，用于恢复最新同步状态。评分使用候选档案，不读取历史训练快照；冻结生成使用被选中任务的最终模型。该策略在 HPO 内部固定，不新增 YAML 参数；普通非 HPO 训练仍保留编号快照。

每个任务的最终 `qnet.pt` / `policy.pt` / `value_net.pt`、`std_scaler.bin`、最新完整断点以及候选档案、账本和结果仍保留。生成任务继续覆盖自己的 `generation_checkpoint.pt`；失败任务可能保留诊断用 `partial_checkpoint.pt`。这次修复消除的是同一任务内的历史全量副本，并不让整个 campaign 的磁盘占用与任务数量无关。原子写入期间需要同时容纳旧断点和新临时文件。

旧版本每步保存全量历史，DQN 中重复保存不断增长的 replay buffer 尤其占空间。旧任务可使用独立标准库脚本清理历史版本：**先停止 HPO 控制器和全部 worker**，将脚本复制到运行机器，先预览，确认输出后执行：

```bash
python scripts/prune_hpo_checkpoints.py --out /path/to/hpo_output
python scripts/prune_hpo_checkpoints.py --out /path/to/hpo_output --apply
```

脚本仅处理有 HPO 清单和作业元数据的目录，保留 `checkpoint.pt` 当前指向的文件和元数据明确引用的文件，不删除最终模型、scaler、生成/诊断断点或实验记录。不会加载 `.pt` 内容。缺失或异常的最新断点会使对应目录跳过清理，避免猜测哪个文件可恢复。不要直接删除所有 `checkpoint-*.pt`：旧 `checkpoint.pt` 是符号链接，删除其目标会破坏恢复和结果校验。

**清理历史文件不改变原实验配置或已记录文件的哈希。** 磁盘仍满时可以只复制此脚本，先释放空间，再更新代码。本次修复登记了 `50c25e9a`（并行功能前，对应合并提交 `f9d593db`）、`5d0d0dae` / `b8e01748`（对应 `ab73f2fd`）以及 `1a3e26fd` 的精确源码兼容关系。旧串行计划只允许把同时缺失的 `plan.parallel_jobs` 和 `plan.sources.parallel_jobs` 解释为 `1/default`；这项规范化只在源码升级已登记、其余字段和 YAML 指纹一致时生效，原 manifest 不重写。

另单独审核了保留六个历史预测器文件的服务器源码组合：`predictors/{composite,dp_property,dp_structure,hea,perovskite,sinter_calcine}.py`。文件内容必须与兼容表的 SHA 完全一致，升级后仍须保留同样的文件集合；不放行任意额外文件或本地代码修改，也不把带旧模块的任务迁移到缺少这些模块的安装。该组合覆盖服务器旧 campaign 指纹 `d495c572...`、并行版本 worker 和 checkpoint 修复版 `5621cee85...`。无需删除这六个文件或改写记录中的哈希。旧任务可以在通过检查后继续并改用单份最新断点。其他源码版本或本地源码改动不会自动放行。配置、预测模型、种子、数值环境和学习/生成规则仍严格校验，没有跳过校验的开关。

更新到此修复并保留原 YAML、资产和运行环境后，先只读检查，再启动恢复：

```bash
rl-matdesign hpo restart --out ./runs/a2c_coarse --check-only
rl-matdesign hpo restart --out ./runs/a2c_coarse
```

如果在出现任务检查结果前报 `Campaign configuration, software, domain or assets changed`，这是 campaign 契约检查未通过，尚不能判断 checkpoint 是否损坏。用运行 CLI 的同一个 Python 环境执行仓库内的只读诊断脚本（不强行切换 `PYTHONPATH`，以便发现实际加载了另一份安装包）：

```bash
python /path/to/rl-matdesign/scripts/diagnose_hpo_restart.py --out ./runs/a2c_coarse
```

脚本输出实际包路径、旧/新源码指纹、是否登记兼容，以及具体变化字段。若源码指纹未登记，`source_evidence` 还列出当前 Git 提交、已修改/缺失的 Python 文件和额外 Python 文件（包括被 Git 忽略的备份），并从最多 12 份旧调用记录读取历史提交。`committed_tree_plus_current_extra_files_matches_saved_digest` 只是“旧提交加上当前额外文件是否能重现旧哈希”的验证结果，不表示这些文件无害或可以自动恢复。它不创建 campaign、不启动 worker、不加载模型张量、不修改实验目录。`campaign_contract_accepted` 仅表示外层契约可以通过，不代表内部断点已同步，之后仍需 `restart --check-only`。输出中的 `configuration_sources` 还会检查原 YAML 的字节指纹；注释变化也会被记录。诊断失败时保留原文件并根据差异定位原因，不改写 manifest 哈希或跳过校验。脚本位于 `scripts/`，新增它不会改变已登记的包源码指纹。

`restart` 按 `state.json` 保存的阶段继续，不从头创建 HPO；原 `resume` 命令同样支持这套机制。检查在启动任何 worker 前读取所有已有任务，输出以下分类：

| 分类 | 含义 |
|---|---|
| `CACHED` | 完成结果及其记录的文件哈希完整，可直接复用 |
| `SYNCHRONIZED` | 完整 checkpoint、候选档案和预算账本同步，可继续；尚未初始化记账的任务也可正常启动 |
| `RECOVERABLE_TERMINAL` | 终态材料档案完整，可完成结果收尾而不新增采样 |
| `BLOCKED` | 存在损坏、缺失、不同步或需检查的失败证据；输出具体任务路径与原因 |

只读检查不写文件、不加载性质预测模型、不采样。实际恢复还会核对 worker 的运行环境和模型契约。存在任何 `BLOCKED` 时不会启动新采样；先处理报告中的任务。版本升级保留原 `manifest.json`、已完成模型和结果哈希，另写一次 `software_migrations.json` 记录审核过的旧/新源码指纹；后续新断点记录真实的新版本。

磁盘写满可能留下未完成的临时文件、截断的候选记录，或已领先模型断点的已付费账本；有 `checkpoint.pt` 并不保证任意中断点都能精确恢复。遇到 `accounting advanced beyond checkpoint`、`archive differs`、缺失/损坏文件或已记录的失败退出，应保留目录与报错逐项检查，不能删除账本、篡改哈希或把诊断 `partial_checkpoint.pt` 当作完整断点。`restart` 不会声称重建从未成功写入磁盘的网络更新，也不会悄悄从零训练或抹掉已花预算。已完成实验的有效结果不必因此全部重做。

如果外层兼容检查已经通过，但某个任务因账本与 checkpoint 不同步而被标为 `BLOCKED`，停止该 campaign 的控制器和 worker 后，针对报告中的 job ID 进一步检查：

```bash
python /path/to/rl-matdesign/scripts/diagnose_hpo_job.py \
  --out ./runs/a2c_coarse --job-id REPORT中的job_id
```

使用运行 CLI 的同一 Python 环境。此脚本只读取指定任务的本地可信 checkpoint（含可能遗留的 `.pt.tmp`），在 CPU 上提取计数与指纹；不运行模型或采样，也不修改文件。报告对比磁盘账本与断点中的各项计数，检查候选档案及断点对应前缀，并列出冻结模型文件指纹和失败日志。退出码 0 仅表示诊断资料读取完成，不表示可以恢复；退出码 2 表示有缺失、损坏或无法读取的诊断资料。

生成任务先保存已评估候选及账本，再保存生成 checkpoint，因此账本领先不能直接判定训练网络或候选丢失。生成使用冻结的训练网络，但精确续跑仍需要同步的随机状态、预测器状态与预算。即使 `.pt.tmp` 可以加载且计数相符，也需要进一步核对全部契约，不能直接重命名替换。保留原文件并提供诊断输出；不回退账本、不删除已付费候选，也不把旧 checkpoint 的计数手动改成新值。

## GlassNet 初次验证与软件 smoke

GlassNet 模板引用用户提供的 benchmark 场景根目录：

```text
/aisi/yxliu/rl-matdesign/glasspy/glassnet_benchmark_nodes_20261106/configs/
```

三个模板分别引用 `coarse/a2c.yaml`、`coarse/dqn_bootstrap.yaml`、`coarse/dqn_mc.yaml`。切换 medium/fine 时修改 `base_config`；模型路径、Tg/CTE 条件和设计空间继续由场景负责。场景 reward 必须明确只有一个非零目标，不能仅凭属性名 `T3` 假定它是主目标。

填写 `evaluation.threshold` 后先运行 `validate`、`plan`，再在具有场景和模型资产的主机启动真实实验。模板中的范围、比例、权重、试验数和容差都是可修改的工程初值，尚未证明适合该真实 GlassNet benchmark。

本地软件流程可使用 `configs/hpo/v1/dummy_a2c.yaml`、`dummy_dqn_bootstrap.yaml`、`dummy_dqn_mc.yaml`，例如：

```bash
rl-matdesign hpo run --hpo-config configs/hpo/v1/dummy_a2c.yaml --out runs/hpo_smoke_a2c
```

dummy 配置使用小范围、小预算验证软件契约；贪心等配置可能因重复而被效率门槛正常拒绝。软件测试通过不代表真实材料目标已达成，也不能代替 GlassNet 验证。

新版完整回归、三个算法的实际流程及恢复验证见 [精简接口验证记录](hpo_v2_validation.md)。
