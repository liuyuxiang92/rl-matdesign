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
  gen_exploration_target: [0, 0.25, 0.5, 0.7, 0.9, 1]
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

场景设置 `value_normalization: popart` 时，每次新训练先用 100 个成功评价的不同候选估计 reward 的均值与尺度。这是所有方法共有的额外校准成本，不占训练或生成预算，也不进入 HPO 训练评分或尝试效率的分子、分母。它与上文 `quality_calibration.json` 的 HPO 评分尺度不同。校准候选不放入 DQN replay，也不预先占用训练的已见候选集合；以后训练再次遇到同一候选时可正常计入训练预算。失败与重复不会填满这 100 个校准名额，实际预测工作还应单独检查。

校准结束后，DQN 按 `warmup_budget_fraction` 从训练预算派生 warmup 数量，剩余预算用于后续训练；校准不再将 warmup 强制提高到 100。新训练仍至少需要 1 个 warmup 候选来拟合状态 scaler。A2C 的整个训练预算用于策略采样，其 scaler 准备采样规则不变。例如 MC 固定训练预算 115、warmup 比例为 0.02–0.20 时，先校准 100 个候选，再收集 2–23 个 warmup 候选和相应的 113–92 个后续训练候选。Bootstrap 固定预算 458 和 A2C 固定预算 916 同样各增加 100 个校准候选。生成预算单独设为 500 时，每次生成作业仍收集 500 个不同候选，复用冻结的 reward 校准结果。

若不优化预算，把相应 `absolute_unique` 的 `min_unique`、`initial_unique`、`max_unique` 设成同一个数。第一步 `--stage ranges` 也使用该固定训练预算诊断参数范围，不另外增大 RL 预算；之后参数搜索保持同一预算。固定单档预算只能比较参数，不能据此证明预算已收敛。

`search.trials_per_round` 是每轮提出的参数组合数；其中前 `startup_random_trials` 组随机探索，已包含在该总数内。后续使用 TPE 按训练评分提案。每轮搜索结束后，程序按评分选出最多三个不同且符合条件的候选参数组合；符合条件的组合不足三个时按实际数量选入。这个上限内部固定，不受 `training.repeats` 影响。程序继续保留原有对照组合，并记录实际复核数量，因此整组复核实验可以包含超过三套参数。验证在完整训练预算网格上复核；推荐预算改变时，在新预算建立下一轮独立搜索。`max_rounds` 包含初始轮，因此 3 轮、每轮 30 组最多是 90 个联合搜索提案，范围、预算、复核和生成实验另计。

`training.search.max_rounds: 3` 不表示固定执行三轮。比如训练预算校准先推荐 1000，第一轮就在 1000 下优化参数；若独立验证仍推荐 1000，则第一轮后结束。如果验证改为推荐 2000，第二轮才在 2000 下重新搜索，因为参数在更大预算下的表现可能不同。第二轮验证仍选 2000 时结束；若改为 4000，则进入第三轮。若第三轮后验证仍推荐另一档预算，就达到轮数上限并报告 `BUDGET_UNRESOLVED`，不会把未稳定的预算写成最终结论。这个上限用于控制“预算改变后重新优化参数”的次数。

预算可向更小或更大改变。结果中的 `selected_budget` 与 `convergence_status` 分开：前者是已测范围内推荐的较小预算，后者说明是否有总分平台证据。平台仍要求足够的更大预算观测，当前内部固定至少两个更大档位。到达上界仍改善、较大预算明显恶化、效率失败或证据缺失均如实报告；不能把推荐某一预算写成已证明全局收敛。预算修订轮数耗尽仍未稳定时报告 `BUDGET_UNRESOLVED`。

## 基于熵目标的生成

`exploration_adaptive: true` 使用 Step 3 自适应熵目标采样。场景中的
`gen_exploration_target` 是初始目标，范围为 `[0,1]`，随后按候选发现结果调整。
HPO 保留这一设置，不会在作业内切换成另一种采样器。

```yaml
gen_exploration_target: 0.7
exploration_adaptive: true
```

HPO 文件中的 `generation.gen_exploration_target` 是要搜索的初始熵目标列表，
必须非空、取值互不重复，且每个值都在 `[0,1]` 内；省略时采用
`[0, 0.25, 0.5, 0.7, 0.9, 1]`。它与上面场景 YAML 顶层的
`gen_exploration_target: 0.7` 不同：场景填写单个数值，HPO 填写候选列表。
数量仍由 `generation.budget` 控制。

```yaml
generation:
  gen_exploration_target: [0, 0.25, 0.5, 0.7, 0.9, 1]
```

在自适应模式中，显式旧协议的 `generation_search` 节也使用
`gen_exploration_target` 列表键，且不接受 `initial_values`。使用更新后的代码启动实验前，应修改
工作副本并使用新的 campaign 输出目录；不要改写已归档的历史配置来恢复旧实验。
范围搜索只能扩展到合法边界 0 或 1；不再搜索原始分数温度、epsilon 或 top fraction。
目标 0 在最高分并列动作中采样，目标 1 为均匀采样；相同分数不能强制达到低熵。
目标值不是随机动作百分比。控制器会记录实际熵以及目标是否可达。

`exploration_adaptive: false` 切换回历史采样器，不提供固定熵目标模式。
此时训练 HPO 恢复 DQN 的 `dqn_eps_min`、`epsilon_anneal_budget_ratio`
和 A2C 的 `pg_entropy_coef`、`pg_entropy_min` 搜索空间。DQN 衰减步数由
所选训练预算减去 warmup 后乘以历史比例并取整；采样时仍按实际 episode 计时。
历史 HPO 固定重复惩罚为零，普通实验可在 false 模式下设置旧重复惩罚参数。

在 false 模式中，旧的生成策略选择顺序及 `generation_selection` 快照恢复。
简化 HPO 使用 `generation.initial_values` 搜索场景选定的 `gen_temperature`、
`gen_epsilon` 或 `gen_top_frac`；显式 v1 协议保持历史温度搜索及其固定参数。
`exploration_target` 和 `gen_exploration_target` 在此模式下拒绝，旧采样参数
在 true 模式下同样拒绝，避免无效设置被默默忽略。切换模式必须启动新实验，
不能跨模式续训。

训练与生成分别记账：生成第一次遇到训练中出现过的材料，仍计入生成预算；同一阶段再次重复只增加尝试。成功评价但不满足材料约束仍消耗有效预算；失败、无效候选和重新采样也消耗尝试。

Reward 校准采用单独记账，不属于上述两个评分阶段。`reward_calibration_observations.json` 保存校准样本，`calibration_checkpoint.pt` 保存可恢复状态；训练候选档案从校准结束后的采样开始。`reward_calibration_budget.json` 记录 `target_candidates`（目标 100）、`valid_candidates`（已得到的不同有效候选）、`attempts_started`（采样尝试，含重复和失败）、`evaluations`（获取校准观测的调用次数，含失败）和 `status`（冻结前 `RUNNING`，冻结后 `COMPLETED`）。这些计数不等于实际后端模型调用次数，缓存与批量推理仍需看预测器记录。

HPO 的 `trials.csv` 分别列出 `calibration_attempts`、`calibration_evaluations`、`calibration_valid_candidates`，`selection.json` 的 `physical_cost.calibration` 汇总完成及未完成作业的校准开销。同一作业复用时不重复累计。校准耗时已经包含在训练作业的 `wall_seconds` 中，不再加一次。

效率统一要求 `尝试次数 / 有效不同候选数 < evaluation.max_attempts_per_unique`，默认严格小于 1.5。目标 1000 个不同候选最多允许 1499 次尝试。不会仅因早期比例偏高就停止；待已启动结果结算后，若即使余下每次都成功也无法在上限内完成，标记 `INFEASIBLE_EFFICIENCY` 并停止。已消耗成本保留，不会放宽上限或重置预算；已声明的探索控制器按其固定协议工作。

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

`restart` 和 `resume` 支持运行时 `--parallel-jobs N`（正整数）。不传时使用原 YAML 的值（默认 1）；上一次覆盖不会自动继承。覆盖只应用于独立作业调度，不修改原 YAML、manifest、实验参数、种子、每项预算或 job ID。完成任务直接复用，中断任务各自恢复；TPE 和自适应边界探测继续遵守依赖顺序。实际独立任务不足 N 个时按可用任务数量运行。`--check-only` 报告原始、请求和实际并发数但不写文件；实际恢复在全部预检通过后把设置变化另存到 `execution_options.json`。并行故障时可能已有更多任务消耗成本，这些成本仍完整记账。

按一次 Ctrl+C 或向 HPO 控制器发送 SIGTERM 时，调度器停止启动新任务，已有 worker 等待完整断点写入后退出。DQN 会完成当前训练 episode；A2C 会完成当前训练 batch；生成会消费完当前 evaluation wave。若还在初始化或 DQN warmup，会等到第一个完整训练断点。等待期间可能继续消耗当前训练单元所需的预算，仍按原配置记账。到达目标预算的任务先完成最终模型和结果导出，再退出。随后直接使用上面的原生 `restart` 命令，无需外部恢复启动器或额外 YAML 参数。

等待保存期间再次按 Ctrl+C 会强制结束进程组；这种强制停止、SIGKILL、断电或磁盘写满仍可能留下不同步文件，不能保证直接恢复。正常停止不会向正在计算的性质预测子进程发送终止信号，也不再用固定 10 秒超时截断尚未完成的批次。

正常停止与恢复用于当前版本创建的 HPO。另外，本次 DQN 性能更新支持从 `ba6bc1cf`（对应先前 HPO 源码）正常停止的任务直接升级续作，详见下文。更早的磁盘损坏、缺失日志等历史任务不在该升级范围内。

新运行使用空输出目录。恢复仅复用配置、代码、资产与结果校验一致的作业；中断继续还要求检查点、随机状态和候选账本同步，不能抹掉已花费额度。配置、模型资产或未经审核的源码发生变化时，使用新输出目录重新运行。ETA 随真实完整作业耗时更新，包含模型初始化、训练和账本持久化。

历史 `protocol: hpo_v1` YAML 仍由兼容读取器识别并发出提示，保留旧显式目标、归一化和阶段种子语义；它们不是当前推荐接口。六份模板已全部更新，路径保持不变。升级为精简配置会改变协议及评分契约，不能继续使用旧 campaign 目录。[第一版实现计划](hpo_v1_implementation_plan.md) 和历史验证报告描述此前版本，不代替本页。

## 恢复预检

`restart` 按 `state.json` 保存的阶段继续，不从头创建 HPO；原 `resume` 命令同样支持这套机制。检查在启动任何 worker 前读取所有已有任务，输出以下分类：

| 分类 | 含义 |
|---|---|
| `CACHED` | 完成结果及其记录的文件哈希完整，可直接复用 |
| `SYNCHRONIZED` | 完整 checkpoint、候选档案和预算账本同步，可继续；尚未初始化记账的任务也可正常启动 |
| `RECOVERABLE_TERMINAL` | 终态材料档案完整，可完成结果收尾而不新增采样 |
| `BLOCKED` | 存在损坏、缺失、不同步或需检查的失败证据；输出具体任务路径与原因 |

只读检查不写文件、不加载性质预测模型、不采样。实际恢复还会核对 worker 的运行环境和模型契约。存在任何 `BLOCKED` 时不会启动新采样；报告会给出具体任务路径与原因。原 `manifest.json`、已完成模型和结果哈希继续用于严格校验。

磁盘写满可能留下未完成的临时文件、截断的候选记录，或已领先模型断点的已付费账本；有 `checkpoint.pt` 并不保证任意中断点都能精确恢复。遇到 `accounting advanced beyond checkpoint`、`archive differs`、缺失/损坏文件或已记录的失败退出，应保留目录与报错逐项检查，不能删除账本、篡改哈希或把诊断 `partial_checkpoint.pt` 当作完整断点。`restart` 不会重建未成功写入磁盘的网络更新，也不会从零训练或抹掉已花预算。需要重算时显式使用新的空输出目录。

将 reward 校准从训练预算中分离改变了训练协议。旧版本中把校准候选计入训练预算的 campaign 或 checkpoint 不能直接按新协议精确恢复；应使用原归档代码继续旧实验，或在新目录按新协议开始。已有的性能优化升级入口不授权这种协议变更。

## Checkpoint 保存与磁盘占用

HPO 训练只原子覆盖一份 `jobs/<id>/checkpoint.pt`，不再逐步累积 `checkpoint-epN.pt`（DQN）或 `checkpoint-iterN.pt`（A2C）。保存频率在 HPO 内部固定为 `--save-checkpoint-freq 1`：DQN 两种方法每完成一次候选 episode 及相应训练处理后保存（含重复候选，不是每个 action）；A2C 每完成一个训练 batch 并更新网络后保存，因此一个 batch 含多个候选时不是每个候选保存一次。断点保存优化器、随机状态、已花预算和 DQN replay buffer，用于恢复最新同步状态。评分使用候选档案，不读取历史训练快照；冻结生成使用被选中任务的最终模型。该策略在 HPO 内部固定，不新增 YAML 参数；普通非 HPO 训练仍保留编号快照。

每个任务的最终 `qnet.pt` / `policy.pt` / `value_net.pt`、`std_scaler.bin`、最新完整断点以及候选档案、账本和结果仍保留。生成任务继续覆盖自己的 `generation_checkpoint.pt`；失败任务可能保留诊断用 `partial_checkpoint.pt`。单份断点避免同一任务内累积历史全量副本；整个 campaign 的磁盘占用仍随任务数量增加。原子写入期间需要同时容纳旧断点和新临时文件。

### DQN 性能更新与现有任务续作

DQN 的完整断点包含 replay buffer。新版按列打包并选择性无损压缩 replay，保持原始 dtype、形状、数值和字段；读取时自动还原原来的字典列表。候选档案 SHA256 改为追加时增量维护，恢复时仍校验已有完整档案；计时状态不再深拷贝随后会被替换的累计集合。bootstrap 的目标值使用有容量上限的缓存，target network 更新后立即清空，网络前向计算方式保持不变。MC 和 bootstrap 共用断点优化，A2C 也受益于档案和计时状态优化。

这些改动不改变学习率、更新次数、replay 容量、预算或断点频率。普通 DQN 与 HPO 使用同一训练实现，因此都受益；普通 CLI 的 `--save-checkpoint-freq` 默认是 0，没有开启保存时不承担周期性全量断点开销。HPO 则仍按上述频率保存最新完整状态。

每个训练任务新增 `checkpoint_timing.jsonl`，分别记录 `checkpoint_prepare_s`（整理状态）、`checkpoint_pack_s`（打包压缩）、`checkpoint_serialize_s`（序列化及写临时文件）、`checkpoint_replace_s`（原子替换）、`checkpoint_total_s` 和 `checkpoint_bytes`。没有外层准备计时的保存调用，其准备及总耗时为 `null`。写入耗时不等同于物理磁盘落盘耗时。此日志用于核对服务器上的实际瓶颈，不能把其他未归类耗时都当作磁盘 I/O。

升级步骤如下，保留原 YAML、模型资产、Python 环境和 `--out`：

1. 在旧程序中按一次 Ctrl+C（或向 HPO 控制器发送 SIGTERM），等所有 worker 保存完整断点并退出。
2. 安装本次性能版本；不要在旧 worker 仍运行时替换源码。
3. 运行只读检查，再正式恢复：

```bash
rl-matdesign hpo restart --out ./runs/dqn_mc_medium --check-only
rl-matdesign hpo restart --out ./runs/dqn_mc_medium
# bootstrap 使用自己的原目录
rl-matdesign hpo restart --out ./runs/dqn_bootstrap_medium --check-only
rl-matdesign hpo restart --out ./runs/dqn_bootstrap_medium
```

升级只接受已经审核的旧、新完整源码清单。服务器上额外保留的六个历史 predictor 文件按确切文件哈希识别；任意未知或修改的源码仍会拒绝。原 campaign manifest、已完成任务、预算账本和模型合同保持原样，正式恢复另存一份 `source_upgrades/dqn-checkpoint-storage-performance-v1.json` 审计记录。`--check-only` 不写该记录。中断任务继续原来的网络、优化器、replay 和 RNG 状态；已经完成或因效率门槛正常终止的任务复用结果。升级不会修复不完整的磁盘写入，也不会放宽模型、运行环境、种子、学习参数或评分设置的校验。

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
