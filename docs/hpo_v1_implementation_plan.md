# 通用 HPO 第一版代码计划与 GlassNet 验证

> 历史实现计划：下文保留最初 `hpo_v1` 的设计依据。当前精简 YAML 接口已改为从场景继承目标和生成方式、自动校准通用质量映射、只用总分容差，并以训练/生成两个重复次数控制种子。使用当前模板与命令时请以 [HPO 使用说明](hpo_v1.md) 为准；升级后的配置和评分契约需要新的输出目录。

本计划实现两个主要部分：先确定训练参数的候选范围，再在这些范围内联合选择训练参数。第二部分内部包含双向预算校准、候选组合的预算复核，以及冻结网络后的生成温度和生成数量选择。对已接入 rl-matdesign 的应用场景，切换应用只修改 YAML，不修改 HPO 或场景代码。核心代码读取场景配置，不内置玻璃性质名称或工业阈值。

本计划的第一版代码已在隔离分支 `codex/hpo-v1` 实现；实际命令、配置模板与恢复限制见 [使用说明](hpo_v1.md)。软件验证采用 dummy 场景和运行契约测试；真实 GlassNet HPO 由用户在现有 benchmark 资产主机运行，T3 阈值仍保留待定。所有 HPO 结果从本次新实验产生，不加载历史优胜参数或旧 RL 权重。本文的实验次数、权重、范围和容差为可配置的第一版工程初值。

**1 需求与边界**

- HPO YAML 指定目标性质、优化方向、合格阈值和最优子集比例。GlassNet 示例选择 T3 最小化，阈值可采用其训练数据 T3 标签中位数。整个 campaign 固定阈值、单位、数据版本及来源，不按每次 RL 结果重新计算中位数。
- CTE、Tg 等约束只在场景 config.yaml 中定义，属于材料设计要求。HPO 通过 base_config 引用场景，读取已有约束判定结果，不复制阈值，也不优化这些约束。更换已接入场景只需修改 HPO YAML 的场景引用、目标与搜索设置；科学条件变化时修改相应场景 YAML。
- 训练参数只按训练收集材料评分。生成参数只按冻结网络生成的材料评分，不把两阶段结果合并充当某一阶段结果。
- 生成只优化 gen_temperature。gen_epsilon、gen_top_frac 和 exploration_gen_eps 均为 0。用户口语中的 gen_temp 对应此规范字段，不新增另一个独立温度参数。
- dqn_grad_steps_per_ep 固定为应用确定的值，预算和采样效率上限不改变这个学习规则。
- 预算优先用不同合法候选的设计空间比例表示。训练和生成都从初始比例向更小、更大的方向测试；0.01% 是起点，不是最小值。计时仅用于预计剩余时间，不根据期望运行时长自动降低预算。
- 尝试效率上限在 HPO YAML 开放，默认 1.5，要求严格小于该值。共用默认值供训练、生成继承，并允许两阶段分别覆盖；最终生效值必须写入 manifest。
- 本轮范围诊断沿用“可重复失稳时内收、边界明显更好时外扩、证据不足时保留”的规则，不假定参数效果单峰，也不将正常低分区域自动删除。

**2 实施基线与现有代码差异**

制定计划时原 checkout 的 HEAD 为 bd4481e9，且已有其他未提交修改。实际实现使用包含预算与训练样本接口的 771d68d7 基线，新建隔离工作树 `/Users/liuyuxiang/.codex/worktrees/hpo-v1/rl-matdesign`，分支为 `codex/hpo-v1`；原 checkout 的其他工作保持不动。

实施时应在包含所需运行模块的隔离分支上工作，或先将这些依赖作为明确变更移植；不能假定当前 checkout 已有同名接口。保留现有未提交工作。启动实现前记录实际基线提交及工作树状态，本计划中的路径均相对于完成依赖整合后的仓库。

可复用：配置解析、环境规范候选身份、预算去重基础、训练样本记录、原始性质快照、预测器批量执行、Optuna 参数采样与 SQLite 存储，以及单组分数网格的精确组成计数。

必须改动的差异：

| 位置 | 现状 | 新流程需要 |
|---|---|---|
| utils/training_budget.py | attempted 仅在成功 rollout 的 record 中增加 | 新增所有实际尝试账本；保留旧成功 episode 计数语义 |
| utils/episode_batch.py | scalar 重试和 batch 预取有各自路径 | 所有真实尝试在进入 rollout 前计数，重试不可绕过额度 |
| training.py | 训练和生成各自计数；生成输出可能过滤，部分原始统计会再请求预测器 | 共用预算契约；使用一次评价的不可变结果；完整保存已评价候选 |
| hpo/metric.py | 主要读取 generated.csv 的 reward top-k | 分别读取训练和生成完整档案中的命名原始性质 |
| hpo/runner.py | TrialResult 主要返回 generated.csv；默认执行训练与生成 | 显式 train_only / generate_only 作业和结构化结果 |
| scripts/hpo.py | 旧阶段缩放，并把 epsilon 衰减尺度截断到在线预算 | 新协议入口不使用这套隐式缩放或截断 |
| utils/design_count.py | 现有便捷适配器只保证单个、无插件过滤的分数网格 | 通过场景计数能力接口返回精确值或明确“不支持精确计数” |

**3 统一预算账本与 1.5 上限**

3.1 三个数量分别记录

- target_unique：本次作业计划成功评价的不同候选数量。
- unique_evaluated：本阶段已成功评价、具有完整有效性质记录的不同候选数量。
- attempts_started：本阶段实际开始的候选采样尝试总数，包括重复、无效候选和失败重试。

一次尝试是从开始构造一个候选到得到成功或失败结果的 episode，不是一个 action、一个性质头、一次 GPU forward，也不是一次 CSV 导出。同一候选的多模型或多性质评价属于同一次候选尝试，其物理调用次数另记。单个候选内部允许的计算重试要另记 evaluator 调用及耗时；重新采样候选的重试必须增加 attempts_started。

成功评价但不满足科学性质约束的候选仍增加 unique_evaluated。性质字段缺失、非有限或评价协议失败不能冒充成功评价。未评价的结构非法候选不增加 unique_evaluated，但其采样尝试已计入。

DQN 有真实 reward 的 warmup 使用训练账本，并按成功评价身份增加 unique。A2C 当前 scaler-only warmup 同样实际采样候选，因此这些采样增加训练 attempts，但因未评价目标性质而不增加 unique；记录 preparation_only 子状态，不把它们伪装成失败或有标签材料。后续训练首次成功评价同一材料时仍可增加 unique。模型加载、scaler 数值拟合等不采样候选的初始化只记时间，不记候选尝试。其他场景也按实际是否采样、是否成功评价处理，不能实现时悄悄豁免准备样本。

预检需要检查固定准备采样是否已经使上限不可能满足。例如 A2C 固定 scaler warmup 为 200 次，目标 unique 为 100，即使后面每次都成功，也至少尝试 300 次，无法满足 1.5。此时提高所有对照使用的共同诊断预算到下一个合法档位并显示原因；若超过空间或预算上限，返回配置不相容，不能自动减少 warmup 或放宽比率。以严格 1.5 为例，200 次不产生 unique 的准备采样至少需要目标 unique 为 401，才可能最终达标。

训练与生成使用独立 seen 集合。同一材料在训练中出现过，仍可在生成阶段首次出现时增加生成 unique_evaluated。同一生成实验内部重复不增加。不同温度、预算或训练 seed 的作业分别记账；不得在整个 HPO 内全局去重后给某个 trial 免费材料。

3.2 严格小于的整数实现

默认 max_attempts_per_unique = 1.5，comparison = lt，要求完成作业时：

~~~text
attempts_started / target_unique < max_attempts_per_unique
~~~

据此计算：

~~~text
max_attempts = ceil(max_attempts_per_unique * target_unique) - 1
~~~

target_unique=1000 时，最多允许 1499 次尝试；1500/1000=1.5 不满足“低于 1.5”。配置校验要求上限有限且大于 1。用十进制或有理数算术计算整数边界，避免浮点误差改变最后一次允许的尝试。

3.3 不因早期比例高就立即终止

例如，目标 1000，目前尝试 20 次、得到 10 个不同材料，当前比例为 2。它仍有机会最终以 1010 次尝试完成，因此只显示当前比值，不按这个前缀直接淘汰。

但可以进行确定性的提前终止判断：

~~~text
最乐观的最终尝试次数
    = 当前尝试次数 + (目标不同材料数 - 当前不同材料数)

如果最乐观的最终尝试次数 > max_attempts：
    即使以后每次都得到新材料也无法达标，终止。
~~~

例：目标 1000、最多 1499 次；当前尝试 1100 次、只有 600 个不同材料。最乐观也要 1100+400=1500 次，因此立即终止，而不是再浪费 399 次尝试。

等价实现是限制“没有带来新有效候选的尝试”总数：本例最多 499 次。该判断只在批量 pending 结果全部结算后执行；未结算的尝试不能误当成永久损失。

3.4 终止后怎么处理

- 保存已发现候选、账本、失败原因和必要的检查点，不删除材料。
- 作业状态为 INFEASIBLE_EFFICIENCY，不是 COMPLETED，也不是程序崩溃。
- 保留未完成结果的诊断指标，但不把它与完整预算的评分混排。
- 不在同一个 trial 内偷偷提高 epsilon、温度、上限或减少目标预算。
- 参数范围诊断中，记录“完整配置在此预算下不合格”；不能仅凭此把一个参数值在所有组合中删除。
- 联合 HPO 中，任何一个必需 seed 因配置导致效率不合格，整个组合不进入优胜候选。TPE 可接收低于所有合法分数的有限惩罚值 -1，并保留显式可行性标签；最终排序必须再次过滤标签。
- 配置导致的数值失稳同样不能入选；模型资产缺失、程序异常等系统问题记录 SYSTEM_ERROR，允许一次明确的相同配置重试，不作为参数低质量证据。
- 若预算档位下没有合格组合，输出 NO_FEASIBLE_CONFIGURATION。可以按既定补点规则测试新的独立配置，达到试验上限仍无解就停止；不自动放宽 1.5。

3.5 账本接口

新增通用 CandidateBudgetLedger，避免修改旧成功 episode 计数后意外改变 DQN epsilon 或 target 更新时钟。

~~~python
ledger = CandidateBudgetLedger(target_unique, ratio_limit, comparison="lt")
ticket = ledger.begin_attempt()        # 在每次实际 rollout 前检查并扣除尝试额度
ledger.finish_attempt(ticket, outcome) # 成功、重复、无效或失败，幂等结算
ledger.check_completion_or_impossibility()
ledger.snapshot()
ledger.state_dict()
~~~

candidate outcome 携带规范身份、评价快照及状态。begin_attempt 不得在 batch 消费成功结果时才调用，否则会漏掉预取失败。批量调度先按剩余尝试额度和剩余 unique 额度限制波次；所有已启动 ticket 都需要结算，不能越过额度后丢弃多算的样本。

**4 场景无关的数据与配置契约**

核心模块只依赖以下信息：

| 契约 | 最低要求 |
|---|---|
| CandidateIdentity | 环境提供稳定的类型化身份，定义什么算同一个材料 |
| CandidateRecord | 作业、阶段、attempt ID、候选身份、原始命名性质、单位、约束结果及评价状态 |
| ObjectiveSpec | HPO YAML 给出的命名目标、minimize/maximize、合格阈值、固定归一化规则 |
| ConstraintSpec | 从 base_config 加载的已有科学约束及判定结果，HPO 不重复定义阈值 |
| DesignSpaceInfo | 精确不同候选总数，或 unknown/infinite 状态及原因；包含设计空间版本摘要 |
| AlgorithmProfile | 生效参数、初始范围、固定值、派生参数规则及候选合法性检查 |

组成环境使用规范组成身份；序列环境保留顺序；字段环境保留字段名称和值类型；结构设计必须保留场景定义中影响候选身份的结构自由度。不能用显示 formula 字符串作为所有场景的去重键。

YAML-only 复用作为验收要求：同一套 HPO 程序，换用另一份 HPO YAML 引用不同的已接入场景，即可完成全流程；核心中不得出现 if scenario == glassnet 等分支。首次实现负责把已有场景的候选身份、命名性质和约束结果接到统一契约，不能把这项工作推给每个 HPO 用户。全新的预测器或环境若当前工程尚不支持，仍需先接入 rl-matdesign；这属于新增场景能力，接入后 HPO 不再需要场景专用代码。

有限且可精确计数的空间支持 fraction_of_space。无法精确计数的空间必须显式提供计数适配器、离散化规则，或由用户选择 absolute_unique 模式。禁止把动作路径数、训练数据行数、近似计数或“预测达标子集大小”悄悄当成精确材料总数。通用 HPO 可支持这些场景，但不能承诺对无限空间计算精确百分比。

设计空间独立分组可在确实独立时组合计数；跨组约束、结构等价性和整数比例约分需要各自适配器或已验证枚举，不能无条件相乘。

完整性质档案应区分训练和生成。保留现有 generated.csv 作为交付视图，但 HPO 从完整 generation_samples 档案评分，包括已评价而未通过性质约束的候选。科学筛选不能改变评分分母。场景若使用生成后过滤，必须明确它是结构合法性检查还是评价后的科学约束。

**5 评分实现**

对完成目标预算的作业，按本阶段不同候选计算：

- qualified_fraction：同时通过科学约束和用户目标阈值的数量，除以全部 unique_evaluated。
- best_fraction_mean：通过科学约束的候选中，按优化方向最好的 ceil(metrics.best_fraction * target_unique) 个的目标平均值，best_fraction 默认 0.10，开放在 HPO YAML。
- best_value：通过科学约束的候选中的最优目标值。
- attempts_per_unique：attempts_started / unique_evaluated。

| 通用指标 | direction=minimize | direction=maximize |
|---|---|---|
| qualified_fraction | 目标值严格小于 threshold 且满足场景约束的比例 | 目标值严格大于 threshold 且满足场景约束的比例 |
| best_fraction_mean | 最低 10% 的均值，越低越好 | 最高 10% 的均值，越高越好 |
| best_value | 最小值，越低越好 | 最大值，越高越好 |
| attempts_per_unique | 越接近 1 越好，并受配置上限约束 | 相同 |

threshold_inclusive 默认 false；设为 true 时分别使用 <= 和 >=，比较方向由 direction 派生，避免阈值比较和排序方向相反。分母包含全部成功评价的不同候选，不仅是通过场景约束的候选。best_fraction 只控制评价指标，与生成采样 gen_top_frac 无关。

最优 10% 的目标槽位由完整预算决定。若只找到 k 个通过约束的候选，而目标槽位为 m，则此项归一化质量分乘 k/m；k=0 时该项与 best_value 质量分均为 0，原始均值和极值为 null，不能捏造数值。

内置 positive_threshold_ratio 转换适用于正阈值和正目标值：最小化使用 quality(value)=threshold/(threshold+value)，最大化使用 quality(value)=value/(threshold+value)。GlassNet YAML 选择前者，阈值与结果均为正的 Kelvin。转换只按方向和数值工作，不识别 T3 名称，也不是概率。

best_fraction_quality 明确定义为 quality(best_fraction_mean) 乘以 min(实际通过约束数量, 槽位数)/槽位数；best_value_quality 为 quality(best_value)。先计算原始均值再做转换，不能换成先转换各材料再求均值。最终报告同时展示原始物理指标和归一化分数。

另内置 sigmoid 转换，中心取固定 threshold、scale 为 YAML 指定的正数；最小化使用 sigmoid((threshold-value)/scale)，最大化使用 sigmoid((value-threshold)/scale)。这支持零和负目标值，无需为普通新场景写转换函数。所有中心、尺度和版本在运行前冻结；负值或零阈值不能误用正值比值公式。不在每个 trial 内重新归一化，也不因更新参数范围而改变评分尺度。若使用多个性质组成的标量目标，由现有场景输出该命名目标；本版不自动增加 Pareto 优化流程。

建议默认权重：

~~~text
训练分 = 0.45 * qualified_fraction
       + 0.30 * best_fraction_quality
       + 0.10 * best_value_quality
       + 0.15 * (unique_evaluated / attempts_started)

生成分 = 0.50 * qualified_fraction
       + 0.40 * best_fraction_quality
       + 0.10 * best_value_quality
~~~

1.5 是先于评分的硬门槛。高质量不能补偿效率超限。生成不额外添加效率权重，但同样受配置中的尝试上限约束。HPO 评分不自动替换 RL 学习所使用的 reward。

**6 三种算法的参数配置**

| 参数 | bootstrap DQN | MC DQN | A2C | 初始探测范围 |
|---|---|---|---|---|
| dqn_lr | 搜索 | 搜索 | 不适用 | 1e-5 至 1e-2，对数 |
| dqn_batch_size | 搜索 | 搜索 | 不适用 | 64、128、256、512 |
| warmup_budget_fraction | 搜索 | 搜索 | 不适用 | 0.02 至 0.20 |
| dqn_eps_min | 搜索 | 搜索 | 不适用 | 0.02 至 0.40 |
| epsilon_anneal_budget_ratio | 搜索 | 搜索 | 不适用 | 0.25 至 2，对数 |
| dqn_target_update_freq | 搜索 | 不生效 | 不适用 | 10、25、50、100、200 |
| pg_lr_actor | 不适用 | 不适用 | 搜索 | 1e-5 至 1e-2，对数 |
| pg_lr_critic | 不适用 | 不适用 | 搜索 | 1e-5 至 1e-2，对数 |
| pg_batch_eps | 不适用 | 不适用 | 搜索 | 8、16、32、64 |
| pg_entropy_coef | 不适用 | 不适用 | 搜索 | 1e-4 至 1e-1，对数 |
| pg_entropy_min | 不适用 | 不适用 | 搜索 | 0 至 0.8 |

初始参考配置自动从范围生成：实数取算术中点，对数参数取几何中点，枚举取排序后的中间档位，偶数个档位取较低的中间项。它只是一次对照的工作配置，不是用户需要另外提供的标准。一轮诊断中保持其他参数的工作值固定；参考失效时先寻找可运行参考，重做受改变影响的对照。

按预算解析 DQN 派生值：

~~~text
warmup_unique = max(1, round(warmup_budget_fraction * train_budget))
dqn_eps_anneal_eps = max(1, round(epsilon_anneal_budget_ratio
                                * (train_budget - warmup_unique)))
~~~

只优化比例，不能同时把比例和对应绝对 episode 数设为自由变量。不将衰减值截断为在线预算。保留实际代码的成功 online episode 衰减时钟，单列全部 attempts，避免误称“按 unique 进度衰减”。

A2C batch 改变时仍保证同一个 train_budget，不能用 batch * 固定 iters 造成不同评价成本。固定 pg_warmup_eps、网络结构、折扣、DQN loss 和 replay 容量；pg_repeat_penalty_coef 第一版固定为 0。固定配置需在 resolved manifest 中逐项列出。MC 不允许惰性的 target-update 参数进入搜索；search_space 中其他不适用字段前置报错。场景配置中为其他算法保留的已知非生效字段仍按现有 schema 保留，不要求用户为切换方法删除它们。

**7 两部分实验调度**

7.1 第一部分确定范围

1. 检查场景、预算、初始工作配置以及实际梯度更新条件。
2. 同一个诊断预算下，运行参考配置和每个参数的低端、高端，每个配置两个配对 seed。bootstrap 初始 13 组，MC 11 组，A2C 11 组。
3. 参考配置系统错误先修复；参考数值失稳则寻找可运行起点，不把所有边界一并收缩。
4. 可重复数值失稳：向最近测试正常点补测；对数参数取几何中点，连续参数取算术中点，枚举只测允许档位。
5. 两个配对 seed 均改善且平均训练分增加至少 0.02：向对应方向扩一档。学习率乘除 10，衰减比例和 warmup 比例乘除 2；有界概率按初始区间半宽移动并裁到合法范围；离散档位仅使用 profile 允许的扩展。
6. 正常低分或差异不明确时保留范围。一个 seed 异常或性能方向冲突时进行配对复核；不将波动假装成可靠边界。
7. 效率超限记录为完整组合不合格，不自动删除某个单参数值。遇到正常/失稳点交错，输出已测档位或明确的条件记录，不声称整段均已验证。
8. 每个参数每个方向最多补两个新取值。输出 ranges.json、range_decisions.jsonl 与证据表，未封闭方向标 unknown。

首批范围测试只评价训练结果，不运行生成。较大预算下如果边界结论失效，需生成有版本的新范围结果；不能悄悄修改已完成 study 的上下文。

边界更新按下面的具体规则执行。“下界”就是当前允许搜索的最小值，“上界”就是最大值；参考值是临时固定其他参数所用的值，不是第三条边界。

| 测试场景 | 下一次实际测试 | 如何修改范围 |
|---|---|---|
| 学习率范围 [1e-5, 1e-2]，参考值约 3.16e-4 正常，上界 1e-2 在两个 seed 中都数值失稳 | 测几何中点 sqrt(1e-2 * 3.16e-4)，约 1.78e-3 | 若正常，上界暂改为 1.78e-3；如仍失稳，再向参考值取几何中点，约 7.50e-4。补点用完后以上界方向已测正常点中最大的值作为本轮保守上界；下界不变。记录这只是当前配套参数下的诊断结论。 |
| 同一范围中 1e-5 正常但分数低，参考值和上界更好 | 不因低分追加向内二分 | 保留 1e-5 下界。低分不等于不能用；它与其他参数联合时仍可能有效。 |
| 1e-2 正常，两个 seed 都优于参考、平均分至少高 0.02 | 向外测 1e-1，其他参数不变 | 若新点也合格且继续改善，上界改为 1e-1，并可再测 1。若新点正常但更差，保留原上界；若新点失稳或效率超限，也保留原上界并记录失败原因。不能因边界好就未经测试把边界移出去。 |
| 下界明显最好，例如 1e-5 胜过参考和上界 | 向下测 1e-6，最多继续到 1e-7 | 仅当补测合格且继续改善才降低下界；上界不变。 |
| 两个边界都比参考好，或参考较好而两端都正常 | 两端满足外扩证据时分别补测；否则交给联合搜索 | 不假设单峰，不把两端一起向参考收缩。 |
| 同一点 seed 11 正常、seed 29 失稳，或两个 seed 的分数改进方向相反 | 对该点及参考增加配对 seed 47 | 若出现至少两次配置导致的数值失稳，按失稳规则处理；仍混合则标记不确定并保留原范围。单次随机异常不直接删除端点。性能外扩需所有已纳入的配对比较方向一致且平均改善达到 0.02；否则不外扩。 |
| dqn_eps_min=0.02 的配置超过效率上限 | 将完整配置和预算写入不合格记录 | 不直接把 eps_min 下界提高到某个值，因为失败还可能来自衰减、学习率或训练预算。联合搜索仍可在原范围中测试其他搭配。 |
| 中间点失稳而两端正常，或稳定性随值交错 | 在每方向最多两个补点的额度内复核相邻点 | 该参数暂按已测且可运行的离散值搜索，并保留原始区间及异常点作为证据；不把未经验证的整段标成稳定。若不同配套参数下结果冲突，仅记录条件证据，最终仍逐 trial 检查。 |
| 参考配置本身不合格 | 先用首批端点配置寻找合格工作配置；没有则在初始合法范围内最多追加 2d 个联合随机配置，每个用两个 seed | 找到后冻结新的参考并重做受影响对照；仍无合格配置则结束范围阶段并报告 NO_FEASIBLE_CONFIGURATION，不虚构 ranges.json 的可用范围。系统错误须先解决，不作为随机搜索理由。 |

离散扩展也要预先声明。第一版建议 dqn_batch_size 的允许扩展档位为 [16, 32, 64, 128, 256, 512, 1024, 2048]，dqn_target_update_freq 为 [1, 2, 5, 10, 25, 50, 100, 200, 500, 1000]，pg_batch_eps 为 [1, 2, 4, 8, 16, 32, 64, 128, 256]；每次只向相邻外侧档位补测，仍受两个新值的上限约束。所有新点必须满足数学合法性、内存和共同预算条件；无法运行的档位记作不可验证，不自动缩小该点的 batch。内收规则在下界方向对称执行。范围截断、外扩、保持和未决均写入 range_decisions.jsonl。

第一部分的直接产物仍是用户预期的“参数选择范围”。例如 dqn_lr 的输出可为 log-float [1e-5, 1.78e-3]、dqn_batch_size 为 categorical [64, 128, 256, 512]。附带证据包括测试预算、参考配置、seed、原边界、新边界和原因；合法性条件只是防止出现 warmup 占满训练预算、MC 配置惰性参数等无意义组合，不是额外的优化目标。范围被标为 proposed、diagnosed 或 unresolved，不能把有限点实验称为整段的数学证明。

7.2 第二部分预算和训练联合搜索

训练和生成使用同一个双向预算控制器，各自有独立配置、账本和结果。第一版采用有限几何档位完整校准：小型代表配置集合在预算上下限内测完，不在每个档位执行完整的 30 组 HPO，也不以相邻两档相近作为提前结束依据。

预算档位的产生规则：

1. HPO YAML 分别配置 initial_fraction、min_fraction、max_fraction、factor、min_unique。初值建议 0.0001，即 0.01%；最小比例示例 0.000001，即 0.0001%；最大比例示例 0.01，即 1%；factor 默认 2。这些都是可改的搜索边界，0.01% 不充当下界。
2. 从初值向下除以 factor，向上乘以 factor，直到两端边界；显式包含 min_fraction 和 max_fraction。每档取 ceil(fraction * exact_space_size)，受整个空间及最大比例对应的整数上限限制。合并相同整数预算并记录实际比例。
3. 下界还须满足 min_unique、固定准备采样的效率限制及共同的实际学习更新条件。初始档不合法时移动到最近合法档，显示原因。A2C 的 200 次无性质评价准备采样在严格 1.5 下要求至少 401 个 unique；这不是小预算质量差，不能通过豁免准备成本继续向下。
4. 超出空间、上下限矛盾或没有任何合法整数预算时，validate 返回配置错误。运行中效率不合格的某一档仅记为该配置的不可行结果，不据此推断其余所有较小或较大档都不可行。
5. 默认调度先测初始档，再交替测下方、上方最近未测档，直至有限列表完成；相同配置、预算、seed、软件和评价契约的同一 campaign 已完成结果可复用。训练的不同预算从头运行，因为 warmup 与衰减日程随预算派生；计时只更新 ETA。

例如空间足够大时，档位包含 0.0025%、0.005%、0.01%、0.02%、0.04%。假设这五档的达标率分别为 0.60、0.72、0.73、0.73、0.73，其他必要指标在 0.005% 起也都达到容差要求，那么可选 0.005%，不被 0.01% 初值限制。完整校准仍覆盖配置允许的更高档；若 0.16% 又显著改善，先前的短平台不能决定最终预算。

“最优预算”的可执行定义是：在本轮已测试的参数、seed 和预算范围内，选择质量与最佳有效结果足够接近的最小预算。比较方式固定如下：

- 所有候选组合的必需 seed 都完成预算并通过硬要求后，才有资格进入比较。每项原始指标及分数按相同 seed 集合取均值。
- 在已测的全部“参数组合、预算”配对中，按平均训练分选参照结果；分差在 score 容差内时优先较小预算，同预算再按较低 attempts_per_unique、较优 best_fraction_mean 和配置 ID 确定顺序。容差带以实际最高分为锚点，不能使用可能不传递的两两近似比较排序。
- 某个配对必须同时满足：训练分距实际最高分不超过 score 容差；qualified_fraction、best_fraction_mean、best_value 和 attempts_per_unique 相对参照结果的单向退步均不超过各自容差。比参照更好不受罚。方向统一处理：最小化的均值/极值越低越好，最大化则越高越好。
- 原始均值或极值为 null 时不能做数值相减：候选 null 而参照有限则不满足该质量条件；候选有限而参照 null 视为改善；两者均 null 时该项没有质量证据，仅允许保留候选作诊断，不能支持平台结论。全部结果都没有满足场景约束的材料时标 NO_FEASIBLE_MATERIAL_EVIDENCE，不导出声称科学质量已饱和的预算。尚未命中目标阈值但已有约束可行材料的情况另行显示 qualified_count=0，不与缺失性质混淆。
- 在合格配对中选预算最小者，再按训练分和上述固定规则选择配置。这样成本只在质量足够接近之后决定选择，不把小预算本身加成到质量分里。
- 建议容差：qualified_fraction 为 0.02、attempts_per_unique 为 0.10、score 为 0.01；best_fraction_mean 与 best_value 的容差由 HPO YAML 以目标物理单位指定。GlassNet 示例可用 0.005 * abs(threshold)，仅作为工程分辨率。零阈值、负值或其他单位也可以直接给正的 absolute 容差，不需要改代码。
- 记录两个独立结果：selected_budget 表示已测范围内的选择，convergence_status 表示证据是否足够。只有选定配对的配置在至少两个更大档均合格完成、各项变化均处于容差内，并且完整档位扫描没有发现更高的实质改善，才标 OBSERVED_PLATEAU。若更大档明显变差，则标 BUDGET_DEGRADATION；较大档因效率失败而缺失数据标 EVIDENCE_INCOMPLETE；到上界仍改善标 BUDGET_LIMIT_REACHED。它们均不能伪装成平台。
- 若最小配置边界就是选定档，额外标 lower_bound_reached，明确更小预算未测试。所有作业都有明确状态才算完成本轮校准；全部不可行时返回 NO_FEASIBLE_CONFIGURATION。选择结论限于已测范围，不称全局最优预算。

与联合训练参数搜索的衔接：

1. 使用参考配置、第一部分最好配置、较慢探索的可运行配置（去重，最多 3 个），以两个配对 seed 校准完整双向预算阶梯，得到暂定预算。
2. 在暂定预算运行 30 个组合：10 个随机起点、20 个 TPE 建议；每组两个固定训练 seed。第一版不按中途低质量剪枝，只处理数值、效率和系统错误。TPE 目标仍是训练分。
3. 将前 3 个组合加入原来的代表配置集合；该集合在全部预算档位上使用相同的 3 个新 seed 复核，再按上述规则联合选择预算和训练参数。原来在小预算表现好的代表配置仍保留，不能只把大预算赢家带过去失败一次就否定小预算。
4. 如果选择的预算改变，在新预算建立独立 study，追加 10 个组合（5 随机、5 TPE），把新入围组合加入复核集合并重新选择。每个新预算的 TPE 仅使用该预算下合法且契约一致的测量；不同预算的分数不混成同一保真度的样本。
5. 默认最多两轮预算改变后的补充搜索，轮数开放配置；达到上限仍发生实质预算或参数变化时标 BUDGET_UNRESOLVED，输出当前候选和证据，不声称已完成稳定选择。稳定时才生成训练阶段的最终选择和冻结模型清单。

第一版没有在每个预算穷举所有参数。它保证向下和向上都实际测量并复核入围配置，不能宣称被测较小预算无论怎样调参都不可能更好。

7.3 第二部分冻结网络后的生成选择

- 使用训练最终配置的 3 个验证网络，记录权重和 scaler 摘要；全部生成试验禁止更新权重。
- 唯一采样参数为 gen_temperature。温度 0 是显式贪心诊断点，正温度采用 softmax(score / temperature)；固定 gen_epsilon=0、gen_top_frac=0、exploration_gen_eps=0。
- 初始建议探测正温度 0.001、0.01、0.1、1、10、100、1000，并包含 0 作为诊断。DQN 用 Q 值、A2C 用策略 logits，因此相同温度不代表相同探索强度。各算法和训练配置分别校准，范围可由用户覆盖。
- 使用 budget.generation 单独构建向下、向上的有限数量阶梯。每个数量都比较相同温度集合和相同的 3 个冻结网络/生成 seed 配对；不能把初始数量上表现差的温度提前排除，再声称它在更小或更大数量上也差。
- 任何档位的最佳合格温度落在正温度边界时，可向外乘除 10 补测，每方向最多两个新值，去重后纳入全部预算档位。所有温度均因效率失败时，可按既定补点额度检查更高温度；每次都是新试验，不在原作业内改温度。动作熵和重复率仅辅助解释。
- 对全部有效“温度、生成数量”配对，使用与训练相同的最小充分预算选择器。比较生成分、qualified_fraction、best_fraction_mean 与 best_value；attempts_per_unique 在生成中作为硬门槛，不另设质量权重或平台改善要求。
- 输出生成数量与该数量对应的温度；初始生成比例不是下界，也不要求等于训练比例。数量按完整合法设计空间计算，不扣除训练已见材料。
- 第一版不同生成数量使用独立作业和账本。温度补点结束且全部已排定作业均有终态后，仅要求所选“温度、生成数量”配对的全部必需网络/seed 作业完成并通过；其他温度或诊断点失败不阻止有效赢家导出。生成评分不能回填训练评分。
- 生成使用与训练相同的 convergence_status 和上下边界标记。质量平台只说明比例、均值和极值改善有限；报告每档不同达标材料数量，不能将其解释为“已经找到足够数量”或“没有更好材料”。

**8 拟新增及修改的文件**

| 文件 | 主要职责 |
|---|---|
| src/rl_matdesign/utils/candidate_budget.py（新增） | 两阶段统一尝试 ticket、unique 集合、严格整数上限、不可挽回判断、恢复状态 |
| src/rl_matdesign/utils/episode_batch.py（修改） | scalar / retry / batch 都通过账本，控制预取额度并结算失败 |
| src/rl_matdesign/utils/training_budget.py（修改） | 与新账本对接，保持旧成功 episode 计数兼容 |
| src/rl_matdesign/utils/candidate_records.py（新增） | 训练/生成统一原始性质记录与版本化身份序列化 |
| src/rl_matdesign/utils/training_samples.py（复用及适配） | 保留既有训练 CSV 导出，接收共享记录契约 |
| src/rl_matdesign/training.py、experiment.py（修改） | 显式阶段运行；包括 scaler warmup 的实际采样记账；冻结生成；不重复请求性质；结构化终止结果 |
| src/rl_matdesign/hpo/spec.py（新增） | 场景、目标、约束、预算、参数 profile、阶段策略的配置校验 |
| src/rl_matdesign/hpo/scoring.py（新增） | 原始性质评分、方向、归一化、固定槽位、合格性检查 |
| src/rl_matdesign/hpo/algorithm_profiles.py（新增） | bootstrap / MC / A2C 生效参数和预算派生规则 |
| src/rl_matdesign/hpo/range_search.py（新增） | 单参数范围诊断和每次边界决策日志 |
| src/rl_matdesign/hpo/budget_search.py（新增） | 双向整数档位生成、共同预算校准、最小充分预算选择和独立收敛状态 |
| src/rl_matdesign/hpo/controller.py（新增） | 两部分状态机、训练 TPE 与冻结生成调度 |
| src/rl_matdesign/hpo/runner.py（修改） | train_only / generate_only 子进程，实时日志和结构化 TrialResult |
| src/rl_matdesign/hpo/search_space.py（扩展） | 类型化范围与已采样比例的显式解析；拒绝无效/重复自由变量 |
| src/rl_matdesign/hpo/reporting.py（新增） | 完整成本、结果表、阶段 ETA、机器可读状态 |
| src/rl_matdesign/utils/design_count.py（扩展） | DesignSpaceInfo 能力接口和已有精确计数适配器 |
| src/rl_matdesign/config_schema.py（修改） | 实际运行需要的预算上限字段和一致性校验 |
| src/rl_matdesign/hpo/cli.py、hpo/__main__.py（新增） | 可安装的 HPO 子命令、模块入口、分阶段运行和恢复；不依赖源码仓库 scripts 目录 |
| src/rl_matdesign/cli.py（修改） | 对 hpo 子命令分派到新入口，保留原实验命令行为；复用 pyproject.toml 已有 rl-matdesign 入口 |
| configs/hpo/profiles/*.yaml、configs/hpo/glassnet_v1.example.yaml（新增） | 算法默认范围与 GlassNet 场景配置示例 |
| tests/test_candidate_budget.py 等（新增及扩展） | 账本、评分、控制器和跨场景测试 |

目录拆分是代码职责划分，不是增加实验阶段。新入口通过配置显式选择协议版本，旧 top-k 生成评分驱动不默默变成新训练评分驱动。

**9 YAML 配置与 CLI**

9.1 两份配置各自负责什么

- 场景 config.yaml 定义环境、设计空间、预测模型、性质输出、材料约束和固定训练设置。GlassNet 的 CTE、Tg 条件只放在这里。
- HPO YAML 用 base_config 引用该场景，定义优化算法、评价目标及方向、阈值、评分权重、参数范围、预算搜索上下限、效率上限、随机种子和搜索次数。
- base_config 相对路径以 HPO YAML 所在目录解析；场景中的资产路径保留场景配置自身的解析基准。导出 trial 配置前解析为绝对路径，不能因输出目录变化失效。
- 所有公开 HPO 选项均有类型化 schema；未知字段、重复键、无效方向、错误单位和冲突的采样方法在运行前报错。默认值和覆盖来源写入 resolved manifest。

9.2 GlassNet A2C 配置示例

这是拟实现的 schema；threshold=null 是待填的用户输入，实际运行前必须给出数值。原有材料约束留在被引用的场景 YAML。

~~~yaml
protocol: hpo_v1
base_config: ../glassnet/config.yaml
method: a2c
parameter_profile: builtin:a2c

objective:
  field: T3
  direction: minimize
  threshold: null
  threshold_inclusive: false
  unit: K
  normalization:
    kind: positive_threshold_ratio

metrics:
  best_fraction: 0.10
  training_weights:
    qualified_fraction: 0.45
    best_fraction_quality: 0.30
    best_value_quality: 0.10
    sampling_efficiency: 0.15
  generation_weights:
    qualified_fraction: 0.50
    best_fraction_quality: 0.40
    best_value_quality: 0.10

efficiency:
  max_attempts_per_unique: 1.5
  comparison: lt
  stop_when_completion_is_impossible: true
  train: {}       # 未指定时继承上面的 1.5；可单独覆盖
  generation: {}  # 例如 {max_attempts_per_unique: 1.8}

budget:
  mode: fraction_of_space
  strategy: bidirectional_grid
  train:
    initial_fraction: 0.0001   # 0.01%，只是起点
    min_fraction: 0.000001    # 0.0001%
    max_fraction: 0.01        # 1%
    factor: 2
    min_unique: 100
  generation:
    initial_fraction: 0.0001
    min_fraction: 0.000001
    max_fraction: 0.01
    factor: 2
    min_unique: 100
  tolerances:
    qualified_fraction: 0.02
    best_fraction_mean: {kind: threshold_fraction, value: 0.005}
    best_value: {kind: threshold_fraction, value: 0.005}
    attempts_per_unique: 0.10
    score: 0.01
  upper_confirmation_levels: 2

range_search:
  reference: initial_midpoint
  seeds: [11, 29]
  confirmation_seed: 47
  max_extra_values_per_direction: 2
  score_improvement: 0.02

# 未覆盖参数沿用所选 profile；所有初始范围均可在本文件改动
search_space:
  pg_lr_actor: {type: float, low: 0.00001, high: 0.01, log: true}
  pg_lr_critic: {type: float, low: 0.00001, high: 0.01, log: true}
  pg_batch_eps: {type: categorical, choices: [8, 16, 32, 64]}
  pg_entropy_coef: {type: float, low: 0.0001, high: 0.1, log: true}
  pg_entropy_min: {type: float, low: 0.0, high: 0.8}

training_search:
  trials: 30
  startup_random_trials: 10
  seeds: [11, 29]
  finalists: 3
  validation_seeds: [101, 211, 307]
  max_budget_revisions: 2
  additional_trials_per_revision: 10
  additional_random_trials: 5

generation_search:
  parameter: gen_temperature
  initial_values: [0, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
  max_extra_values_per_direction: 2
  fixed: {gen_epsilon: 0, gen_top_frac: 0, exploration_gen_eps: 0}
  paired_seeds: [401, 503, 607]
~~~

切换到最大化场景时，在 HPO YAML 中更换 base_config、objective.field、direction、threshold、unit 和适合数值域的内置归一化即可。例如最大化允许负值的目标可选 normalization: {kind: sigmoid, scale: 10.0}；原始物理容差可写 {kind: absolute, value: 0.5}。不要求新增 Python 指标函数。预算模式为 absolute_unique 时，改用 initial_unique、min_unique、max_unique、factor，不允许同时填写比例字段。min_unique 在此模式也是明确的搜索下界，实际可运行下界仍接受预检。

每个方法单独建立 campaign。公开方法名为 a2c、dqn-bootstrap、dqn-mc；a2c 映射到场景运行配置 method=a2c，后两者分别映射到 method=dqn 与 dqn_target_mode=bootstrap/mc，不使用训练函数内部的 rl_method 参数名充当场景配置字段。MC 不暴露无效 target-update 参数。HPO 的 method 可来自 YAML 或 --method；两处同时填写必须一致，否则报错，不静默把 A2C 参数表拿去跑 DQN。CLI 不再提供另一套性质阈值和评分覆盖项，科学与 HPO 设置分别由对应 YAML 统一管理。

9.3 推荐命令与分阶段运行

项目当前已安装的入口是 rl-matdesign，本次增加 hpo 子命令。推荐完整运行命令：

~~~bash
rl-matdesign hpo run --method a2c --hpo-config hpo_a2c.yaml --out runs/glassnet_a2c
~~~

--stage 默认为 all；method 已在 YAML 中给定时可省略 --method。不注册容易与其他工具冲突的全局 hpo 命令。保留已有 rl-matdesign 实验命令、旧 scripts/hpo.py 及其旧两阶段语义，不能把新范围阶段与旧“小预算筛选”静默混用。

| --stage | 执行内容 | 前置条件 |
|---|---|---|
| all（默认） | 范围确定 → 双向训练预算 → 联合训练寻优及预算复核 → 冻结生成温度和数量寻优 | 一份合法配置，自动执行全部依赖 |
| ranges | 第一部分：确定训练参数范围，完成后退出 | 一份合法配置 |
| search | 第二部分：预算、联合训练参数及生成选择 | 同一 campaign 已有完整的 ranges 结果 |
| train-budget | 只做双向训练预算校准 | 已有 ranges 结果 |
| train-search | 训练联合寻优、验证、预算改变后的补充搜索，产出冻结模型清单 | 已有 ranges 和预算校准结果 |
| generation | 只优化冻结网络的温度与双向生成数量 | 已有完整训练选择及对应模型/scaler |

第一、第二部分分别执行：

~~~bash
rl-matdesign hpo run --hpo-config hpo_a2c.yaml --out runs/glassnet_a2c --stage ranges
rl-matdesign hpo run --hpo-config hpo_a2c.yaml --out runs/glassnet_a2c --stage search
~~~

需要更细的控制时：

~~~bash
rl-matdesign hpo run --hpo-config hpo_a2c.yaml --out runs/glassnet_a2c --stage train-budget
rl-matdesign hpo run --hpo-config hpo_a2c.yaml --out runs/glassnet_a2c --stage train-search
rl-matdesign hpo run --hpo-config hpo_a2c.yaml --out runs/glassnet_a2c --stage generation
~~~

检查、查看和恢复：

~~~bash
rl-matdesign hpo validate --hpo-config hpo_a2c.yaml
rl-matdesign hpo plan --hpo-config hpo_a2c.yaml
rl-matdesign hpo status --out runs/glassnet_a2c
rl-matdesign hpo resume --out runs/glassnet_a2c
~~~

validate 检查配置、可获得的模型元数据与运行契约，不采样候选；plan 额外展示生效参数、双向整数预算档位、依赖关系、已有计时下的 ETA 和待定实验数量，不启动 HPO。资产检查需要加载模型时单独显示。支持 python -m rl_matdesign.hpo 作为相同入口，确保非 editable 安装也能使用，不依赖仓库 scripts 目录。

阶段执行与恢复契约：

- all 自动执行依赖；指定单独阶段时若前置结果缺失，立即说明缺少哪个阶段及应执行的命令，不擅自运行用户没有选择的前置实验。
- 同一 out 内合法的后续阶段使用已有结果，不重新运行已完成阶段。空目录创建 campaign；已有其他内容或不匹配 campaign 的目录拒绝覆盖。
- resume 读取保存的 resolved manifest 和原执行范围，继续未完成的作业；不能清空 attempts/seen 或重新开始已消耗额度的 trial。已成功完成 ranges 的 campaign 不会因 resume 自动变成全流程；运行 --stage search 才会进入第二部分。
- 已完成阶段及模型都有摘要。修改目标、约束、参数范围、预算、效率上限或模型后，不能冒用原 campaign 状态；新建 out。软件或资产不匹配时明确拒绝恢复。新预算实验是新作业，不是给旧检查点换预算。
- 输出严格区分“请求的实验已经完成”和“科学上观察到了平台”；没有平台可以输出带状态的已测最佳，不得伪称收敛。BUDGET_UNRESOLVED 或没有可行配置时不自动进入依赖稳定训练结果的生成阶段。

**10 状态与结果文件**

每个作业输出 result.json，至少包含：

~~~text
status / reason_code / algorithm / phase / seeds
requested_unique / achieved_unique / attempts_started / max_attempts
duplicate_attempts / invalid_attempts / failed_attempts / pending_attempts
constraint_pass_count / qualified_count / raw_metrics / normalized_metrics
score / eligible_for_selection
model_digest / scenario_digest / scoring_digest / effective_parameters
checkpoint_paths / records_path / timing
~~~

状态至少区分 COMPLETED、INFEASIBLE_EFFICIENCY、INFEASIBLE_NUMERICAL、INVALID_CONFIG、SYSTEM_ERROR、INTERRUPTED、TIME_CENSORED。外部运行时间限制导致的中断不是效率失败；计时本身不触发自动预算削减。campaign 另记录 NO_FEASIBLE_CONFIGURATION、OBSERVED_PLATEAU、BUDGET_LIMIT_REACHED、BUDGET_DEGRADATION、EVIDENCE_INCOMPLETE 和 BUDGET_UNRESOLVED 等结果，不把“进程退出 0”当成完成证据。

campaign 目录保存 manifest.json、ranges.json、range_decisions.jsonl、budget_results.csv、trials.csv、generation_trials.csv、selection.json、完整候选档案及分阶段物理成本。budget_results 与 selection 包含原始比例、整数预算、实际比例、最小充分预算选择依据、convergence_status 和上下界是否触及；训练、生成分开记录。采用原子写入和版本化状态，避免中断留下半份最佳配置。

只有所有选择条件满足时才输出 best_training.yaml、best_generation.yaml 及它们适用的预算。没有合格生成设置时可保留训练结果，但不能发布一份声称全流程完成的最终配置。

ETA 使用当前算法、参数规模、缓存命中和实际波次的时间记录更新。显示当前作业、已排定任务以及再增加一个预算档位的预计开销；尚未决定的自适应任务显示范围或待定，不伪装成固定总时长。所有预热、失败、重跑、重复候选和系统初始化时间都进入成本报告。

**11 测试与首次 GlassNet 验证**

11.1 无模型依赖的单元测试

- strict ratio 边界：目标 1000，1499 次完成通过，1500 次不通过；目标 1、奇数预算、接近 1 的上限和非法 NaN/Inf/bool。
- 当前 20 次得到 10 个不同候选不能误停；当前 1100 次只得到 600 个则必须提前终止。
- scalar、失败重试、batch 预取、全部失败、cache hit 与 batch 内重复都准确计数，不突破额度。
- DQN 带 reward warmup 与 A2C scaler-only warmup 分别正确增加 attempts/unique；固定准备次数已经使比率不可能达标时预检拒绝，不能免除或虚增 unique。
- 训练出现过的候选在生成首次出现时计一次；生成内部重复不计 unique。
- 科学约束不通过仍计成功评价预算；缺失性质不是通过约束；重复身份与显示列变化无关。
- resume 恢复 seen、attempts、pending 和原配置，不能因重启获得新额度；跨预算作业不伪装成同一条训练曲线。
- 评分方向、零达标、最优 10% 槽位不足、负值目标、无有效约束样本与固定归一化。
- 只改 YAML 的 direction 即自动改变阈值比较、最优子集排序、极值、归一化方向和预算容差比较；threshold_inclusive 的等号边界一致。换属性名后不得走 GlassNet 特例。
- 双向比例档位、最小/最大边界、整数舍入去重、准备成本下界、无法计数的 absolute_unique 模式；0.01% 起点必须能选出 0.005% 或更低预算。
- 预设曲线覆盖中途短平台后改善、只在较小预算有优胜配置、预算上下限仍未封闭、较大预算效率失败、质量与成本权衡，不能把任何缺失数据当成平台。
- 原始指标的 null/有限值组合按明确规则比较；全部无约束可行材料不能输出平台；温度 0 等诊断点失败不阻止其他有效温度导出。
- 生成的温度排序随数量改变时，在相同温度集合和冻结网络配对上找到相应数量的最优温度；不得只保留初始数量的前几名。
- YAML 的共用 1.5 默认与阶段覆盖生效，改变上限不改变计数语义；CLI 与 YAML method 冲突报错。
- 全流程与分阶段执行得到同一任务图及相同结果；缺少前置阶段时明确失败；resume 不重复已完成任务、不扩大原执行范围、不清零账本。
- 一个必需 seed 不合格不能平均掉；系统异常与算法不可行不混淆；所有 trial 不可行时不输出假最佳。
- 每个范围更新分支有预设结果序列驱动的确定性测试，检查下一个测试值与最终边界。
- epsilon 比例解析不截断；MC target-update 参数被拒绝；A2C batch 改变不改变 unique 预算。
- 生成温度 0、极大温度、稳定 softmax、epsilon/top-frac 冲突配置和冻结权重摘要。

有限空间中即使目标不超过总数，接近全覆盖的随机采样也可能天然难以满足 1.5；确定性生成的实际支持集也可能远小于整个设计空间。测试应让这些情形诚实返回效率不可行，而不是靠提高尝试数隐藏问题。

11.2 跨场景契约验证

采用可枚举的小型确定性预测器，覆盖分数网格组成、整数约分组成、顺序敏感序列、命名字段及多组耦合约束。检查规范身份、精确计数或 unknown 返回、minimize/maximize、负值目标、约束过滤和相同预算。加入带随机噪声的评价器检查 eval_seed 与原始记录冻结，不将预测噪声误当成计数变化。

验收必须通过同一 CLI 分别加载多个场景 YAML，包括最小化正值性质、最大化性质和含负值性质；期间不能修改 HPO 源码。另做非 editable wheel 安装的入口验证，保证 rl-matdesign hpo 和 python -m rl_matdesign.hpo 均可工作，原实验入口不受影响。这些测试验证核心不依赖 T3/玻璃列名；不声称每个真实预测模型都已完成性能验证。

11.3 GlassNet 接入检查

- 使用已有 GlassNetPredictor 和同一个命名模型实例返回 T3、Tg、CTEbelowTg，三个性质不是三份材料预算。
- 验证 T3/Tg 为 K，CTEbelowTg 已由 log10(K^-1) 转成 K^-1，不能二次转换。
- 校验模型资产、源版本、scaler、设计域及输出模式摘要；不重新训练性质预测器。
- 用户若提供训练标签文件用于计算阈值，工具只从明确的训练子集和目标列提取有限有效标签，输出中位数、样本数、文件摘要及单位。注意原 SciGlass T4 与模型输出 T3 的映射；未经确认不能用全数据集、随机 RL 样本或预测值代替训练标签。
- 小规模真实运行分别覆盖 bootstrap、MC、A2C 的训练，以及每种模型的冻结温度生成。使用实际设计空间比例和统计下限；软件冒烟验证只检查记录、状态、预算、评分和冻结，不因随机性能未达目标就判软件实现失败。
- 设计一个可控制的重复场景，验证 1.5 终止原因、部分材料保存及下一 trial 正常启动；对实际模型额外重算档案指标并与在线结果逐项核对。

11.4 首次完整验证

在用户给定的 GlassNet 设计空间、CTE/Tg 约束与固定 T3 阈值上，从空 campaign 执行两部分流程。记录范围更新、预算曲线、全部合格/不合格组合、3 个独立训练验证 seed 和温度结果。真实实验可得出“在已测试配置和预算下最好”的结论；若全部被 1.5 排除或到上限仍改善，应报告这些实际结果。

不要求一次随机实验必然找到优于既有材料的候选作为软件验收条件，也不使用历史 HPO 结果设置更容易通过的范围。

**12 开发顺序与交付条件**

| 开发任务 | 内容 | 完成条件 |
|---|---|---|
| 1 运行依赖与账本 | 整合已核对预算/样本模块，新增真实尝试与严格上限 | scalar/retry/batch/resume 账本测试通过；原学习时钟不变 |
| 2 通用契约与评分 | identity、原始记录、约束、目标、计数能力与评分器 | 多类合成场景得到可复算的相同口径结果 |
| 3 阶段执行 | train_only / generate_only、冻结模型、完整生成档案 | 三种算法均能独立运行和准确终止 |
| 4 实验控制器 | 范围诊断、双向预算搜索、TPE、候选复核和温度搜索 | 预设结果驱动的端到端状态机测试通过 |
| 5 配置与报告 | CLI、manifest、恢复、日志、ETA、选择导出 | 未完成或不合格作业不能导出假最佳；配置可追溯 |
| 6 GlassNet 验证 | 真实预测器接入、预算验收、首次完整实验 | 提交实际实验记录与局限，不用预设成功替代结果 |

先完成任务 1 至 5 的软件验证，再启动真实 GlassNet HPO。若在现有运行基线缺少必要接口，任务 1 明确包含移植，不跳过依赖。规划阶段不修改训练算法、不运行实际 HPO，也不合并任何分支。

源码核对入口：

- [运行版本的预算账本](/Users/liuyuxiang/.codex/worktrees/budget-training-samples/rl-matdesign/src/rl_matdesign/utils/training_budget.py)
- [运行版本的批量与重试执行](/Users/liuyuxiang/.codex/worktrees/budget-training-samples/rl-matdesign/src/rl_matdesign/utils/episode_batch.py)
- [训练与生成实现](/Users/liuyuxiang/.codex/worktrees/budget-training-samples/rl-matdesign/src/rl_matdesign/training.py)
- [GlassNet 输出契约](/Users/liuyuxiang/.codex/worktrees/budget-training-samples/rl-matdesign/docs/glassnet_oracle.md)
- [已有精确组成计数](/Users/liuyuxiang/.codex/worktrees/budget-training-samples/rl-matdesign/src/rl_matdesign/utils/design_count.py)
