# 直接调用 gp_physchem

RL 与 GP 使用同一个 Python 环境。RL 通过现有的 FQN 插件接口导入
`gp_physchem.api:PeptidePredictor`。RL 使用名称列表定义允许替换的单体，GP 负责把名称
解析为结构、手性和描述符。GP 默认读取自己目录中的 `examples/new_monomers.csv`，
RL 无需填写该文件的路径。
不需要运行准备命令，也不生成或读取 `catalog.csv`、manifest、GP worker 或描述符缓存文件。

## 安装与运行

在准备运行 RL 的 Python 环境中，安装具有 Python API 的 GP 项目：

```bash
python -m pip install -e /path/to/rl-matdesign
python -m pip install -e /path/to/gp_physchem
```

需要同步本次 GP API 改动，包括包内新单体 CSV 的默认加载、已知单体复用和
`sequence_constraints` 接口。只有旧版 `PeptidePredictor` 还不够：它会把 CSV 名称
范围当作全部允许输入，并且不会自动提供巯基约束。旧版仅有脚本的 GP 还需要安装配置。
RL 的 Git 仓库不包含外部 GP 项目，提交或更新 RL 不会自动同步 GP 源码。
GP 包声明自己的依赖版本，RL 所需的 PyTorch 等依赖也应安装在同一个环境中。
建议为这套组合单独建立环境，避免改变其他材料模型使用的依赖。

本地配置保存在 `configs/peptide_gp.yaml`，由 `.gitignore` 忽略。
其中与 GP 有关的配置形状如下。单体选择只需填写候选名称、母序列和替换次数，
化学定义文件由 GP 自行定位：

```yaml
groups:
  - name: n_terminal
    kind: categorical
    choices:
      - name: treatment
        element: treatment
        values: [H, Ac]
  - name: peptide
    kind: sequence
    length: 11
    tokens: [A, V, F, K, W, HCys, D-HCys]
    parent: [A, HCys, F, K, V, W, W, A, P, R, b-HCys]
    num_mutations: 5
    allow_repeats: true

properties:
  - name: pIC50
    predictor: gp_physchem.api:PeptidePredictor
    direction: max
    objective: mean_minus_kstd
k: 1.0
```

每个 episode 从同一条母肽开始，执行 1～5 次替换。`num_mutations` 表示上限。
每次先选一个未修改的位置，再从 `tokens` 中选择一个与原残基不同的名称。
至少完成一次替换、且当前序列满足最终约束后，RL 可以选择“结束替换”；达到 5 次
则自动结束。每个位置最多改一次，替换名称可以不同，也可以在不同位置重复。
位置、替换残基和是否结束都由现有 RL 策略及探索设置决定；不保证 1～5 次等频出现，
也不保证一批结果包含每一种替换次数。约束不允许的次数不会生成。

旧实现中的 `num_mutations` 表示恰好替换次数。本次增加了结束动作并更新 checkpoint
语义校验，需要在新输出目录重新训练 RL；GP 模型及其接口无需为这项改动更新。

母肽中不在 `tokens` 内的名称可以保留，但不会成为替换动作。例如这里的 `b-HCys`
由 GP 内置字典解析，仍能留在母肽中。去掉 `parent` 和 `num_mutations` 两项就恢复
从头选择 11 个残基；其他配置保持同一种写法。

如果候选列表含 `BB103` 等新单体，只需在 `tokens` 中显式列出希望选择的名称，
并确保 GP 自己的 `examples/new_monomers.csv` 中已有对应定义。GP 读取该文件的
`monomer`、结构 SMILES 和手性等化学定义；文件包含的其他
单体不会自动加入 RL 动作范围。GP 能识别的全部单体才是可预测范围。与内置名称重复
的定义必须保持化学身份一致，否则启动时报错，不能覆盖已有训练描述符。

GP 从候选与母肽名称对应的化学结构自动提供“最终恰好两个游离巯基残基”的数量约束，
无需在 RL 配置里再维护 `thiol_tokens`。不要从 ID 或 `type2` 猜测巯基归属。环境允许
可被剩余替换修复的中间计数，例如暂时有一个或三个巯基，只有最终合法序列才进入 GP。

旧 `tokens: {csv: ..., id_column: ...}` 和 CSV 列 `where` 约束已经移除。将候选改为
名称列表，删除重复维护的巯基列表，保留 GP 所需的新单体化学定义文件即可。
HPO 原样传递名称列表、母肽和替换次数。GP 按自身安装位置定位 CSV，
不依赖启动目录或 trial 目录；RL 配置不需要 `new_monomers` 字段。

在同一 YAML 中保留所需训练和生成预算，然后在 RL 仓库根目录运行：

```bash
rl-matdesign \
  --config configs/peptide_gp.yaml \
  --method dqn --dqn-target-mode mc --out runs/peptide_gp_mc \
  --train-seed 0 --gen-seed 0
```

A2C 和 REINFORCE 只需改变 `--method`。GP 的训练表、词典和新单体 CSV 默认都来自
安装的 GP 包。CSV 扩展 GP 能理解的化学定义；RL 的动作范围始终由 `tokens` 决定。
直接调用 GP API 的其他应用仍可显式覆盖输入文件，但这套 RL 配置无需设置这些路径。

## MC 与 HPO 起步流程

MC 是 DQN 的回报目标选项，使用 `--method dqn --dqn-target-mode mc`，
或设置 `method: dqn`、`dqn_target_mode: mc`。本地 `configs/peptide_gp.yaml`
目前已按用户选择切换到 MC。终点奖励任务使用 `dqn_gamma: 1.0`，
使每个动作的 MC 目标都等于该条完整多肽的终点奖励；`k: 1.0` 仍保持奖励为 mean − std。
这样不会因折扣而额外偏好更早结束的同分候选。
序列排列会改变分子，因此 `dqn_augment_permutations` 必须为 0。

自由构建模式中，一个 episode 包含一次 N 端选择和 11 次残基选择，共 12 个
transition。母肽修改模式中，每次替换有“位置、残基”两次决策，上限为 n 时，
加一次 N 端选择最多 `2n + 1` 个 transition。若在 k < n 次替换后主动结束，
还包含一次结束动作，总共 `2k + 2` 步；完成 n 次则自动结束，共 `2n + 1` 步。
例如 `num_mutations: 5` 的 1～5 次替换分别需要 4、6、8、10、11 步。
两种模式都只在终点得到一次 GP 分数。`dqn_warmup_eps` 是额外的随机轨迹，不能包含在
`dqn_num_train_eps` 中重复计数。replay buffer 的容量单位是 transition，
同样的容量能容纳多少轨迹取决于实际替换次数，不能再按固定步数换算。

本地还准备了两份 Git 忽略的配置：
`configs/hpo/peptide_gp_mc_pilot.yaml` 用于计时，
`configs/hpo/peptide_gp_mc.yaml` 用于两阶段搜索。

HPO 为每个 trial/seed 调用当前环境 PATH 中的 `rl-matdesign` CLI，
不依赖 `scripts/run_experiment.py` 的源码路径。先激活同时安装 RL、GP 和 Optuna 的环境；
`rl-matdesign --help` 应能正常运行。先在实际远端机器运行计时配置：

```bash
python scripts/hpo.py \
  --hpo-config configs/hpo/peptide_gp_mc_pilot.yaml \
  --out runs/hpo/peptide_gp_mc_pilot_v1 \
  --skip-stage2 --no-plots
```

这只跑一个短 trial：100 个 warmup、300 个训练 episode，最终生成 100 个候选。
用于确认安装、MC 分支和耗时，不用于选择最佳参数。GP 初始化会在每个 trial/seed
的进程启动时重复一次；因此不能只用 GP 单次推理速度估算整轮 HPO 时间。
HPO 默认自动选择 CUDA（可用时）或 CPU；先保持 `n_parallel_trials: 1`。

2026-10-16 已在本地 CPU 验证旧自由构建模式的短流程：100 warmup + 300 MC 训练 + 100 个
独特候选，运行器耗时 61.9 秒（初始化 23.3 秒，训练 22.6 秒）；没有启动正式搜索。
这些时间仅对应本机和该 trial，不能直接当作远端机器或更大 batch/更新次数的耗时。
证据位于 `runs/hpo/peptide_gp_mc_pilot_local_20261016/validation.json`。

建议的第一轮预算是以下起点，并非已经证明能够收敛的预算：

| 阶段 | 参数组数 | 每组种子数 | 每种子 warmup | 每种子训练 episode | 最终候选数 |
|---|---:|---:|---:|---:|---:|
| 初筛 | 20 | 1 | 500 | 2,000 | 200 |
| 复核 | 前 3 组 | 3 | 1,000 | 8,000 | 200 |

两个阶段合计名义上生成并评分 136,800 条轨迹（含 warmup 和最终生成）；
缓存命中、重复候选和生成重试会使实际唯一 GP 计算量不同。
搜索学习率、batch size、每 episode 梯度更新次数、探索衰减时长和生成温度这 5 个参数。
固定网络大小、gamma、奖励中的 k 及每阶段采样预算；MC 不需要搜索 target-network 更新频率。

```bash
python scripts/hpo.py \
  --hpo-config configs/hpo/peptide_gp_mc.yaml \
  --out runs/hpo/peptide_gp_mc_v1 \
  --skip-stage2
```

查看 `stage1_summary.csv` 后，使用相同命令并移除 `--skip-stage2` 开始复核。
复核会从头训练，最终配置路径在 `final_report.md`。更改预算、搜索空间或原始模型输入时，
应使用新的输出目录；已经完成的 stage1 会被状态文件跳过，单改 `n_trials` 不会继续搜索。

HPO 当前比较每个 run 的 `generated.csv` 中 reward 最高的 10 个候选的均值。
要同时核对 `n_generated_used` 为 10、生成数量及去重情况；少于 10 行时指标会退化为
现有行的均值。该指标衡量最终生成策略，不包含训练过程中发现的所有候选。
跨 seed 的分数标准差反映训练随机性，与每条候选的 GP `obj_pIC50_std` 是不同量。

复核使用 seed 0、1、2，其中 seed 0 与初筛重合。选定参数后，用新 seed 3、4、5
检查稳定性，并在相同评价规则下比较 2,000、4,000、8,000 个训练 episode 的收益。
只有收益仍持续改善时再扩到 16,000；判断时看 top-10 reward、候选多样性和随机采样基线，
不要只根据训练 loss 决定采样量。

## 运行时发生什么

1. GP 接口读取原始词典和自己目录中的 `examples/new_monomers.csv`；调用原有化学函数，在内存中计算描述符。
2. 用原始训练表调用 `train_matrix` 和 `fit_gp`，每个 predictor 实例只拟合一次。
3. 环境初始化时，GP 的 `sequence_constraints` 根据候选与母肽名称提供巯基数量约束。
4. RL 选择 N 端，并自由构建序列或对母序列执行 1～n 次“选位置、选残基”；过程不调用 GP 评分。
5. 完成后，GP 接口从完整序列推导两个巯基位置，调用原始 `validate_candidates → featurise → predict_gp`。
6. 返回原始预测均值和 GP 不确定性 `std`；RL 计算奖励并写入正常运行结果。

描述符、GP 和 scaler 在本次进程内复用。每次启动新进程都会重新计算描述符，
包含 3D 构象计算，因此启动阶段可能需要数分钟；之后的候选预测无需重复初始化。
GP 接口自身不写文件，正常的 RL 输出（`generated.csv`、模型权重、训练日志等）保持不变。

`sequence_schema.json` 保存本次 RL 使用的母序列及替换次数上限。实际替换次数是最终
序列与配置母序列逐位置比较后的差异数。GP 输出中的 `mutation_count_vs_parent`
仍相对于 GP 固定训练母肽；只有两条母序列相同时，它才等于 RL 实际替换次数。
`num_mutations` 是上限，不是每条结果的实际次数。单体应使用统一名称，避免同一结构
的两个别名被计为一次替换。

## 已确认的模型约定

| 项目 | 约定 |
|---|---|
| 动作单体库 | RL `tokens` 中显式列出的名称；母肽中的其他名称只能保留 |
| 序列 | 11 个有序位置，允许重复 |
| N 端动作 / C 端 | H 或 Ac / 固定 NH2 |
| 游离巯基单体 | 由 GP 化学定义识别，自动转成 RL 的最终数量约束 |
| 二硫键 | 恰好两个巯基残基；位置从 1 开始，i < j |
| 目标 | 原 GP 的伪 pIC50，越大越好 |
| std | 原 GP 直接返回的预测不确定性 |

在原始 138 行 CSV 中，游离巯基名称为 BB1、BB6、BB55、BB56、BB103、BB137。
更换 CSV、名称或候选子集后应以其实际结构为准。例如原 BB110 的 `type2` 写着 Cys，
但结构为硫醚，不属于游离巯基。
原预测器没有最小环大小为 4 的限制；该限制仅属于原项目的候选枚举脚本。

默认奖励为 `mean - k * std`。`objective: mean` 只优化均值，
`mean_plus_kstd` 鼓励探索不确定区域；GP 接口始终返回未经奖励变换的原始 mean/std。

## 输出

| generated.csv 字段 | 含义 |
|---|---|
| `candidate_json` | 完整有序序列和 N 端动作 |
| `obj_pIC50_mean`, `obj_pIC50_std` | 原 GP 均值、不确定性 |
| `reward` | 配置的优化奖励 |
| `meta_monomers` | 原 GP 的单体串，以竖线分隔 |
| `meta_n_term`, `meta_c_term` | 端基处理 |
| `meta_disulfide_pos_i`, `meta_disulfide_pos_j` | 传入 GP 的两个位置 |
| `meta_smiles` | 原化学代码组装的完整分子 |
| `meta_nearest_training_distance_scaled` | 到训练集的标准化特征距离 |
| `meta_unseen_position_monomer` | 未见过的位置与单体名称组合 |
| `metadata_json` | 完整诊断信息 |

`meta_pred_pIC50` 和 `meta_posterior_sd` 保留 GP 的原始字段名，与 `obj_pIC50_*` 对应。
`dp_mean` 是已有奖励引擎的聚合字段；本单目标配置中等于原始均值。
多目标情况下，metadata 按 property 名和模型索引嵌套，避免同名字段覆盖。

## 模型解释边界

GP API 增加名称解析、可选新单体定义和约束提供接口，保留原核函数、标签、特征
和预测公式。已知单体使用字典内原描述符，不因重复导入定义而重新覆盖。以下是原模型的现有性质：

- 标签是单点抑制率换算的伪 pIC50，并非实测 IC50。
- 96 条训练记录只有 2 条 Ac；删失记录的数值标签也参与普通 GP 拟合。
- GP 返回的 std 包含 WhiteKernel 的噪声贡献。
- 二硫键位置先校验；显式全局特征包含环大小和环外残基数，位置特征包含单体及巯基标记。
- BB 名称与训练词典不同；即便化学结构相同，原 GP 仍按名称编码。别名统一应作为独立模型修订。
- 训练表有 92 条 11 肽、3 条 10 肽、1 条 12 肽；原特征函数只读取前 11 个位置。

## 验证方式

轻量 RL 回归不要求安装外部 GP。安装 GP 后，在同一个 Python 环境中明确启用真实检查：

```bash
PYTHONDONTWRITEBYTECODE=1 \
GP_PHYSCHEM_TEST_ROOT=/path/to/gp_physchem \
python -m pytest tests/test_gp_physchem_adapter.py -o addopts='' -q
```

检查直接 API 对原始 GP 函数的均值/std、序列顺序、端基、二硫键位置和分子结构一致性。
接入验证与小规模训练用于确认软件行为，不代表活性或优化收敛验证。

2026-10-15 的直接调用版验证结果：通用回归 458 项通过、2 项缺少已有外部夹具而跳过；
共用 GP/PyTorch 环境中的序列及元数据检查 22 项通过；GP 接口检查 27 项通过。
已安装 wheel 对 280 个候选覆盖全部 138 个单体、H/Ac 和 7 组二硫键位置，
与原始 GP 的均值最大绝对差为 7.99e-15，std 为 2.22e-16；280 个组装结构一致。
DQN、REINFORCE、A2C 各导出 4 个有效候选，原始 mean/std、诊断字段及奖励一致。
检查确认 GP 不生成中间文件；源码写回后也验证了原有科学函数未改变、旧预测脚本仍可启动。
本地验证记录位于 `runs/gp_physchem_direct_validation_20261015/validation.json`。
