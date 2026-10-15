# 真实 gp_physchem 模型接入

本适配使用用户提供的 `gp_physchem` 源码与数据，调用原有的
`load_library → train_matrix → fit_gp → validate_candidates → featurise → predict_gp`。
不修改原 GP 的核函数、训练标签、预处理、单体名称或位置编码。

## 已确认的模型与设计空间

| 项目 | 实际约定 |
|---|---|
| 动作单体库 | `examples/new_monomers.csv` 中 138 个 ID：BB1–BB138 |
| 序列 | 11 个有序位置，允许重复选择 |
| N 端动作 | `H` 或 `Ac` |
| C 端 | 固定 `NH2` |
| 巯基单体 | BB1、BB6、BB55、BB56、BB103、BB137 |
| 数量约束 | 恰好两个带游离巯基的残基，位置可变 |
| 二硫键位置 | 从 1 开始，`i < j`；由两个残基的位置推导 |
| 目标 | 原始 `pred_pIC50` 越大越好 |
| 不确定性 | 原始 `posterior_sd`，包含拟合的 WhiteKernel 噪声 |

训练词典中的 33 个单体仅供原模型构造训练和参考特征，**不会额外加入动作空间**。
单体巯基标记来自 GP 原有的结构解析和原子映射，不能按含硫元素或 `type2` 判断。
例如 BB110 的 `type2` 写着 Cys，但其结构为硫醚，不属于游离巯基。

原预测器不要求固定 2–11 桥，也未设置最小环大小为 4；后者仅出现在原项目的一个
候选枚举脚本里。此适配只执行原预测器的有效性约束，不加入该枚举限制。

## 分工

- `src/rl_matdesign/sequence.py`：通用序列、顺序、重复及精确数量约束。
- `src/rl_matdesign/predictors/json_process.py`：通用持久子进程协议，不包含多肽逻辑。
- `integrations/gp_physchem/worker.py`：GP 专用边界，使用原项目处理描述符、端基、
  二硫键位置、组装 SMILES 和预测。可以独立放到 GP 一侧部署。
- `scripts/prepare_gp_physchem.py`：生成经核对的单体表和可直接运行的 RL YAML。

GP 模块使用 `config`、`chemistry` 等顶层导入，同时依赖独立的 RDKit/sklearn 版本。
子进程隔离这些模块及依赖；GP 环境无需安装 PyTorch，RL 环境无需安装 RDKit。
原 GP 目录只读，现有数据和结果文件不被覆盖。

## 准备与运行

先准备与 GP `requirements.txt` 相符的 Python 环境。例如：

```bash
python -m venv /path/to/gp-venv
/path/to/gp-venv/bin/python -m pip install -r /path/to/gp_physchem/requirements.txt
```

在 RL 仓库根目录，用 RL 的 Python 生成配置：

```bash
python scripts/prepare_gp_physchem.py \
  --gp-root /path/to/gp_physchem \
  --gp-python /path/to/gp-venv/bin/python \
  --out runs/gp_physchem/prepared \
  --config configs/peptide_gp.yaml

python scripts/run_experiment.py \
  --config configs/peptide_gp.yaml \
  --method a2c --out runs/peptide_gp_a2c \
  --train-seed 0 --gen-seed 0
```

准备脚本默认读取 `GP_ROOT/examples/new_monomers.csv`；可用 `--new-monomers` 覆盖。
`--smoke` 会生成微型预算的配置，用于验证接入，不能用于评价优化效果。
三种算法使用同一配置；可将 `--method` 改为 `dqn` 或 `reinforce`。

当前工作区已经生成 `configs/peptide_gp.yaml` 和 `runs/gp_physchem/prepared/`。
验证使用的 GP Python 为 `/private/tmp/rl-gp-physchem-20261015/bin/python`；这是临时环境，
长期运行前请按上述步骤创建固定环境，并重新生成配置。Python 版本和依赖版本不变时，
可以复用准备目录；版本变化时使用新的 `--out` 目录。

默认奖励为 `mean - 1.0 * std`，目标方向为 `max`。
这是明确可编辑的起始设置，并未调参：`objective: mean` 只优化均值；
`mean_plus_kstd` 鼓励不确定区域的探索；`k` 控制权重。

准备阶段逐行调用 GP 化学接口，保留全部原始 ID 和顺序。描述符写入
`catalog.csv`，原行内容保存在 `source_*` 列。任何单体失败都会报错，绝不静默删行。
缓存记录原始源码、训练数据、词典、新单体表、适配代码和导出文件的 SHA-256，
以及 Python、RDKit、NumPy、pandas、SciPy、sklearn 版本。完全一致时复用缓存；
改变后需要新建准备目录，防止混用不同预处理结果。

每次启动一个 worker 时拟合一次原 GP，随后在内存中复用 GP、scaler 和单体特征。
一次预测后的均值、标准差和诊断信息也会缓存；记录 CSV 不会再次拟合模型。
通用奖励层同步缓存统计量和诊断信息，避免底层缓存淘汰后两者来自不同预测。

## 输出与检查

`generated.csv` 保留现有 RL 字段，并增加：

| 字段 | 含义 |
|---|---|
| `candidate_json` | 完整有序序列和 N 端动作 |
| `obj_pIC50_mean`, `obj_pIC50_std` | 原模型均值和标准差 |
| `reward` | 配置的优化奖励 |
| `meta_monomers` | 原 GP 接受的 `\|` 分隔单体串 |
| `meta_n_term`, `meta_c_term` | H/Ac 与 NH2 |
| `meta_disulfide_pos_i`, `meta_disulfide_pos_j` | 真实传入原验证函数的位置 |
| `meta_smiles` | 原 GP 化学代码组装的完整多肽 |
| `meta_nearest_training_distance_scaled` | 到训练集的标准化特征距离 |
| `meta_unseen_position_monomer` | 未见过的位置与单体名称组合 |
| `metadata_json` | 完整诊断信息 |

`dp_mean` 属于现有奖励引擎的聚合输出；请使用 `obj_pIC50_*` 查看原始 GP 统计量。
若要交给原 `scripts/predict.py` 再预测，将上表对应 `meta_*` 字段去掉前缀，
再增加唯一的 `compound_id`；调用时必须传入 `--new-monomers`，原词典才会加载 BB 单体：

```bash
/path/to/gp-venv/bin/python /path/to/gp_physchem/scripts/predict.py \
  --input candidates.csv --output predictions.csv \
  --new-monomers /path/to/gp_physchem/examples/new_monomers.csv
```

## 本模型自身的解释边界

这些是直接检查源码和训练表得到的事实，不是新增的 RL 限制：

- 标签是单点抑制率换算的伪 pIC50；适配层保留它，不能把输出理解为实测 IC50。
- 96 条训练记录中只有 2 条 Ac。原 GP 将包括删失记录在内的数值标签用于普通 GP 拟合。
- 二硫键输入先校验；模型的显式全局特征使用 `j-i+1` 和环外残基数，位置特征还包含
  单体和巯基标记。并非另加两列独立的 i/j 数值。
- BB 名称与训练词典名称不同。审计发现 30 个化学结构及手性对应的词典别名，
  但原 GP 按名称生成 one-hot。为保持原脚本预测一致，本适配未自动合并这些别名。
  如果要改进这点，应在 GP 端统一训练/推理的名称和描述符，并单独验证模型变化。
- 原训练表包含 92 条 11 肽、3 条 10 肽、1 条 12 肽；原特征函数只读取前 11 个位置。
  此接入保留原行为；
  清理训练数据或重新设计特征应作为 GP 模型修订处理。

## 验证

真实 RDKit 环境下，138/138 单体映射成功，828/828 组端基/位置组装检查通过。
测试覆盖每种单体位于首、中、末位置，分别使用 H 和 Ac。
另以原始 CSV 独立重算全部描述符并直接调用原 GP，对比 280 个候选，覆盖全部 138 单体、
H/Ac、7 种二硫键拓扑和相同组成的顺序交换。均值最大绝对误差为 `7.99e-15`，
标准差为 `3.33e-16`，280 个 SMILES/端基/二硫键位置全部一致。
DQN、REINFORCE、A2C 均已完成真实 GP 的微型训练，每种输出 4 个候选；
逐条验证了完整输入、硬约束、位置和 `reward = mean - std`。
常规回归分两组执行，共 505 项通过、11 项默认跳过；其中 9 项真实 GP 检查
在独立环境另行启用并全部通过，剩余 2 项需要缺失的原有材料测试文件。
完整证据保存在 `runs/gp_physchem_validation_20261015/`。
软件接入测试与微型训练不等价于实际活性或优化收敛验证。

快速测试默认不加载外部 GP。真实 GP 测试可明确启用：

```bash
GP_PHYSCHEM_TEST_PYTHON=/path/to/gp-venv/bin/python \
GP_PHYSCHEM_TEST_ROOT=/path/to/gp_physchem \
GP_PHYSCHEM_TEST_PREPARED="$PWD/runs/gp_physchem/prepared" \
python -m pytest tests/test_gp_physchem_adapter.py -o addopts='' -q
```
