# 直接调用 gp_physchem

RL 与 GP 使用同一个 Python 环境。RL 通过现有的 FQN 插件接口导入
`gp_physchem.api:PeptidePredictor`，直接读取 GP 的原始 `new_monomers.csv`。
不需要运行准备命令，也不生成或读取 `catalog.csv`、manifest、GP worker 或描述符缓存文件。

## 安装与运行

在准备运行 RL 的 Python 环境中，安装具有 Python API 的 GP 项目：

```bash
python -m pip install -e /path/to/gp_physchem
```

该 GP 版本新增了安装配置和 `gp_physchem.api.PeptidePredictor`。旧版仅有脚本的
GP 目录需要同步这些源码改动后安装；RL 的 Git 仓库不包含外部 GP 项目。
GP 包声明自己的依赖版本，RL 所需的 PyTorch 等依赖也应安装在同一个环境中。
建议为这套组合单独建立环境，避免改变其他材料模型使用的依赖。

本地配置保存在 `configs/peptide_gp.yaml`，由 `.gitignore` 忽略。
其中与 GP 有关的配置形状如下，远端使用时替换原始 CSV 的路径：

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
    tokens:
      csv: &monomers /path/to/gp_physchem/examples/new_monomers.csv
      id_column: monomer
    allow_repeats: true
    count_constraints:
      - tokens: &thiols [BB1, BB6, BB55, BB56, BB103, BB137]
        min: 2
        max: 2

properties:
  - name: pIC50
    predictor: gp_physchem.api:PeptidePredictor
    new_monomers: *monomers
    thiol_tokens: *thiols
    direction: max
    objective: mean_minus_kstd
k: 1.0
```

示例中的 YAML 锚点让 RL 和 GP 使用同一份原始单体表和同一组约束标记。
这里六个单体来自对当前 138 行原始 CSV 的化学核对，不是通过 ID 名称或 `type2` 猜测。
GP 接口启动时重新从结构识别巯基，并校验 `thiol_tokens` 与实际结果完全一致；
如果替换单体表后标记改变，应相应调整配置。建议 CSV 使用绝对路径，避免插件路径与
相对于 YAML 解析的序列 CSV 路径混淆。

在同一 YAML 中保留所需训练和生成预算，然后在 RL 仓库根目录运行：

```bash
python scripts/run_experiment.py \
  --config configs/peptide_gp.yaml \
  --method a2c --out runs/peptide_gp_a2c \
  --train-seed 0 --gen-seed 0
```

DQN 和 REINFORCE 只需改变 `--method`。GP 的训练表和词典默认来自安装的 GP 包，
也可在 property 中通过 `train`、`dictionary` 指定原始文件。`new_monomers` 指定动作单体库，
不会把训练词典中的其他单体加入可选动作。

## 运行时发生什么

1. GP 接口读取原始词典和新单体 CSV，调用原有化学函数，在内存中计算描述符。
2. 用原始训练表调用 `train_matrix` 和 `fit_gp`，每个 predictor 实例只拟合一次。
3. RL 逐步选择 N 端和 11 个有序单体；通用数量约束保证恰好两个巯基残基。
4. GP 接口从序列推导两个位置，调用原始 `validate_candidates → featurise → predict_gp`。
5. 返回原始预测均值和 GP 不确定性 `std`；RL 计算奖励并写入正常运行结果。

描述符、GP 和 scaler 在本次进程内复用。每次启动新进程都会重新计算描述符，
包含 3D 构象计算，因此启动阶段可能需要数分钟；之后的候选预测无需重复初始化。
GP 接口自身不写文件，正常的 RL 输出（`generated.csv`、模型权重、训练日志等）保持不变。

## 已确认的模型约定

| 项目 | 约定 |
|---|---|
| 动作单体库 | 原始 CSV 的 138 个 ID：BB1–BB138 |
| 序列 | 11 个有序位置，允许重复 |
| N 端动作 / C 端 | H 或 Ac / 固定 NH2 |
| 游离巯基单体 | BB1、BB6、BB55、BB56、BB103、BB137 |
| 二硫键 | 恰好两个巯基残基；位置从 1 开始，i < j |
| 目标 | 原 GP 的伪 pIC50，越大越好 |
| std | 原 GP 直接返回的预测不确定性 |

例如 BB110 的 `type2` 写着 Cys，但结构为硫醚，不属于游离巯基。
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

此次 GP 源码修改增加可安装 Python API 和包内导入兼容性，保留原核函数、标签、
特征和预测公式。以下是原模型的现有性质：

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
