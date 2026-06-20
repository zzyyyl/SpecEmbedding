# SpecEmbedding 重排序方案设计

本文档记录在现有 SpecEmbedding 谱图-分子对齐模型之上增加 molecule reranker 的完整方案。目标是在不引入分子生成模型的前提下，利用当前模型已经召回的候选分子列表，提升 Top-1、MRR 和 Top-K 内排序质量。

## 1. 背景与动机

当前 `SpecMolAlignModel` 通过双塔结构将 MS/MS 谱图和分子图映射到同一向量空间，并使用余弦相似度完成 spectrum-to-molecule retrieval：

```text
spectrum -> spec_encoder -> f_spec
molecule -> mol_encoder  -> f_mol
score = cosine(f_spec, f_mol)
```

这种方式简单高效，但最终排序完全依赖跨模态相似度。由于谱图和分子结构属于不同模态，即使整体 embedding 空间已经对齐，Top-1 排序仍可能受到 modality gap 影响。

从现有实验结果看，SpecEmbedding 在 Top-20 上通常明显高于 Top-1，说明真实分子经常已经进入候选前列，但排序还不够精准。因此 reranker 的核心目标不是重新解决召回问题，而是在已有候选集内提升排序质量。

## 2. 方案定位

建议实现一个轻量级判别式 reranker：

```text
MS/MS spectrum
    -> SpecEmbedding 初筛 top-K candidate molecules
    -> context-aware molecule reranker 重新打分
    -> 输出最终排序
```

该方案不生成 SMILES，不引入 ChemFormer decoder，也不改变当前双塔模型的主训练流程。

| 维度 | 当前 SpecEmbedding | 增加 reranker 后 |
|---|---|---|
| 检索方式 | 直接用谱图-分子余弦相似度排序 | 先初筛，再对 top-K 候选重排序 |
| 是否生成分子 | 否 | 否 |
| 主要优化目标 | 跨模态 embedding 对齐 | 候选列表内排序 |
| 计算成本 | 低 | 中等，取决于 top-K |
| 主要收益 | 候选召回 | Top-1 和 MRR |
| 主要限制 | modality gap | 不能恢复 pre-retrieval 未召回的真值分子 |

## 3. 问题定义

对每个查询谱图 `s_i`，给定候选分子集合：

```text
C_i = {m_i1, m_i2, ..., m_iK}
```

现有模型给出：

```text
q_i = f_spec(s_i)              # [d]
e_ij = f_mol(m_ij)             # [d]
b_ij = cosine(q_i, e_ij)       # base score
r0_ij = base rank              # initial rank
```

reranker 学习新的打分函数：

```text
R(q_i, e_ij, b_ij, r0_ij, C_i) -> rerank_score_ij
```

最终按 `rerank_score_ij` 降序排序。

## 4. 模型结构

推荐第一版实现 `QueryConditionedCandidateTransformer`。该模型以一个谱图 embedding 和一个候选分子列表为输入，输出每个候选分子的重排序分数。

### 4.1 输入特征

对第 `j` 个候选分子，构造 pair feature：

```text
x_j = concat(
    q,
    e_j,
    q * e_j,
    abs(q - e_j),
    base_score_j,
    rank_embedding(rank_j)
)
```

其中：

| 特征 | 作用 |
|---|---|
| `q` | 谱图表示 |
| `e_j` | 候选分子表示 |
| `q * e_j` | 逐维交互 |
| `abs(q - e_j)` | 距离特征 |
| `base_score_j` | 原始双塔相似度 |
| `rank_embedding(rank_j)` | 原始排序位置 |

### 4.2 候选上下文建模

每个候选先通过 MLP 投影成 token：

```text
h_j = pair_mlp(x_j)
```

然后将候选 token 序列送入 TransformerEncoder：

```text
H = TransformerEncoder([h_1, h_2, ..., h_K])
```

最后对每个候选输出新分数：

```text
delta_j = score_head(H_j)
score_j = alpha * base_score_j + delta_j
```

保留 `alpha * base_score_j` 是为了让 reranker 学习对原始排序的修正，而不是完全推翻已有模型。这样训练更稳，也更容易解释。

### 4.3 推荐默认参数

```yaml
rerank:
  prepare:
    pre_top_k: 256
  train:
    train_k: 128
    hidden_dim: 512
    rank_emb_dim: 32
    n_layers: 2
    n_heads: 8
    dropout: 0.1
    alpha_init: 1.0
```

## 5. 候选样本构造

训练样本格式：

```python
{
    "query_id": int,
    "true_smiles": str,
    "spec_emb": Tensor[d],
    "candidate_smiles": list[str],
    "candidate_embs": Tensor[K, d],
    "base_scores": Tensor[K],
    "base_ranks": Tensor[K],
    "label": int,
    "positive_in_base_topk": bool,
}
```

### 5.1 训练集构造

训练时建议强制加入真实分子：

1. 用当前 `SpecMolAlignModel` 对训练集 query 进行初筛。
2. 从候选库中取 top-K hard negatives。
3. 如果真实分子已经在 top-K 中，保留原始列表。
4. 如果真实分子不在 top-K 中，用真实分子替换最后一个候选。
5. 记录 `positive_in_base_topk`，用于后续分析。

这样做是为了保证每个训练样本都有正例，避免 reranker 在大量无正例样本上无法学习。

### 5.2 验证与测试构造

验证和测试时不能强制加入真实分子：

```text
如果 true molecule 不在 pre-retrieval top-K 中，reranker 无法救回，直接视为 miss。
```

这样评估才符合真实检索流程。

### 5.3 候选顺序

训练时建议随机打乱候选顺序，同时保留 `rank_embedding`。这样模型不会仅仅记住输入顺序，但仍能利用原始 rank 信息。

## 6. Loss 设计

### 6.1 主 loss：Listwise Cross Entropy

对每个 query，在候选列表内做分类：

```text
L_ce = CrossEntropy(score_1...score_K, true_index)
```

这个 loss 直接优化“真实分子在候选列表中排第一”的目标。

### 6.2 辅助 loss：Pairwise Ranking Loss

可以加入 pairwise loss，加强正负样本间隔：

```text
L_pair = mean(log(1 + exp(score_neg - score_pos + margin)))
```

总 loss：

```text
L = L_ce + lambda_pair * L_pair
```

推荐第一版：

```yaml
lambda_pair: 0.2
margin: 0.1
```

如果训练不稳定，先只使用 `L_ce`。

## 7. 训练流程

建议第一版冻结当前双塔模型，只训练 reranker：

```text
1. 训练或加载已有 SpecMolAlignModel
2. 预计算 train/val/test 的 spec embedding 和 candidate molecule embedding
3. 生成 rerank cache
4. 训练 CandidateReranker
5. 在验证集上按 Top-1 或 MRR early stopping
6. 在测试集上评估 reranked 排序
```

不建议第一版端到端微调 `spec_encoder` 或 `mol_encoder`。当前 embedding 空间已经具备较强召回能力，端到端训练可能破坏已有对齐。

### 7.1 推荐训练参数

```yaml
rerank:
  train:
    batch_size: 16
    epochs: 30
    lr: 0.0001
    weight_decay: 0.0001
    patience: 5
    grad_clip: 1.0
    metric_for_best: "mrr"
```

如果显存不足，优先降低 `train_k` 或 `batch_size`。

## 8. 推理流程

推理时流程如下：

```text
1. 对 query spectrum 编码，得到 q
2. 对候选分子编码，得到 e_j
3. 用 cosine(q, e_j) 得到 base scores
4. 取 pre_top_k 个候选
5. 输入 reranker 得到 rerank scores
6. 重新排序 top-K
7. 计算 Top-1、Top-5、Top-10、Top-20、MRR、MCES@1
```

`pre_top_k` 必须大于评估最大 K。当前评估包含 Top-20，因此建议：

```yaml
pre_top_k: 256
```

对于候选集本身小于 256 的情况，直接重排序完整候选集。

## 9. 评估方案

主指标：

| 指标 | 说明 |
|---|---|
| Top-1 Accuracy | 最重要，reranker 的主要优化目标 |
| Top-5/Top-10/Top-20 | 检查 reranker 是否破坏候选召回 |
| MRR | 反映整体排序质量 |
| MCES@1 | 检查 top-1 分子结构是否更接近真实分子 |

必须同时报告：

```text
pre-retrieval top-K recall upper bound
```

原因是 reranker 不能恢复未被初筛召回的真值分子。比如 `pre_top_k=256` 时，如果真实分子不在前 256，那么 reranker 的理论上限已经失败。

## 10. 必做消融实验

| 实验 | 目的 |
|---|---|
| Base SpecEmbedding | 原始双塔检索基线 |
| Pointwise MLP reranker | 验证是否需要候选上下文 |
| Candidate Transformer reranker | 主方法 |
| 去掉 `base_score` residual | 验证 residual 排序修正的作用 |
| 不打乱候选顺序 vs 打乱候选顺序 | 检查是否依赖输入顺序 |
| `train_k = 32/64/128/256` | 分析候选数量影响 |
| `pre_top_k = 64/128/256/512` | 分析召回上限和计算成本 |
| mass candidate vs formula candidate | 分析不同候选难度 |
| cross-dataset evaluation | 检查泛化能力 |

## 11. 预期收益与上限

reranker 主要提升 Top-1 和 MRR，不应期待显著提升 Top-20。理论上，Top-1 的上限由 pre-retrieval top-K recall 决定。

以当前 MassSpecGym 结果为例：

| Candidate type | Base Top-1 | Base Top-20 | 含义 |
|---|---:|---:|---|
| mass | 约 48% | 约 78% | 若真实分子已在前 20，reranker 有较大提升空间 |
| formula | 约 58% | 约 82% | formula 候选更强，Top-1 仍有提升空间 |

合理预期：

| Candidate type | Top-1 预期提升 |
|---|---:|
| mass | +5 到 +12 |
| formula | +4 到 +8 |

实际提升取决于候选集难度、正样本是否进入 pre_top_k，以及 hard negative 的结构相似度。

## 12. 实现计划

建议新增以下文件：

| 文件 | 作用 |
|---|---|
| `SpecEmbedding/models_rerank.py` | 定义 `CandidateReranker`、`PointwiseReranker` |
| `SpecEmbedding/data/datasets_rerank.py` | 读取 rerank cache，构造 batch |
| `prepare_rerank_cache.py` | 预计算 top-K 候选、embedding 和 base scores |
| `train_rerank.py` | 训练 reranker |
| `eval_rerank.py` | 评估 reranker 后的结果 |

### 12.1 第一阶段实现

第一阶段只实现离线 cache 训练，避免重复编码谱图和分子：

```text
prepare_rerank_cache.py
    -> train_rerank.py
    -> eval_rerank.py
```

这样工程风险最低，也方便快速对比不同 reranker 结构。

### 12.2 第二阶段实现

如果第一阶段有效，再考虑在线推理版本：

```text
eval_rerank.py
    -> load align checkpoint
    -> online encode query and candidates
    -> rerank
```

在线版本更适合最终发布，但调试成本更高。

## 13. 风险与处理

| 风险 | 处理方式 |
|---|---|
| true molecule 不在 pre_top_k | 增大 `pre_top_k`，并报告 upper bound |
| reranker 过拟合候选分布 | 使用 dropout、候选顺序打乱、cross-dataset validation |
| 强制加入正样本导致训练/测试不一致 | 仅训练强制加入，测试严格真实评估 |
| 大候选集计算慢 | 先用 base model 截断到 top-256 或 top-512 |
| reranker 破坏已有好排序 | 使用 `score = alpha * base_score + delta` residual 设计 |
| 不同数据集 candidate 构造不一致 | 每个实验明确记录 candidate source 和 filtering 逻辑 |

## 14. 论文表述建议

建议命名：

```text
SpecEmbedding-Rerank
```

可写成：

```text
We introduce a lightweight context-aware molecular reranker on top of SpecEmbedding. Instead of generating molecules, the reranker directly optimizes listwise ordering among hard molecular candidates retrieved by the base spectrum-molecule alignment model.
```

中文表述：

```text
我们在 SpecEmbedding 的谱图-分子对齐模型之上引入一个轻量级候选分子重排序模块。该模块不依赖分子生成，而是在初筛得到的 hard candidate list 内进行上下文感知的 listwise reranking，从而提升 Top-1 命中率与排序质量。
```

## 15. 最小可行实验

第一轮实验建议只做 MassSpecGym：

```text
Dataset: MassSpecGym
Candidate type: mass, formula
Base checkpoint: 当前最好 SpecMolAlignModel
pre_top_k: 256
train_k: 128
Model: Candidate Transformer reranker
Metrics: Top-1/5/10/20, MRR, MCES@1, pre_top_k recall upper bound
```

如果 MassSpecGym 上 Top-1 有稳定提升，再扩展到 MassBank、GNPS、MoNA 和 NPLIB1。

## 16. 当前代码入口

第一版实现采用离线 cache 流程，对应 3 个入口脚本：

| 脚本 | 作用 |
|---|---|
| `prepare_rerank_cache.py` | 用已有 `SpecMolAlignModel` checkpoint 生成 rerank cache |
| `train_rerank.py` | 读取 train/val cache 训练 reranker |
| `eval_rerank.py` | 读取 test cache 和 reranker checkpoint，比较 base 与 rerank 结果 |

重排序相关超参数统一写在 `params.yaml` 的 `rerank.prepare`、`rerank.train`、`rerank.eval` 中。命令行只保留本次运行经常变化的路径、split、候选来源和设备等参数。设备参数支持 `cpu`、`cuda`、`cuda:0`、`cuda:1` 等写法，例如：

```bash
python train_rerank.py \
  --train_cache rerank_cache/massspecgym_formula_train.pt \
  --val_cache rerank_cache/massspecgym_formula_val.pt \
  --device cuda:1
```

### 16.1 生成训练 cache

训练 cache 建议打开 `--force_include_positive`，保证每个训练样本都有正例。`pre_top_k`、batch size、候选 chunk size、embedding dtype 等从 `params.yaml` 读取：

```bash
python prepare_rerank_cache.py \
  --checkpoint checkpoints_align/run/best_model_stage2.pth \
  --dataset_type massspecgym \
  --split train \
  --data_path /path/to/processed \
  --candidate_type formula \
  --force_include_positive \
  --save_path rerank_cache/massspecgym_formula_train.pt
```

### 16.2 生成验证和测试 cache

验证和测试 cache 不要打开 `--force_include_positive`，否则会高估真实效果：

```bash
python prepare_rerank_cache.py \
  --checkpoint checkpoints_align/run/best_model_stage2.pth \
  --dataset_type massspecgym \
  --split val \
  --data_path /path/to/processed \
  --candidate_type formula \
  --save_path rerank_cache/massspecgym_formula_val.pt

python prepare_rerank_cache.py \
  --checkpoint checkpoints_align/run/best_model_stage2.pth \
  --dataset_type massspecgym \
  --split test \
  --data_path /path/to/processed \
  --candidate_type formula \
  --save_path rerank_cache/massspecgym_formula_test.pt
```

### 16.3 训练 reranker

训练超参数从 `params.yaml` 的 `rerank.train` 读取：

```bash
python train_rerank.py \
  --train_cache rerank_cache/massspecgym_formula_train.pt \
  --val_cache rerank_cache/massspecgym_formula_val.pt \
  --save_dir checkpoints_rerank/massspecgym_formula
```

### 16.4 评估 reranker

评估的 `top_k`、batch size、worker 数和是否默认计算 MCES 从 `params.yaml` 的 `rerank.eval` 读取：

```bash
python eval_rerank.py \
  --cache rerank_cache/massspecgym_formula_test.pt \
  --checkpoint checkpoints_rerank/massspecgym_formula/best_reranker.pth \
  --save_dir checkpoints_rerank/massspecgym_formula
```

如果只想快速验证排序指标，可以先跳过 MCES：

```bash
python eval_rerank.py \
  --cache rerank_cache/massspecgym_formula_test.pt \
  --checkpoint checkpoints_rerank/massspecgym_formula/best_reranker.pth \
  --no-mces
```
