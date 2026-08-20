# SpecEmbedding 项目记忆

最后更新：2026-08-20

当前修订状态：

- 分支：`codex/transfer-2026-0831`
- 暂定论文完成日期：2026-08-31
- ADMA 2026 截止日期和 `dev`/`a2280d2` 状态只作为历史实验背景；当前转投进度以
  `paper/TRANSFER_2026_PLAN.md`、专项计划和 Git 历史为准。

本文档用于后续开发、实验和论文协作时快速恢复上下文。它是内部协作材料，不应直接放入 ADMA 双盲补充材料。若本文档与代码、`params.yaml` 或实验日志冲突，以代码和原始日志为准。

## 1. 当前总目标

项目已经从“谱图嵌入”扩展为一个完整的 MS/MS-to-molecule retrieval 系统：

1. 使用 SpecEmbedding 编码 MS/MS 谱图。
2. 使用 GINE 编码候选分子图。
3. 通过跨模态对比学习对齐谱图和分子。
4. 使用基础相似度从候选库中检索 top-$K$ 分子。
5. 使用非生成式残差 reranker 重新排序 hard candidates，并受控比较 pointwise 与候选集合 Transformer 变体。

当前目标是完成转投稿件，暂定完稿日期为 2026-08-31；ADMA 2026 及其主题
`Data mining for bioinformatics` 是原始投稿背景。英文标题暂定为：

> Non-Generative Learning to Rerank for MS/MS-Based Molecule Retrieval

三随机种子受控消融完成后，论文需要强调的证据边界是：

- 在已有跨模态检索器之上直接优化候选列表排序。
- 结合 base score、base rank 和显式谱图--候选交互特征。
- 使用 residual listwise reranking，在不生成分子的情况下修正基础排序。
- pointwise 与候选集合 Transformer 在 alignment seeds 42/43/44 的 12/12 个
  alignment--candidate--learned-model 聚合单元中均同时改善对应 base 的 Recall@1 和 MRR；
  Transformer 相对 pointwise 在两类候选和两项指标上都只有 2/3 alignment 方向为正。
- Recall/MRR 使用本地 exact-target-SMILES 单正例规则，不等价于参考二维 InChIKey/可能
  多正例 evaluator；差异影响尚未量化。
- 因此可被当前实验支持的核心是“监督残差学习排序有效”，而不是“候选间关系是主要增益来源”。

## 2. 协作与工程约定

### 2.1 环境

- Conda 环境：`specembedding`
- Python：3.12
- 设备参数支持：`cpu`、`cuda`、`cuda:0`、`cuda:1` 等
- 已验证 `cuda:1` 可用于 alignment 训练
- 主要实验硬件：NVIDIA GeForce RTX 4090

激活环境：

```bash
conda activate specembedding
```

### 2.2 配置

- 训练、评估和 rerank 超参数统一保存在 `params.yaml`。
- 命令行只保留频繁变化的运行参数，例如路径、候选类型、设备、运行模式和调试 limit。
- 配置项不应提供静默 fallback；缺失配置应尽早失败，避免隐藏错误。
- 业务代码优先直接访问 `config.xxx`。
- checkpoint 中必须保存构建 reranker 所需的完整模型配置。

### 2.3 代码复用

不要从 `train.py`、`eval.py` 等入口脚本中导入公共函数。公共逻辑已经逐步抽取到：

- `SpecEmbedding/utils/runtime.py`
- `SpecEmbedding/utils/model.py`
- `SpecEmbedding/utils/align.py`
- `SpecEmbedding/utils/rerank.py`
- `SpecEmbedding/utils/providers.py`
- `SpecEmbedding/trainer/`

入口脚本只负责参数解析和流程编排，避免重复实现模型加载、设备解析、候选加载、指标统计和日志初始化。

### 2.4 Git 与文档

- 提交信息使用中文 Conventional Commit，例如：

```text
feat(rerank): 增加候选集合重排序
docs(paper): 更新 ADMA2026 论文稿件
```

- 用户要求提交时，如无特殊说明，提交后直接推送远端。
- 当前主要开发分支为 `codex/transfer-2026-0831`。
- 本环境没有安装 `gh`，但普通 `git push` 可用；无法自动创建 GitHub/GitLab PR。
- LaTeX 工作编译产物放入已被 Git 忽略的 `paper/build/`；审计或投稿里程碑使用
  `paper/build_release.sh` 强制重建，并把 PDF 与构建清单留存在受跟踪的 `paper/release/`。

## 3. 基础模型

### 3.1 SpecEmbedding 谱图编码器

SpecEmbedding 使用峰序列表示而不是固定 binning：

- 选取峰的 $m/z$ 和 intensity。
- 使用 sinusoidal $m/z$ embedding。
- 使用 Transformer encoder 建模峰间关系。
- 使用同一分子的重复谱图作为监督对比学习正例。
- 原始 SpecEmbedding 工作还结合 Tanimoto structural similarity loss。

对应参考论文：

> Supervised Contrastive Learning Leads to More Reasonable Spectral Embeddings
>
> DOI: 10.1021/acs.analchem.5c02655

论文 PDF 位于 `paper/ref/SpecEmbedding.pdf`。

### 3.2 谱图--分子对齐模型

主要实现位于：

- `SpecEmbedding/models_align.py`
- `SpecEmbedding/loss_align.py`
- `train_align.py`
- `eval_align.py`

`SpecMolAlignModel` 是双塔结构：

```text
MS/MS spectrum -> SiameseModel/Transformer -> spec projection -> z_s
molecule graph -> GINEEncoder             -> mol projection  -> z_m
```

分子编码器使用：

- 原子和化学键离散特征 embedding。
- 多层 GINEConv。
- residual connection、normalization 和 dropout。
- global mean pooling。
- 显式图规模特征，补偿 mean pooling 丢失的 size signal。

两个模态投影到 512 维，并在检索时 L2 normalize。基础分数是：

```text
base_score = dot(normalize(z_s), normalize(z_m))
```

对齐损失是双向 multi-positive contrastive loss：

- spectrum-to-molecule 和 molecule-to-spectrum 两个方向取平均。
- 同一 molecular label 的多个谱图视为多正例。
- 使用可学习 temperature/logit scale。

训练流程支持：

- 有预训练：先冻结谱图编码器训练分子塔和投影头，再端到端微调。
- `--no-pretrain`：跳过冻结阶段，从零开始进行端到端 alignment。

当前论文中的主要实验使用 `--no-pretrain`。

## 4. Molecule Reranker

完整设计文档位于 `docs/reranker_solution_zh.md`。主要代码：

- `SpecEmbedding/models_rerank.py`
- `SpecEmbedding/data/datasets_rerank.py`
- `SpecEmbedding/utils/rerank.py`
- `prepare_rerank_cache.py`
- `train_rerank.py`
- `eval_rerank.py`
- `run_rerank_pipeline.py`
- `run_rerank_multiseed.py`
- `run_rerank_overlap_sensitivity.py`
- `freeze_adma2026_artifacts.py`：验证并生成不含绝对路径或主机身份信息的内部实验 artifact manifest；manifest 仅作 inventory，其引用的原始 status/log 仍需在匿名发布前脱敏。

### 4.1 设计目标

reranker 不生成 SMILES，也不访问 top-$K$ 之外的分子。它只解决：

> 当真实分子已经被基础模型召回时，如何在 hard candidate list 中提高 Top-1 和 MRR。

它不能提高候选召回上界，也不能恢复基础 top-$K$ 中不存在的真实分子。

### 4.2 输入特征

对每个候选 $m_i$：

```text
x_i = concat(
    z_s,
    z_m_i,
    z_s * z_m_i,
    abs(z_s - z_m_i),
    base_score_i,
    rank_embedding(base_rank_i)
)
```

随后：

```text
h_i       = pair_mlp(x_i)
h_context = TransformerEncoder(h_1, ..., h_K)
delta_i   = score_head(h_context_i)
score_i   = alpha * base_score_i + delta_i
```

主要设计点：

- `CandidateReranker` 使用候选 self-attention。
- `PointwiseReranker` 将 Transformer 层数设为 0，用作消融。
- residual base score 防止 reranker 完全推翻已有强排序。
- padding mask 支持可变候选数量。
- 训练时随机打乱候选张量顺序，但保留 base rank embedding，避免直接记住输入位置。

### 4.3 Loss

```text
L = L_listwise_cross_entropy + lambda_pair * L_pairwise
```

当前参数：

- `lambda_pair = 0.2`
- `margin = 0.1`

pairwise loss 使用 softplus：

```text
softplus(score_neg - score_pos + margin)
```

### 4.4 Cache 与正例协议

基础模型冻结，谱图和候选分子 embedding 离线缓存。

训练集：

- 允许 `force_include_positive=true`。
- 如果真实分子不在 base top-$K$，用真实分子替换最后一个候选。
- 保留真实分子在完整候选池中的原始 base rank。
- 这是监督训练采样，不是测试协议。

验证集和测试集：

- 必须使用 `force_include_positive=false`。
- 真实分子未进入 top-$K$ 时直接记为 miss。
- 所有查询都进入最终指标，缺失正例的查询贡献 0。
- 必须同时报告 pre-retrieval recall upper bound。

这是当前判断是否存在候选数据泄露的关键边界。

## 5. 常用运行方式

### 5.1 基础流水线

用户实际运行过：

```bash
python run_pipeline.py --no-pretrain --device cuda:1
```

alignment 训练完成后，主要使用：

```text
best_model_stage2.pth
```

作为 rerank cache 的基础 checkpoint。

### 5.2 Rerank 流水线

推荐统一入口：

```bash
python run_rerank_pipeline.py massspecgym \
  --align_save_dir checkpoints_align/<run> \
  --candidate_type formula \
  --topk 40 \
  --device cuda:1
```

候选类型：

- `mass`
- `formula`

运行模式：

- `--mode prepare`
- `--mode train`
- `--mode eval`
- `--mode all`

调试：

```bash
python run_rerank_pipeline.py massspecgym \
  --align_save_dir checkpoints_align/<run> \
  --candidate_type formula \
  --topk 40 \
  --limit 100 \
  --device cuda:1
```

输出目录会自动增加 `_topkN` 和 `_limitN`，防止覆盖正式实验。

### 5.3 论文编译

英文稿：

```bash
latexmk -pdf \
  -interaction=nonstopmode \
  -halt-on-error \
  -cd -outdir=build paper/main.tex
```

中文稿：

```bash
latexmk -pdf -xelatex \
  -interaction=nonstopmode \
  -halt-on-error \
  -cd -outdir=build paper/main_cn.tex
```

输出：

- `paper/build/main.pdf`
- `paper/build/main_cn.pdf`
- 可审计副本：`paper/release/specembedding-adma2026-en.pdf`、
  `paper/release/specembedding-adma2026-zh.pdf`
- 构建来源、工具链、页数和 SHA-256：`paper/release/build-manifest.yaml`

## 6. 已完成的主要实验

### 6.1 实验设置

- 数据集：MassSpecGym
- 数据划分：官方 structure-disjoint split
- 训练谱图：194,119
- 验证谱图：19,429
- 测试谱图：17,556
- 唯一分子：32,010
- canonical 基础 checkpoint：`checkpoints_align/a2280d2_massspecgym_nopretrain_valoverlapclean/best_model_stage2.pth`
- alignment：无额外 SpecEmbedding 预训练
- rerank top-$K$：40
- canonical/组件表 alignment 随机种子：42（固定 checkpoint）
- 跨 alignment 分层分析：alignment seeds 42、43、44；每个 checkpoint 内先聚合
  reranker seeds 42、43、44
- reranker 随机种子：42、43、44
- reranker hidden dim：512
- rank embedding dim：32
- Transformer：2 层、8 heads
- dropout：0.1
- batch size：16
- learning rate：$10^{-4}$
- early stopping：validation MRR，patience 5

MassSpecGym 候选文件来自其官方数据集发布：

- `MassSpecGym_retrieval_candidates_mass.json`
- `MassSpecGym_retrieval_candidates_formula.json`

本项目预处理后保存为：

- `candidates_mass.pkl`
- `candidates_formula.pkl`

### 6.2 本地结果

Recall 指标以百分比表示。当前 overlap-clean 主结果表的 MRR 保留日志中 $[0,1]$ 原始量纲；后续历史表为了与旧稿一致将 MRR 乘以 100。MCES@1 是结构距离，保留原始量纲，越低越好。Recall/MRR 采用 MassSpecGym 提供候选文件上的本地 exact-target-SMILES 单正例规则；该规则与参考二维 InChIKey/可能多正例 evaluator 不等价，影响尚未量化。

2026-07-12 完成 commit `a2280d2` 的 train--val overlap-clean 全流水线。该流水线排除 validation 索引 7686/7687/7688/8464/8465/8466，重训 alignment、重建 mass/formula top-40 cache，并完成 2 种候选库 $×$ 2 种 reranker $×$ 3 个 seeds 的 12 组训练与评估。12/12 组均为 `complete`，`errors=[]`。当前主结果应采用下表的 overlap-clean 数字：

| Candidate | Model | Upper bound | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR | MCES@1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mass | Clean Base | 83.8802 | 47.4596 | 63.5111 | 71.0242 | 77.7626 | 0.5503 | 15.3681 |
| mass | Pointwise | 83.8802 | 67.4432±0.7357 | 77.0202±0.5911 | 80.1055±0.2077 | 82.2624±0.1588 | 0.7179±0.0066 | 未计算 |
| mass | Set Transformer | 83.8802 | 68.5824±0.2035 | 77.3582±0.3610 | 79.9935±0.3405 | 82.2036±0.0668 | 0.7260±0.0025 | 7.7057（seed 42） |
| formula | Clean Base | 89.2572 | 63.1009 | 75.7861 | 80.8669 | 84.9339 | 0.6886 | 5.4389 |
| formula | Pointwise | 89.2572 | 73.9709±0.5910 | 81.9359±0.2712 | 84.4934±0.0786 | 86.7813±0.2651 | 0.7764±0.0049 | 未计算 |
| formula | Set Transformer | 89.2572 | 74.5671±0.2819 | 81.9682±0.2645 | 84.3757±0.0821 | 86.7719±0.1392 | 0.7802±0.0026 | 3.0913（seed 42） |

overlap-clean 受控结论：

- Pointwise 相对 Clean Base 的 Recall@1 增益为 mass +19.9836、formula +10.8700 个百分点；Set Transformer 增益为 mass +21.1228、formula +11.4662 个百分点。监督残差 reranking 的核心收益保持。
- Set Transformer 相对 Pointwise 的平均 Recall@1 差异为 mass +1.1392、formula +0.5962 个百分点，在 6/6 个成对 seed--candidate Recall@1 比较中获胜；MRR raw 平均差分别为 +0.0081/+0.0038。
- 上述 Transformer 优势与历史 `d4c1f70` 固定 alignment 结果（mass -0.13、formula +0.21 个百分点，胜场 3/6）不一致。因此只能说 clean run 呈现小幅一致优势，但尚不能确认 candidate self-attention 具有稳健独立收益。
- 本表的 $±$ 只覆盖 reranker seeds 42/43/44，并固定 canonical alignment seed 42；
  跨 alignment 证据另见下方分层结果，不能把两种统计单位混用。
- overlap-clean MCES@1 已使用同一 `a2280d2` 流水线补算完成：mass Clean Base/seed-42 Set Transformer 为 15.3681/7.7057，绝对降低 7.6624（49.86%）；formula 为 5.4389/3.0913，绝对降低 2.3476（43.16%）。四项均完成 17,556/17,556 条查询，batch 状态为 `complete`。

#### 跨 alignment 分层结果（2026-08-17）

alignment seeds 42/43/44 均已完成两种候选协议、两种监督 reranker 和 reranker seeds
42/43/44，共 36/36 组。下表先在每个 alignment 内平均三个 reranker seeds；R@1/MRR
均以百分比显示，不把九次 alignment--reranker 组合展平为九个 alignment 样本。

| Alignment | Candidate | Base R@1 | Pointwise R@1 | Transformer R@1 | Base MRR | Pointwise MRR | Transformer MRR |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | mass | 47.46 | 67.44 | 68.58 | 55.03 | 71.79 | 72.60 |
| 42 | formula | 63.10 | 73.97 | 74.57 | 68.86 | 77.64 | 78.02 |
| 43 | mass | 35.92 | 66.92 | 67.19 | 44.33 | 70.42 | 70.58 |
| 43 | formula | 45.08 | 69.64 | 69.30 | 51.85 | 72.14 | 71.96 |
| 44 | mass | 23.91 | 57.78 | 57.42 | 32.26 | 61.22 | 60.91 |
| 44 | formula | 28.73 | 59.76 | 59.82 | 36.87 | 63.14 | 63.34 |

- 三个 alignment、两种候选、两种学习模型构成的 12 个 alignment-level 聚合单元中，
  Pointwise/Transformer 的三 reranker-seed 均值均同时改善对应 base 的 R@1 与 MRR（12/12）。
- Transformer-minus-Pointwise 的 alignment-level 配对均值在 mass/formula 的 R@1 与 MRR
  上都只有 2/3 alignment 为正，不支持 self-attention 的一致独立收益。
- 三个 checkpoints 使用相同参数哈希，但来自两个 source commits；三个 alignment-level
  estimates 仅是内部描述性敏感性证据，不构成置信区间、显著性或一般端到端稳定性证明。
- 版本化证据：`analysis/transfer2026_alignment_multiseed/`。

Train--val 模型选择审计：

- 6 条重叠 validation 查询对应 2 个分子组；TokenSet validation keys 由 3386 过滤为 3384。
- clean alignment 的最佳 epoch 为 16，best validation contrastive loss 为 1.5294826125，epoch 21 early stop；历史 alignment 也是 best/stop epoch 16/21。由于 validation 集已变，两者 validation loss 数值不可直接比较。
- clean reranker 12/12 组全部 early stop；9/12 组的 best/stop epoch 对与历史流水线不同，3/12 组相同。因此不能声称新旧下游模型选择逐项一致；该差异不能单独归因于 6 条重叠查询。
- 新旧流水线对比同时包含 alignment 重训、checkpoint 和 cache 变化。因此测试指标差异只能解释为完整流水线对 validation 协议的敏感性，不是删除 6 条查询的隔离因果效应。

下表是历史 `d4c1f70` 代表性 seed-42 Set Transformer 结果，仅保留为已完成 MCES@1 的追溯记录：

| Candidate | Method | Upper bound | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR | MCES@1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mass | Base | 82.15 | 43.78 | 61.10 | 68.43 | 75.55 | 51.96 | 16.42 |
| mass | Set Transformer (seed 42) | 82.15 | 67.91 | 76.00 | 78.44 | 80.45 | 71.62 | 8.02 |
| formula | Base | 87.30 | 58.49 | 71.49 | 76.86 | 81.81 | 64.65 | 6.14 |
| formula | Set Transformer (seed 42) | 87.30 | 74.33 | 80.64 | 82.89 | 85.07 | 77.27 | 3.13 |

mass MCES@1 的原始输出为 Base 16.4199、代表性 seed-42 Set Transformer 8.0209；表中按两位小数展示。这些 MCES@1 不能与 overlap-clean 主结果混用。

历史 `d4c1f70` 固定同一 alignment checkpoint 和 top-40 cache 时，reranker seeds 42/43/44 的均值 $\pm$ 样本标准差为：

| Candidate | Model | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mass | Pointwise | 67.89±0.44 | 75.79±0.19 | 78.36±0.11 | 80.63±0.05 | 71.53±0.35 |
| mass | Set Transformer | 67.75±0.31 | 75.88±0.18 | 78.41±0.12 | 80.63±0.15 | 71.52±0.24 |
| formula | Pointwise | 73.77±0.19 | 80.55±0.12 | 82.93±0.23 | 84.97±0.15 | 76.88±0.16 |
| formula | Set Transformer | 73.98±0.84 | 80.27±0.54 | 82.75±0.33 | 85.07±0.05 | 76.95±0.69 |

该历史 checkpoint 上的受控结论：

- 两种学习式 reranker 都大幅优于固定 base retriever。
- Set Transformer 相对 Pointwise 的 Recall@1 平均差异为 mass $-0.13$ 个百分点、formula $+0.21$ 个百分点。
- Transformer 在 6 个成对的 seed--candidate Recall@1 比较中只获胜 3 次，MRR 差异也接近于零。
- 该历史结果不支持“candidate self-attention 带来稳定独立收益”；不应将整体 reranking 提升归因于 self-attention。
- 该误差只覆盖 reranker 训练变异，不覆盖 alignment 或完整端到端流水线的方差。

2026-07-12 完成 train--test 完全相同输入的剔除敏感性评估。统一排除
test cache 索引 5908、5909 和 5910，查询数从 17,556 降至 17,553；
12/12 组实验均为 `complete`，`errors=[]`。三种子平均 Recall@1 为：

| Candidate | Model | 原始 | 剔除后 | 变化（百分点） |
| --- | --- | ---: | ---: | ---: |
| mass | Pointwise | 67.8856 | 67.8801 | -0.0055 |
| mass | Set Transformer | 67.7546 | 67.7491 | -0.0055 |
| formula | Pointwise | 73.7735 | 73.7690 | -0.0045 |
| formula | Set Transformer | 73.9804 | 73.9760 | -0.0045 |

mass/formula 的 upper bound 分别从 82.1542%/87.3035% 变为
82.1512%/87.3013%。单个 seed 的 Recall@1 最大绝对变化为 0.0056 个
百分点，MRR 最大绝对变化为 0.0001（raw）；Transformer 的成对 Recall@1
胜场仍为 3/6。因此，剔除这 3 条 train--test 输入重叠后，监督残差重排序
有效的核心结论不变；在该历史 checkpoint 上的 self-attention 胜场也不变。该批次按协议
未重算 MCES@1。

历史 `d4c1f70` 代表性 seed-42 Set Transformer 相对基础检索器：

- mass Recall@1：+24.13 个百分点
- mass MRR：+19.66 个百分点
- mass MCES@1：16.42 降至 8.02，绝对降低 8.40，相对降低 51.15%（越低越好）
- formula Recall@1：+15.84 个百分点
- formula MRR：+12.62 个百分点
- formula MCES@1：6.14 降至 3.13（越低越好）

测试 top-40 平均候选数：

- mass：40.00
- formula：35.73

训练集在插入正例前的 base top-40 recall：

- mass：93.78%
- formula：90.42%

训练 cache 插入后 labeled coverage 为 100%；验证和测试没有插入。

### 6.3 结果路径

overlap-clean 最终流水线：

```text
checkpoints_align/a2280d2_massspecgym_nopretrain_valoverlapclean/
checkpoints_align/a2280d2_massspecgym_nopretrain_valoverlapclean/alignment_selection.json
rerank_cache/a2280d2_massspecgym_nopretrain_valoverlapclean_mass_topk40/
rerank_cache/a2280d2_massspecgym_nopretrain_valoverlapclean_formula_topk40/
checkpoints_rerank/a2280d2_massspecgym_nopretrain_valoverlapclean_topk40_multiseed/
checkpoints_rerank/a2280d2_massspecgym_nopretrain_valoverlapclean_topk40_multiseed/summary.csv
checkpoints_rerank/a2280d2_massspecgym_nopretrain_valoverlapclean_topk40_multiseed/summary_aggregate.csv
checkpoints_rerank/a2280d2_massspecgym_nopretrain_valoverlapclean_topk40_multiseed/batch_status.json
checkpoints_rerank/a2280d2_massspecgym_nopretrain_valoverlapclean_topk40_mces_seed42_transformer/
checkpoints_rerank/a2280d2_massspecgym_nopretrain_valoverlapclean_topk40_mces_seed42_transformer/batch_status.json
```

该批次于 2026-07-12T22:21:42+08:00 完成，记录的完整代码 commit 为
`a2280d2828ce872da1f69319b49e0ef7f1bed572`，`params.yaml` SHA-256 为
`b6260dad043f9c0f1cb4ff36dc7b7bf8bac4481998aa36b4d55f44392e1b0bc6`，alignment checkpoint SHA-256 为
`ad5d1eb76805c51563349f259a4b4c935336064b6171a4e650472b78aeeaa06f`。

MCES@1 batch 于 2026-07-13T15:35:46+08:00 完成，source commit 为 `a2280d2`，batch 与 mass/formula 两个子任务状态均为 `complete`；Base 与 reranker 四项均完成 17,556/17,556 条查询。

以下为历史 `d4c1f70` 路径，用于追溯旧结果和 train--test 敏感性实验。

基础 alignment：

```text
checkpoints_align/d4c1f70_massspecgym_nopretrain/
```

mass rerank：

```text
rerank_cache/d4c1f70_massspecgym_nopretrain_mass_topk40/
checkpoints_rerank/d4c1f70_massspecgym_nopretrain_mass_topk40/
```

formula rerank：

```text
rerank_cache/d4c1f70_massspecgym_nopretrain_formula_topk40/
checkpoints_rerank/d4c1f70_massspecgym_nopretrain_formula_topk40/
```

三种子 pointwise/Transformer 受控实验：

```text
checkpoints_rerank/d4c1f70_massspecgym_nopretrain_topk40_multiseed/
checkpoints_rerank/d4c1f70_massspecgym_nopretrain_topk40_multiseed/summary.csv
checkpoints_rerank/d4c1f70_massspecgym_nopretrain_topk40_multiseed/summary_aggregate.csv
checkpoints_rerank/d4c1f70_massspecgym_nopretrain_topk40_multiseed/batch_status.json
```

该批次共 12 组实验（2 candidate types $\times$ 2 reranker variants $\times$ 3 seeds），全部以 `complete` 状态结束，无失败 attempt。

Train--test 输入重叠敏感性评估：

```text
checkpoints_rerank/d4c1f70_massspecgym_nopretrain_topk40_input_overlap_sensitivity/
checkpoints_rerank/d4c1f70_massspecgym_nopretrain_topk40_input_overlap_sensitivity/summary.csv
checkpoints_rerank/d4c1f70_massspecgym_nopretrain_topk40_input_overlap_sensitivity/summary_aggregate.csv
checkpoints_rerank/d4c1f70_massspecgym_nopretrain_topk40_input_overlap_sensitivity/batch_status.json
```

该批次于 2026-07-12T10:30:23+08:00 完成，使用完整 commit
`b20a2e842bf20034e1ba3321d26ccb06ae408d06`。

关键日志：

```text
checkpoints_rerank/<run>/train_rerank.log
checkpoints_rerank/<run>/eval_rerank.log
rerank_cache/<run>/prepare_rerank_cache_{train,val,test}.log
```

## 7. 候选协议与数据泄露结论

### 7.1 当前可以辩护的部分

- 使用 MassSpecGym 官方候选文件。
- train/val/test 使用官方结构划分。
- 基础模型和 reranker 只在训练集上训练。
- reranker 只在训练 cache 中强制加入正例。
- 验证、测试不插入正例。
- 指标包含正例未被召回的查询。
- 报告 top-$K$ recall upper bound。

2026-07-11 的再审计结果：

- mass 和 formula 的 top-40 cache meta 与生成日志均记录 train
  `force_include_positive=true`、val/test `false`。
- val/test 中 `label is not None` 与 `positive_in_base_topk` 完全一致，两类
  cache 均为 0 mismatch；所有已标注 label 均指向 `true_smiles`。
- MassSpecGym 原始 processed split 包含 194,119/19,429/17,556 条
  train/val/test 谱图；三个划分的 identifier、原始 SMILES、InChIKey 和
  RDKit 规范化 isomeric/non-isomeric SMILES 两两交集均为 0，fold 标记
  无错配，未发现同分子标签跨 split 重复。
- 按 mzs、intensities、precursor m/z、adduct、instrument 和 collision energy
  组成的完全输入签名检查，发现 6 个 train--val 和 3 个 train--test
  重复。这 9 个 eval 查询的标签分子与 train 不同，且全部为
  `simulation_challenge=true`；它们不是正标签结构泄露，但属于输入样本重叠，
  需要单独评估。三条 test cache 的统一 0-based 索引为
  5908、5909 和 5910。
- 非空 Murcko scaffold 在 train--val/train--test/val--test 间分别有
  115/128/34 个交集。因此只能称使用 MassSpecGym 官方 structure-disjoint
  相似性分组，不能进一步声称 scaffold-disjoint。

2026-07-12 已完成 3 条 train--test 重叠输入的剔除敏感性评估：排除后
Recall@1 的三种子均值仅变化 $-0.0045$ 至 $-0.0055$ 个百分点，主结论
不变。因此，“训练时保证监督列表有正例”本身不构成测试正标签泄露，且
train--test 输入重叠不会实质改变当前结果。

2026-07-12 也已完成 6 条 train--val 重叠输入的 overlap-clean 全流水线审计。clean 与历史 alignment 均选择 best/stop epoch 16/21，但 reranker 9/12 组的 best/stop epoch 对不同。这说明新旧完整流水线的模型选择并非逐项一致，同时 overlap-clean 流水线仍保持对 base 的数值大幅改善。由于 alignment 同时重训，新旧差异只能报告为流水线级敏感性，不应单独归因于 validation 协议或作为 6 条查询的隔离因果估计。

### 7.2 必须明确披露的限制

- formula candidate 设置假设已知真实分子式，应称为 formula-conditioned retrieval，不能描述成完全开放的未知分子鉴定。
- reranker 是 closed-library 方法，不能生成候选库之外的新分子。
- canonical 主表和组件消融固定 alignment seed 42，其 $±$ 只覆盖三个 reranker seeds；
  另有 alignment seeds 42/43/44 分层分析，但仅形成三个内部描述性 estimates，不是一般
  端到端方差、置信区间或显著性结果。
- 跨 alignment 分析中 candidate self-attention 的 Recall@1/MRR 方向均只有 2/3
  alignment 为正，不能将整体增益归因于候选交互。
- JESTR 和 GLMR 尚未在本代码库统一复现；当前正式采用 reported-only、不可直接比较的
  降级边界，不再作为本轮开放 P0。
- train--test 剔除敏感性和 train--val overlap-clean 模型选择审计均已完成；对 train--val 结果的解释仍必须保留“alignment 重训使其不是隔离因果效应”的限制。

## 8. 与 JESTR 和 GLMR 的比较

### 8.1 JESTR

论文：

> JESTR: Joint Embedding Space Technique for Ranking Candidate Molecules for the Annotation of Untargeted Metabolomics Data

位置：`paper/ref/JESTR.pdf`

主要机制：

- 分子：GCN。
- 谱图：1 Da binning 后使用 MLP。
- 使用 CMC/InfoNCE 对齐谱图和分子。
- 在最后 3% epoch 使用同分子式 PubChem candidates 做 regularization。
- 最终仍对每个候选独立计算 spectrum--molecule cosine similarity。
- 没有学习独立的 downstream ranker。

JESTR 自身论文在 MassSpecGym 报告：

- mass rank@1：15.13
- formula rank@1：11.85

GLMR 论文重新报告的 JESTR 数字略有不同，因此引用数字时必须注明来源。

### 8.2 GLMR

论文：

> Breaking the Modality Barrier: Generative Modeling for Accurate Molecule Retrieval from Mass Spectra

位置：`paper/ref/11055-AAAI26.ZhangY-AD.pdf`

主要机制：

1. 使用 Transformer spectrum encoder 和 ChemFormer molecule encoder 做 contrastive pre-retrieval。
2. 取 top-$K$ 分子作为 contextual priors。
3. 使用 cross-fusion 和 ChemFormer decoder 自回归生成 SMILES。
4. 将生成分子重新编码。
5. 按 generated-molecule 与候选分子的 cosine similarity 重排序。

GLMR 的核心是把跨模态检索转为分子--分子同模态相似度，但代价是：

- 需要自回归生成。
- 结果依赖生成质量和有效性。
- 推理成本更高，但当前项目尚未完成同硬件实测。

### 8.3 GLMR Table 1 外部结果

这些数字来自 GLMR 论文，不是本地复现：

| Candidate | Method | Recall@1 | Recall@5 | Recall@20 | MRR | MCES@1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mass | JESTR（reported；未复现） | 17.62 | 40.36 | 64.76 | 29.12 | 15.82 |
| mass | GLMR（reported；未复现） | 64.17 | 72.96 | 78.78 | 67.82 | 11.14 |
| formula | JESTR（reported；未复现） | 11.77 | 33.26 | 61.01 | 22.83 | 11.73 |
| formula | GLMR（reported；未复现） | 68.48 | 78.09 | 84.22 | 72.47 | 5.05 |

这些外部数值只记录 GLMR 论文给出的点估计，不用于对本地与外部流水线作相对性能排序，因为：

- 基础 retriever 不同。
- 分子 canonicalization 和候选处理细节可能不同。
- GLMR/JESTR 未在本地统一复现。
- canonical 数字与跨 alignment 描述性汇总使用不同统计单位，且外部协议均未匹配。

### 8.4 最稳妥的论文定位

| 方法 | 最终排序信号 | 候选间关系 | 生成分子 | downstream listwise ranker |
| --- | --- | --- | --- | --- |
| JESTR | spectrum--candidate cosine | 无 | 否 | 否 |
| GLMR | generated--candidate cosine | 生成器 cross-attention | 是 | 否 |
| SpecEmbedding-Rerank | explicit pair/retrieval features 的 residual learned score | 候选集合 Transformer 变体可选 self-attention | 否 | 是 |

可以主张：

- 直接、非生成式的监督残差 learning-to-rank。
- 结合显式谱图--候选交互特征、base score 和 base rank 进行重排序。
- 使用离线候选 embedding cache，可作为现有双塔检索器后的独立第二阶段。
- alignment seeds 42/43/44 中两个监督 reranker 在 12/12 个候选--模型聚合单元均改善对应 base；Transformer 相对 Pointwise 的 Recall@1/MRR 方向均仅 2/3 alignment 为正。

不要主张：

- 首次学习谱图--分子联合嵌入。
- 首次使用候选分子训练。
- candidate self-attention 已被证明是主要增益来源。
- 仅凭当前外部数字已经严格达到 SOTA。
- 在没有 latency 实验时声称一定比 GLMR 更快。

## 9. 转投稿件状态（ADMA 2026 历史背景）

官方征稿页面：

> https://adma2026.github.io/call_for_research_papers.html

重要日期：

- 论文截止：2026-07-17 AoE
- 通知：2026-09-02
- Camera-ready：2026-09-11
- 会议：2026-11-13 至 2026-11-15
- 补充材料：可选；若提交，截止为 2026-07-20 AoE（论文截止后 3 天）

关键要求：

- 英文。
- Springer LNAI/LNCS 格式。
- 总长度不超过 15 页，包含参考文献、致谢和 limitation claims。
- 双盲评审。
- PDF 和补充材料不得泄露作者、单位或身份元数据。
- 评审稿不得包含致谢和基金信息。
- 作者列表在提交时即为最终版本。
- 禁止同时投稿其他 archival venue。
- 评审期间不要新上传 arXiv 或个人主页。
- 使用 AI 生成或修改的文本必须披露。
- 若提交补充材料，CMT 单文件不超过 20 MB 且必须匿名；也可使用匿名仓库。

当前完成状态：

- 英文稿：`paper/main.tex`
- 中文稿：`paper/main_cn.tex`
- 参考文献：`paper/references.bib`
- 待办清单：`paper/ADMA2026_TODO.md`
- 上一版受跟踪 release evidence 对应 source commit `e597d89`，英文 17 页、中文 16 页；
  这是历史审计快照，不再代表当前初稿。
- 2026-08-20 从干净 source commit `033ac22`（论文实质 commit `076c5c5`）强制完整重建
  当前初稿：英文 17 页、479,007 bytes、SHA-256
  `68eaebb36afa5c9fe000b7b948b72080df390f81b1bc1901ff5ca92bfad7dd44`；中文 15 页、
  421,262 bytes、SHA-256
  `75870b7dde764559f9ba6626b82ecabbe0500a016200a7fd2212051158ffe197`。两份 PDF、工具链、
  页数和哈希清单已留存在 `paper/release/`。
- 第三轮独立审计基于远端 `5a1fe38` 复跑 88 项测试、临时 cwd 总管线、Ruff、artifact check
  和双语发布证据；结合用户明确“页数可以延后且不阻塞审计”的范围决定，本轮最终无 P0/P1
  阻断项。英文 17 页仅作为独立非阻断投稿整理事项，待确认转投 venue 页数规则后处理。
- 已忽略的 `paper/build/main.pdf` 与 `main_cn.pdf` 已由 release build 同步为 17/15 页；
  发布证据以受跟踪的 `paper/release/` 为准。
- 页数压缩：按用户决定延后到完稿后统一微调；最终投稿合规仍未完成。
- 英文 PDF 作者元数据：空
- 致谢和基金：未加入
- AI assistance disclosure：已移入 Introduction，并明确覆盖所有章节、代码编辑、实验编排和一致性审计；数值来自软件流水线，作者核验并承担全部责任。
- SpecEmbedding 增量审计：已完成；正文在引言、相关工作和方法中就地归因继承的峰序列 Transformer，区分原工作的重复谱图 SupCon + Tanimoto-MSE 与本文从零训练的跨模态目标，并将本文贡献限定为第二阶段非生成式残差 learning-to-rank。
- SpecEmbedding 书目信息：已按 ACS 正式页面补齐为 Analytical Chemistry 2025, 97(37), 20137--20146。
- 初稿内容收口：全部 25 条参考文献已逐项核对一手来源；Related Work 在保留全部引用的
  前提下净压缩约 80 个英文词；英文语言、证据边界及中英文一致性复核均已完成。
- 正式方法架构图：已使用共享 TikZ 源 `paper/figures/method_overview.tex` 替换中英文稿文本占位图；图中包含三阶段流程、两种 reranker 变体、残差跳连和训练/测试协议。
- overlap-clean mass/formula MCES@1：已完成（mass Base/seed-42 Set Transformer 15.3681/7.7057；formula 5.4389/3.0913；四项均为 17,556/17,556，batch 状态 `complete`）
- 最终实验 artifact manifest：已生成 `paper/adma2026_artifact_manifest.json`，60 个 canonical artifacts 校验通过；2026-08-17 加入评价身份协议和 source-commit `params.yaml` blob 后，SHA-256 为 `ad92596e1ca60e8edc6b7be594bde7cbe314f5d08a0421551ba88eaad8207875`；仅作内部 inventory，匿名附件不得直接打包其引用的原始 status/log。
- overlap-clean alignment/cache/reranker 全流水线：12/12 组 pointwise/Transformer 实验已完成
- alignment seeds 42/43/44 跨 alignment：36/36 组完成并形成分层分析；监督 reranking
  相对 base 的 12/12 聚合单元均改善，Transformer--Pointwise 方向不一致。
- canonical 核心组件消融：五项移除、两种候选、三个 reranker seeds 共 30/30 组完成；
  没有组件通过严格独立收益门槛。
- 匿名补充代码包：已从 source commit `c565d29` 以 40 文件显式 allowlist 构建；包体
  63,991 bytes，SHA-256 为
  `0e6b775a15ca0d54a58d6072ce1fbce0deb0951c0a357958a7d99e3f5c8baf52`。身份扫描
  零命中，独立解包后 7 项测试、Ruff、compileall 与合成
  `prepare -> train -> eval` CPU cold smoke 通过；证据见
  `reproducibility/anonymous-supplement-validation.yaml`。该 smoke 不复现论文指标，
  本轮也未再次从 lock 新建 Conda 环境。
- train--test 输入重叠敏感性：12/12 组完成；剔除 3 条重叠输入后主结论不变
- train--val 输入重叠：overlap-clean 审计已完成；clean 与历史 alignment best/stop epoch 相同，reranker 9/12 组选择不同
- 中心主张：已收窄为本地 exact-target-SMILES 协议下非生成式残差 learning-to-rank 的框架级
  收益；不主张官方 evaluator 等价、稳健 self-attention 独立收益或 SOTA。

当前双语内容初稿及其 release evidence 已经完成，但尚不是最终投稿版本。后续仅处理转投
venue 确认与格式适配、作者列表和 COI，以及定稿后的 PDF/源码/补充归档联合匿名检查；
页数调整暂不阻断。本阶段不新增实验、外部基线复现或论文主张。

## 10. 论文文件与结构

### 10.1 文件

- `paper/main.tex`：正式英文稿，LNCS/LNAI。
- `paper/main_cn.tex`：内部讨论中文稿，不用于正式投稿。
- `paper/figures/method_overview.tex`：中英文稿共享的 TikZ 正式方法架构图，参数 `0/1` 切换英文/中文标签。
- `paper/references.bib`：JESTR、GLMR、MassSpecGym、SpecEmbedding、MIST、CMSSP 等参考文献。
- `paper/ref/JESTR.pdf`：JESTR 原文。
- `paper/ref/11055-AAAI26.ZhangY-AD.pdf`：GLMR 原文。
- `paper/ref/SpecEmbedding.pdf`：原 SpecEmbedding 论文。
- `paper/ADMA2026_TODO.md`：投稿待办和实验记录。

### 10.2 当前英文稿结构

1. Abstract
2. Introduction
3. Related Work
4. Problem Formulation
5. Method
6. Experiments
7. Limitations and Responsible Evaluation
8. Conclusion

论文已经：

- 在机制层面对照 JESTR、GLMR 与本文排序方式，不作匹配性能比较。
- 本地结果与外部 reported results 视觉拆表；外部明确未复现、不可直接比较。
- 写明训练正例插入和测试无插入协议。
- 写明 formula-conditioned setting 的限制。
- 披露跨划分输入重叠，并报告 3 条 train--test 重叠的剔除敏感性结果。
- 完成 6 条 train--val 重叠的 overlap-clean 模型选择审计，并将 9/12 组 reranker 选择差异及“非隔离因果效应”限制同步到中英文稿。
- 使用共享 TikZ 矢量源完成三阶段正式方法图，并明确两种 reranker 变体、残差基础分数路径、训练时正例插入及验证/测试只重排协议。
- 完成 SpecEmbedding 既有贡献核对，在引言和方法处明确 backbone 继承、随机初始化及跨模态适配边界，并补齐正式卷期页。
- 将 AI 披露移入 Introduction，并按 ADMA 与 Springer 要求覆盖实际辅助范围和作者责任。
- 避免将联合嵌入本身作为本文创新。

## 11. 当前最高优先级待办

完整清单以 `paper/ADMA2026_TODO.md` 为准。当前优先事项：

1. 确认转投 venue，并据其模板、页数和披露规则完成格式适配；页数当前非阻断。
2. 确认最终作者列表、单位、COI 及 venue 要求的声明。
3. 当前初稿 release evidence 已生成；待 venue 格式与作者信息定稿后，重建最终提交副本并
   联合复扫 PDF、源码和匿名补充归档。
4. 不新增实验目标；top-$K$、效率、排名迁移及 JESTR/GLMR 独立复现均不属于当前待办，
   除非用户以后另行开启任务。

## 12. 已知风险与容易混淆的地方

### 12.1 设计默认值与已完成实验不同

`docs/reranker_solution_zh.md` 和当前 `params.yaml` 中可能以 `pre_top_k=256` 作为设计默认值，但已写入论文的主要结果使用 `topk40`。

引用实验时必须看运行目录和日志，不要仅根据默认配置推断。

### 12.2 旧设计文档中的预期数字已经过时

`docs/reranker_solution_zh.md` 中的“合理预期提升”是实现前估计，不是最终实验结果。实际结果以第 6 节和日志为准。

### 12.3 多种子作用域与 self-attention 证据

- canonical 与组件表中，reranker seeds 42/43/44 共享 seed-42 alignment checkpoint 和
  cache，因此其标准差只表示 reranker-level variation。
- 跨 alignment 分层分析已经覆盖 alignment seeds 42/43/44，并在每个 checkpoint 内聚合
  相同三个 reranker seeds；三个 alignment-level estimates 来自两个 source commits。
- 两种学习式 reranker 相对 base 的 12/12 聚合单元均改善，但 Set Transformer 相对
  Pointwise 在 mass/formula 的 Recall@1 与 MRR 上都只有 2/3 alignment 为正。
- 因此现有证据是内部描述性敏感性结果，不是置信区间、显著性或一般完整流水线稳定性；
  组件消融仍未跨 alignment 重复，不能将整体增益归因于候选间交互。

### 12.4 跨论文结果不能作为受控消融

我们的 base retriever 已经与 JESTR/GLMR 的 pre-retriever 不同，因此：

- reranker 相对本地 base 的提升可以归因于 reranker。
- 本地端到端结果与 GLMR 的差异不能全部归因于 reranker。

### 12.5 输入重叠审计的解释边界

3 条 train--test 输入重叠的剔除敏感性评估已完成，结果变化可忽略。6 条 train--val 输入重叠的 overlap-clean 模型选择审计也已完成：clean 与历史 alignment 选择 epoch 相同，但 reranker 9/12 组 best/stop epoch 对不同，反映了新旧完整流水线的模型选择敏感性。

本次审计重训了 alignment，新旧流水线差异还包含训练随机性、checkpoint 和 cache 变化。可以声称“重叠问题已完成敏感性审计且 clean 主结论保持”，但不能声称新旧测试差异是删除 6 条 validation 查询造成的隔离因果效应。

### 12.6 匿名补充材料风险

仓库中可能存在：

- 用户名或本地绝对路径。
- Git 远端地址和提交历史。
- README 中的公开仓库、Figshare 或演示链接。
- checkpoint、日志或配置中的本地目录。

匿名补充材料应使用无 `.git` 历史的独立归档，并执行身份字符串扫描。

2026-08-20 已落实为 `reproducibility/anonymous-allowlist.txt` 和确定性 ZIP builder；
归档默认生成到 Git 忽略的 `dist/`，不把提交 artifact 反向纳入源码历史。当前包已通过
用户名、hostname、Git remote/author、绝对路径、邮箱/ORCID、GPU UUID、公开托管链接和
敏感文件类型扫描。提交前仍需从固定 source commit 重建最终副本，并与最终 PDF/源码一起
复扫；当前验证使用既有事后验收环境，不等于又完成了一次从 lock 的全新安装。

### 12.7 AI 披露

当前英文稿已将全局声明置于 Introduction 内，说明 OpenAI Codex 用于所有章节的文字
起草/修改，以及代码编辑、实验编排和结果一致性审计；同时明确实验数值来自软件流水线、
作者核验 AI 辅助内容并承担全部责任。这与 2026-07-14 核对的 ADMA CFP 和 Springer
Nature book AI policy 一致。若实际使用范围继续扩大，提交前必须同步更新声明。

### 12.8 SpecEmbedding 继承边界

SpecEmbedding 原工作是谱图--谱图表示学习：以同分子增强/重复谱图作为监督对比正例，
并额外拟合 Tanimoto 结构相似度。本文只继承峰序列 Transformer 架构，不加载其预训练
checkpoint，也不沿用原谱图--谱图目标；第一阶段改为从零训练的谱图--分子跨模态对齐，
论文主张则集中在第二阶段残差 learning-to-rank。方法描述必须持续保留这一归因边界。

## 13. 关键提交历史

- `a8d7ecc`：添加重排序方案设计文档。
- `638f029`：添加分子重排序训练与评估流程。
- `49c6c10`：统一重排序配置与公共工具。
- `d4c1f70`：抽取评估与对齐公共工具。
- `9cd4754`：添加重排序流水线脚本。
- `c1dfad4`：支持重排序候选 top-k 参数。
- `7eed946`：添加论文 LaTeX 框架。
- `59cbab1`：添加中文论文稿。
- `a2ccd8c`：加入参考论文 PDF。
- `1669f4b`：按 ADMA 2026 要求完善中英文论文。
- `91877d6`：添加 ADMA 2026 投稿待办清单。
- `96fbaa9`：回填质量候选 MCES@1 结果。
- `cafb70c`：支持多随机种子 pointwise/Transformer 批量实验。
- `4c22bc6`：回填多种子结果并将论文核心结论收窄为非生成式残差学习排序。
- `ddd5153`：增加 feature/loss 批量消融、输入重叠敏感性评估入口和相关测试。
- `b20a2e8`：添加可恢复的 train--test 输入重叠敏感性评估脚本及审计检查。
- `4528fab`：回填 train--test 输入重叠敏感性结果。
- `a2280d2`：支持 train--val 输入排除、alignment 选择记录和 overlap-clean 全流水线脚本。

## 14. 维护本记忆文档

发生以下事件时更新本文档：

- 最终实验参数改变。
- 新增主要数据集或候选协议。
- 完成新的随机种子、消融或 baseline。
- 论文主张、标题或投稿 venue 改变。
- 发现数据泄露、评估偏差或复现差异。
- 提交 camera-ready 或公开代码。

每次更新至少修改：

- 顶部日期和当前 commit。
- 第 6 节实验结果。
- 第 9 节投稿状态。
- 第 11 节最高优先级待办。
- 第 12 节风险。
