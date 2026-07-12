# SpecEmbedding 项目记忆

最后更新：2026-07-12

本次更新前代码状态：

- 分支：`dev`
- 最新提交：`b20a2e8 feat(rerank): 添加输入重叠敏感性评估脚本`
- 工作区：干净

本文档用于后续开发、实验和论文协作时快速恢复上下文。它是内部协作材料，不应直接放入 ADMA 双盲补充材料。若本文档与代码、`params.yaml` 或实验日志冲突，以代码和原始日志为准。

## 1. 当前总目标

项目已经从“谱图嵌入”扩展为一个完整的 MS/MS-to-molecule retrieval 系统：

1. 使用 SpecEmbedding 编码 MS/MS 谱图。
2. 使用 GINE 编码候选分子图。
3. 通过跨模态对比学习对齐谱图和分子。
4. 使用基础相似度从候选库中检索 top-$K$ 分子。
5. 使用非生成式残差 reranker 重新排序 hard candidates，并受控比较 pointwise 与 set-aware 变体。

当前论文目标是投稿 ADMA 2026，主题选择为 `Data mining for bioinformatics`。英文标题暂定为：

> Non-Generative Learning to Rerank for MS/MS-Based Molecule Retrieval

三随机种子受控消融完成后，论文需要强调的证据边界是：

- 在已有跨模态检索器之上直接优化候选列表排序。
- 结合 base score、base rank 和显式谱图--候选交互特征。
- 使用 residual listwise reranking，在不生成分子的情况下修正基础排序。
- pointwise 与 set-aware Transformer 都大幅优于固定 base，但 self-attention 没有显示稳定额外收益。
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
- 当前主要开发分支为 `dev`。
- 本环境没有安装 `gh`，但普通 `git push` 可用；无法自动创建 GitHub/GitLab PR。
- LaTeX 编译产物统一放入 `paper/build/`，该目录已被 Git 忽略。

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

## 6. 已完成的主要实验

### 6.1 实验设置

- 数据集：MassSpecGym
- 数据划分：官方 structure-disjoint split
- 训练谱图：194,119
- 验证谱图：19,429
- 测试谱图：17,556
- 唯一分子：32,010
- 基础 checkpoint：`checkpoints_align/d4c1f70_massspecgym_nopretrain/best_model_stage2.pth`
- alignment：无额外 SpecEmbedding 预训练
- rerank top-$K$：40
- alignment 随机种子：42（固定 checkpoint）
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

除 MCES@1 外，表中数值均以百分比表示；MRR 在日志中原始范围为 $[0,1]$，表中乘以 100。MCES@1 是结构距离，保留原始量纲，越低越好。

下表是代表性 seed-42 Set Transformer 结果，也是 MCES@1 的计算对象：

| Candidate | Method | Upper bound | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR | MCES@1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mass | Base | 82.15 | 43.78 | 61.10 | 68.43 | 75.55 | 51.96 | 16.42 |
| mass | Set Transformer (seed 42) | 82.15 | 67.91 | 76.00 | 78.44 | 80.45 | 71.62 | 8.02 |
| formula | Base | 87.30 | 58.49 | 71.49 | 76.86 | 81.81 | 64.65 | 6.14 |
| formula | Set Transformer (seed 42) | 87.30 | 74.33 | 80.64 | 82.89 | 85.07 | 77.27 | 3.13 |

mass MCES@1 的原始输出为 Base 16.4199、代表性 seed-42 Set Transformer 8.0209；表中按两位小数展示。

固定同一 alignment checkpoint 和 top-40 cache，reranker seeds 42/43/44 的均值 $\pm$ 样本标准差为：

| Candidate | Model | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mass | Pointwise | 67.89±0.44 | 75.79±0.19 | 78.36±0.11 | 80.63±0.05 | 71.53±0.35 |
| mass | Set Transformer | 67.75±0.31 | 75.88±0.18 | 78.41±0.12 | 80.63±0.15 | 71.52±0.24 |
| formula | Pointwise | 73.77±0.19 | 80.55±0.12 | 82.93±0.23 | 84.97±0.15 | 76.88±0.16 |
| formula | Set Transformer | 73.98±0.84 | 80.27±0.54 | 82.75±0.33 | 85.07±0.05 | 76.95±0.69 |

受控结论：

- 两种学习式 reranker 都大幅优于固定 base retriever。
- Set Transformer 相对 Pointwise 的 Recall@1 平均差异为 mass $-0.13$ 个百分点、formula $+0.21$ 个百分点。
- Transformer 在 6 个成对的 seed--candidate Recall@1 比较中只获胜 3 次，MRR 差异也接近于零。
- 当前结果不支持“candidate self-attention 带来稳定独立收益”；不应将整体 reranking 提升归因于 self-attention。
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
有效而 candidate self-attention 无稳定额外收益的主结论不变。该批次按协议
未重算 MCES@1。

代表性 seed-42 Set Transformer 相对基础检索器：

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

- mass：92.59%
- formula：89.40%

训练 cache 插入后 labeled coverage 为 100%；验证和测试没有插入。

### 6.3 结果路径

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
train--test 输入重叠不会实质改变当前结果。剩余 6 条 train--val 输入重叠
仍需评估其对 validation MRR、early stopping 和 checkpoint selection 的影响。

### 7.2 必须明确披露的限制

- formula candidate 设置假设已知真实分子式，应称为 formula-conditioned retrieval，不能描述成完全开放的未知分子鉴定。
- reranker 是 closed-library 方法，不能生成候选库之外的新分子。
- reranker 已使用 3 个随机种子，但 alignment 仍只有 seed 42，当前误差不覆盖端到端方差。
- candidate self-attention 相对 pointwise 对照没有稳定优势，不能将整体增益归因于候选交互。
- JESTR 和 GLMR 尚未在本代码库统一复现。
- train--test 输入重叠敏感性已通过，但 6 条 train--val 输入重叠对模型选择的影响尚未量化。

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
| mass | JESTR | 17.62 | 40.36 | 64.76 | 29.12 | 15.82 |
| mass | GLMR | 64.17 | 72.96 | 78.78 | 67.82 | 11.14 |
| formula | JESTR | 11.77 | 33.26 | 61.01 | 22.83 | 11.73 |
| formula | GLMR | 68.48 | 78.09 | 84.22 | 72.47 | 5.05 |

当前本地 reranker 数字高于 GLMR 论文报告值，但不能直接据此声称严格超越，因为：

- 基础 retriever 不同。
- 分子 canonicalization 和候选处理细节可能不同。
- GLMR/JESTR 未在本地统一复现。
- 当前多种子只覆盖 reranker，alignment 仍固定为 seed 42。

### 8.4 最稳妥的论文定位

| 方法 | 最终排序信号 | 候选间关系 | 生成分子 | downstream listwise ranker |
| --- | --- | --- | --- | --- |
| JESTR | spectrum--candidate cosine | 无 | 否 | 否 |
| GLMR | generated--candidate cosine | 生成器 cross-attention | 是 | 否 |
| SpecEmbedding-Rerank | explicit pair/retrieval features 的 residual learned score | set-aware 变体可选 self-attention | 否 | 是 |

可以主张：

- 直接、非生成式的监督残差 learning-to-rank。
- 结合显式谱图--候选交互特征、base score 和 base rank 进行重排序。
- 轻量级、可缓存、可插入现有双塔检索器。
- 三种子受控实验中 pointwise 与 Set Transformer 表现相当。

不要主张：

- 首次学习谱图--分子联合嵌入。
- 首次使用候选分子训练。
- candidate self-attention 已被证明是主要增益来源。
- 仅凭当前外部数字已经严格达到 SOTA。
- 在没有 latency 实验时声称一定比 GLMR 更快。

## 9. ADMA 2026 投稿状态

官方征稿页面：

> https://adma2026.github.io/call_for_research_papers.html

重要日期：

- 论文截止：2026-07-17 AoE
- 通知：2026-09-02
- Camera-ready：2026-09-11
- 会议：2026-11-13 至 2026-11-15
- 补充材料：论文截止后 3 天

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
- 补充材料单文件不超过 20 MB，或使用匿名仓库。

当前完成状态：

- 英文稿：`paper/main.tex`
- 中文稿：`paper/main_cn.tex`
- 参考文献：`paper/references.bib`
- 待办清单：`paper/ADMA2026_TODO.md`
- 英文编译：成功，10 页
- 中文编译：成功，9 页
- 英文 PDF 作者元数据：空
- 致谢和基金：未加入
- AI assistance disclosure：已加入
- mass MCES@1：已完成（Base 16.42，代表性 seed-42 Set Transformer 8.02）
- 固定 alignment 的 reranker 多种子：12/12 组 pointwise/Transformer 实验已完成
- train--test 输入重叠敏感性：12/12 组完成；剔除 3 条重叠输入后主结论不变
- train--val 输入重叠：6 条尚待处理，其对 validation-based 模型选择的影响未量化
- 中心主张：已收窄为非生成式残差 learning-to-rank，不再将 self-attention 作为已验证增益来源

当前论文仍不是最终可提交版本；未完成消融、外部基线复现、方法图与投稿检查已明确记录为
当前证据范围之外的待办。

## 10. 论文文件与结构

### 10.1 文件

- `paper/main.tex`：正式英文稿，LNCS/LNAI。
- `paper/main_cn.tex`：内部讨论中文稿，不用于正式投稿。
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

- 直接比较 JESTR、GLMR 和本文排序机制。
- 分开标注本地结果与外部 reported results。
- 写明训练正例插入和测试无插入协议。
- 写明 formula-conditioned setting 的限制。
- 披露跨划分输入重叠，并报告 3 条 train--test 重叠的剔除敏感性结果。
- 将 6 条 train--val 重叠对模型选择的潜在影响保留为明确限制。
- 避免将联合嵌入本身作为本文创新。

## 11. 当前最高优先级待办

完整清单以 `paper/ADMA2026_TODO.md` 为准。当前优先事项：

1. 处理 6 条 train--val 完全相同输入，评估其对 validation MRR、early stopping 和 checkpoint selection 的影响。
2. 完成 feature、loss、候选顺序和 top-$K$ 消融。
3. 尽可能在统一协议下复现 JESTR 和 GLMR。
4. 若不能复现，保留 `reported` 标记并撤回严格 SOTA 表述。
5. 测量参数量、显存、cache 时间、rerank latency 和端到端 latency。
6. 制作正式方法图，替换 LaTeX 文本框。
7. 对匿名补充材料执行身份信息清理。

## 12. 已知风险与容易混淆的地方

### 12.1 设计默认值与已完成实验不同

`docs/reranker_solution_zh.md` 和当前 `params.yaml` 中可能以 `pre_top_k=256` 作为设计默认值，但已写入论文的主要结果使用 `topk40`。

引用实验时必须看运行目录和日志，不要仅根据默认配置推断。

### 12.2 旧设计文档中的预期数字已经过时

`docs/reranker_solution_zh.md` 中的“合理预期提升”是实现前估计，不是最终实验结果。实际结果以第 6 节和日志为准。

### 12.3 多种子作用域与 self-attention 证据

- seeds 42/43/44 只重训 reranker，所有实验共享同一 seed-42 alignment checkpoint 和 cache。
- 因此已报告的标准差是 reranker-level uncertainty，不是 end-to-end uncertainty。
- Pointwise 与 Set Transformer 表现相当，不能将增益归因于候选间 self-attention。
- 如需主张完整流水线稳定性，必须重训多个 alignment seeds 并为每个 checkpoint 重建 cache。

### 12.4 跨论文结果不能作为受控消融

我们的 base retriever 已经与 JESTR/GLMR 的 pre-retriever 不同，因此：

- reranker 相对本地 base 的提升可以归因于 reranker。
- 本地端到端结果与 GLMR 的差异不能全部归因于 reranker。

### 12.5 输入重叠尚未全部闭环

3 条 train--test 输入重叠的剔除敏感性评估已完成，结果变化可忽略；但
6 条 train--val 输入重叠仍可能影响 early stopping 和 checkpoint selection。
完成对应模型选择审计前，不能声称跨划分输入重叠问题已全部解决。

### 12.6 匿名补充材料风险

仓库中可能存在：

- 用户名或本地绝对路径。
- Git 远端地址和提交历史。
- README 中的公开仓库、Figshare 或演示链接。
- checkpoint、日志或配置中的本地目录。

匿名补充材料应使用无 `.git` 历史的独立归档，并执行身份字符串扫描。

### 12.7 AI 披露

当前稿件使用一个全局声明说明 OpenAI Codex 用于全文起草和语言修改。ADMA CFP 的措辞较严格，提交前应再次确认该全局披露是否足以覆盖“任何使用 AI 文本的章节”。

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
