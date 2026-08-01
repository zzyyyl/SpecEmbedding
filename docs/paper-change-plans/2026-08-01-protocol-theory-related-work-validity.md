# 论文修改计划：协议澄清、理论分解、直接相关工作与效度边界

- 状态：`已执行`
- 创建日期：2026-08-01
- 最后更新：2026-08-01
- 负责人：Codex
- 关联论文：`paper/main.tex` / `paper/main_cn.tex`
- 关联参考文献：`paper/references.bib`
- 基线 commit：`9985aee docs(workflow): 规范论文修改计划流程`
- 计划约束：不新增模型训练，不重新运行模型评价，不改动既有结果数值；允许代码与固定工件的静态审计、代数推导、文献核验和 LaTeX 编译

## 1. 修改目标与动机

本轮修改落实前一轮“在不进行新实验的前提下完善论文”的审阅结论。目标不是继续堆叠通用背景，而是补齐论文的四个证据闭环：

1. 准确限定候选池、排名和正例身份协议，消除“完整化学库排名”及“完全等同官方评价协议”的过强暗示；
2. 用覆盖率—条件排序能力分解形式化 reranker 的能力边界；
3. 补齐质谱候选重排的直接前序工作，收窄并增强本文创新边界；
4. 补足第一阶段复现信息、方法性质及内部/构念/协议/外部效度威胁。

修改后，读者应能准确回答：模型实际在哪个候选集合内排序、怎样定义唯一正例、reranker 能和不能改善什么、本文相对既有重排工作的具体差异、结果能支持多强的结论，以及复现当前流水线还需要哪些信息。

## 2. 当前证据与问题定位

### 2.1 候选池不是完整 PubChem 或完整化学库

- `prepare_rerank_cache.py:358--364` 读取每条查询对应的 MassSpecGym 候选文件。
- `prepare_rerank_cache.py:140--173` 只对该查询的候选集合打分并保留 top-$K$。
- MassSpecGym 论文明确将每条检索候选集合限制为至多 256 个分子，而不是对 PubChem 全库排序。
- 因此当前英文稿中的 `full-library base rank` 必须改为 `base rank within the full supplied candidate pool`；中文同步改为“所提供完整候选池中的基础排名”。

### 2.2 本地 positive identity 与 MassSpecGym 当前默认实现不同

- `prepare_rerank_cache.py:38--56` 按原始 SMILES 字符串建立唯一索引。
- `prepare_rerank_cache.py:136--147` 通过 `smiles_to_idx.get(true_smiles)` 确定目标索引并按整数索引去重。
- `prepare_rerank_cache.py:177--200` 只把 `top_indices == true_idx` 的一个位置设为正例。
- MassSpecGym 当前参考实现的 `RetrievalDataset` 默认使用 `MolToInChIKey()`，而该转换默认 `twod=True`；同一二维连接性的多个候选可同时为正例。
- 本轮不重新评价，因此不能量化身份规则差异对 Recall/MRR 的影响，也不能恢复“官方指标完全等价”的主张。安全做法是明确报告：使用 MassSpecGym 划分和候选文件，但本地 cache/evaluator 按 exact-SMILES 字符串定义唯一正例。
- 不写“该差异必然使结果更低”；只写“协议不完全等价、影响未量化”。

官方参考实现核验版本为 commit `4f501b3207c9f466bc745a0353f61dbfaf150d32`（访问日期 2026-08-01）：

- `RetrievalDataset` 默认标签转换及候选布尔标签：<https://github.com/pluskal-lab/MassSpecGym/blob/4f501b3207c9f466bc745a0353f61dbfaf150d32/massspecgym/data/datasets.py>
- `MolToInChIKey(twod=True)`：<https://github.com/pluskal-lab/MassSpecGym/blob/4f501b3207c9f466bc745a0353f61dbfaf150d32/massspecgym/data/transforms.py>

### 2.3 候选池包含目标与训练后置强制加入是两个不同操作

- `prepare_rerank_cache.py:140--147` 对所有划分以 `[true_smiles, *candidates]` 构造待评分池并按 exact SMILES 去重。
- 静态文件审计确认本轮冻结的两个候选工件中，32,010 个 molecule-keyed 列表均已包含 exact target，且没有 exact-SMILES 重复，因此上述 prepend 在当前工件中没有实际增加候选。
- `prepare_rerank_cache.py:177--195` 的 post-retrieval forcing 只在训练时启用：若目标不在 base top-$K$，才替换截断列表末项并保留其 full-pool rank。
- 验证/测试中目标未进入 top-40 时 label 缺失，Recall/MRR 对该查询记零；当前论文对此部分的表述基本准确，但应避免把两个层次混为一谈。

### 2.4 当前问题定义只给出了上界，尚未形式化条件排序能力

现稿定义了 top-$K$ 覆盖率 $U_K$，并在 Discussion 中用 $mathrm{R@1}/U_{40}$ 解释条件成功率，但没有正式定义该量。加入

\[
A_{k\mid K}=
\frac{\sum_n \mathbb{I}[m_n^+\in\mathcal C_{s_n},
\operatorname{rank}_g(m_n^+)\le k]}
{\sum_n\mathbb{I}[m_n^+\in\mathcal C_{s_n}]},
\qquad
\mathrm{Recall@}k=U_KA_{k\mid K}\le U_K
\]

即可形成代数闭环。该式是指标定义的恒等分解，不需要新实验或外部引用。

### 2.5 直接重排先例不足，创新边界仍可更精确

现稿已引用 MS2Query，但只将其描述为 analogue search，没有说明其 top-2000 检索后使用五个特征和随机森林重排。现稿也缺少：

- MetFusion：融合 MetFrag 与 MassBank 谱库证据以改善候选排名；
- LC-MS2Struct：将已有 MS/MS scorer 与同一 LC 实验中的保留顺序结合，进行结构化候选重排。

这些文献说明“候选分数修正/第二阶段重排”本身并非首次出现。本文创新边界应改为：在冻结谱图—分子跨模态表示和检索先验之上，对 query-specific spectrum-to-structure top-$K$ 列表进行非生成式残差学习排序，并可选地建模同一查询内部候选交互；不依赖谱库邻居或保留时间。

### 2.6 第一阶段复现信息不完整

代码与 `params.yaml` 的静态核验结果：

- 未加载预训练谱图 checkpoint，实际跳过冻结谱图塔的 stage 1，只执行最多 100 epoch 的端到端 stage 2；
- 谱图 Transformer 为 4 层、16 heads、宽度 512；分子 GINE 为 4 层、隐藏宽度 128、dropout 0.2；
- AdamW，batch size 128，学习率 $10^{-4}$，weight decay $10^{-4}$，cosine schedule，patience 5，gradient-norm clipping 1；
- 数据集长度为唯一分子标签数的 10 倍，每次按标签随机抽取一条关联谱图；
- 谱图增强和图增强分别以 0.5 概率触发；谱图增强删除归一化强度低于 0.3 的候选峰数量的 20%（向下取整）并进行 $\pm15\%$ 强度扰动；图增强为 10% node dropout 和 10% directed edge-entry dropout；
- 验证关闭增强，但仍按分子标签随机抽取关联谱图，因此 checkpoint 选择包含由固定随机流控制的 replicate-spectrum sampling variation。

准确性边界：图边条目是对双向 `edge_index` 条目独立采样，不能写成保持无向键对称的 `bond dropout`。

### 2.7 方法分数与统计解释仍需限定

- $b_i$ 同时进入输入特征 $\mathbf{x}_i$ 和残差 shortcut，因此 $\alpha$ 不能单独解释为“保留基础分数的比例”。
- listwise cross-entropy 是一正例、多候选的 top-one listwise likelihood，不监督负例之间的完整全序。
- 两项损失均对同一查询全部 $q_i$ 加相同常数保持不变，因此输出是相对排序分数，不是校准概率。
- 三个 reranker seeds 的样本标准差和 paired wins 只作描述性统计，不是置信区间或显著性检验。

## 3. 修改范围

### 3.1 涉及文件与章节

- [x] `paper/main.tex`：摘要、引言贡献边界、相关工作、问题定义、候选协议、方法性质、实现细节、指标协议、Discussion、Limitations。
- [x] `paper/main_cn.tex`：与英文稿逐项同步，保持数字、公式、引用键和证据强度一致。
- [x] `paper/references.bib`：新增 MetFusion、LC-MS2Struct、conformal molecular retrieval 三条已核验期刊文献。
- [x] 本计划文件：执行过程中更新步骤、偏差、验证结果和论文修改 commit。

### 3.2 明确不做的事项

- 不训练任何模型，不生成新 checkpoint/cache，不重新运行 Recall、MRR 或 MCES 评价。
- 不修改任何已有实验结果、表格数值或图。
- 不重新计算 2D InChIKey 多正例指标；只披露本地 exact-SMILES 协议及与参考实现的差异。
- 不声称 PubChem/full-library 排名、官方 leaderboard 指标等价、SOTA、统计显著性、校准置信度、开放集能力或 self-attention 因果收益。
- 不新增正文表格、图、附录或冗长 Future Work，以控制 15 页限制。
- 不加入 DreaMS、ListMLE、Scheubert FDR、MSI reporting standards 等次优先文献；其中 Scheubert 工作针对谱库匹配而非当前结构候选重排，当前页数预算下由更直接的 conformal retrieval 文献承担单查询可靠性讨论。
- 不修改代码、配置、数据或实验工件。

## 4. 具体内容设计

### 4.1 摘要与引言

- 将英文摘要的结果写成 `mean $\pm$ sample-SD Recall@1`；中文同步为“三种子均值 $\pm$ 样本标准差”。
- 将引言中 `Under MassSpecGym's official candidate protocol` 降级为：使用 MassSpecGym 划分和候选文件，并采用本地 cache 实现的 exact-SMILES 身份规则。
- 保持主要贡献不变，不在摘要加入新的背景或未来工作。

### 4.2 Related Work 与方法定位

用紧凑段落替换现有 MS2Query 开头段落的一部分，而不是新增小节：

1. MetFusion 融合 in-silico fragmentation 与谱库证据以修正候选排序；
2. MS2Query 先按 MS2DeepScore 选 top 2000，再用五特征随机森林重排谱库匹配；
3. LC-MS2Struct 将 MS/MS scorer 与保留顺序结合，在同一 LC 实验内联合排序结构注释；
4. 明确这些工作使用的候选对象和额外证据不同，本文不使用谱库邻居或保留时间；
5. 将定位句改为“贡献不是 reranking per se，而是冻结跨模态表示上的 query-specific、非生成式残差重排；候选自注意力只是被评价的可选项”。

不得把 MetFusion 称为神经重排器，也不得把 LC-MS2Struct 的跨分析物保留顺序依赖写成单查询候选 self-attention。

### 4.3 Problem Formulation

- 保留 $U_K$ 定义。
- 紧接其后加入 $A_{k\mid K}$ 和 $\mathrm{Recall@}k=U_KA_{k\mid K}\le U_K$，英文标签 `eq:coverage_decomposition`，中文标签 `eq:coverage_decomposition_cn`。
- 说明固定 cache 的 reranker 只能改变 $A_{k\mid K}$，不能改变 $U_K$。
- Discussion 改为直接引用 $A_{1\mid40}$，减少重复解释 $mathrm{R@1}/U_{40}$。

### 4.4 Candidate Construction and Training Protocol

- `candidate library` 在描述本地实际排序范围时改为 `supplied candidate pool`。
- `full-library base rank` 改为 `base rank within the full supplied candidate pool` / `full-pool base rank`。
- 保留“严格高于目标分数的候选数加一”的排名定义；如篇幅允许，说明该 strict-greater convention 对并列分数赋予相同的最佳排名。
- 区分候选池中 exact target 的存在与训练专用 post-top-40 forcing，不暗示验证/测试执行截断后强制加入。

### 4.5 Residual Reranker 与 Learning Objective

- 将置换等变性限定为 `At deterministic inference`。
- 在残差式后说明：由于 $b_i$ 也进入 $\mathbf{x}_i$，残差分支可学习额外的 score-dependent effect，故不对 $\alpha$ 作机制解释。
- 在损失后说明：listwise 项是 top-one likelihood，不规定负例完整顺序；两项损失对 query-wise additive constant 不变，输出不是校准概率。
- 保留现有解析复杂度和“不作实测效率声明”的边界，不新增延迟结论。

### 4.6 Dataset、Implementation Details 与 Metrics

- 将 `official retrieval settings` 等容易暗示完整官方指标等价的措辞改为 `MassSpecGym-supplied candidate settings/files`。
- 以紧凑文字报告冻结候选工件的静态事实：候选池至多 256；测试查询在 mass/formula 下截断前平均 253.88/165.75，少于 40 个候选的查询为 1/3,159；截断后平均 40.00/35.73。明确这是固定文件描述，不是模型实验。
- 说明本地 cache 按 exact target-SMILES equality 定义唯一正例，因此 Recall/MRR 是 MassSpecGym 候选文件上的 exact-SMILES 本地协议，不是参考 evaluator 身份规则完全等价的复现。
- 补齐第一阶段架构、优化器、采样和增强设置，控制为一个紧凑段落。
- 不把当前环境中的 `myopic_mces` 版本扩展成“完整软件环境已冻结”；现有 `environment.yml` 未锁定 PyTorch、RDKit、PyG 和 `myopic_mces` 的确切版本，Limitations 中简要披露该复现边界。

### 4.7 Discussion

- 用正式定义的 $A_{1\mid40}$ 替换“R@1/U40 正好是……”的重复推导。
- 保留现有 headroom 分析及“不是新增估计、不能作因果分离”的限定。
- 不增加新的结果数字或跨模型优越性解释。

### 4.8 Limitations and Responsible Evaluation

以替换和压缩现有段落为主，按以下四类组织，但是否显示 `\paragraph{}` 标题由最终页数决定：

1. **Internal validity**：三种子样本标准差与 paired wins 只是描述性统计；共享一个 alignment checkpoint；候选集合模型与 pointwise 容量不匹配。
2. **Protocol and reproducibility**：训练 forcing 改变训练列表分布且不能解决 $1-U_K$；exact-SMILES 与 2D InChIKey 默认规则不等价且影响未量化；exact-signature 审计不能排除近重复或相关 provenance；依赖版本未完全锁定。
3. **Construct validity**：闭集候选池按构造含目标，未评价数据库缺失目标时的开放集拒识；top-$k$ 是数据集级排序统计，不是单谱可靠性；输出无概率校准或 abstention。
4. **External validity**：训练近似按分子平衡、最终指标按谱图查询加权；外部基线未在相同协议复现；采集条件、其他检索器、身份规则、候选池构造及外部库迁移未经验证。

加入一条谨慎的 conformal prediction 讨论：已有正式工作可在 exchangeability 且目标属于候选集合等假设下，将候选排名转换为谱图特异预测集；该保证不能自动处理本文目标未进入 top-40 的覆盖失败。不得写成当前模型已经具有保证。

## 5. 引用文献与真实性核验

| 引用键 | 文献 | 支持的论断 | 一手来源 | 核验状态 |
|---|---|---|---|---|
| `gerlich2013metfusion` | Gerlich and Neumann, “MetFusion: integration of compound identification strategies,” *Journal of Mass Spectrometry* 48(3):291–298, 2013 | 融合 MetFrag 和 MassBank 谱库证据改善候选排名；不是神经重排 | <https://doi.org/10.1002/jms.3123> | 已核验出版社题名、作者、卷期、页码、年份、DOI 和摘要 |
| `dejonge2023ms2query` | de Jonge et al., “MS2Query: reliable and scalable MS2 mass spectra-based analogue search,” *Nature Communications* 14:1752, 2023 | MS2DeepScore top-2000 后用五特征随机森林重排；已有 BibTeX | <https://doi.org/10.1038/s41467-023-37446-4> | 已核验正文 workflow；现有条目准确 |
| `bach2022lcms2struct` | Bach, Schymanski, and Rousu, “Joint structural annotation of small molecules using liquid chromatography retention order and tandem mass spectrometry data,” *Nature Machine Intelligence* 4:1224–1237, 2022 | 将已有 MS/MS scorer 与观测保留顺序结合进行结构化候选排序 | <https://doi.org/10.1038/s42256-022-00577-2> | 已核验出版社题名、作者、页码、年份、DOI 和方法边界 |
| `rakhshaninejad2026conformal` | Rakhshaninejad et al., “Reliable Molecular Retrieval from Mass Spectra Using Conformal Prediction,” *Journal of Chemical Information and Modeling* 66(10):5788–5800, 2026 | 在 exchangeability/候选包含目标等假设下提供谱图特异预测集和边际覆盖；该保证不会自动覆盖本文 top-40 的覆盖失败 | <https://doi.org/10.1021/acs.jcim.6c00727> | 已核验为 2026-05-11 正式发表期刊论文，不是预印本 |
| `bushuiev2024massspecgym` | Bushuiev et al., MassSpecGym, NeurIPS 2024 | benchmark 候选集合、每条至多 256、检索任务和划分 | <https://doi.org/10.52202/079017-3491> | 现有条目准确；已核验论文任务定义 |
| `schymanski2014confidence` / `hoffmann2022cosmic` | 现有置信度与谱图库缺失结构文献 | 区分 computational prioritization 与确认鉴定；支撑现有限制 | 现有 DOI | 已在上一轮核验，保留现有条目与有限论断 |

新增 BibTeX 必须逐字符核对作者变音符号：`J{\"u}rgens`、`F{\'e}lix` 等。本轮实际新增的三条文献均为正式发表论文，不标记为预印本。

## 6. 实验与计算边界

- 是否新增训练实验：否。
- 是否新增评价运行：否。
- 是否重新计算 2D InChIKey 多正例指标：否。
- 是否进行静态数据统计：仅使用已经完成并与冻结候选工件 SHA-256 绑定的候选列表描述统计；不新增模型输出。
- 是否进行代数计算：仅加入指标恒等分解并复用现有结果中的条件成功率解释。
- 是否只修改文字、公式或引用：是，另执行 LaTeX 编译验证。

任何需要重新生成 cache、运行 `eval_rerank.py`、重新计算 Recall/MRR/MCES、训练模型或改变既有结果数值的情况，都超出本计划，必须将状态改为 `已偏离待确认` 并询问用户。

## 7. 分步执行清单

- [x] 步骤 1：创建并完整核验本计划，保持状态 `未执行`；确认论文尚未修改。
- [x] 步骤 2：将计划状态改为 `执行中`，记录开始时间。
- [x] 步骤 3：在 `references.bib` 新增三条已核验文献，检查键唯一、元数据和变音符号。
- [x] 步骤 4：修改英文 Related Work、Positioning、协议术语和正例身份说明。
- [x] 步骤 5：修改英文 Problem Formulation、Residual Reranker、Learning Objective 和 Discussion 的理论性质。
- [x] 步骤 6：补充英文第一阶段复现细节并重写/压缩 Limitations。
- [x] 步骤 7：逐项同步中文稿，核对公式标签、数字、引用键和结论强度。
- [x] 步骤 8：核验所有新增引用及 BibTeX，搜索并消除不准确的 `full-library` / `official protocol` 等措辞。
- [x] 步骤 9：编译英文和中文稿，检查页数、未定义引用/交叉引用、LaTeX error 和 overfull box。
- [x] 步骤 10：检查 `git diff --check`、完整 diff 与工作区，确保没有代码、数据、实验工件或无关改动。
- [x] 步骤 11：使用 Conventional Commit 提交论文实质修改。
- [x] 步骤 12：回填论文 commit、最终页数、验证结果和偏差，将状态设为 `已执行`，再以独立文档 commit 归档计划。

## 8. 风险、证据边界与待确认事项

### 8.1 当前无需用户确认的已知调整

- “之前的论文修改计划”按上一轮建议解释为本计划覆盖的零实验修订范围。
- exact-SMILES 与 2D InChIKey 的差异是新发现的准确性风险，但可通过如实披露本地协议解决，不需要改动结果数值。
- 当前英文 14 页、中文 13 页；新增内容采用替换和压缩方式，英文净新增目标为 450--550 词，最多新增三篇参考文献。

### 8.2 会触发 `已偏离待确认` 的条件

- 用户要求把结果改称官方 MassSpecGym evaluator 等价结果；这需要按二维 InChIKey 多正例重新评价。
- 发现现有结果表与 frozen artifacts 不一致，或必须修改结果数值才能保持论文准确。
- 新证据否定候选池、正例、forcing、alignment 参数或文献支持关系中的任何关键前提。
- 修改需要增加模型实验、重新评价、代码变更或超过本计划的文献/章节范围。
- 为满足 15 页限制必须删除身份协议、覆盖率分解等核心准确性内容；此时应暂停而不是隐去关键边界。

### 8.3 页数降级策略

若英文达到 16 页，按以下顺序压缩，不视为改变目标：

1. 合并 Limitations 句子并移除显式 `\paragraph{}` 小标题；
2. 缩短静态候选池统计和增强细节，但保留身份规则、候选池上限和关键复现参数；
3. 删除 conformal future-work 的扩展解释，仅保留一条准确引用句；
4. 不删除 exact-SMILES 协议差异、full-pool 术语修正和覆盖率分解。

## 9. 验证方案

- [x] 英文稿编译：

  ```bash
  cd /home/zyl/MYPROS/SpecEmbedding/paper
  latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
  ```

- [x] 中文稿编译：

  ```bash
  cd /home/zyl/MYPROS/SpecEmbedding/paper
  latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build main_cn.tex
  ```

- [x] 页数：`pdfinfo paper/build/main.pdf` 必须不超过 15 页；记录中文页数但不设投稿上限。
- [x] 日志：检查 `undefined`、`Citation ... undefined`、`Reference ... undefined`、`LaTeX Error` 和 `Overfull`；不得新增这些问题。
- [x] 引用：确认三条新键均被引用、无重复键；题名、作者、年份、期刊、卷期页码和 DOI 与一手来源一致。
- [x] 协议措辞：搜索 `full-library`、`full library`、`official candidate protocol`、`official retrieval settings`，逐处确认不再过度声称。
- [x] 双语一致性：逐节比对 Related Work、公式、候选协议、实现细节、Discussion 和 Limitations。
- [x] 差异检查：`git diff --check` 通过；`git diff` 只包含计划内四个文件。
- [x] 不运行训练/评价命令，不产生新实验工件。

## 10. 执行记录

- 2026-08-01：完成只读证据复核和计划编写；检查工作区后确认 `paper/main.tex`、`paper/main_cn.tex` 与 `paper/references.bib` 尚未修改。
- 2026-08-01：计划状态由 `未执行` 切换为 `执行中`，开始按步骤实施。
- 2026-08-01：完成中英文协议术语、覆盖率分解、直接相关工作、方法性质、第一阶段复现信息和效度边界修改；未改动结果表数字，未运行训练或模型评价。
- 2026-08-01：新增并逐条核验 `gerlich2013metfusion`、`bach2022lcms2struct`、`rakhshaninejad2026conformal`；三条均为正式发表期刊论文，并在中英文稿各引用一次。MassSpecGym 默认身份规则固定核验到官方 commit `4f501b3207c9f466bc745a0353f61dbfaf150d32`。
- 2026-08-01：候选池静态统计对应 manifest 条目 `data.candidates_formula`（SHA-256 `0b0d8ffff166eeda9c9dc4060618f823ae4984a71d045119b13b01b5dd5f8194`）和 `data.candidates_mass`（SHA-256 `b4ddf39783ea2dceb32217eab68b71133572f63cc482ba9d17ed9a530c906a8f`）。
- 2026-08-01：英文首次编译为 16 页，按本计划 8.3 节压缩重复措辞后稳定为 15 页；中文为 14 页。两稿均无 undefined citation/reference、LaTeX error 或 overfull box。保留的非阻断 package warning 为英文 `amsmath` 无法重定义 `\vec`，中文 Fandol 字体的两条 `fontspec` CJK script warning。
- 2026-08-01：最终 `git diff --check`、高风险协议措辞、BibTeX 重复键和双语一致性检查通过；工作区仅含计划内三份论文文件与本计划。
- 2026-08-01：论文实质修改以 `75e57ba docs(paper): 澄清检索协议与证据边界` 提交；随后回填本计划并将状态更新为 `已执行`，计划以独立文档 commit 归档。

## 11. 最终结果

- 完成日期：2026-08-01
- 最终状态：`已执行`
- 英文页数：基线 14 页；修改后 15 页
- 中文页数：基线 13 页；修改后 14 页
- 引用验证：三条新增 BibTeX 元数据、唯一键、双语引用位置和一手来源均已核验
- LaTeX 验证：英文 pdfLaTeX 与中文 XeLaTeX 均通过；无 undefined、error 或 overfull，保留上述非阻断 package warning
- 论文修改 commit：`75e57ba docs(paper): 澄清检索协议与证据边界`
- 计划归档 commit：无需在本文件中自我引用
- 相对原计划的偏差：为满足英文 15 页上限并消除首次编译中的 overfull，按既定降级策略压缩重复说明；英文相对基线净增约 350 词，低于 450--550 词目标，但未删除 exact-SMILES 协议、覆盖率分解、三条新增引用或效度边界。该调整不改变目标、论点或证据范围。
