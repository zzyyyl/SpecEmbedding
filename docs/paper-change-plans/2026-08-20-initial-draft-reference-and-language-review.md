# 论文修改计划：初稿引用核验、相关工作压缩与双语语言收口

- 状态：`未执行`
- 创建日期：2026-08-20
- 最后更新：2026-08-20
- 负责人：Codex
- 关联论文：`paper/main.tex` / `paper/main_cn.tex`
- 关联参考文献：`paper/references.bib`
- 关联待办：`paper/ADMA2026_TODO.md` / `docs/project_memory_zh.md`
- 基线 commit：`01945f7 docs(repro): 归档匿名补充包验收`
- 计划约束：不新增训练或评价实验，不改变结果数字、评价协议或核心结论；页数只记录、不作为本轮阻断

## 1. 修改目标与动机

本轮目标是完成现有论文初稿的写作收口，而不是增加研究目标。具体包括：

1. 对当前全部 25 条参考文献逐条核对题名、作者、年份、出版物、卷期页码、DOI/稳定链接与发表状态；
2. 压缩 Related Work 中重复的定位文字，同时保留“reranking 并非本文首创”及与 JESTR/GLMR 的必要机制边界；
3. 修复英文稿中会导致协议、组件方向、配对计数或实验层级误读的措辞，并做一次人工语言润色；
4. 同步修正中文稿的实质内容，尤其是当前误写为“跳过 alignment 阶段”的实现描述；
5. 清理已与当前决定冲突的投稿前注释：不再把 top-$K$/效率实验或当前页数写成初稿阻断项；
6. 编译并审计双语稿，确认没有未定义引用、交叉引用、致命 LaTeX 错误或新增证据越界。

完成后，论文应达到“完整初稿”状态：章节完整、现有结果和限制形成闭环、引用可追溯、英文可通读、中文实质同步。作者列表、COI、目标 venue 格式、最终匿名复扫和页数压缩仍属于投稿整理，而不是本轮初稿内容阻断。

## 2. 当前证据与问题定位

### 2.1 结构、数字和证据边界已经成立

- 英文稿包含 Abstract、Introduction、Related Work、Problem Formulation、Method、Experiments、Discussion、Limitations 与 Conclusion；没有未完成章节或占位结果。
- 中英文稿使用完全相同的 25 个引用键；25 个 BibTeX 条目均被引用，没有未使用或缺失键。
- 独立双语审读未发现摘要、结果、讨论、局限和结论之间的数值冲突。
- 必须保留的边界已经一致：本地 exact-target-SMILES 单正例协议不等价于参考二维 InChIKey/可能多正例 evaluator；cross-alignment 只有三个描述性 alignment 单位；组件消融固定 alignment seed 42；JESTR/GLMR 未独立复现且不可直接比较；没有实测延迟、显存或效率优势。

### 2.2 顶部注释仍保留过时阻断项

- `paper/main.tex:3--24` 和 `paper/main_cn.tex:3--20` 仍以 ADMA 2026 为当前投稿检查表，并写“补齐 top-$K$ 消融和效率结果”。
- 当前已转投且目标 venue 尚未确定；用户已明确不增加研究目标，页数在完稿后调整。
- 正文已经诚实说明未报告 top-$K$/candidate-order 与效率测量。因此顶部注释应改成历史来源说明和当前非阻断整理事项，不能暗示还必须启动实验。

### 2.3 英文存在可能改变读者理解的措辞

- 摘要首次使用 `canonical`，但未说明其为 overlap-clean alignment-seed-42 checkpoint/cache。
- 摘要的 `gives none a full-model MRR advantage` 方向含混；“六个配对”应明确为两个候选池乘三个 reranker seeds，每个配对同时比较 Recall@1 与 MRR。
- `Fixed pairwise cosine scores do not directly optimize...` 容易忽略已有方法的候选监督；更精确的是“candidate-wise cosine scoring does not explicitly train a second-stage listwise objective over the retrieved set”。
- `avoiding fixed cosine-only ranking` 可能被误读为本文不使用余弦基础分数；应改为 `avoiding cosine-only final ranking`。
- `exact-SMILES` 同时被用于候选字符串去重和评价身份规则。评价规则应统一为 `exact-target-SMILES single-positive protocol`，仅去重操作保留 `exact-SMILES-deduplicated`。
- `earlier checkpoint` 指代不清；应改为 `pre-overlap-clean fixed checkpoint`，相应流水线称 `overlap-clean` 与 `pre-clean`。
- `negligible` 是主观判断；已知变化全部低于 0.006 个百分点，直接报告该范围更可靠。

### 2.4 中文有两处事实/术语错误

- `paper/main_cn.tex:430--432` 当前写成“跳过……alignment 阶段”，但实际只跳过冻结预训练谱图塔的第一训练阶段，后续仍端到端训练两个编码塔进行 alignment。
- `paper/main_cn.tex:480` 的“规范实验选择 `PULP_CBC_CMD`”是 `canonical runs` 的误译，应改为“主结果实验”或“canonical 实验”。

### 2.5 Related Work 有重复但引用本身必要

- Annotation Paradigms 三段已覆盖 MetFusion、MS2Query、LC-MS2Struct、CSI:FingerID、MIST、CMSSP、JESTR、MVP、SpecEmbedding、MassFormer、MSNovelist 与 GLMR；引用均有作用，但部分句子可合并。
- Learning to Rank 段需保留 RankNet/ListNet 与 Set Transformer/SetRank 两组直接先例。
- 独立的 `Positioning of Our Method` 小节与前段重复“贡献不在 reranking/self-attention 本身”。计划删除该小节标题，将定位压成表格前的一段，从而减少层级和重复，而不改变机制表或论点。
- 目标压缩约 100--150 个英文词；不以删掉证据边界换取篇幅。

### 2.6 参考文献核验结论

- 25 个条目均真实可追溯，未发现伪造 DOI、错误年份、错误题名或把预印本误写成期刊论文。
- 需要补齐的书目信息：GLMR 期号与官方 proceedings 名称；MassSpecGym 完整作者；CMC 的 LNCS 系列/卷；Attention Is All You Need 的页码与稳定链接；LC-MS2Struct 的期号；Set Transformer 与 GNN pretraining 的官方稳定链接。
- `liu2026massspecgymwild` 仍是 arXiv preprint；即使有非归档 workshop 展示，也不改为正式会议论文。
- SpecEmbedding 论文支持峰序列 Transformer 架构；本文 tokenizer 的逐项细节来自本地实现。正文应把细节明确写成“in our implementation”，避免让单条论文引用承担未直接核验的代码级细节。

### 2.7 当前静态检查

- `texcount -inc -sum paper/main.tex`：总计 4,861 词，其中正文 4,425 词。
- 在 `paper/` 目录执行 ChkTeX，只报告两处表格中 `align. seed 42` 的缩写空格提示。
- LaCheck 另报告一处句点前空格、`GINE.` 的句末间距提示和模板宏花括号提示；前两项可安全修复，模板/宏提示不作为错误。

## 3. 修改范围

### 3.1 涉及文件与章节

- [ ] `paper/main.tex` 顶部注释：改成历史 ADMA 检查来源和当前转投整理边界。
- [ ] `paper/main.tex` Abstract/Introduction：定义 canonical，澄清组件配对与余弦排序表述，压缩重复语言。
- [ ] `paper/main.tex` Related Work：压缩三类方法描述，合并 Positioning 小节但保留机制表和全部必要引用。
- [ ] `paper/main.tex` Method/Experiments/Limitations/Conclusion：统一协议名、canonical 名称、pool-by-seed 配对计数和 pre-clean 指代；修复小型排版告警。
- [ ] `paper/main_cn.tex`：同步所有会影响主张、协议、实验层级或相关工作结构的修改，并修复 alignment/`canonical runs` 误译。
- [ ] `paper/references.bib`：仅补齐已核验元数据，不新增或删除文献。
- [ ] `paper/ADMA2026_TODO.md`：勾选相关工作压缩、英文润色、全部引用核验；保留实验、页数和投稿账户事项原状态或明确非本轮范围。
- [ ] `docs/project_memory_zh.md`：记录初稿写作收口结果和剩余投稿整理事项。
- [ ] 本计划：记录执行、验证、论文 commit 与偏差。

### 3.2 明确不做的事项

- 不启动候选顺序、$K=20/40/100/256$、预训练开关、coverage 分层、效率或案例分析实验。
- 不重新训练 alignment/reranker，不重新计算 Recall、MRR 或 MCES，不改变任何结果数值。
- 不新增引用，不扩大 Related Work 的主题范围。
- 不更改 exact-target-SMILES 身份规则、候选池、数据划分或 evaluator。
- 不将 rank-free 变体事后升级为主模型，不新增因果、显著性、SOTA、稳健 self-attention 或效率主张。
- 不把页数压缩作为本轮完成条件；只记录编译后的实际页数。
- 不完成作者列表、COI、目标 venue 格式或最终提交操作。

## 4. 具体内容设计

### 4.1 顶部注释与摘要

- 将 ADMA 日期、15 页和补充材料规则明确标记为历史检查记录；写明目标 venue 确定后再做最终格式、页数和匿名检查。
- 删除“必须补 top-$K$/效率实验”的注释，改成：未测量项目保留为限制，不据此作主张。
- 将摘要中的 canonical 首次出现改成 `the overlap-clean alignment-seed-42 first-stage checkpoint and caches`，后文才可简称 canonical。
- 将组件结论改为：完整模型相对任一移除变体均未取得六个 pool-by-reranker-seed MRR 配对全胜；rank-free 变体在每个配对中同时改善两项指标。
- 保留 local protocol、cross-alignment、负向组件结果、测试不强制正例等关键信息；只删除重复修饰。

### 4.2 Introduction

- 将候选余弦评分的限制收窄为“没有显式训练检索列表上的第二阶段 listwise objective”。
- 贡献点使用 `cosine-only final ranking`，明确本文仍使用 base cosine score 和 residual shortcut。
- 评价身份统一为 `exact-target-SMILES single-positive protocol`。

### 4.3 Related Work 与机制表

- 第一段保留三个 reranking 先例及证据差异，但减少逐条过渡词和重复的场景解释。
- 第二段用三句区分 fingerprint prediction、joint embedding 与 SpecEmbedding 架构继承；保留 MVP 四视图的准确描述。
- 第三段紧凑区分 spectrum simulation、de novo generation 与本文 closed-set reranking。
- Learning to Rank 段将 `explicit pair interactions` 改成 `explicit spectrum--candidate pair features`。
- 对 SetRank 的描述采用原文直接支持的“self-attention 建模 cross-document interactions”，不让该引用单独承担本文实现置换性质的证明。
- 删除 `Positioning of Our Method` 小节标题，把定位段与机制表保留在 Learning to Rank 小节末尾。
- 表题从 `closest` 改为 `selected related`；GLMR 的候选交互列改成“retrieved candidates condition the generator”，避免与本文的候选集合交互混淆。

### 4.4 Method 与 Implementation Details

- 在 SpecEmbedding 架构归因后写明 tokenizer 细节是 `In our implementation` 的行为。
- 统一 `candidate-set Transformer`，减少未定义的 `set-aware variant`。
- 在实现细节首次正式定义 canonical：overlap-clean alignment seed 42 及其 caches。
- 中文明确“跳过冻结预训练谱图塔的第一训练阶段”，而不是跳过 alignment。

### 4.5 Results、Discussion 与 Limitations

- 将“all six differences”统一改为六个 `pool-by-reranker-seed pairs` 中两项指标都朝同一方向。
- 移除实验同样使用 `pool-by-reranker-seed pair`，防止把两个指标合计误算成六项。
- 将 `earlier checkpoint/pipeline` 改为 `pre-overlap-clean fixed checkpoint` / `pre-clean pipeline`。
- 将 `negligible` 改为“所有变化均低于 0.006 个百分点”。
- 将内部审计式的 `parameter hash/source commits` 改为“相同记录超参数、来自两个代码修订”，但保留 provenance 边界。
- Limitations 中的组件名称与表格统一为 `joint base-score-pathway removal`，拆分过长句。
- 保留所有未测量实验、开放集、候选覆盖、环境历史、identity rule 和外部基线限制。

### 4.6 中文同步原则

- 所有会改变协议、主张强度、canonical 定义、配对计数、组件方向或 Related Work 结构的修改逐项同步。
- 纯英文冠词、搭配和断句可不机械映射；中文以准确、自然为准。
- 引用键、公式、表格数字和限制结论不得发生差异。

## 5. 引用文献与真实性核验

| 引用键 | 文献与已核验元数据 | 支持的附近论断 | 一手来源 | 核验状态/计划动作 |
|---|---|---|---|---|
| `xiong2025specembedding` | Xiong, Xu, Zheng; *Analytical Chemistry* 97(37):20137--20146, 2025 | 峰序列 Transformer 谱图架构与谱图表征学习 | [ACS](https://doi.org/10.1021/acs.analchem.5c02655) | 已核验；BibTeX 正确；代码级 tokenizer 细节改写为本文实现事实 |
| `kalia2025jestr` | Kalia, Chen, Krishnan, Hassoun; *Bioinformatics* 41(7):btaf354, 2025 | CMC 联合空间与候选余弦排序 | [Oxford Academic](https://doi.org/10.1093/bioinformatics/btaf354) | 已核验；无需修改 |
| `zhang2026glmr` | Zhang et al.; *Proceedings of the AAAI Conference on Artificial Intelligence* 40(2):1561--1569, 2026 | 预检索、条件生成、按生成分子相似度重排及外部表数值 | [AAAI](https://doi.org/10.1609/aaai.v40i2.37132) | 已核验；补期号并统一官方 proceedings 名称 |
| `bushuiev2024massspecgym` | Bushuiev et al.; *NeurIPS* 37:110010--110027, 2024 | benchmark、结构划分、数据量、候选池 | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c6c31413d5c53b7d1c343c1498734b0f-Abstract-Datasets_and_Benchmarks_Track.html) | 已核验；展开完整 30 人作者表 |
| `goldman2023mist` | Goldman et al.; *Nature Machine Intelligence* 5(9):965--979, 2023 | MIST 预测分子指纹用于候选检索 | [Nature](https://doi.org/10.1038/s42256-023-00708-3) | 已核验；无需修改 |
| `chen2024cmssp` | Chen et al.; *Analytical Chemistry* 96(42):16871--16881, 2024 | 对比式谱图--结构预训练 | [ACS](https://doi.org/10.1021/acs.analchem.4c03724) | 已核验；无需修改 |
| `tian2020cmc` | Tian, Krishnan, Isola; ECCV 2020, LNCS 12356:776--794 | Contrastive Multiview Coding | [Springer](https://doi.org/10.1007/978-3-030-58621-8_45) | 已核验；补 LNCS 系列、卷号与出版地 |
| `vaswani2017attention` | Vaswani et al.; *NeurIPS* 30:5998--6008, 2017 | Transformer/self-attention 架构先例 | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) | 已核验；补页码、稳定 URL 与作者重音 |
| `kretschmer2025coverage` | Kretschmer et al.; *Nature Communications* 16:554, 2025 | thresholded myopic-MCES 及 stronger bound 边界 | [Nature](https://doi.org/10.1038/s41467-024-55462-w) | 已核验；无需修改 |
| `duhrkop2015csifingerid` | Dührkop et al.; *PNAS* 112(41):12580--12585, 2015 | 谱图到分子指纹并搜索结构数据库 | [PNAS](https://doi.org/10.1073/pnas.1509788112) | 已核验；无需修改 |
| `young2024massformer` | Young, Röst, Wang; *Nature Machine Intelligence* 6(4):404--416, 2024 | 从分子图模拟 tandem mass spectrum | [Nature](https://doi.org/10.1038/s42256-024-00816-8) | 已核验；无需修改 |
| `dejonge2023ms2query` | de Jonge et al.; *Nature Communications* 14:1752, 2023 | top-2000 MS2DeepScore 后五特征随机森林重排 | [Nature](https://doi.org/10.1038/s41467-023-37446-4) | 已核验；无需修改 |
| `gerlich2013metfusion` | Gerlich, Neumann; *Journal of Mass Spectrometry* 48(3):291--298, 2013 | 融合 in-silico fragmentation 与谱库证据 | [Wiley](https://doi.org/10.1002/jms.3123) | 已核验；无需修改 |
| `bach2022lcms2struct` | Bach, Schymanski, Rousu; *Nature Machine Intelligence* 4(12):1224--1237, 2022 | MS/MS scorer 与 LC retention order 联合候选排序 | [Nature](https://doi.org/10.1038/s42256-022-00577-2) | 已核验；补 `number={12}` |
| `rakhshaninejad2026conformal` | Rakhshaninejad et al.; *JCIM* 66(10):5788--5800, 2026 | exchangeability/target-in-pool 下的谱图特异预测集 | [ACS](https://doi.org/10.1021/acs.jcim.6c00727) | 已核验为正式期刊论文；无需修改 |
| `stravs2022msnovelist` | Stravs et al.; *Nature Methods* 19(7):865--870, 2022 | 从 MS/MS-derived fingerprints 生成结构 | [Nature](https://doi.org/10.1038/s41592-022-01486-3) | 已核验；无需修改 |
| `chen2026mvp` | Yan Zhou Chen, Soha Hassoun; *Analytical Chemistry* 98(9):6598--6606, 2026 | graph/fingerprint/individual/consensus spectrum 四视图对比学习 | [ACS](https://doi.org/10.1021/acs.analchem.5c05675) | 已核验为正式期刊论文；无需修改 |
| `burges2005ranknet` | Burges et al.; ICML 2005:89--96 | pairwise preference ranking | [ACM](https://doi.org/10.1145/1102351.1102363) | 已核验；无需修改 |
| `cao2007listnet` | Cao et al.; ICML 2007:129--136 | listwise probability model/top-one loss | [ACM](https://doi.org/10.1145/1273496.1273513) | 已核验；无需修改 |
| `pang2020setrank` | Pang et al.; SIGIR 2020:499--508 | self-attention 建模 cross-document interactions | [ACM](https://doi.org/10.1145/3397271.3401104) | 已核验；收窄正文措辞以避免过度依赖引用证明本文实现性质 |
| `lee2019settransformer` | Lee et al.; PMLR 97:3744--3753, 2019 | set-structured attention blocks | [PMLR](https://proceedings.mlr.press/v97/lee19d.html) | 已核验；补稳定 URL |
| `hu2020pretraininggnn` | Hu et al.; ICLR 2020 | 带 atom/bond attributes 的 GIN/GINE 分子图编码先例 | [OpenReview](https://openreview.net/forum?id=HJlWWJSFDH) | 已核验；补稳定 URL，不额外强调非必要 Spotlight 状态 |
| `schymanski2014confidence` | Schymanski et al.; *Environmental Science & Technology* 48(4):2097--2098, 2014 | 候选优先级与确认鉴定置信度的区别 | [ACS](https://doi.org/10.1021/es5002105) | 已核验；无需修改 |
| `hoffmann2022cosmic` | Hoffmann et al.; *Nature Biotechnology* 40(3):411--421, 2022 | 谱图库缺失结构的高置信度注释边界 | [Nature](https://doi.org/10.1038/s41587-021-01045-9) | 已核验；无需修改 |
| `liu2026massspecgymwild` | Liu et al.; arXiv:2606.19624 [cs.LG], v1, 2026-06-17 | MassSpecGym leakage/shortcut/implementation/metric risks与 v1.5 | [arXiv](https://arxiv.org/abs/2606.19624) | 已核验；继续标为 preprint，不改成正式会议论文 |

## 6. 实验与计算边界

- 是否新增训练实验：否。
- 是否新增评价运行：否。
- 是否重新计算已有指标：否。
- 是否新增静态数据统计：否；仅复用已审计的行数、词数、页数和现有实验工件。
- 是否只修改文字、结构和引用：是；另做 LaTeX/BibTeX/静态一致性验证。
- 相关工作压缩与参考文献元数据补全均不得改变任何模型结果或比较结论。

## 7. 分步执行清单

- [ ] 步骤 1：提交本计划，确认论文文件尚未修改。
- [ ] 步骤 2：将状态改为 `执行中`，记录开始执行的计划 commit。
- [ ] 步骤 3：补齐 `references.bib` 的已核验元数据，检查 25 键唯一且引用集合不变。
- [ ] 步骤 4：修改英文顶部注释、摘要和引言，修正 canonical、余弦排序、协议名与组件配对表述。
- [ ] 步骤 5：压缩英文 Related Work 并合并 Positioning 小节，保留全部必要引用和机制表。
- [ ] 步骤 6：修改英文 Method/Experiments/Discussion/Limitations/Conclusion 的术语、指代与语言问题。
- [ ] 步骤 7：逐项同步中文稿并修复 alignment 与 canonical runs 误译。
- [ ] 步骤 8：执行引用集合、BibTeX 字段、协议词、数字和中英文边界检查。
- [ ] 步骤 9：完整编译双语稿，记录页数及 warning；页数超出未来 venue 上限不阻断本轮。
- [ ] 步骤 10：运行 ChkTeX/LaCheck、`git diff --check` 并人工审阅完整 diff。
- [ ] 步骤 11：使用 Conventional Commit 提交论文、BibTeX、TODO 与记忆文档的实质修改。
- [ ] 步骤 12：回填论文 commit、验证结果与实际偏差，将状态改为 `已执行`，再以独立文档 commit 归档计划。
- [ ] 步骤 13：推送当前分支，并进行一次只读收口审计，判断完整初稿是否成立。

## 8. 风险、证据边界与待确认事项

- 删除 Related Work 层级属于结构调整，但只合并重复定位，不删除引用支撑或扩大创新主张。
- 任何发现现有引用不支持附近论断、需要新增/替换文献、修改结果数字或重开实验的情况，均超出本计划；应把状态设为 `已偏离待确认` 并停止相关修改。
- `and others` 展开为 MassSpecGym 完整作者表可能改变参考文献换行和页数，但属于书目准确性修复；页数变化只记录。
- 英文润色不得删掉 local protocol、external reported-only、cross-alignment 层级、组件容量混淆、训练正例替换或 top-40 coverage 等限制。
- `canonical` 只指 overlap-clean alignment-seed-42 checkpoint/cache；不得泛化成全部 alignment 或端到端标准环境。
- 参考文献核验时间点为 2026-08-20；`liu2026massspecgymwild` 的发表状态未来可能变化，投稿前仍需复查。
- 目标 venue 尚未确定，因此 AI 披露、格式和页数不按某一新 venue 擅自改写；本轮保留披露正文。

## 9. 验证方案

- [ ] 英文稿编译：在 `paper/` 执行 `latexmk -gg -pdf -interaction=nonstopmode -halt-on-error main.tex`。
- [ ] 中文稿编译：在 `paper/` 执行 `latexmk -gg -xelatex -interaction=nonstopmode -halt-on-error main_cn.tex`。
- [ ] 发布构建：执行 `bash paper/build_release.sh`，检查当前双语 PDF、页数、SHA-256 和构建日志；按需要更新受跟踪 manifest。
- [ ] 引用/BibTeX：比较两稿 `\\cite{}` 键集合与 `references.bib` 条目集合，要求均为同一 25 键；日志不得有 undefined citation/reference。
- [ ] 证据边界：搜索 `exact-SMILES`、`exact-target-SMILES`、`official`、`SOTA`、`significant`、`robust`、`latency`、`efficiency`、`all six`、`earlier checkpoint`，逐处人工核验。
- [ ] 中英文一致性：逐项核对摘要、贡献、Related Work 结构、canonical 定义、pool-by-seed 计数、pre-clean 敏感性、Limitations 与 Conclusion。
- [ ] 静态语言检查：在 `paper/` 运行 ChkTeX 与 LaCheck；区分模板/宏噪声和可修复警告。
- [ ] 匿名性：对最终 PDF 文本和本轮修改文件扫描用户名、本地绝对路径、远端地址与作者元数据；最终提交联合复扫仍另行保留。
- [ ] 变更范围：`git diff --check`、`git status -sb`、完整 `git diff`，确认未混入代码、数据、checkpoint 或用户改动。

## 10. 执行记录

尚未执行。计划基于 2026-08-20 的双语全文只读审读、25 条一手文献核验、ChkTeX/LaCheck 与引用集合检查形成；论文文件在创建计划前未修改。

## 11. 最终结果

- 完成日期：尚未完成
- 最终状态：`未执行`
- 验证结果：尚未验证
- 论文修改 commit：尚未提交
- 计划归档 commit：无需在本文件中自我引用
- 相对原计划的偏差：尚未记录
