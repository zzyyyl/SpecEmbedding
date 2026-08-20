# 论文修改计划：专家预审后的证据披露与结果口径修订

- 状态：`未执行`
- 创建日期：2026-08-21
- 最后更新：2026-08-21
- 负责人：Codex
- 关联论文：`paper/main.tex` / `paper/main_cn.tex`
- 关联实现与补充说明：`prepare_rerank_cache.py`、`README.md`、`reproducibility/anonymous-supplement-README.md`
- 关联分析工件：`analysis/transfer2026_core_ablations/report.md`、现有 `summary_long.csv` 与 `analysis_manifest.json`
- 计划约束：不新增训练、评价、外部 baseline 复现或 MCES 运行；只披露已有静态事实、重组已有结果并收窄结论

## 1. 修改目标与动机

2026-08-21，评审专家对远端 HEAD `01204ac97368b1278011bd904411bc5229f146f5` 完成独立只读预审，结论为 Major Revision 后再投稿；无 P0，但有六项投稿前 P1。核心数值、统计层级、引用和双语内容总体可靠。此次计划只修复披露、结果呈现和复现边界，不扩大研究目标。

目标如下：

1. 明确训练期正例强制插入导致的 training-only label-correlated rank shortcut 和 train--evaluation distribution shift，区分该风险与测试泄漏；
2. 解释 alignment seed 42 是 `params.yaml` 的项目默认种子，将 canonical 结果称为 single default-seed audit，不暗示按测试指标挑选或预注册；统一称 reranker training seeds；
3. 将 canonical 主结果的 Recall/MRR 三种子均值±样本 SD 提升为主表，并把只计算一次的 MCES@1 独立为 single-checkpoint audit；
4. 利用已有组件工件的 P/Z/N paired counts，使摘要和正文的 six-of-six 结论可由受跟踪表格核验；
5. 在正文增加准确的 Code/Data/Artifact Availability，明确匿名补充包能验证什么、不能复现什么，并列出已有 provenance/config/精度/训练入口事实；
6. 删除未复现外部数值表，仅保留机制背景，避免跨论文 canonicalization 风险被误读为受控结果；
7. 同步中英文稿、README、分析报告和补充说明，保持 local protocol、reported-only、n=3 描述性边界不变。

## 2. 当前证据与问题定位

### 2.1 训练正例插入与 rank shortcut

- `prepare_rerank_cache.py:169--184` 在训练 target 不在 base top-40 时，将 target 替换末位并保留完整池 `true_rank`；验证/测试的 `force_include_positive=false` 不执行该替换。
- canonical 训练 split 的 top-40 base recall 为 mass 93.78%、formula 90.42%，因此约 6.22%/9.58% 的训练查询受训练期插入影响；这些数值已在正文实现细节中出现，不需重算。
- 被插入 target 的 full-pool rank 大于 40，而同列表的原 top-40 候选 rank 为 1--39，rank embedding 可携带确定性 label-correlated shortcut。该现象是训练/评价分布偏移，不是测试标签泄漏。
- 现有 no-rank-embedding 消融在 2 个候选池×3 个 reranker training seeds 的 6 个配对中同时改善 Recall@1/MRR；不重训，只把现有风险如实升级披露并停止 rank prior 正收益主张。

### 2.2 canonical seed 与统计层级

- `params.yaml:1--2` 的项目默认 `general.seed` 为 42；alignment seeds 43/44 是后续 sensitivity checkpoints。历史记录没有预注册或性能独立选择证据，因此正文改称 single default-seed audit，不写 representative/best-seed 选择依据。
- alignment-42/43/44 的 base R@1 为 mass 47.46/35.92/23.91、formula 63.10/45.08/28.73；U40 为 mass 83.88/80.98/72.20、formula 89.26/81.65/75.60。全部数字已在 `cross_alignment_summary.csv` 或现有论文表中，新增 U40 列只做静态重排。
- 固定 alignment 的 reranker seeds 42/43/44 同时控制初始化、DataLoader 顺序、候选 shuffle 和 dropout；全文将 `initializations` 统一为 `training seeds`。
- cross-alignment 仍为来自两个 source revisions 的三个历史 checkpoints，先在 checkpoint 内聚合 reranker seeds；不称一般端到端随机种子稳定性。

### 2.3 主结果与 MCES 口径

- 当前 `tab:main_results` 混用单 reranker seed 42 的 Recall/MRR/MCES，而摘要与组件表使用三种子均值±SD。
- `analysis/transfer2026_alignment_multiseed/summary_long.csv` 已包含 canonical alignment 42 下 pointwise/Transformer 三种子 R@1/R@5/R@10/R@20/MRR；只需静态聚合并把均值±SD放入主表。
- MCES@1 仅有 overlap-clean base 与 Transformer reranker seed 42 的 single-checkpoint 计算，solver 为 `PULP_CBC_CMD`，日志失败数为 0；不虚构 seed-level MCES 均值或 SD。

### 2.4 six-of-six 可核验性

- `analysis/transfer2026_core_ablations/report.md:26--40` 已包含 full-minus-ablation 的 P/Z/N：no residual 2/0/4、no base-score pathway 3/0/3、no rank embedding 0/0/6、no interaction features 2/0/4、listwise only 3/0/3。
- 现有组件表只有均值±SD，读者无法独立核验摘要中的 six-of-six；新增紧凑 MRR paired-direction 表，不改任何数字。
- six-of-six 首次出现时明确分母为 2 个 candidate pools × 3 个 reranker training seeds；Transformer--Pointwise 的 2/3 仍指 4 个 pool×metric 的 alignment-level summaries，不是 reranker runs。

### 2.5 复现边界

- `reproducibility/anonymous-supplement-README.md` 已诚实说明：补充包不含 raw spectra、candidate pools、checkpoints、caches、logs 或 internal inventories；smoke 只验证 source inspection 和 tiny synthetic CPU pipeline，不复现论文指标。
- 当前论文缺少同等明确的 Code/Data/Artifact Availability 段落。新增内容列出：mass/formula 分开 reranker、candidate artifact/version fingerprint、现有 source commits/config hash、atom/bond/graph-size features、alignment projection/tau=0.07、reranker weight decay 1e-4、gradient clip 1.0、spectrum FP32 与 molecule FP16 cache，以及 `run_rerank_pipeline.py` 入口。
- 不把匿名包称为端到端复现包，不承诺公开数据、候选池、checkpoint/cache 或历史训练环境可按位重现。

### 2.6 外部基线风险

- 当前外部表的数字来自 GLMR Table 1，虽然已标 `reported/not reproduced/not directly comparable`，仍容易被读者作相对排序。
- 本计划删除双语 external numeric table 及其跨论文数值段，只保留 Related Work 的机制比较和一句 reported-only 边界；因此不新增外部 canonicalization 论断或引用。

## 3. 修改范围

### 3.1 涉及文件与章节

- [ ] `paper/main.tex`：训练 shortcut、default-seed rationale、主结果/MCES 表、U40 列、P/Z/N 表、availability 段、外部数值表删除、P2/P3措辞同步。
- [ ] `paper/main_cn.tex`：逐项同步上述协议、统计、复现边界、表格和结论；保证中文不把六个配对理解成六类候选池。
- [ ] `README.md`：同步 default-seed audit、availability 和复现边界，避免 README 与论文口径不一致。
- [ ] `analysis/transfer2026_core_ablations/report.md`：在已有 P/Z/N 结果段补充分母和 six-of-six 解释（不改 CSV/实验数值）。
- [ ] `reproducibility/anonymous-supplement-README.md`：仅在需要时与正文 availability 文字逐项对齐；不扩大 allowlist 或 smoke 功能。
- [ ] 本计划：记录逐项执行、验证、专家复审意见、commit 与偏差。

### 3.2 明确不做的事项

- 不重训 alignment/reranker，不重跑官方二维 InChIKey/multi-positive evaluator，不重算全部 seeds 的 MCES。
- 不增加 alignment seeds、组件跨 alignment、K/order/latency/显存/吞吐、coverage 分层或案例实验。
- 不独立复现 JESTR/GLMR，不把 reported-only 数字重新用于排序、SOTA 或效率结论。
- 不把 no-rank-embedding 变体升级为新主模型；继续将组件收益写成未建立或候选依赖。
- 不新增文献；除非发现现有 `liu2026massspecgymwild` 不支持附近风险句，否则不改引用集合。
- 不把匿名 source smoke 写成论文指标复现，不将私有数据/候选/模型工件打包进投稿附件。
- 页数仍不作为本轮阻断；摘要/表格布局只做可读性调整。

## 4. 具体内容设计

### 4.1 Abstract/Introduction

- `three reranker initializations` 改为 `three reranker training seeds`。
- six-of-six 首次出现改为“2 candidate pools × 3 reranker training seeds”；保留 no-rank 变体的负向结果，不主张 rank prior 收益。
- contribution 1 的 `We formulate` 改成 `We instantiate and systematically audit`，强调受控实证而非 reranking 首创。

### 4.2 Candidate construction / Implementation / Limitations

- 在正例替换段落明确：被强制插入的训练 target 保留 full-pool rank >40，而其余 top-40 rank 为 1--39，形成 training-only label-correlated rank shortcut；6.22%/9.58% 是训练期覆盖缺失比例；验证/测试不插入，因此这是 train--evaluation shift、不是 test leakage。
- 将 canonical 定义改为：`params.yaml` project-default seed 42 的 single default-seed audit；不声明按测试指标选择、不声明预注册；alignment 43/44 是历史 sensitivity checkpoints。
- 增加 U40 到 cross-alignment 表；caption 说明三个 alignment-level estimates 来自两个 source revisions。
- 将模型身份/精度列入 availability 段，明确 GINE atom/bond/graph-size features、alignment projection 512/tau 0.07、reranker weight decay 1e-4、clip 1.0、spectrum FP32/molecule FP16 cache。
- exact-target-SMILES 限制增加模型层含义：当前图特征未显式编码 atom chirality/bond stereochemistry，某些立体差异可能被 local identity rule 与图表示共同放大；这是静态限制推断，不写成已测量结果。
- 将 32,010 明确为 molecule-keyed candidate lists；同一目标的重复谱图复用该列表。
- `rank-free` 改 `no-rank-embedding`；保留 no residual/base-score pathway 和 no interaction/self-attention 的消融定义。

### 4.3 Main Results / MCES / Component evidence

- 主表改为固定 base 与 pointwise/Transformer 三种 reranker training seeds 的 Recall@1/@5/@20/MRR mean±sample-SD，不再把 seed-42 单次值与均值混列。
- 新增独立 MCES audit 表：overlap-clean alignment seed 42、reranker seed 42 的 base/Transformer，thresholded myopic-MCES@1，PULP_CBC_CMD，0 failed pairs；caption 明确 lower is better、single checkpoint、无 seed uncertainty。
- 新增 MRR full-minus-ablation P/Z/N 表：no residual 2/0/4、no base-score pathway 3/0/3、no rank embedding 0/0/6、no interaction features 2/0/4、listwise only 3/0/3。
- `uncertainty` 改 `seed dispersion`/`seed-to-seed variation`；保留 SD 不是置信区间的限制。

### 4.4 External context / Availability

- 删除 external numeric table 及其数值段；机制表继续保留 JESTR/GLMR 的 reported-only、not reproduced、not directly comparable 语义。
- 新增 Code/Data/Artifact Availability：说明 anonymous source supplement 的 allowlist、environment locks、unit tests 和 synthetic smoke；明确缺少 real-data inputs/checkpoints/caches/logs/analysis artifacts，smoke 不复现报告指标；给出现有真实运行入口和 provenance/config 记录。

### 4.5 Bilingual synchronization

- 中英文使用同一结果数字、表格行、P/Z/N counts、U40、MCES 单次审计和 availability 边界。
- 中文摘要/正文第一次出现六个配对时写“2 种候选池 × 3 个 reranker training seeds”；中文方法段明确“未加载已发布 checkpoint，后续 alignment 阶段端到端训练两个塔”。
- AI disclosure 责任表述在中英文保持同一范围；不额外增加中文独有的原创性声明。

## 5. 引用文献与真实性核验

本轮不新增引用，继续使用上一计划已核验的 25 条一手来源。`zhang2026glmr` 只支持 GLMR/JESTR 的机制与 reported-table 来源，不再支持本地数值排序；`liu2026massspecgymwild` 只支持数据/实现/指标风险背景，不将审计预印本写成 GLMR 代码复现或定量校正。

| 引用键 | 文献 | 本轮附近论断 | 一手来源 | 核验状态 |
|---|---|---|---|---|
| `zhang2026glmr` | Zhang et al., AAAI 2026, GLMR | 生成式条件候选机制与外部 reported 数字来源；本轮删除数值表后仍保留机制 | AAAI DOI `10.1609/aaai.v40i2.37132` | 已核验 |
| `liu2026massspecgymwild` | Liu et al., arXiv:2606.19624, v1 (2026-06-17) | MassSpecGym 数据泄漏/shortcut/实现/指标风险背景；不作定量校正 | arXiv `https://arxiv.org/abs/2606.19624` | 已核验为 preprint |

不添加新文献，不改变 25 键闭包；BibTeX、双稿引用集合和 DOI 元数据需重新检查。

## 6. 实验与计算边界

- 是否新增训练实验：否。
- 是否新增评价运行：否。
- 是否进行静态数据统计：是；只聚合现有 CSV 中的三种子均值/SD、P/Z/N、U40 并核对现有工件，不产生新模型结果。
- 是否只修改文字、公式、表格或引用：是；删除外部数值表并新增已有证据的 availability/MCES/P-Z-N 呈现。

## 7. 分步执行清单

- [ ] 步骤 1：提交本计划并保持稿件未修改。
- [ ] 步骤 2：将计划状态改为 `执行中`，记录起始 commit。
- [ ] 步骤 3：从现有 CSV/manifest 静态核对均值、SD、U40 和 P/Z/N，形成修改用数字清单。
- [ ] 步骤 4：修改英文训练 shortcut、seed rationale、统计层级和 availability；重组主表/MCES/P-Z-N 表并删除外部数值表。
- [ ] 步骤 5：逐项同步中文稿、README、分析报告和匿名补充说明。
- [ ] 步骤 6：运行 citation/number/boundary/anonymous diff 审计，确认无新实验数字或主张越界。
- [ ] 步骤 7：完整编译双语稿，运行严格 release build，检查 PDF/manifest/哈希与页数。
- [ ] 步骤 8：用 Conventional Commit 提交论文及配套说明实质修改。
- [ ] 步骤 9：推送并通知评审专家对话再次只读复审；若有同范围意见继续收口，否则归档本计划。
- [ ] 步骤 10：回填验证、评审反馈、commit 和偏差，将状态改为 `已执行`，以独立文档 commit 归档计划。

## 8. 风险、证据边界与待确认事项

- 若无法从现有工件稳定导出主表三种子均值/SD或 P/Z/N，必须停止并设为 `已偏离待确认`，不能手填猜测。
- 若 seed-42 的历史选择记录出现按测试指标挑选证据，必须删除 default-seed rationale 中的独立性措辞并明确 selection bias；不重算实验。
- 若删除 external numeric table 使某个引用孤立，保留机制引用或从 BibTeX 中删除条目，但不得新增替代文献。
- availability 段不能暗示匿名包包含真实数据/模型工件；所有运行入口必须注明需要外部 supplied paths。
- 任何要求重训、统一官方 evaluator、补算全 seed MCES、扩展 alignment 或加入新 baseline 的意见均超出本计划，设为 `已偏离待确认`。
- 表格重组可能改变页数和浮动位置；页数仍是非阻断投稿整理事项。

## 9. 验证方案

- [ ] 英文稿：`latexmk -gg -pdf -interaction=nonstopmode -halt-on-error main.tex`。
- [ ] 中文稿：`latexmk -gg -xelatex -interaction=nonstopmode -halt-on-error main_cn.tex`。
- [ ] 严格发布：`bash paper/build_release.sh`；检查 fatal/undefined/Overfull/Underfull、页数、作者元数据、私有路径与 manifest 哈希。
- [ ] 引用/BibTeX：双稿 citation keys 与 `references.bib` 均为同一 25 条，无 undefined citation/reference。
- [ ] 数字闭包：主表三种子均值/SD、U40、MCES 单次值、P/Z/N 与现有 CSV/报告逐项一致；结果数字无意外修改。
- [ ] 边界审计：搜索 train-only/rank shortcut/distribution shift/test leakage、local exact-target-SMILES、official evaluator、n=3、12/12、2/3、reported-only、not reproduced、not directly comparable、availability。
- [ ] 双语一致性：摘要、方法、主结果、MCES、组件、局限、结论和 availability 逐项对照。
- [ ] 匿名性：PDF/源码/补充包扫描身份字符串和元数据；不把内部 manifest/log/data 纳入投稿包。
- [ ] `git diff --check`、ChkTeX/LaCheck、artifact check、完整 diff 与工作区状态。

## 10. 执行记录

2026-08-21：收到评审专家对远端 `01204ac` 的 Major Revision 预审意见；尚未修改论文，先创建本计划。

## 11. 最终结果

- 完成日期：尚未完成
- 最终状态：`未执行`
- 验证结果：尚未验证
- 论文修改 commit：尚未提交
- 计划归档 commit：无需在本文件中自我引用
- 相对原计划的偏差：无/尚未记录
