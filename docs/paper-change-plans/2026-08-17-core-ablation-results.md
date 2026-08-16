# 论文修改计划：核心组件消融结果入稿

- 状态：`已执行`
- 创建日期：2026-08-17
- 最后更新：2026-08-17
- 负责人：Codex（作者负责最终学术判断与文字确认）
- 关联论文：`paper/main.tex` / `paper/main_cn.tex`
- 计划约束：不新增训练或评价实验；仅使用已冻结的 canonical full-model 与 30 组核心消融结果；不改变核心主张或事后升级主模型

## 1. 修改目标与动机

将 2026-08-15 完成的 30 组核心组件消融转换为可由原始工件重建、数字可反查、
中英文口径一致的论文证据。当前论文只报告 Pointwise 与 Set Transformer 的受控比较，
并明确把 feature/loss ablation 留在证据范围之外；继续保留该表述会遗漏已经完成的实验，
也无法核验论文中每个模型组件是否有独立证据支持。

本轮首先冻结确定性的分析口径和内容寻址 manifest，再依据结果决定哪些组件只能作为
设计选择、哪些结果允许作有限的描述性陈述。核心结论仍限定为：在冻结跨模态检索器之上，
监督、非生成式的第二阶段 learning-to-rank 在当前协议下有效。不得因查看消融结果后发现某个
变体更好，就把该变体事后升级为新的主模型。

## 2. 当前证据与问题定位

### 2.1 已完成实验

- canonical full-model 对照位于
  `checkpoints_rerank/a2280d2_massspecgym_nopretrain_valoverlapclean_topk40_multiseed/`，
  固定 alignment seed 42，包含 mass/formula 两类候选和 reranker seeds 42/43/44。
- 核心消融位于
  `checkpoints_rerank/transfer2026_canonicalalignseed42_topk40_core_ablations/`，覆盖
  5 ablations × 2 candidate types × 3 reranker seeds = 30 组。
- `retry_status.json` 的最终状态为 `complete`，30/30 个实验的最新 attempt 均为
  `complete`；历史 8 个失败 attempt 均由 GPU 门禁或抢占造成并已恢复。
- 最终 `batch_status.json` 为 `complete`、`errors=[]`。其中 29 个 `skipped` 表示最后一次
  supervisor 重入复用了已经成功的 attempt，不表示结果缺失；最后 1 个结果在本次重入完成。
- `summary.csv` 含 30 个唯一的 seed-level 消融结果；`summary_aggregate.csv` 含 10 个
  candidate × ablation 组合，每组均为 3 个种子。
- full 对照提交为 `a2280d2828ce872da1f69319b49e0ef7f1bed572`，消融提交为
  `8bd2ca439bc972c0ada88b6ad8b746372040df14`。两提交间 `params.yaml`、
  `SpecEmbedding/`、`train_rerank.py`、`eval_rerank.py` 和
  `run_rerank_multiseed.py` 无差异；source commit 仍须分别如实记录。
- 两批结果的 `params.yaml` SHA-256 均为
  `b6260dad043f9c0f1cb4ff36dc7b7bf8bac4481998aa36b4d55f44392e1b0bc6`，
  并复用相同 canonical seed-42 top-40 caches、validation 排除索引
  `7686 7687 7688 8464 8465 8466`、模型选择指标和训练预算。

### 2.2 实现语义边界

- `no_residual` 只关闭 residual base-score shortcut，仍保留 base score 作为 MLP 输入。
- `no_base_score` 同时关闭 base-score feature 与 residual shortcut，不能表述为纯粹的
  单因素 base-score feature 消融。
- `no_rank_embedding` 只关闭 rank embedding，MLP 输入宽度以零填充方式保持不变。
- `no_interaction_features` 关闭 product 与 absolute-difference 显式跨模态特征，
  不关闭 candidate self-attention，不能表述为“无候选交互”。
- `listwise_only` 令 pairwise loss 权重 `lambda_pair=0`，其余模型结构不变。

### 2.3 当前稿件问题

- 英文 `paper/main.tex` 的 Controlled Reranker Ablation 小节仍称 feature/loss
  removal 未报告；中文稿存在对应旧表述。
- 贡献、讨论和结论尚未依据消融结果审计组件级主张。
- 原始 summary 位于 Git 忽略的 checkpoint 目录；若不生成受版本控制、带输入哈希的
  派生分析，论文数字无法只通过仓库中的分析产物反查。
- 现有结果只有 3 个 reranker seeds 和一个固定 alignment checkpoint；它们支持受控的
  描述性比较，不支持显著性、因果性或跨 alignment 的组件稳健性结论。

## 3. 修改范围

### 3.1 涉及文件与章节

- [x] `analyze_core_ablations.py`：新增确定性的原始工件审计、同 seed 配对统计、
  组件主张门槛和 manifest 生成入口。
- [x] `tests/test_core_ablation_analysis.py`：覆盖矩阵完整性、配对方向、语义门槛、
  非有限数值/协议漂移拒绝和确定性输出。
- [x] `analysis/transfer2026_core_ablations/`：保存 `summary_long.csv`、
  `paired_deltas.csv`、`ablation_summary.csv`、`report.md` 和
  `analysis_manifest.json`。
- [x] `paper/TRANSFER_2026_PLAN.md`：回填 30/30 消融完成记录、实验冻结状态和新的
  执行顺序；top-K、效率、排名迁移与 venue 未完成项保持真实状态。
- [x] `paper/main.tex`：更新 Contributions、Controlled Reranker Ablation、Discussion、
  Limitations 和 Conclusion；仅在组件结果确实要求收窄核心主张时同步修改 Abstract。
- [x] `paper/main_cn.tex`：与英文稿同步相同数字、方向、术语和证据边界。

### 3.2 明确不做的事项

- 不新增训练、评价、MCES、top-K、效率、排名迁移或 P2 消融。
- 不改变 candidate identity、cache、split、validation exclusion 或 test-query 协议。
- 不把表现更好的 removal 变体升级为新的主模型，也不补跑其跨-alignment 矩阵。
- 不改外部 baseline 表或引入新的 SOTA 比较。
- 不声称统计显著性、因果贡献、跨数据集泛化或跨 alignment 的组件稳健性。
- 本轮不新增文献；若论文修改需要新引用、扩大章节或改变核心主张，则触发
  `已偏离待确认`。

## 4. 具体内容设计

### 4.1 分析口径

1. 以 `(candidate_type, reranker_seed)` 配对 full Set Transformer 与每个 removal。
2. delta 固定定义为 `full - ablation`；Top-k 单位为百分点，MRR 使用 raw [0, 1]。
3. 每个 candidate × ablation 报 full mean/sample-SD、ablation mean/sample-SD、
   paired-delta mean/sample-SD，以及 3 个配对差值的正/零/负方向计数。
4. 不展平或伪增样本量；三个 reranker seeds 只作为当前固定 alignment 内的描述性重复。
5. 主文紧凑报告 Recall@1 与 MRR；Recall@5/10/20 保留在可追溯分析文件中。

### 4.2 组件主张门槛

- 只有当 full 相对相应 removal 在 mass 和 formula 两种候选协议下的 MRR 平均差均为正，
  且每种协议的三个同-seed 差值方向一致为正时，才标记为“通过严格描述性组件门槛”。
- 若均值方向一致但 seed 方向混合，只能写“均值倾向于 full，但随机种子方向不稳”。
- 若 mass/formula 均值方向不同，写“候选协议间方向混合，未显示稳健独立收益”。
- 若 removal 在两类候选、全部 seeds 上均优于 full，必须明确写“该组件未获当前结果支持”，
  并从贡献性措辞中删除；不得事后重定义主模型。
- 所有门槛均为描述性决策规则，不是置信区间或假设检验。

### 4.3 表格与正文

- 在现有 Controlled Reranker Ablation 小节内用紧凑表替换“未报告 feature/loss
  ablation”的旧文字，每类候选报告 full 与五个 removal 的 Recall@1/MRR mean ± SD。
- 正文解释配对 delta 和方向，特别区分 residual shortcut、联合 base-score pathway、
  显式 product/absolute-difference 特征和 candidate self-attention。
- Contributions 只保留由多-alignment 主结果和本轮组件审计共同支持的贡献；组件未过门槛时
  改为设计描述，不称为已验证的独立贡献。
- Discussion/Limitations 明确本轮固定 alignment seed 42、仅三个 reranker seeds、
  无显著性检验和未选择 removal 作为新主模型的冻结后边界。
- Conclusion 与摘要不得超出结果表和 claim gate；中英文保持相同数字精度与方向。
- 英文当前约 15 页，优先替换旧限制性文字并压缩重复解释，不无约束增加篇幅。

## 5. 引用文献与真实性核验

本轮不新增、删除或修改引用与 BibTeX。现有文献不承担本轮内部实验数值的证据功能；
所有新增数字仅引用原始实验工件和生成的 analysis manifest。若执行时需要新增文献，
必须将计划改为 `已偏离待确认`，逐条核验题名、作者、年份、出版物和 DOI 后再继续。

| 引用键/暂定键 | 文献 | 支持的论断 | 一手来源 | 核验状态 |
|---|---|---|---|---|
| 无新增引用 | 不适用 | 本轮仅报告内部受控实验 | 原始 status/log/summary 与 analysis manifest | 不适用 |

## 6. 实验与计算边界

- 是否新增训练实验：否。
- 是否新增评价运行：否。
- 是否进行静态数据统计：是；从已冻结 full/ablation summary 重算 mean、sample-SD、
  同-seed paired delta 和方向计数，并验证原始工件/输出 SHA-256。
- 是否只修改文字、公式或引用：否；还会新增确定性分析脚本、测试和派生表格。

不得将上述静态重算写成新的模型实验，也不得把方向计数解释为显著性检验。

## 7. 分步执行清单

- [x] 步骤 1：核验 full 与 30 组消融的矩阵、status、日志、checkpoint、cache、params、
  validation/test 协议及两个 source commits 间的运行时代码兼容性。
- [x] 步骤 2：实现 `analyze_core_ablations.py` 与单元测试，固定配对方向、单位、
  组件门槛和失败条件。
- [x] 步骤 3：从原始工件生成 `analysis/transfer2026_core_ablations/`，重复运行确认
  字节级确定性，并核对所有 input/output hashes。
- [x] 步骤 4：回填 `paper/TRANSFER_2026_PLAN.md` 的核心消融验收记录、实验冻结状态和
  下一步顺序；仍缺失项目不误勾选。
- [x] 步骤 5：提交分析与计划回填，记录分析 commit；真正修改论文前将本计划状态改为
  `执行中`。
- [x] 步骤 6：按 claim gate 重写英文稿相关表格、贡献、结果、讨论、限制与结论。
- [x] 步骤 7：同步中文稿件，自动反查所有表格数字、delta、符号和单位。
- [x] 步骤 8：核验本轮无新增引用且现有 BibTeX/交叉引用未被破坏。
- [x] 步骤 9：编译双语稿并检查页数、引用、交叉引用、LaTeX error 与版面警告。
- [x] 步骤 10：检查 `git diff`，确认无新实验、无无关改动、无证据越界。
- [x] 步骤 11：使用 Conventional Commit 提交论文实质修改。
- [x] 步骤 12：回填论文 commit 和最终结果，将状态设为 `已执行`，再以独立文档
  commit 归档本计划。

## 8. 风险、证据边界与待确认事项

- full 与 removal 使用相同 cache 和参数配置，但只覆盖 alignment seed 42；组件级结论
  不得外推到 alignment seeds 43/44。
- `no_base_score` 是联合 removal，无法把影响唯一归因于 feature input 或 residual shortcut。
- `no_interaction_features` 不移除 self-attention；不得用它证明候选间注意力有无价值。
- 三个 seeds 只能给出 sample-SD 和配对方向，不能支撑显著性或稳健泛化结论。
- 若工件审计发现 fingerprint/cache/protocol 不一致、缺失日志/checkpoint、summary 与日志
  不符，立即标记 `已偏离待确认` 并停止入稿。
- 若结果要求改变主模型、补跑跨-alignment 消融、引入新实验或改变核心结论，立即标记
  `已偏离待确认` 并等待用户确认。
- top-K、效率和排名迁移尚未完成；本计划不以它们为由推迟 8/23 内容冻结，也不替它们
  补写未经测量的结论。
- 目标 venue 仍须尽快由作者确认；若 venue 模板或页数规则实质改变表格范围，先更新本计划。

## 9. 验证方案

- [x] 原始工件审计：30/30 latest complete、每组 seeds 42/43/44、`errors=[]`、
  `params_sha256`/cache/split/exclusions 一致；test log 为 17,556 queries、MCES skipped。
- [x] 数字重算：使用 sample SD (`ddof=1`)；同-seed delta=`full-ablation`；CSV 与报告显示值
  可从 seed-level rows 反查。
- [x] 确定性与 provenance：连续两次生成产物字节一致；manifest 记录全部输入和输出的
  repository-relative path、size 与 SHA-256。
- [x] 代码检查：`ruff check analyze_core_ablations.py tests/test_core_ablation_analysis.py`、
  `python -m py_compile analyze_core_ablations.py tests/test_core_ablation_analysis.py`、
  相关及完整单元测试通过。
- [x] 英文稿编译：在 `paper/` 中使用现有 `latexmk`/构建方式生成 `main.pdf`，检查
  LaTeX Error、undefined citation/reference、Overfull 和最终页数。
- [x] 中文稿编译：在 `paper/` 中使用 XeLaTeX 现有构建方式生成中文 PDF并检查同类错误。
- [x] 引用和 BibTeX 检查：确认 `paper/references.bib` 无改动，编译日志无引用错误。
- [x] 中英文内容一致性检查：自动提取消融表数字并人工核对术语、方向、限制和贡献表述。
- [x] 旧表述扫描：清除“feature/loss removal 未报告”及中文对应过时句，但不扩大其它章节。
- [x] Git 检查：`git diff --check`、完整 diff、工作区与提交范围检查。

## 10. 执行记录

### 2026-08-17：论文修改前的分析准备完成

- 新增 `analyze_core_ablations.py`、9 项定向测试和
  `analysis/transfer2026_core_ablations/` 五个受版本控制的派生产物。
- 审计 6 个 full Transformer attempts、30 个最终消融 attempts、38 个历史 attempt
  statuses、6 个 canonical cache hashes、训练/评价日志、checkpoint、selection、
  params、validation exclusions、17,556-query test 协议和 source-commit 运行代码。
- 36 条长表、30 个同 seed 配对差值和 10 个聚合单元全部通过独立数字反查；delta
  固定为 `full - ablation`，sample SD 使用 `ddof=1`。
- claim gate 为 `component_claims_require_narrowing`，0/5 组件通过严格描述性门槛；
  `no_rank_embedding` 在两候选协议、全部六个配对单元同时提高 Top-1 和 MRR。
- 连续两次全量生成的五个输出字节一致；manifest 中 241 个唯一文件的路径、大小和
  SHA-256 经独立审计全部匹配，且未包含绝对路径、主机名、用户名或 GPU UUID。
- 验证通过：70 项完整单元测试、Ruff、`py_compile`、`git diff --check`。
- 分析与转投计划回填 commit：
  `2701782a1e193d442e09a9fc49668fbd161416f5`。
- 截至分析提交时仍未修改 `paper/main.tex` 或 `paper/main_cn.tex`。

### 2026-08-17：开始双语稿修改

- 用户确认继续下一步；专项计划在任何论文正文改动前切换为 `执行中`。
- 后续严格按步骤 6–12 执行，不扩大实验、引用或论文主张范围。

### 2026-08-17：核心消融完成双语入稿与验收

- 用一张 8 行横向表同步替换中英文受控 reranker 表：Base、Pointwise、完整候选集合
  Transformer 与五项移除设置共同报告质量/分子式候选的 Recall@1 和 MRR；所有学习行均为
  reranker seeds 42--44 的均值 $\pm$ 样本标准差。
- 自动从 `ablation_summary.csv` 重建表内 full/removal 数字并逐格匹配双稿；再次从
  `paired_deltas.csv` 验证严格 MRR 门槛为 0/5，且去除排名嵌入在六个候选协议与种子配对中
  同时提高 Recall@1 和 MRR。
- 将摘要、贡献、方法定位、实现细节、结果、讨论、局限与结论的最强主张统一收窄为：当前证据
  支持监督、非生成式第二阶段重排序框架，但不建立 self-attention 或任一受审计设计的独立、
  因果或跨 alignment 收益；不事后把无排名嵌入版本升级为新主模型。
- 明确五项实现语义，特别记录 `no_residual` 仍保留基础分数 MLP 输入、`no_base_score` 是联合
  移除、显式乘积/绝对差特征移除仍保留 self-attention，以及 listwise-only 令
  $\lambda=0$。
- 未新增或修改引用；`paper/references.bib` 无 diff。旧的“feature/loss removal 未报告”双语
  表述已清除；top-$K$、候选顺序和实测效率仍如实保留为未完成项。
- 英文以 `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` 编译为 16 页；中文以
  `latexmk -xelatex -interaction=nonstopmode -halt-on-error main_cn.tex` 编译为 15 页。两份最终
  日志均无 LaTeX error、undefined citation/reference、Overfull 或 Underfull。仅保留英文
  `amsmath` 的 `\vec` 重定义警告与中文 Fandol 字体的 CJK script 警告。
- 用户明确要求当前以完成既定待办和保证内容充实可靠为先，英文页数上限暂不作为阻塞条件，
  待完稿后统一微调；本轮不为页数新增目标或删减证据边界。
- 英文、中文审阅代理各一名及一名数字审阅代理完成只读终审，未发现阻塞性数字、语义、表结构或
  双语一致性问题。最终 `git diff --check` 通过，生成的编译中间文件未纳入版本控制。
- 论文实质修改 commit：
  `bc0dbf6f6064760e51cb32000c7d9494197b9048`（`docs(paper): 纳入核心组件消融结果`）。

## 11. 最终结果

- 完成日期：2026-08-17
- 最终状态：`已执行`
- 验证结果：分析工件、双语数值反查、引用/交叉引用、双语编译、版面日志、完整 diff 与
  提交范围均已核验；英文 16 页、中文 15 页，页数微调按用户决定延后
- 分析 commit：`2701782a1e193d442e09a9fc49668fbd161416f5`
- 论文修改 commit：`bc0dbf6f6064760e51cb32000c7d9494197b9048`
- 计划归档 commit：无需在本文件中自我引用
- 相对原计划的偏差：未新增实验、引用或模型选择；英文最终为 16 页，用户明确将页数上限
  微调延后，因此不作为本轮完成阻塞项
