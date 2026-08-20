# ADMA 2026 投稿待办（历史基线；转投修订中）

最后更新：2026-08-20

- 当前转投暂定完稿日期：2026-08-31
- 下列 ADMA 截止日期仅作历史记录，不再是当前投稿日程。

- 论文截止：2026-07-17 AoE
- 补充材料：可选；若提交，截止为 2026-07-20 AoE（正文截止后 3 天）
- 官方征稿页面：https://adma2026.github.io/call_for_research_papers.html
- 英文稿：`paper/main.tex`
- 中文稿：`paper/main_cn.tex`
- 工作编译目录：`paper/build/`（Git 忽略）
- 可审计发布构建：`paper/release/`；执行 `bash paper/build_release.sh`，构建来源、工具链、
  页数和 SHA-256 见 `paper/release/build-manifest.yaml`
- 可选匿名代码包：执行 `python reproducibility/build_anonymous_supplement.py` 生成
  `dist/specembedding-anonymous-supplement.zip`；固定源码、哈希、独立解包测试和边界见
  `reproducibility/anonymous-supplement-validation.yaml`

状态约定：

- `[ ]`：未完成
- `[x]`：已完成
- `[-]`：决定不做，需要在条目后记录原因

## P0：投稿前必须完成

- [x] 计算历史 `d4c1f70` checkpoint 质量候选场景的 MCES@1（Base：16.4199；代表性 seed-42 Set Transformer：8.0209；越低越好）。
- [x] 使用 `run_overlap_clean_mces.sh` 完成最终 overlap-clean `a2280d2` checkpoint 的 MCES@1 重算：mass Base/seed-42 Transformer 为 15.3681/7.7057，formula 为 5.4389/3.0913；四项均完成 17,556/17,556 条查询，batch 状态为 `complete`。
- [-] 在统一候选集、SMILES canonicalization 和评价代码下复现 JESTR；当前采用
  reported-only 降级方案，不再作为本轮转投 P0，除非后续明确开启新复现任务。
- [-] 在相同协议下复现 GLMR；同上，外部数字仅作未复现、不可直接比较的跨论文背景。
- [x] 当前采用无法完成统一复现时的降级方案：全文保留 `reported results` 标记，不声称严格 SOTA。
- [x] 全文、双语 README、canonical manifest 和版本化分析报告统一标注本地
  exact-target-SMILES 单正例规则；与参考二维 InChIKey/可能多正例 evaluator 的差异
  尚未量化，因此不声称官方 evaluator 等价。
- [x] 在固定 alignment checkpoint 上为 pointwise/Transformer reranker 运行 seeds 42/43/44，报告均值和样本标准差。
- [x] 确认验证集和测试集始终使用 `force_include_positive=false`；mass/formula cache meta、生成日志和 label 一致性检查均通过。
- [x] 完成分子标签重复审计；identifier、原始/规范化 SMILES 和 InChIKey 两两交集均为 0，fold 标记无错配。
- [x] 使用 `run_rerank_overlap_sensitivity.py` 对 3 个 train--test 完全相同谱图输入完成剔除敏感性评估；12/12 组完成，排除 test cache 索引 5908/5909/5910 后三种子平均 Recall@1 仅变化 $-0.0045$ 至 $-0.0055$ 个百分点，主结论不变；实验代码 commit 为 `b20a2e8`。
- [x] 完成 6 个 train--val 完全相同输入的 overlap-clean 全流水线审计：排除 val 索引 7686/7687/7688/8464/8465/8466，alignment validation molecule keys 由 3386 降至 3384；best/stop epoch 仍为 16/21，reranker 12/12 完成、无错误，其中 9/12 组 best/stop epoch 对与历史流水线不同。实验代码 commit 为 `a2280d2`；该对比包含 alignment 重训，只能解释为流水线级敏感性，不是 6 条查询的隔离因果效应。
- [x] 已运行 `python freeze_adma2026_artifacts.py` 并生成 `paper/adma2026_artifact_manifest.json`：60 个 canonical artifacts 全部通过校验；2026-08-17 加入评价身份协议并将历史 `params.yaml` 绑定到 source-commit blob 后，manifest SHA-256 为 `ad92596e1ca60e8edc6b7be594bde7cbe314f5d08a0421551ba88eaad8207875`。该 manifest 仅作内部 inventory；其中引用的原始 status/log 可能含主机、用户、进程和 GPU 信息，不能直接按清单打包匿名附件。
- [x] 将正文中的 `Required Ablations`、`we will` 等待办式内容替换为已完成的三种子 pointwise/Transformer 结果。
- [x] 使用共享 TikZ 矢量源 `paper/figures/method_overview.tex` 制作正式方法架构图，并替换中英文稿的文本框占位图；图中区分跨模态对齐、top-40 检索/缓存和监督残差重排序，标明 pointwise/候选集合 Transformer 两种变体、基础分数跳连以及训练/验证测试协议。
- [x] 对照 SpecEmbedding 原文与 ACS 正式书目信息完成引用和增量审计：在引言、相关工作和方法处就地归因峰序列 Transformer backbone，准确区分其“重复谱图 SupCon + Tanimoto-MSE”与本文从零训练的跨模态目标；将本文贡献收窄为第二阶段非生成式残差 learning-to-rank；补齐 `97(37):20137--20146`。
- [x] 对照 ADMA CFP 与 Springer Nature AI policy 修正 AI 使用披露：声明移入 Introduction，覆盖全部章节，以及代码编辑、实验编排和一致性审计；明确数值来自软件流水线、作者核验全部 AI 辅助内容并承担责任。

## 历史 P1：核心消融实验与非当前可选分析

- [x] 基础检索器 vs. pointwise MLP vs. Transformer listwise reranker（3 个 reranker seeds）。
- [x] 比较候选间 self-attention：canonical alignment seed 42 内 Transformer 呈小幅优势；
  但 alignment seeds 42/43/44 分层后，mass/formula 的 alignment-level Recall@1 与 MRR
  差值都只有 2/3 为正，因此不将整体增益归因于 self-attention。
- [x] 分别审计只移除 residual shortcut，以及联合移除基础分数输入与 residual 路径；后者不是单因素 feature 消融。
- [x] 移除 rank embedding；六个同种子候选协议配对中 Recall@1/MRR 均提高，因此不把该设计写成已验证贡献。
- [x] 联合移除元素乘积与绝对差特征；仍保留 self-attention，不解释为无候选交互。
- [x] 仅 listwise CE vs. `CE + pairwise loss`。
- [-] 开启/关闭候选顺序随机打乱；当前初稿不新增实验，仅保留为未来可选分析。
- [-] 比较 `K=20/40/100/256`；当前初稿不新增实验，仅保留为未来可选分析。
- [-] 比较有无 SpecEmbedding 预训练；当前初稿不新增实验，仅保留为未来可选分析。
- [-] 分析不同候选召回上界下 reranker 的实际增益；当前初稿不新增实验，仅保留为未来可选分析。
- [x] 已完成 alignment seeds 42/43/44 的基础检索器、cache 与每个 alignment 下
  reranker seeds 42/43/44，共 36/36 组，并生成分层分析。三个 alignment-level estimates
  只提供内部描述性敏感性证据，不构成置信区间、显著性或一般端到端稳定性证明。

## 历史 P1：效率与可解释性（非当前目标）

- [-] 统计基础模型、reranker 和完整模型参数量；当前已有 reranker 参数量边界，完整效率画像不属于本轮待办。
- [-] 测量 cache 构造时间和磁盘占用；未测量并已在正文限制中披露。
- [-] 测量单查询及完整测试集 rerank latency；未测量并已在正文限制中披露。
- [-] 测量端到端 latency、峰值显存和吞吐量；未测量并已在正文限制中披露。
- [-] 与 GLMR 的生成式推理成本进行同硬件比较；不作未测量的效率主张。
- [-] 选择成功和失败案例，分析 reranker 调整排名的原因；不新增案例分析目标。
- [-] 统计真实分子从不同原始排名提升到 Top-1/5 的分布；不新增排名迁移目标。

## P1：论文完善

- [x] 补充固定基础检索器上三个 reranker seeds 的均值和样本标准差。
- [x] 将本地结果与外部 reported 结果拆为独立表；外部表明确 `not reproduced / not directly comparable`。
- [x] 补充数据集、完整候选池规模和平均候选数量统计。
- [x] 统一使用 `Recall@K`、`MRR`、`MCES@1` 等术语。
- [x] 2026-08-20 压缩并重组相关工作：保留 25 个既有引用及候选重排、联合嵌入、
  模拟/生成和 learning-to-rank 的证据链，合并重复的 Method Positioning 小节；英文
  Related Work 净减少约 80 词，未新增文献或扩大主张。
- [x] 根据受控消融收窄核心论点：在三个审计 alignment 内，两种监督残差 reranker
  均改善对应 base；Transformer 相对 Pointwise 的 Recall@1/MRR 方向都只有 2/3
  alignment 为正，因此不主张 self-attention 的稳健独立收益。
- [x] 2026-08-20 完成全文英文语言与证据边界润色；修正 canonical 定义、协议名、
  pool-by-reranker-seed 配对计数、pre-clean 指代和组件方向表述。
- [x] 检查中英文稿内容一致性（2026-08-17 逐表、逐证据边界复核；2026-08-20 在
  相关工作压缩和语言收口后再次同步复核）。
- [x] 2026-08-20 对全部 25 条参考文献逐项核对一手来源、作者、题名、年份、卷期页码、
  DOI/稳定链接和发表状态；补齐 7 条不完整元数据，MassSpecGym in the Wild 继续明确为
  arXiv preprint。

## 投稿合规

- [x] 使用 Springer LNCS/LNAI 模板。
- [x] 2026-08-20 从干净 source commit `033ac22`（论文实质 commit `076c5c5`）强制完整
  重建为英文 17 页、中文 15 页，并将当前 PDF、页数和 SHA-256 留存在 `paper/release/`；
  `paper/build` 已同步。用户决定完稿后再统一微调页数，故最终页数合规仍未完成。
- [x] 当前留存 PDF 的作者元数据为空或缺失；正文中的作者和单位已隐藏。
- [x] 未包含致谢和基金信息。
- [x] 已在 Introduction 中加入覆盖全部章节和实际辅助范围的 AI 使用披露。
- [ ] 最终确认正文、参考文献和 limitation 总计不超过 15 页。
- [ ] 检查 PDF、源文件和补充材料中是否含姓名、用户名、单位或本地路径；当前初稿 PDF、
  源码和匿名代码包均已完成零命中扫描，但最终提交副本仍需联合复查。
- [x] 清理当前受跟踪源码、配置和归档 notebook 中的私有机器绝对路径。
- [ ] 最终匿名检查 Git 远端地址、公开仓库链接及 Git 历史信息；当前 allowlist 代码包
  已通过上述扫描且不含 `.git`，最终提交副本仍需复扫。
- [x] 准备不含 `.git` 历史的匿名代码压缩包：source commit `c565d29`，40 个 allowlisted
  源文件，63,991 bytes，SHA-256
  `0e6b775a15ca0d54a58d6072ce1fbce0deb0951c0a357958a7d99e3f5c8baf52`；从包外 cwd
  独立执行 7 项测试和合成 `prepare -> train -> eval` CPU smoke 均通过。
- [x] 当前匿名压缩包不超过历史 20 MB 限制（63,991 bytes）；目标 venue 确定后仍需
  按其实际规则复核文件格式和上限。
- [-] 当前采用匿名单文件压缩包，不建立匿名仓库；若目标 venue 改要求仓库，再重新开放冻结检查。
- [ ] 目标 venue 确定后，在其投稿系统中确认最终作者列表。
- [ ] 目标 venue 确定后，按其规则完整申报利益冲突。
- [ ] 确认稿件未同时投稿其他 archival venue。
- [ ] 由作者确认是否存在与本稿相关的既有公开 preprint、paper announcement 或 workshop 版本；目标 venue 确定后按其披露规则处理。
- [ ] 目标 venue 确定后，遵守其评审期间公开预印本和宣传政策。

## 实验结果记录

在完成实验后，将结果路径和关键指标记录在这里，避免只保留终端日志。MCES@1 为结构距离，越低越好。当前主结果采用 commit `a2280d2` 的 overlap-clean 全流水线；下表 MRR 保留日志中的 $[0,1]$ 量纲，$±$ 为 3 个 reranker seeds 的样本标准差。

| Candidate | Model | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR | MCES@1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mass | Clean Base | 47.4596 | 63.5111 | 71.0242 | 77.7626 | 0.5503 | 15.3681 |
| mass | Pointwise | 67.4432±0.7357 | 77.0202±0.5911 | 80.1055±0.2077 | 82.2624±0.1588 | 0.7179±0.0066 | 不计算 |
| mass | Transformer | 68.5824±0.2035 | 77.3582±0.3610 | 79.9935±0.3405 | 82.2036±0.0668 | 0.7260±0.0025 | 7.7057（seed 42） |
| formula | Clean Base | 63.1009 | 75.7861 | 80.8669 | 84.9339 | 0.6886 | 5.4389 |
| formula | Pointwise | 73.9709±0.5910 | 81.9359±0.2712 | 84.4934±0.0786 | 86.7813±0.2651 | 0.7764±0.0049 | 不计算 |
| formula | Transformer | 74.5671±0.2819 | 81.9682±0.2645 | 84.3757±0.0821 | 86.7719±0.1392 | 0.7802±0.0026 | 3.0913（seed 42） |

overlap-clean 的 mass/formula top-40 upper bound 分别为 83.8802%/89.2572%；Pointwise 相对 Clean Base 的 Recall@1 增益为 +19.9836/+10.8700 个百分点，Transformer 为 +21.1228/+11.4662 个百分点。完整逐项与聚合结果分别见
`checkpoints_rerank/a2280d2_massspecgym_nopretrain_valoverlapclean_topk40_multiseed/summary.csv`
和
`checkpoints_rerank/a2280d2_massspecgym_nopretrain_valoverlapclean_topk40_multiseed/summary_aggregate.csv`。

overlap-clean MCES@1 batch 于 2026-07-13T15:35:46+08:00 完成，状态为 `complete`，四项均评估 17,556/17,556 条查询；source commit 为 `a2280d2`。mass 从 15.3681 降至 7.7057，绝对降低 7.6624（49.86%）；formula 从 5.4389 降至 3.0913，绝对降低 2.3476（43.16%）。结果与完整状态见
`checkpoints_rerank/a2280d2_massspecgym_nopretrain_valoverlapclean_topk40_mces_seed42_transformer/batch_status.json`。

以下 `d4c1f70` 表格仅保留为历史 MCES@1 和 train--test 敏感性的追溯记录，不应与 overlap-clean 的 Recall/MRR 混合用于同一主结果表。

| 实验 | 候选类型 | 随机种子 | Recall@1 | Recall@5 | Recall@20 | MRR | MCES@1 | 结果路径 | 状态 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Base retriever | mass | 42 | 43.78 | 61.10 | 75.55 | 51.96 | 16.42 | `checkpoints_rerank/d4c1f70_massspecgym_nopretrain_mass_topk40/eval_rerank.log` | 已完成 |
| Set Transformer | mass | 42 | 67.91 | 76.00 | 80.45 | 71.62 | 8.02 | `checkpoints_rerank/d4c1f70_massspecgym_nopretrain_mass_topk40/eval_rerank.log` | 已完成 |
| Base retriever | formula | 42 | 58.49 | 71.49 | 81.81 | 64.65 | 6.14 | `checkpoints_rerank/d4c1f70_massspecgym_nopretrain_formula_topk40/eval_rerank.log` | 已完成 |
| Set Transformer | formula | 42 | 74.33 | 80.64 | 85.07 | 77.27 | 3.13 | `checkpoints_rerank/d4c1f70_massspecgym_nopretrain_formula_topk40/eval_rerank.log` | 已完成 |
| Pointwise reranker | mass | 42/43/44 | 67.89±0.44 | 75.79±0.19 | 80.63±0.05 | 71.53±0.35 | 未重算 | `checkpoints_rerank/d4c1f70_massspecgym_nopretrain_topk40_multiseed/summary_aggregate.csv` | 已完成 |
| Set Transformer | mass | 42/43/44 | 67.75±0.31 | 75.88±0.18 | 80.63±0.15 | 71.52±0.24 | 未重算 | `checkpoints_rerank/d4c1f70_massspecgym_nopretrain_topk40_multiseed/summary_aggregate.csv` | 已完成 |
| Pointwise reranker | formula | 42/43/44 | 73.77±0.19 | 80.55±0.12 | 84.97±0.15 | 76.88±0.16 | 未重算 | `checkpoints_rerank/d4c1f70_massspecgym_nopretrain_topk40_multiseed/summary_aggregate.csv` | 已完成 |
| Set Transformer | formula | 42/43/44 | 73.98±0.84 | 80.27±0.54 | 85.07±0.05 | 76.95±0.69 | 未重算 | `checkpoints_rerank/d4c1f70_massspecgym_nopretrain_topk40_multiseed/summary_aggregate.csv` | 已完成 |

Train--test 输入重叠敏感性评估保留 17,553 条查询。三种子平均 Recall@1 的原始值
$\rightarrow$ 剔除后值分别为：mass Pointwise 67.8856 $\rightarrow$ 67.8801
（$-0.0055$ 个百分点）、mass Transformer 67.7546 $\rightarrow$ 67.7491
（$-0.0055$）、formula Pointwise 73.7735 $\rightarrow$ 73.7690（$-0.0045$）、
formula Transformer 73.9804 $\rightarrow$ 73.9760（$-0.0045$）。mass/formula
upper bound 分别从 82.1542%/87.3035% 变为 82.1512%/87.3013%；单个 seed
的最大绝对 Recall@1 变化为 0.0056 个百分点，MRR 最大绝对变化为 0.0001。
逐项与聚合结果分别见
`checkpoints_rerank/d4c1f70_massspecgym_nopretrain_topk40_input_overlap_sensitivity/summary.csv`
和
`checkpoints_rerank/d4c1f70_massspecgym_nopretrain_topk40_input_overlap_sensitivity/summary_aggregate.csv`；
该敏感性评估未重算 MCES@1。

Train--val overlap-clean 审计排除 6 条 validation 查询（两个分子组），alignment validation molecule keys 从 3386 降至 3384。clean 与历史 alignment 均在 epoch 16 达到最佳并在 epoch 21 early stop；clean validation contrastive loss 为 1.5294826125，与原 validation loss 不可直接数值比较。clean reranker 12/12 组完成且全部 early stop，其中 9/12 组的 best/stop epoch 对与历史流水线不同。这说明新旧完整流水线的模型选择并非逐项一致；由于 alignment 也重训，该差异不能单独归因于 validation 协议或这 6 条查询。详情见
`checkpoints_align/a2280d2_massspecgym_nopretrain_valoverlapclean/alignment_selection.json`
与
`checkpoints_rerank/a2280d2_massspecgym_nopretrain_valoverlapclean_topk40_multiseed/batch_status.json`。

## 论文数字来源

| 方法 | 数字来源 | 是否本地复现 | 使用限制 |
| --- | --- | --- | --- |
| SpecMolAlign | 本项目日志 | 是 | 可用于受控比较 |
| SpecEmbedding-Rerank canonical/组件表 | 本项目日志 | 是 | alignment 固定为 seed 42；reranker 使用 seeds 42/43/44；组件只在此 alignment 审计 |
| SpecEmbedding-Rerank 跨 alignment | 版本化分层分析 | 是 | alignment seeds 42/43/44；每个 alignment 内先聚合 reranker seeds 42/43/44；三个 estimates 仅描述性 |
| JESTR | GLMR 论文 Table 1 | 否 | 必须标注 `reported; not reproduced; not directly comparable` |
| GLMR | GLMR 论文 Table 1 | 否 | 必须标注 `reported; not reproduced; not directly comparable` |

## 建议时间线

- [x] 7 月 10 日至 12 日：完成质量候选场景的 MCES@1。
- [x] 7 月 10 日至 12 日：完成固定 alignment 的 reranker 多随机种子实验。
- [x] 7 月 12 日：完成 train--val overlap-clean alignment/cache/reranker 全流水线及 12 组多种子评估。
- [x] 7 月 13 日：执行 `run_overlap_clean_mces.sh`，补算 overlap-clean 代表性 checkpoint 的 mass/formula MCES@1。
- [x] 7 月 14 日：执行 `python freeze_adma2026_artifacts.py`，生成不含本机绝对路径或 hostname 的内部实验 artifact manifest；60 个 artifacts 校验通过。其中仍记录 Git commit，且所引用的原始 status/log 需要脱敏，不得未审查就直接放入匿名补充材料。
- [x] 7 月 14 日：完成共享中英文标签的 TikZ 正式方法图，替换两稿文本占位图；当时英文/中文稿编译为 11/10 页。
- [x] 7 月 14 日：完成 SpecEmbedding 贡献边界、正式书目信息和 ADMA/Springer AI 披露审计，并同步修正中英文稿。
- [x] 7 月 10 日至 12 日：确定 reported-only 降级方案；2026-08-17 已正式执行为当前证据边界，JESTR/GLMR 统一复现不再作为本轮开放待办。
- [x] 2026-08-20：完成显式 allowlist 匿名代码包、身份扫描、确定性重建及独立解包
  `prepare -> train -> eval` CPU cold smoke；验证记录见
  `reproducibility/anonymous-supplement-validation.yaml`。
- [-] 7 月 13 日至 14 日：消融、效率和案例分析；历史日程已结束，未完成部分不转为当前实验目标。
- [-] 7 月 15 日：更新其余图表并完成全文核对；历史日程已结束，当前以转投计划为准。
- [-] 7 月 16 日：执行最终双盲、页数、引用和可选补充材料检查；历史日程已结束，待目标 venue 确定后按其规则执行。
- [-] 7 月 17 日 AoE：提交论文；ADMA 历史截止已结束，不再执行。
- [-] 7 月 20 日 AoE 前：若选择提交补充材料，上传匿名单文件或冻结匿名仓库；ADMA 历史截止已结束，不再执行。

## 临时记录

在此处记录实验异常、参数调整和需要进一步确认的问题。

<!-- 在此添加记录。 -->
