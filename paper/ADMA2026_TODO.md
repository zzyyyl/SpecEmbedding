# ADMA 2026 投稿待办

最后更新：2026-07-14

- 论文截止：2026-07-17 AoE
- 补充材料：可选；若提交，截止为 2026-07-20 AoE（正文截止后 3 天）
- 官方征稿页面：https://adma2026.github.io/call_for_research_papers.html
- 英文稿：`paper/main.tex`
- 中文稿：`paper/main_cn.tex`
- 编译目录：`paper/build/`

状态约定：

- `[ ]`：未完成
- `[x]`：已完成
- `[-]`：决定不做，需要在条目后记录原因

## P0：投稿前必须完成

- [x] 计算历史 `d4c1f70` checkpoint 质量候选场景的 MCES@1（Base：16.4199；代表性 seed-42 Set Transformer：8.0209；越低越好）。
- [x] 使用 `run_overlap_clean_mces.sh` 完成最终 overlap-clean `a2280d2` checkpoint 的 MCES@1 重算：mass Base/seed-42 Transformer 为 15.3681/7.7057，formula 为 5.4389/3.0913；四项均完成 17,556/17,556 条查询，batch 状态为 `complete`。
- [ ] 在统一候选集、SMILES canonicalization 和评价代码下复现 JESTR。
- [ ] 在相同协议下复现 GLMR。
- [x] 当前采用无法完成统一复现时的降级方案：全文保留 `reported results` 标记，不声称严格 SOTA。
- [x] 在固定 alignment checkpoint 上为 pointwise/Transformer reranker 运行 seeds 42/43/44，报告均值和样本标准差。
- [x] 确认验证集和测试集始终使用 `force_include_positive=false`；mass/formula cache meta、生成日志和 label 一致性检查均通过。
- [x] 完成分子标签重复审计；identifier、原始/规范化 SMILES 和 InChIKey 两两交集均为 0，fold 标记无错配。
- [x] 使用 `run_rerank_overlap_sensitivity.py` 对 3 个 train--test 完全相同谱图输入完成剔除敏感性评估；12/12 组完成，排除 test cache 索引 5908/5909/5910 后三种子平均 Recall@1 仅变化 $-0.0045$ 至 $-0.0055$ 个百分点，主结论不变；实验代码 commit 为 `b20a2e8`。
- [x] 完成 6 个 train--val 完全相同输入的 overlap-clean 全流水线审计：排除 val 索引 7686/7687/7688/8464/8465/8466，alignment validation molecule keys 由 3386 降至 3384；best/stop epoch 仍为 16/21，reranker 12/12 完成、无错误，其中 9/12 组 best/stop epoch 对与历史流水线不同。实验代码 commit 为 `a2280d2`；该对比包含 alignment 重训，只能解释为流水线级敏感性，不是 6 条查询的隔离因果效应。
- [x] 已运行 `python freeze_adma2026_artifacts.py` 并生成 `paper/adma2026_artifact_manifest.json`：60 个 canonical artifacts 全部通过校验，manifest SHA-256 为 `e00634b13d79ab1f4e7cbb38339f89302a98082e75f5daf54e3e0daef4702245`。该 manifest 仅作内部 inventory；其中引用的原始 status/log 可能含主机、用户、进程和 GPU 信息，不能直接按清单打包匿名附件。
- [x] 将正文中的 `Required Ablations`、`we will` 等待办式内容替换为已完成的三种子 pointwise/Transformer 结果。
- [x] 使用共享 TikZ 矢量源 `paper/figures/method_overview.tex` 制作正式方法架构图，并替换中英文稿的文本框占位图；图中区分跨模态对齐、top-40 检索/缓存和监督残差重排序，标明 pointwise/set-aware 两种变体、基础分数跳连以及训练/验证测试协议。
- [x] 对照 SpecEmbedding 原文与 ACS 正式书目信息完成引用和增量审计：在引言、相关工作和方法处就地归因峰序列 Transformer backbone，准确区分其“重复谱图 SupCon + Tanimoto-MSE”与本文从零训练的跨模态目标；将本文贡献收窄为第二阶段非生成式残差 learning-to-rank；补齐 `97(37):20137--20146`。
- [x] 对照 ADMA CFP 与 Springer Nature AI policy 修正 AI 使用披露：声明移入 Introduction，覆盖全部章节，以及代码编辑、实验编排和一致性审计；明确数值来自软件流水线、作者核验全部 AI 辅助内容并承担责任。

## P1：核心消融实验

- [x] 基础检索器 vs. pointwise MLP vs. Transformer listwise reranker（3 个 reranker seeds）。
- [x] 比较候选间 self-attention：overlap-clean 流水线中 Transformer 平均 Recall@1 比 Pointwise 高 1.1392（mass）/0.5962（formula）个百分点，6/6 个成对 seed 获胜；但这与历史 checkpoint 的 3/6 胜场不一致，因此仅能报告小幅优势且对 alignment/checkpoint 敏感，不将整体增益归因于 self-attention。
- [ ] 移除基础检索分数。
- [ ] 移除 rank embedding。
- [ ] 移除元素乘积特征。
- [ ] 移除绝对差特征。
- [ ] 仅 listwise CE vs. `CE + pairwise loss`。
- [ ] 开启/关闭候选顺序随机打乱。
- [ ] 比较 `K=20/40/100/256`。
- [ ] 比较有无 SpecEmbedding 预训练。
- [ ] 分析不同候选召回上界下 reranker 的实际增益。
- [ ] 若要报告端到端不确定性，还需使用多个 alignment seeds 重训基础检索器并重建 cache。

## P1：效率与可解释性

- [ ] 统计基础模型、reranker 和完整模型参数量。
- [ ] 测量 cache 构造时间和磁盘占用。
- [ ] 测量单查询及完整测试集 rerank latency。
- [ ] 测量端到端 latency、峰值显存和吞吐量。
- [ ] 与 GLMR 的生成式推理成本进行同硬件比较。
- [ ] 选择成功和失败案例，分析 reranker 调整排名的原因。
- [ ] 统计真实分子从不同原始排名提升到 Top-1/5 的分布。

## P1：论文完善

- [x] 补充固定基础检索器上三个 reranker seeds 的均值和样本标准差。
- [x] 在结果表中明确区分本地复现结果和论文报告结果。
- [ ] 补充数据集、完整候选池规模和平均候选数量统计。
- [ ] 统一使用 `Recall@K`、`MRR`、`MCES@1` 等术语。
- [ ] 压缩相关工作，避免 JESTR/GLMR 方法介绍喧宾夺主。
- [x] 根据受控消融收窄核心论点：监督残差重排序稳定有效；overlap-clean 运行中 Transformer 有小幅一致优势，但优势对 alignment/checkpoint 敏感，暂不主张 self-attention 的稳健独立收益。
- [ ] 完成人工英文润色。
- [ ] 检查中英文稿内容一致性。
- [ ] 核对全部参考文献的作者、年份、页码和 DOI。

## 投稿合规

- [x] 使用 Springer LNCS/LNAI 模板。
- [x] 英文稿当前为 11 页，低于 15 页限制。
- [x] 作者、单位和 PDF 作者元数据已隐藏。
- [x] 未包含致谢和基金信息。
- [x] 已在 Introduction 中加入覆盖全部章节和实际辅助范围的 AI 使用披露。
- [ ] 最终确认正文、参考文献和 limitation 总计不超过 15 页。
- [ ] 检查 PDF、源文件和补充材料中是否含姓名、用户名、单位或本地路径。
- [ ] 清理 `/data1/zyl`、Git 远端地址、公开仓库链接及 Git 历史信息。
- [ ] 准备不含 `.git` 历史的匿名代码压缩包。
- [ ] 确认补充材料不超过 20 MB；超过时改用匿名仓库。
- [ ] 确认匿名仓库在补充材料截止后不再修改。
- [ ] 在 CMT 中确认最终作者列表。
- [ ] 在 CMT 中完整申报利益冲突。
- [ ] 确认稿件未同时投稿其他 archival venue。
- [ ] 由作者确认是否存在与本稿相关的既有公开 preprint、paper announcement 或 workshop 版本；若存在，按官方 CFP 至少提前 24 小时联系 Program Chair 并完成披露。
- [ ] 评审期间不新上传 arXiv 或个人主页。

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
| SpecEmbedding-Rerank | 本项目日志 | 是 | alignment 固定为 seed 42；reranker 使用 seeds 42/43/44 |
| JESTR | GLMR 论文 Table 1 | 否 | 必须标注 `reported` |
| GLMR | GLMR 论文 Table 1 | 否 | 必须标注 `reported` |

## 建议时间线

- [x] 7 月 10 日至 12 日：完成质量候选场景的 MCES@1。
- [x] 7 月 10 日至 12 日：完成固定 alignment 的 reranker 多随机种子实验。
- [x] 7 月 12 日：完成 train--val overlap-clean alignment/cache/reranker 全流水线及 12 组多种子评估。
- [x] 7 月 13 日：执行 `run_overlap_clean_mces.sh`，补算 overlap-clean 代表性 checkpoint 的 mass/formula MCES@1。
- [x] 7 月 14 日：执行 `python freeze_adma2026_artifacts.py`，生成不含本机绝对路径或 hostname 的内部实验 artifact manifest；60 个 artifacts 校验通过。其中仍记录 Git commit，且所引用的原始 status/log 需要脱敏，不得未审查就直接放入匿名补充材料。
- [x] 7 月 14 日：完成共享中英文标签的 TikZ 正式方法图，替换两稿文本占位图；英文/中文稿编译后仍为 11/10 页。
- [x] 7 月 14 日：完成 SpecEmbedding 贡献边界、正式书目信息和 ADMA/Springer AI 披露审计，并同步修正中英文稿。
- [x] 7 月 10 日至 12 日：确定保留 reported-results 降级方案，不声称严格 SOTA；JESTR/GLMR 统一复现仍作为独立待办。
- [ ] 7 月 13 日至 14 日：消融、效率和案例分析。
- [ ] 7 月 15 日：更新其余图表并完成全文核对。
- [ ] 7 月 16 日：执行最终双盲、页数、引用和可选补充材料检查。
- [ ] 7 月 17 日 AoE：提交论文。
- [ ] 7 月 20 日 AoE 前：若选择提交补充材料，上传匿名单文件或冻结匿名仓库。

## 临时记录

在此处记录实验异常、参数调整和需要进一步确认的问题。

<!-- 在此添加记录。 -->
