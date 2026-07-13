# ADMA 2026 投稿待办

最后更新：2026-07-13

- 论文截止：2026-07-17 AoE
- 补充材料截止：正文截止后 3 天
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
- [ ] 执行 `run_overlap_clean_mces.sh`，使用最终 overlap-clean `a2280d2` checkpoint 重算 mass/formula 的 Base 和代表性 seed-42 Transformer MCES@1，不与历史 checkpoint 的 MCES@1 混用。
- [ ] 在统一候选集、SMILES canonicalization 和评价代码下复现 JESTR。
- [ ] 在相同协议下复现 GLMR。
- [x] 当前采用无法完成统一复现时的降级方案：全文保留 `reported results` 标记，不声称严格 SOTA。
- [x] 在固定 alignment checkpoint 上为 pointwise/Transformer reranker 运行 seeds 42/43/44，报告均值和样本标准差。
- [x] 确认验证集和测试集始终使用 `force_include_positive=false`；mass/formula cache meta、生成日志和 label 一致性检查均通过。
- [x] 完成分子标签重复审计；identifier、原始/规范化 SMILES 和 InChIKey 两两交集均为 0，fold 标记无错配。
- [x] 使用 `run_rerank_overlap_sensitivity.py` 对 3 个 train--test 完全相同谱图输入完成剔除敏感性评估；12/12 组完成，排除 test cache 索引 5908/5909/5910 后三种子平均 Recall@1 仅变化 $-0.0045$ 至 $-0.0055$ 个百分点，主结论不变；实验代码 commit 为 `b20a2e8`。
- [x] 完成 6 个 train--val 完全相同输入的 overlap-clean 全流水线审计：排除 val 索引 7686/7687/7688/8464/8465/8466，alignment validation molecule keys 由 3386 降至 3384；best/stop epoch 仍为 16/21，reranker 12/12 完成、无错误，其中 9/12 组 best/stop epoch 对与历史流水线不同。实验代码 commit 为 `a2280d2`；该对比包含 alignment 重训，只能解释为流水线级敏感性，不是 6 条查询的隔离因果效应。
- [ ] 固化最终 checkpoint、`params.yaml`、代码 commit 和候选文件版本。
- [x] 将正文中的 `Required Ablations`、`we will` 等待办式内容替换为已完成的三种子 pointwise/Transformer 结果。
- [ ] 制作正式方法架构图，替换当前文本框占位图。
- [ ] 检查 SpecEmbedding 既有工作的引用和增量说明，避免重复声明已有贡献。
- [ ] 确认 AI 使用披露形式满足 ADMA 要求；必要时咨询 Program Chair。

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
- [x] 英文稿当前为 10 页，低于 15 页限制。
- [x] 作者、单位和 PDF 作者元数据已隐藏。
- [x] 未包含致谢和基金信息。
- [x] 已加入 AI 使用披露。
- [ ] 最终确认正文、参考文献和 limitation 总计不超过 15 页。
- [ ] 检查 PDF、源文件和补充材料中是否含姓名、用户名、单位或本地路径。
- [ ] 清理 `/data1/zyl`、Git 远端地址、公开仓库链接及 Git 历史信息。
- [ ] 准备不含 `.git` 历史的匿名代码压缩包。
- [ ] 确认补充材料不超过 20 MB；超过时改用匿名仓库。
- [ ] 确认匿名仓库在补充材料截止后不再修改。
- [ ] 在 CMT 中确认最终作者列表。
- [ ] 在 CMT 中完整申报利益冲突。
- [ ] 确认稿件未同时投稿其他 archival venue。
- [ ] 评审期间不新上传 arXiv 或个人主页。

## 实验结果记录

在完成实验后，将结果路径和关键指标记录在这里，避免只保留终端日志。MCES@1 为结构距离，越低越好。当前主结果采用 commit `a2280d2` 的 overlap-clean 全流水线；下表 MRR 保留日志中的 $[0,1]$ 量纲，$±$ 为 3 个 reranker seeds 的样本标准差。

| Candidate | Model | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR | MCES@1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mass | Clean Base | 47.4596 | 63.5111 | 71.0242 | 77.7626 | 0.5503 | 待重算 |
| mass | Pointwise | 67.4432±0.7357 | 77.0202±0.5911 | 80.1055±0.2077 | 82.2624±0.1588 | 0.7179±0.0066 | 不计算 |
| mass | Transformer | 68.5824±0.2035 | 77.3582±0.3610 | 79.9935±0.3405 | 82.2036±0.0668 | 0.7260±0.0025 | 待重算 |
| formula | Clean Base | 63.1009 | 75.7861 | 80.8669 | 84.9339 | 0.6886 | 待重算 |
| formula | Pointwise | 73.9709±0.5910 | 81.9359±0.2712 | 84.4934±0.0786 | 86.7813±0.2651 | 0.7764±0.0049 | 不计算 |
| formula | Transformer | 74.5671±0.2819 | 81.9682±0.2645 | 84.3757±0.0821 | 86.7719±0.1392 | 0.7802±0.0026 | 待重算 |

overlap-clean 的 mass/formula top-40 upper bound 分别为 83.8802%/89.2572%；Pointwise 相对 Clean Base 的 Recall@1 增益为 +19.9836/+10.8700 个百分点，Transformer 为 +21.1228/+11.4662 个百分点。完整逐项与聚合结果分别见
`checkpoints_rerank/a2280d2_massspecgym_nopretrain_valoverlapclean_topk40_multiseed/summary.csv`
和
`checkpoints_rerank/a2280d2_massspecgym_nopretrain_valoverlapclean_topk40_multiseed/summary_aggregate.csv`。

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
- [ ] 7 月 13 日：执行 `run_overlap_clean_mces.sh`，补算 overlap-clean 代表性 checkpoint 的 mass/formula MCES@1。
- [x] 7 月 10 日至 12 日：确定保留 reported-results 降级方案，不声称严格 SOTA；JESTR/GLMR 统一复现仍作为独立待办。
- [ ] 7 月 13 日至 14 日：消融、效率和案例分析。
- [ ] 7 月 15 日：更新图表和全文。
- [ ] 7 月 16 日：执行双盲、页数、引用和补充材料检查。
- [ ] 7 月 17 日 AoE：提交论文。
- [ ] 7 月 20 日 AoE 前：提交匿名补充材料。

## 临时记录

在此处记录实验异常、参数调整和需要进一步确认的问题。

<!-- 在此添加记录。 -->
