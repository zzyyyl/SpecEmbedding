# ADMA 2026 投稿待办

最后更新：2026-07-10

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

- [ ] 计算质量候选场景的 MCES@1。
- [ ] 在统一候选集、SMILES canonicalization 和评价代码下复现 JESTR。
- [ ] 在相同协议下复现 GLMR。
- [ ] 如果无法完成 JESTR/GLMR 复现，全文保留 `reported results` 标记，不声称严格 SOTA。
- [ ] 主要实验至少运行 3 个随机种子，报告均值和标准差。
- [ ] 确认验证集和测试集始终使用 `force_include_positive=false`。
- [ ] 再次核验训练、验证、测试分子划分不存在样本或结构泄露。
- [ ] 固化最终 checkpoint、`params.yaml`、代码 commit 和候选文件版本。
- [ ] 将正文中的 `Required Ablations`、`we will` 等待办式内容替换为真实实验结果。
- [ ] 制作正式方法架构图，替换当前文本框占位图。
- [ ] 检查 SpecEmbedding 既有工作的引用和增量说明，避免重复声明已有贡献。
- [ ] 确认 AI 使用披露形式满足 ADMA 要求；必要时咨询 Program Chair。

## P1：核心消融实验

- [ ] 基础检索器 vs. pointwise MLP vs. Transformer listwise reranker。
- [ ] 移除候选间 self-attention。
- [ ] 移除基础检索分数。
- [ ] 移除 rank embedding。
- [ ] 移除元素乘积特征。
- [ ] 移除绝对差特征。
- [ ] 仅 listwise CE vs. `CE + pairwise loss`。
- [ ] 开启/关闭候选顺序随机打乱。
- [ ] 比较 `K=20/40/100/256`。
- [ ] 比较有无 SpecEmbedding 预训练。
- [ ] 分析不同候选召回上界下 reranker 的实际增益。

## P1：效率与可解释性

- [ ] 统计基础模型、reranker 和完整模型参数量。
- [ ] 测量 cache 构造时间和磁盘占用。
- [ ] 测量单查询及完整测试集 rerank latency。
- [ ] 测量端到端 latency、峰值显存和吞吐量。
- [ ] 与 GLMR 的生成式推理成本进行同硬件比较。
- [ ] 选择成功和失败案例，分析 reranker 调整排名的原因。
- [ ] 统计真实分子从不同原始排名提升到 Top-1/5 的分布。

## P1：论文完善

- [ ] 补充主要结果的标准差或置信区间。
- [ ] 在结果表中明确区分本地复现结果和论文报告结果。
- [ ] 补充数据集、完整候选池规模和平均候选数量统计。
- [ ] 统一使用 `Recall@K`、`MRR`、`MCES@1` 等术语。
- [ ] 压缩相关工作，避免 JESTR/GLMR 方法介绍喧宾夺主。
- [ ] 强化核心论点：候选集合关系能够提供独立于 pairwise 相似度的排序信号。
- [ ] 完成人工英文润色。
- [ ] 检查中英文稿内容一致性。
- [ ] 核对全部参考文献的作者、年份、页码和 DOI。

## 投稿合规

- [x] 使用 Springer LNCS/LNAI 模板。
- [x] 英文稿当前为 9 页，低于 15 页限制。
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

在完成实验后，将结果路径和关键指标记录在这里，避免只保留终端日志。

| 实验 | 候选类型 | 随机种子 | Recall@1 | Recall@5 | Recall@20 | MRR | MCES@1 | 结果路径 | 状态 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Base retriever | mass | 42 | 43.78 | 61.10 | 75.55 | 51.96 | 待计算 | `checkpoints_align/d4c1f70_massspecgym_nopretrain/` | 已完成 |
| Full reranker | mass | 42 | 67.91 | 76.00 | 80.45 | 71.62 | 待计算 | `checkpoints_rerank/d4c1f70_massspecgym_nopretrain_mass_topk40/` | 已完成 |
| Base retriever | formula | 42 | 58.49 | 71.49 | 81.81 | 64.65 | 6.14 | `checkpoints_align/d4c1f70_massspecgym_nopretrain/` | 已完成 |
| Full reranker | formula | 42 | 74.33 | 80.64 | 85.07 | 77.27 | 3.13 | `checkpoints_rerank/d4c1f70_massspecgym_nopretrain_formula_topk40/` | 已完成 |
| Pointwise reranker | mass | - | - | - | - | - | - | - | 未运行 |
| Pointwise reranker | formula | - | - | - | - | - | - | - | 未运行 |

## 论文数字来源

| 方法 | 数字来源 | 是否本地复现 | 使用限制 |
| --- | --- | --- | --- |
| SpecMolAlign | 本项目日志 | 是 | 可用于受控比较 |
| SpecEmbedding-Rerank | 本项目日志 | 是 | 当前只有 seed 42 |
| JESTR | GLMR 论文 Table 1 | 否 | 必须标注 `reported` |
| GLMR | GLMR 论文 Table 1 | 否 | 必须标注 `reported` |

## 建议时间线

- [ ] 7 月 10 日至 12 日：基线复现、MCES 和多随机种子实验。
- [ ] 7 月 13 日至 14 日：消融、效率和案例分析。
- [ ] 7 月 15 日：更新图表和全文。
- [ ] 7 月 16 日：执行双盲、页数、引用和补充材料检查。
- [ ] 7 月 17 日 AoE：提交论文。
- [ ] 7 月 20 日 AoE 前：提交匿名补充材料。

## 临时记录

在此处记录实验异常、参数调整和需要进一步确认的问题。

<!-- 在此添加记录。 -->
