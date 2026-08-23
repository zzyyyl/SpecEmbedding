# 论文修改计划：ScholarGPT 方法重构与实验复核

- 状态：`已执行`
- 创建日期：2026-08-23
- 关联论文：`paper/main.tex` / `paper/main_cn.tex`
- 当前归档：正文、实验边界、发布工件和匿名补充包已完成本轮核验。

## 1. 修改目标与动机

将论文主线收紧为：在固定跨模态检索器已召回的候选列表内，使用监督式、非生成式 residual learning-to-rank 改善 Recall/MRR。方法不生成候选，不把 candidate self-attention 或 relative branch 写成已证实的独立收益。

## 2. 当前证据与问题定位

- 旧设计含 rank embedding、训练期 positive forcing 和过长的 candidate-set Transformer 叙事；当前主线改为 rank-free relative/pointwise，并把 legacy 协议单独标记。
- 主评价使用 MassSpecGym official retrieval JSON 候选顺序、二维 InChIKey 身份规则和保存的 alignment embeddings；这是 official-compatible candidate/identity audit，不是 fresh loader 重编码、alignment 重训或外部方法复现。
- `formula` 协议假设已知分子式；reranker 是 closed-library 方法。local exact-target-SMILES 仅作敏感性视图。

## 3. 修改范围与不做事项

已修改或核验：

- relative reranker、no-rank 特征、候选对反对称关系、候选置换等变性及测试；
- no-forcing pipeline 默认语义、显式 legacy forcing 开关和匿名 CPU smoke；
- 英文/中文正文、README、项目记忆、引用闭包、release PDF 和匿名补充验证记录。

明确不做：

- 不重新训练 alignment 或补做未具备统一输入的外部 baseline；
- 不把静态检查、匿名 smoke、候选身份审计或前向拆解写成新的科学实验结果；
- 不声称 SOTA、统计显著性、端到端稳定性、部署 latency 或 relative 模块的独立普遍收益。

## 4. 具体修改内容

- 将方法叙事收紧为固定候选池内的监督 residual learning-to-rank；同步英文稿、中文稿和 README。
- 把 no-rank relative/pointwise、候选对反对称性、候选置换等变性和 no-forcing 语义写入实现说明与复现入口。
- 将 official-compatible candidate/identity audit、local sensitivity、外部 reported-only baseline 和实现审计明确分层。

## 5. 引用文献与真实性核验

- 当前正文和 `paper/references.bib` 使用 24 个引用键；本轮不新增文献，未使用条目已删除。

## 6. 实验与计算边界

- 本轮方法、测试和发布验收使用已有代码、缓存、检查点和工件；不新增外部 baseline 或跨数据集实验。
- 实验结果必须按 candidate type、checkpoint、seed scope、cache 协议和日志核对；计划文档不替代原始工件。
- 训练/验证/测试的正例 forcing 语义、候选覆盖上界和 miss 处理必须在方法和复现材料中保持一致。

## 7. 执行清单

- [x] 完成无 rank、无 forcing、最多 256 候选 coarse-to-fine relative pilot 与 pointwise 对照。
- [x] 完成等变性、反对称性、谱图依赖、数据协议和 pipeline 回归测试。
- [x] 完成官方候选顺序/二维身份的保存 embedding 兼容审计，并保留证据边界。
- [x] 同步英文稿、中文稿、README、项目记忆和投稿记录。
- [x] 重建匿名补充包，完成 allowlist 扫描、独立解包测试和 synthetic CPU smoke。
- [x] 重建双语 release，更新构建清单和 SHA-256。

## 8. 风险与待确认事项

- 当前英文 release 为 16 页，是否满足目标 venue 的页数上限仍需按最终 venue 核对。
- 作者列表、COI、公开 preprint 政策和最终投稿副本的联合匿名检查仍由投稿前流程完成。
- 外部 JESTR/GLMR、candidate-aware alignment、跨数据集验证和完整 official-loader 端到端重跑仍不在证据范围内。

## 9. 验证结果

- 仓库：`pytest -q tests` 为 112 passed、9 warnings；Ruff、compileall、`git diff --check` 通过。
- 匿名包：40 个 allowlisted 源文件，包内 13 项测试通过，relative CPU smoke 通过；详细记录见 `reproducibility/anonymous-supplement-validation.yaml`。
- 双语 release：英文 16 页、中文 14 页；PDF、构建工具链和哈希见 `paper/release/build-manifest.yaml`。
- 计划相关提交：代码修复 `1d9b5eb`；发布与引用归档 `88e37c5`；匿名包最终源版本记录 `69df121`、`5e5ced9`。

## 10. 执行记录

- 代码修复提交：`1d9b5eb`。
- 发布与引用归档提交：`88e37c5`。
- 匿名包最终源版本和验证记录：`69df121`、`5e5ced9`。
- 本次文档精简只修改 `docs/`，不改变论文源码、实验数字或发布工件。

## 11. 最终结果

本计划已执行并归档。后续只在目标 venue 确定后进行格式、作者/COI、页数和最终联合匿名检查；若改变论文主张、评价协议或实验范围，应另建计划，不在本文件中追加新的历史过程。
