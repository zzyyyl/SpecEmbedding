# 投稿待办

最后更新：2026-09-07。目标 venue 尚未确定；旧 ADMA 和 8 月转投日程已结束，不再作为执行期限。
当前工作状态见 [项目记忆](../docs/project_memory_zh.md)，实验数字仅引用 [证据索引](../analysis/README.md)。

## 科学证据

- [ ] 按 [MassSpecGym 全量重训计划](../docs/paper-change-plans/2026-09-07-MassSpecGym全量重训.md)
  完成新版全量训练与评价；该计划当前暂停，文档整理不恢复实验。
- [ ] 区分旧 20,000-query/top-40/训练 forcing 结果、新版 top-256/no-forcing 默认配置和
  独立 pilot；旧结果不能替代尚未产生的全量结果。
- [ ] 按 [NPLIB1 计划](../docs/paper-change-plans/2026-09-05-NPLIB1跨数据集增强.md)
  完成第二套 in-domain 派生协议；去重 zero-shot 只作小样本敏感性分析，不预告跨数据集收益。
- [ ] 每张保留结果表核对候选来源/顺序、身份标签、checkpoint、训练样本数、forcing、coverage
  与 seed 层级。所有测试 query 纳入指标；MCES 若仅单 checkpoint 计算须单独说明。
- [ ] 所有贡献、摘要、讨论与结论同步到实际结果；不能把 relative 的结构性质写成独立实验优势。
  cutoff 截断不是 K-specific 重训，forward 开销不是端到端 latency。
- [ ] 若纳入 GLACIER，先完成 [交接清单](../docs/glacier_reproduction_handoff_zh.md) 中的推理与
  协议匹配；JESTR-style 本地打分控制和 reported-only GLMR 不作为官方模型复现。
- [ ] 完整 official-loader 重编码、独立外部 retriever、candidate-aware alignment 和实质化学案例
  仍无对应正式证据。只有在明确纳入后续范围时执行，不能把旧评审建议视为已完成实验。

## 稿件与投稿规则

- [ ] 确定 venue，依据当时的官方来源核对模板、截止时间、页数口径、匿名要求、补充材料、
  AI 使用、预印本与宣传政策；不沿用旧 ADMA 页数/包体积限制。
- [ ] 逐项核对中英文摘要、方法、表图、指标量纲、引用和限制；保留 SpecEmbedding 继承架构归属。
- [ ] 作者核验参考文献一手来源及发表状态，完成英文终审和独立技术审读。
- [ ] 作者确认最终作者列表、COI、基金/致谢、既有公开版本和无重复投稿。
- [ ] AI assistance disclosure 覆盖实际辅助范围，并由作者确认责任说明。

## 发布与复现验收

- [ ] 按 [构建说明](BUILDING.md) 生成最终 PDF，核对页数、引用、警告、源 commit 和 SHA-256。
- [ ] 在干净环境验证依赖安装、相关测试和最小 synthetic smoke；已有锁与验收记录只是指定
  版本的快照，见 [复现说明](../reproducibility/README.md)。
- [ ] 从最终 allowlist 重建匿名源码包，在独立目录完成身份扫描、解包测试、shell 语法、
  Ruff、compileall 和 synthetic CPU smoke，记录新验收结果。
- [ ] 联合扫描最终 PDF、源码与归档中的作者/单位、用户名、hostname、本地路径、邮箱/ORCID、
  GPU UUID、remote、公开托管链接和 Git 历史；公开项目仓库本身不是匿名补充包。
- [ ] 核对提交副本的格式、文件大小和 checksum。数据、checkpoint、cache、日志与内部 manifest
  不进入匿名源码 allowlist；若另行提交真实实验工件，须按 venue 规则独立审查和脱敏。

已完成的历史修改、旧日程与实验表不在待办中重复保存，追溯入口见
[计划索引](../docs/paper-change-plans/README.md)。论文重大修改继续遵循先计划、后修改、
先提交实质内容、后回填归档的流程。
