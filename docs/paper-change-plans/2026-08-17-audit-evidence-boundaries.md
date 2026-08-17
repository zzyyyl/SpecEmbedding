# 论文修改计划：审计证据边界与投稿记录修订

- 状态：`执行中`
- 创建日期：2026-08-17
- 最后更新：2026-08-17
- 负责人：Codex（作者负责最终学术判断与投稿决策）
- 关联论文：`paper/main.tex` / `paper/main_cn.tex`
- 关联审计：`docs/audit-report-2026-08-17.md`
- 计划约束：不新增训练或评价实验，不改写已有实验数值，不新增引用；只使用已冻结的
  exact-SMILES 结果、alignment seeds 42/43/44 分层分析和现有一手来源核验记录

## 1. 修改目标与动机

落实 2026-08-17 项目与论文审计中 P0-2、P1-2、P1-3、P1-4 和 P2-1 的修改意见，
在不增加实验目标的前提下形成一致、可追溯的投稿口径：

1. 让摘要、表格、结论、README 和版本化分析工件均自包含地说明：Recall/MRR 使用
   MassSpecGym 提供候选文件上的本地 exact-target-SMILES 单正例规则，不等价于参考
   loader 默认的二维 InChIKey/可能多正例 evaluator，差异影响尚未量化；
2. 把已经完成的 alignment seeds 42/43/44 分层分析正式纳入双语稿，区分 canonical
   seed-42 主表、固定 alignment 内 reranker seed 统计和跨 alignment 描述性证据；
3. 将未独立复现的 JESTR/GLMR 外部数字与本地结果在视觉和叙述上分离，消除公平 benchmark
   或严格超越的暗示；
4. 将代表性 seed-42、三 reranker-seed 均值和三 alignment-level 估计三种统计口径明确分开；
5. 把投稿待办、项目记忆和审计报告统一到当前源稿英文 17 页、中文 16 页，并保留用户
   “完稿后再微调页数”的决定。

## 2. 当前证据与问题定位

### 2.1 身份协议已有正文披露，但关键独立阅读位置仍不自包含

- `paper/main.tex` 和 `paper/main_cn.tex` 的数据集/协议段已说明本地 evaluator 以 exact
  target-SMILES 定义唯一正例，并说明 MassSpecGym 参考 loader 默认使用二维 InChIKey
  等价且可能产生多正例。
- 双语摘要、受控消融表 caption 和结论没有重复这一限定；两份 README 以及
  `analysis/transfer2026_alignment_multiseed/`、
  `analysis/transfer2026_core_ablations/` 的报告/manifest 也没有协议字段。
- 现有协议核验固定到 MassSpecGym 官方 commit
  `4f501b3207c9f466bc745a0353f61dbfaf150d32`；本计划不重新联网核验，也不新增引用。

### 2.2 多 alignment 实验已经完成，但论文与状态文档仍按未完成描述

- `analysis/transfer2026_alignment_multiseed/report.md` 已按 alignment seed 42/43/44 分层，
  每个 alignment 内对 reranker seeds 42/43/44 配对，不把九次运行伪展平为独立重复。
- 六个 alignment × candidate 单元中，Transformer 相对 Pointwise 的 alignment-level
  差值方向混合；mass/formula 的 Top-1 与 MRR 均只有 2/3 alignment 为正，严格门槛失败。
- 相比同一 alignment 的 base，Pointwise 与 Transformer 在两类候选、三个 alignment 的
  12/12 aggregate cells 中均提高 Top-1 和 MRR；这支持“监督第二阶段重排序在三个内部
  alignment 中均改善基础检索”的描述性边界，不支持 self-attention 的一致独立收益。
- `paper/TRANSFER_2026_PLAN.md` 已记录 36/36 组完成，但 `paper/ADMA2026_TODO.md` 和
  `docs/project_memory_zh.md` 仍有“alignment 只有 seed 42/多 alignment 待做”的过时表述。

### 2.3 外部 reported 结果仍与本地结果同表

- 双语正文已写明 JESTR/GLMR 数字取自 GLMR 论文 Table 1、未独立复现且协议可能不同。
- 表格仍把外部行和本地行置于同一连续区块，正文还直接列出高/低的精确差值，快速阅读时
  容易被理解为 matched comparison。

### 2.4 页数与统计口径记录过时

- 8 月 1 日遗留 `paper/build` PDF 为英文 15 页、中文 14 页；8 月 17 日审计修订后，
  当前源稿最后验证为英文 17 页、中文 16 页。
- `paper/ADMA2026_TODO.md` 和 `docs/project_memory_zh.md` 仍保留 11/10 页或多 alignment
  未完成记录；历史日期记录可以保留，但必须标明“当时”。
- 主结果表是代表性 reranker seed 42；组件表是固定 alignment seed 42 下 reranker
  seeds 42--44 的均值与样本标准差；跨 alignment 表将以三个 alignment-level 估计作
  描述性汇总。三者不得混用或解释为端到端置信区间。

## 3. 修改范围

### 3.1 涉及文件与章节

- [ ] `paper/main.tex`：摘要、协议说明、主要结果表、跨 alignment 结果、受控消融表、
  Discussion、Limitations、Conclusion。
- [ ] `paper/main_cn.tex`：逐项同步相同数字、术语、表结构与证据强度。
- [ ] `analyze_alignment_multiseed.py` 及相应测试/派生产物：为报告和 manifest 加入固定的
  identity-protocol 声明，重新生成确定性工件。
- [ ] `analyze_core_ablations.py` 及相应测试/派生产物：加入相同声明，并明确统计对象为
  canonical alignment seed 42 下的 reranker seeds 42--44。
- [ ] `freeze_adma2026_artifacts.py` / `paper/adma2026_artifact_manifest.json`：在 canonical
  工件清单中加入相同的结构化评价身份协议，避免最上游结果 manifest 缺少协议标签。
- [ ] `README.md` / `docs/README_zh.md`：在论文主结果复现入口附近加入本地身份规则、
  非官方 evaluator 等价和 fixed/cross-alignment 统计口径说明。
- [ ] `paper/ADMA2026_TODO.md` / `docs/project_memory_zh.md`：同步多 alignment 完成状态、
  外部 reported 降级决策、当前页数和后续页数微调状态。
- [ ] `docs/audit-report-2026-08-17.md`：记录本轮修复、验证和提交。
- [ ] 本计划文件：按步骤更新状态、实际执行、偏差、验证和论文 commit。

### 3.2 明确不做的事项

- 不新增或重跑 exact-SMILES 与二维 InChIKey/多正例敏感性评价；因此不声称官方 evaluator
  等价，身份差异继续标为未量化。
- 不重训 alignment、reranker，不重建 cache，不重新计算 Recall、MRR 或 MCES。
- 不复现 JESTR/GLMR，不作 SOTA、显著性、因果性或同协议优越性声明。
- 不把九个 alignment × reranker 运行展平为九个独立 alignment 样本，不报告置信区间。
- 不修改模型、数据划分、候选池、身份规则、实验参数或已有表格数值。
- 不新增参考文献或 BibTeX；若必须新增文献、实验或改变核心结论，状态改为
  `已偏离待确认` 并暂停。
- 不在本轮强行压缩到目标页数；按用户决定，完稿后统一微调。

## 4. 具体内容设计

### 4.1 全局协议标签

在双语摘要、关键表 caption、结论、README 和两份分析报告/manifest 使用同一事实口径：

> Recall/MRR use MassSpecGym-supplied candidate files with a local exact-target-SMILES
> single-positive rule. The reference loader defaults to 2D InChIKey equivalence and may
> yield multiple positives; the unquantified difference means these are not
> official-evaluator-equivalent results.

中文同步为等强度陈述。表 caption 可压缩，但必须同时保留 `local`、`exact-SMILES`、
`single-positive`、`not official-evaluator-equivalent` 四层含义。分析 manifest 用结构化字段
记录正例规则、参考规则、影响是否量化以及官方等价状态，报告由脚本生成，禁止只手工改产物。

### 4.2 主要结果与外部背景分离

- 将代表性 seed-42 表拆成视觉分区，或拆为“本地结果”和“外部 cross-paper context”两表；
  外部区标题/方法名/caption 明写 `reported; not reproduced; not directly comparable`。
- 本地方法名显式加入 `(reranker seed 42)`，避免与相邻三种子均值表混淆。
- 删除“比 GLMR 高 4.57/6.27、MCES 更低”式方向性精确比较，改为：外部数字仅显示不同
  已发表流水线的数值范围，因 pre-retriever、候选处理、身份规则和 MCES 实现未匹配，
  不作 higher/lower 或方法归因。

### 4.3 已有跨 alignment 结果入稿

- 在受控结果附近新增紧凑表，逐 alignment seed 42/43/44、candidate type 报 Base、
  Pointwise、Transformer 的 Top-1，以及 Pointwise/Transformer MRR；学习方法值为各自
  三个 reranker seeds 的均值。
- 正文报告：监督 reranking 相对 base 在 12/12 aggregate cells 的 Top-1/MRR 均改善；
  Transformer 相对 Pointwise 的 Top-1/MRR 方向在 mass 和 formula 下均为 2/3 alignment
  为正，故不支持一致的候选交互收益。
- 描述性汇总以三个 alignment-level paired estimates 为单位，样本标准差不是置信区间，
  不从内部三个 alignment 外推一般随机稳定性。
- canonical seed-42 主表继续作为可追溯的代表性配置；组件移除仍只覆盖该 alignment，
  不把跨 alignment 主结果误写为跨 alignment 组件审计。

### 4.4 讨论、局限与结论

- 将“固定 alignment 下 reranker 初始化稳定性”与“已有三个 alignment 的分层敏感性”分开。
- 最强支持结论限定为：在本地 exact-SMILES、MassSpecGym supplied-candidate、top-40 协议
  下，监督残差重排序在三个内部 alignment 的已完成结果中一致改善 base；Pointwise 已取得
  大部分增益，候选 self-attention 的额外收益随 alignment 改变。
- 保留组件审计只覆盖 canonical alignment seed 42、无组件独立因果收益的边界。
- 结论不再让读者仅凭摘要或最后一节误解为官方 evaluator 等价、端到端显著性或 SOTA。

### 4.5 状态与页数记录

- `ADMA2026_TODO.md` 将多 alignment 训练/分析标记为已完成，把“论文整合”在本轮完成后
  勾选；JESTR/GLMR 统一复现采用已确认的 reported-only fallback，不继续作为当前 P0。
- `project_memory_zh.md` 在历史记录保留当时状态，同时新增当前状态，清除会误导下一步执行的
  “alignment 仍只有 seed 42”陈述。
- 当前页数记录为“源稿最近编译 17/16；旧 build PDF 15/14；页数压缩延后”，最终投稿
  合规项继续未完成，不以旧 11/10 页作完成依据。

## 5. 引用文献与真实性核验

本轮不新增、删除或修改引用与 BibTeX。身份规则继续引用此前已核验的 MassSpecGym 官方
代码 commit；JESTR/GLMR 外部数字继续引用现有 GLMR 论文条目，但只承担 cross-paper
context，不承担公平比较。

| 引用键 | 文献/来源 | 支持的论断 | 一手来源 | 核验状态 |
|---|---|---|---|---|
| `bushuiev2024massspecgym` | MassSpecGym, NeurIPS 2024 | benchmark、候选文件与任务背景 | 现有 DOI 与官方代码 | 已在既有计划核验；本轮不改元数据 |
| `zhang2026glmr` | GLMR, AAAI 2026 | JESTR/GLMR 的外部 reported 数字来源 | 仓库原文与现有 BibTeX | 已在既有计划核验；只作背景 |
| 官方代码 commit | MassSpecGym `datasets.py` / `transforms.py` | 默认二维 InChIKey、可能多正例 | commit `4f501b3...` | 已在 2026-08-01 计划核验 |

## 6. 实验与计算边界

- 是否新增训练实验：否。
- 是否新增评价运行：否。
- 是否进行静态数据统计：否；仅复用已版本化的跨 alignment 聚合与核心消融派生产物。
- 是否重新生成派生产物：是；只为加入协议元数据，由现有分析脚本确定性重写，不改变数值。
- 是否只修改文字、表格或元数据：是。

重新生成报告不是新模型实验；不得把已有描述性聚合解释为显著性检验。

## 7. 分步执行清单

- [x] 步骤 1：提交本计划；开始任何论文正文修改前将状态改为 `执行中`。
- [x] 步骤 2：修改 canonical freeze 与两份分析脚本和测试，加入身份协议与统计单位字段；
  重新生成 manifest/派生产物并验证数值、输入/输出哈希和字节级确定性。
- [x] 步骤 3：修改英文稿的全局协议标签、外部结果视觉分区、跨 alignment 结果、讨论、
  局限和结论。
- [x] 步骤 4：逐项同步中文稿，核对表格数字、正负方向、术语和结论强度。
- [x] 步骤 5：更新双语 README、投稿 TODO、项目记忆与审计修复记录。
- [x] 步骤 6：确认无新增引用，检查 BibTeX、交叉引用和表格引用。
- [x] 步骤 7：运行分析脚本测试和完整 `python -m pytest -q`。
- [x] 步骤 8：编译英文/中文稿，记录当前页数、错误、undefined citation/reference、
  Overfull/Underfull 和已知 package warning。
- [x] 步骤 9：检查协议词扫描、双语一致性、`git diff --check` 和完整 diff，确保无新实验、
  无数值漂移、无无关文件。
- [x] 步骤 10：使用 Conventional Commit 提交论文及关联实质修改。
- [ ] 步骤 11：将论文、工程复现与审计记录推送后，通知原审计会话复审；若有阻断意见，
  在同一计划内继续修复、验证和复审。
- [ ] 步骤 12：复审无阻断项后回填最终结果，将状态设为 `已执行`，再以独立文档 commit
  归档本计划。

## 8. 风险、证据边界与待确认事项

- 三个 alignment checkpoints 来自两个 source commits，但分析已保留每个 checkpoint 的
  provenance；不得省略这一事实或把结果写成同一 commit 的完全重复实验。
- 每个 alignment 内的三个 reranker seeds 先聚合为 alignment-level estimate；不得展平
  九次运行夸大样本量。
- 跨 alignment 结果支持内部描述性复现，不支持置信区间、显著性、一般泛化或组件因果性。
- exact-SMILES 与二维 InChIKey 的影响没有量化；任何官方 evaluator 等价主张都会触发
  `已偏离待确认` 并要求新增评价计划。
- 外部 baseline 未统一复现；任何 higher/lower、SOTA 或机制优越性主张都会触发暂停。
- 若重新生成派生产物导致已有数值、输入哈希或 claim gate 改变，立即设为
  `已偏离待确认`，不得继续入稿。
- 用户已决定页数暂不设硬阻塞；若 venue/模板要求在本轮实质改变内容范围，再单独确认。

## 9. 验证方案

- [x] 派生产物：两次生成字节一致；manifest 的数值、路径、SHA-256 与输入工件匹配。
- [x] 数字核验：跨 alignment 表逐格反查 `alignment_level.csv` 或对应受控 CSV；
  Base/Pointwise/Transformer 和 delta 方向与报告一致。
- [x] 协议扫描：摘要、所有结果表 caption、结论、README、analysis report/manifest 均出现
  本地 exact-SMILES 和非官方 evaluator 等价限定。
- [x] 外部基线：表内视觉分区明确，正文无直接 higher/lower 或 SOTA 解释。
- [x] 英文稿编译：`latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`。
- [x] 中文稿编译：`latexmk -xelatex -interaction=nonstopmode -halt-on-error main_cn.tex`。
- [x] 引用检查：日志无 undefined citation/reference，`paper/references.bib` 无计划外 diff。
- [x] 双语一致性：表格数字自动比对，人工核对协议、统计单位、方向与限制性措辞。
- [x] 测试：`conda run -n specembedding python -m pytest -q` 全部通过。
- [x] Git：`git diff --check` 通过，提交只包含计划内文件。

## 10. 执行记录

### 2026-08-17：开始执行

- 计划已先以 commit `72af18e` 提交，尚未修改论文正文。
- 按用户要求采用不新增实验的严格协议标注方案，并将已有 alignment seeds 42/43/44
  分层结果纳入论文；状态在正文修改前切换为 `执行中`。

### 2026-08-17：协议工件与双语稿完成

- canonical freeze、核心消融与跨 alignment 分析统一加入结构化身份协议；相关提交为
  `ae8efe1`、`0e02992`、`48316f2`。canonical manifest 共 60 个工件，配置证据固定到历史
  source commit 的 `params.yaml` blob；两次确定性生成哈希一致。
- 英文和中文稿同步加入本地 exact-target-SMILES 单正例边界、跨 alignment 表与分层统计单位；
  外部 JESTR/GLMR 拆为 reported-only 独立表，不作相对性能排序。
- 数字逐格反查版本化 CSV：两个监督模型在 12/12 个 alignment--candidate--model 聚合单元
  均改善对应 base 的 Recall@1/MRR；Transformer-minus-Pointwise 在两种候选和两个指标上
  均仅 2/3 alignment 为正。三个 checkpoint 来自两个 source commits，故仅作描述性证据。
- 中英文最终编译为 17/16 页；无 fatal error、undefined citation/reference 或
  Overfull/Underfull。保留英文 `amsmath` 与中文 Fandol `fontspec` 已知 package warning。
- 未修改 `paper/references.bib`，未新增引用、训练、评价或数值；页数压缩按用户决定延后。
- canonical `specembedding` 环境完整测试为 `86 passed, 5 warnings`；其中新增路径可移植性
  测试属于工程审计修复，不改变论文实验。

### 2026-08-17：实质提交与工程复现闭环

- 双语论文、投稿 TODO 与项目记忆以 commit `1c0dee2` 提交；该 commit 是本计划要求的
  论文实质修改 commit。
- 路径可移植性、无导入时目录创建、完整 CLI 路径透传、legacy notebook 输出目录、依赖
  固定和 Linux x86-64 显式锁以 commit `f5e6e78` 提交。该工程修复来自同一审计，但没有
  改变论文方法、实验数值或证据边界。
- canonical 环境最终测试为 `86 passed, 5 warnings in 33.48s`；从显式 Conda/PyPI 锁
  重建的全新环境为 `86 passed, 5 warnings in 26.92s`，Ruff 通过。
- 双语 PDF 再确认 17/16 页；无 fatal error、undefined citation/reference 或
  Overfull/Underfull。tracked 私有机器根扫描为零命中。
- 下一步是推送并请求原审计会话独立复审；在复审结论返回前保持状态 `执行中`。

### 2026-08-17：第二轮独立审计反馈与续修决定

- 原审计会话以远端 `7b824c4` 为只读基线复核，判定“有条件通过”：无 P0；提出两项 P1：
  `run_rerank_pipeline.py` 仍依赖调用 cwd，以及当前 17/16 页双语 PDF 缺少可审计留存。
- 同轮 P2 指出：论文仍写“确切依赖版本未锁定”，与新增的事后验收锁表面矛盾；应改为
  “当前验收环境已锁定，但历史训练环境没有逐包证明，不能声称 bitwise retraining”。
- 续修不新增实验、引用、性能数字或结论：为 rerank 总管线固定 repository cwd/绝对子脚本
  并增加临时 cwd dry-run 测试；强制从当前源稿完整重编两稿，将最终 PDF、页数、SHA-256、
  source commit 和 TeX 工具链作为 tracked release evidence 留存；同步覆盖本地旧 build PDF，
  避免误认；只精确改写双语 limitation 的环境边界句。
- 上述修改仍属于本计划的“证据边界、局限与投稿记录”范围，不改变方法定义、评价协议、
  实验数值或研究结论，状态保持 `执行中`。

执行时按日期记录状态转换、派生产物核验、双语修改、页数、测试、提交和复审反馈。

## 11. 最终结果

- 完成日期：尚未完成
- 最终状态：`执行中`
- 验证结果：本地与显式锁环境均已通过；第二轮审计提出两项 P1，正在续修并等待再次复审
- 论文修改 commit：`1c0dee2`
- 工程复现 commit：`f5e6e78`
- 计划归档 commit：无需在本文件中自我引用
- 相对原计划的偏差：同步完成同一审计中的路径可移植性和依赖锁修复；未扩大论文论点、
  未新增实验或引用。按用户要求增加“通知原审计会话并迭代至无阻断项”的验收步骤。
