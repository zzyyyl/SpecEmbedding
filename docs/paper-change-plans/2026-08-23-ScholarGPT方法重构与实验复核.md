# 论文修改计划：ScholarGPT 方法重构与实验复核

- 状态：`已执行`
- 创建日期：2026-08-23
- 最后更新：2026-08-23
- 负责人：Codex
- 关联意见：`docs/paper-change-plans/2026-08-21-ScholarGPT修改意见.md`、外部评审附件 `pasted-text-1.txt`
- 关联论文：`paper/main.tex` / `paper/main_cn.tex`
- 计划约束：优先完成评审明确要求的无捷径训练和候选相对判别核心；本次继续执行将使用当前可访问的 MassSpecGym 1.3.1 官方数据快照补做正式身份评价和困难候选分层，只把实际运行产生的结果写入论文。缺失外部代码、统一 checkpoint 或额外数据的项目仍保留为未完成边界，不伪造结论。

## 1. 修改目标与动机

将论文从“固定 top-40 上的 generic candidate-set Transformer 比较”重构为可检验的谱图条件候选相对判别方法。核心方法应满足：

1. 不使用 rank embedding；
2. 训练和评价均不执行 positive forcing，使用自然进入候选池的标签；
3. 在现有最多 256 候选的缓存上先做 pointwise coarse scoring，再做显式成对相对判别；
4. 成对偏好通过反对称构造保证 `p_ij = -p_ji`，候选顺序变化时输出按同样方式置换；
5. 增加谱图依赖约束，避免把候选先验或 base score 校准误写成谱图匹配能力。

论文主张必须由新实验实际支撑；若新方法不能在现有本地协议下稳定优于容量可比 pointwise，则回退为诚实的失败/条件性结果，不包装为成功的方法贡献。

## 2. 当前证据与问题定位

- 现有 `prepare_rerank_cache.py` 支持 `pre_top_k=256`，但历史 canonical cache 使用 `force_include_positive=True`；训练期强制插入造成 rank/base-score shortcut。
- `SpecEmbedding/models_rerank.py` 的现有模型含 rank embedding 和可选 generic Transformer；没有显式候选对关系，也没有 spectrum-dependency loss。
- 工作区存在本地不可提交的完整 rerank cache、alignment checkpoint 和 RTX 4090 GPU；缓存元数据和路径必须通过实际读取核验，不以 manifest hash 冒充原始数据。
- 本地 `processed/MassSpecGym/` 数据入口已实际生成无 forcing 全池 cache；结果记录只保留仓库相对路径、字节数和 SHA-256，不把机器绝对路径写入发布材料。
- 现有论文已明确 local exact-SMILES 与官方二维 InChIKey 协议边界；官方 evaluator、外部强 baseline、容量匹配、query bootstrap 只有在真实输入和运行完成后才能升级为主结果。

## 3. 修改范围

### 3.1 涉及文件与章节

- [x] `prepare_rerank_cache.py`、`SpecEmbedding/data/datasets_rerank.py`：无 forcing 设为新实验配置；训练只使用自然有标签 query，保留全池覆盖元数据。
- [x] `SpecEmbedding/models_rerank.py`、`SpecEmbedding/utils/rerank.py`、`train_rerank.py`、`eval_rerank.py`：实现无 rank 的谱图条件相对候选 reranker、反对称 pair preference、谱图依赖损失和可审计配置。
- [x] 新增 `analysis/transfer2026_scholargpt_method/`：保存配置、参数量、训练/评价结果、指纹和证据边界；未覆盖既有 canonical 工件。
- [x] `tests/`：增加模型等变性、反对称性、无 forcing 数据集、谱图依赖损失和参数统计测试。
- [x] `paper/main.tex` / `paper/main_cn.tex`：按真实 pilot 结果重写标题、摘要、引言、方法、主结果、限制和结论；中英文同步，并移除正文旧版大表以保持精炼。
- [x] `README.md`、`docs/project_memory_zh.md`：记录新命令、数据/检查点/匿名工件边界和未完成项目。
- [x] `paper/release/`：源稿变化后重新生成双语发布 PDF 和构建清单。

### 3.2 明确不做的事项

- 不把现有 forced top-40 结果改名为无 forcing 新方法结果。
- 没有完整 loader 环境、外部模型或统一重训输入时，不声称外部 baseline 或 official-evaluator 端到端等价；本轮仅从本地 MassSpecGym 1.3.1 快照核对候选集合/顺序，并在保存嵌入上单独记录交叉核验范围。
- 不把新模型的单次运行或同 alignment 多次运行写成一般稳定性；种子、alignment checkpoint 和训练配置逐项记录。
- 不要求本轮无依据地新增文献；现有 25 条引用继续核验闭包。

## 4. 具体内容设计

### 4.1 方法

- 使用 `h_i=[z_s,z_{m_i},z_s\odot z_{m_i},|z_s-z_{m_i}|,b_i]` 的绝对分支，不输入 rank。
- 对候选对 `(i,j)` 构造 spectrum-conditioned relation，包括 `z_{m_i}-z_{m_j}`、`b_i-b_j`、候选 embedding 相似度/差异及谱图条件；用 `g(r_ij)-g(r_ji)` 形成反对称 `p_ij`。
- 用候选对权重聚合相对偏好：`q_i=alpha b_i+f_abs(h_i)+beta sum_j w_ij p_ij`，mask padding 候选；证明候选置换等变性和 `p_ij=-p_ji`。
- 对自然进入 256 候选池的训练 query 进行 pointwise coarse + relative refinement；未进入池的 query 不参与 reranker loss，并在 coverage 中单独报告。
- 训练目标为 listwise/pairwise ranking 加谱图依赖约束；错误谱图只作为 batch 内对照，不把该约束写成已证实的因果保证。

### 4.2 实验

- 主对照至少包括 base cosine、无 rank pointwise、相对候选模型；若预算允许加入 generic Transformer、capacity-matched pointwise 和 base-score-only/candidate-only。
- 报告 mass/formula、自然候选覆盖、R@1/MRR/U256/U40；同时保留 local protocol 标记。
- 对可用的 top-40/50/100/256 缓存做 K 敏感性或至少明确其不是同一训练协议的等价 sweep；不能把截断 cache 当 full-pool 训练结果。
- 记录每个模型参数量、训练时间、推理时间和峰值显存；效率结果只写实际测量值。
- 做 candidate order permutation、pair antisymmetry、spectrum shuffle/依赖损失的机制验证；分层分析只在真实 query-level 输出可得时加入。

### 4.3 论文叙事

- 研究问题改为“在质量/分子式约束造成结构相近候选时，如何利用谱图条件的相对候选差异区分 hard negatives”。
- 摘要只保留问题、方法、性质和真实主结果；seed、审计和协议限定放入实验与限制。
- 贡献点只能写已由新结果支持的内容；若相对模块不稳定，则明确将其作为负面机制结论，主线回退为监督式 reranking。
- JESTR/GLMR 保留 reported-only 机制背景，不展示未统一协议的外部数值。

## 5. 引用文献与真实性核验

| 引用键 | 文献 | 支持的论断 | 一手来源 | 核验状态 |
|---|---|---|---|---|
| `bushuiev2024massspecgym` | MassSpecGym, NeurIPS 2024 | 数据集、候选池和 evaluator 背景 | 原有论文/官方代码 | 已核验，沿用现有 25-key 闭包 |
| `kalia2025jestr` | JESTR, Bioinformatics 2025 | 外部机制背景，非本项目复现 | 原有论文 | 已核验，沿用 |
| `zhang2026glmr` | GLMR, AAAI 2026 | 外部机制背景，reported-only | 原有论文 | 已核验，沿用 |
| 其余现有 22 条 | `paper/references.bib` | 方法、数据和评价背景 | 原有 DOI/出版商来源 | 不新增，保持闭包 |

## 6. 实验与计算边界

- 是否新增训练实验：是；仅使用仓库已有缓存/检查点和用户已确认可用的 GPU，长任务使用 detached `tmux`。
- 是否新增评价运行：是；新结果必须保存配置、日志、checkpoint hash 和结果 JSON。
- 是否进行静态数据统计：是；统计自然 label coverage、候选池大小、参数量、复杂度和 identity 可行性。
- 是否只修改文字、公式或引用：否；论文正文在新实验结果后修改。

## 7. 分步执行清单

- [x] 步骤 1：提交本计划并将状态设为 `执行中`（`389ee58`）。
- [x] 步骤 2：确认无 forcing 256-cache、原始候选源和 RTX 4090 GPU 运行入口。
- [x] 步骤 3：实现无 rank 的 relative reranker、谱图依赖损失及无泄漏数据集。
- [x] 步骤 4：完成形式性质、mask、loss 和训练配置测试；运行 Ruff/compileall。
- [x] 步骤 5：通过 detached `tmux` 生成 mass/formula 新 cache，完成 relative 与 pointwise pilot 对照。
- [x] 步骤 6：完成主指标、coverage、谱图打乱审计和工件指纹记录；未把未完成的效率/难度分析写成结果。
- [x] 步骤 7：按真实结果重写英文稿，同步中文稿、README 和项目记忆，并删除正文旧版大审计表。
- [x] 步骤 8：完成双语构建、引用/匿名性/工件检查和测试。
- [x] 步骤 9：使用中文 Conventional Commit 提交本轮论文实质修改（`42ab734`、`620bf2b`）。
- [x] 步骤 10：更新发布证据（`574d2e1`），回填本计划实际结果并用独立中文文档 commit 归档。
- [x] 步骤 11：在当前可访问的官方 MassSpecGym 1.3.1 数据快照上完成候选来源/顺序核对、保存嵌入上的官方顺序交叉评价，并补充结构相似度、基础分数间隔、候选池大小和谱峰数分层；已同步论文和证据清单。该核验不等同于完整 official loader 端到端重跑。
- [x] 步骤 12：按严格最低验收补充完成二维身份等价的三 seed 官方候选顺序评价、效率拆解和主表重组；外部 baseline、candidate-aware alignment 与 fresh loader 重编码仍按边界保留。

## 8. 风险、证据边界与待确认事项

- 新模型可能无法稳定优于 pointwise；这是评审验收条件，不能用叙事修补。
- 全池缓存和相对 pair 计算可能受显存/磁盘限制；先用小限额 smoke test，再提交后台长任务。
- 现有缓存路径包含历史机器绝对路径元数据；结果记录应使用仓库相对路径和哈希，不把私有路径写入论文或 release PDF。
- 若实现需要改变官方 evaluator、数据身份规则或引入新外部模型，计划应暂停为 `已偏离待确认`，先向用户报告；本次只复用已缓存的官方数据和官方 2D InChIKey 规则，不引入外部模型。

## 9. 验证方案

- [x] `conda run -n specembedding python -m pytest -q tests`：110 passed, 9 warnings。
- [x] `conda run -n specembedding python -m ruff check .`：All checks passed。
- [x] `python freeze_adma2026_artifacts.py --check`：60 canonical artifacts validated，manifest matches；这是旧 canonical 工件的历史校验，不覆盖新 pilot。
- [x] `python -m compileall`：通过；`git diff --check`：通过。
- [x] `bash paper/build_release.sh`（detached `tmux`）：英文 16 页、中文 14 页；最终日志无 LaTeX error、undefined citation/reference、Overfull/Underfull；仅保留已知 amsmath/Fandol 非阻断警告。页数仅作记录，不假定 venue 上限。
- [x] PDF 匿名性与哈希核对：Author 为空/缺失，私有路径扫描无命中；发布稿与 manifest 已同步。英文 16 页、479372 bytes、SHA-256 `c066ed41b657a2aaa3957295c6ed064e7bd31aa83fe2319846cb9b6db86a5dcc`；中文 14 页、416682 bytes、SHA-256 `f1e1fe30d8481184f1007a25c669cb2f2d3843650d7646f4ec0046a5d2de3376`。

## 10. 执行记录

2026-08-23：读取外部评审附件，确认需补做评价协议、容量匹配、信息来源消融和 query-level 分析；本计划由初始 `389ee58` 进入第二阶段执行。

2026-08-23：完成无 forcing full-pool cache 与 rank-free relative 实现；代码/配置/消融开关提交 `a10a02c`、`69a5521`、`26e6294`、`684039c`、`488a04a`、`8ce2655`。缓存日志确认 mass/formula 各 194,119 条 valid query，`force_include_positive=False`，训练标签覆盖率 1.0。

2026-08-23：在 detached `tmux`、两张 RTX 4090 上完成同一 `d4c1f70` alignment checkpoint 的 relative 与容量匹配 pointwise seeds 42--44；每池 5,000 training queries、3 epochs。三 seed 汇总和逐 query bootstrap/K 敏感性写入 `analysis/transfer2026_scholargpt_review/multiseed_summary.json` 与本目录的 `second_stage_report.md`。

2026-08-23：完成 seed-42 信息来源/机制消融：score-only、embedding-only、embedding+base、candidate-only、no-spectrum-conditioning、no-molecular-relation、no-antisymmetric；两个候选池均有 R@1/MRR 结果，未将单 seed 方向写成因果组件结论。

2026-08-23：对新 relative full-pool test cache 应用 MassSpecGym 1.3.1 2D InChIKey 变换；两个候选池均无 multiple-positive query 或 identity collision，local/reference 指标一致。结果为 cache-level identity audit，不称为完整 official-loader rerun。

2026-08-23：论文英文/中文正文、README、项目记忆已同步三 seed 主结果、信息来源分析、K 敏感性、前向资源审计和 identity 边界；页数继续仅作为后续排版事项。

2026-08-23：读取本地 Hugging Face MassSpecGym 1.3.1 快照，核对 `MassSpecGym.tsv` 测试顺序和两份 retrieval JSON。候选集合对全部 17,556 个目标完全一致（平均 set Jaccard 1.0），但有序列表完全匹配为 mass 0/17,556、formula 231/17,556。使用官方 JSON 顺序和保存的 alignment 嵌入完成 seed-42 relative/pointwise 交叉评价；结果写入 `official_candidate_eval_all.json`，不声称完整 loader 重编码或 alignment 重训。

2026-08-23：从已保存的 seed-42 query predictions 完成候选数量、Base 名次、top-1 Morgan 相似度、Base 分数间隔和谱峰数分层；结果写入 `difficulty_analysis.json`。困难 query 上 relative 增益更明显，但容易 query 上有下降；仅作 local exact-SMILES 协议下描述性证据。

2026-08-23：完成三 seed 官方兼容评价。`official_protocol_eval.json` 使用官方 retrieval JSON 顺序、稳定降序并列规则和保存的 alignment embeddings；同一 full-pool `relative_identity_eval.json` 审计证明两个候选池均为 17,556/17,556 正例 query、0 多正例、0 碰撞，因此 exact-target 标签与二维身份标签等价。mass relative/pointwise 的 R@1/MRR 为 50.98±0.87/0.5871、51.37±0.54/0.5903；formula 为 66.32±0.46/0.7129、66.65±1.30/0.7166。该评价没有 fresh loader 重编码或 alignment 重训。

2026-08-23：完成效率拆解。RTX 4090、batch 64、2,048 queries、排除数据加载时，mass relative coarse/relation/full 为 0.261/0.111/0.372 ms/query（2,685 q/s，302 MB），pointwise 为 0.030 ms/query（33,450 q/s，244 MB）；formula relative 为 0.026/0.050/0.076 ms/query（13,235 q/s，302 MB），pointwise 为 0.026 ms/query（38,492 q/s，244 MB）。这些是 implementation audit，不是端到端效率主张。

2026-08-23：论文英文/中文正文、README、项目记忆和证据 manifest 已同步；论文实质提交为 `42ab734`、表注排版修正为 `620bf2b`，双语 release evidence 为 `574d2e1`。最终 `pytest -q tests` 为 110 passed、9 warnings；Ruff、compileall、`git diff --check` 均通过。

## 11. 当前结果与待归档

- 当前状态：`已执行`；论文正文、官方兼容三 seed 候选顺序评价、二维身份审计、困难候选分层、效率拆解、双语构建、发布 manifest、测试和静态核验均已完成。当前英文 16 页、中文 14 页仅作构建快照，不假定具体 venue 上限；用户已明确后续按目标 venue 再做精炼排版。
- 主要结论：relative/pointwise 均改善 base；pointwise 在主 R@1/MRR 汇总略高，relative 独立普遍收益未建立。
- 证据边界：三 reranker seeds 不是 alignment 重复；二维身份等价由同一 full-pool cache 审计建立，但官方兼容表仍使用保存嵌入、未重跑完整 loader 或重训 alignment；JESTR/GLMR 仍 reported-only；前向 latency/显存不是端到端效率主张。
- 外部 baseline、候选感知 alignment 重训、跨数据集验证和完整 official loader 端到端重跑仍不纳入本次范围，不写成已完成结果；本次只完成官方候选 JSON 顺序与保存嵌入的交叉核验。

## 12. 最终核验步骤

1. 对新增 Python/分析脚本运行 Ruff、compileall 和相关单元测试；完整 `pytest -q tests` 为 110 passed, 9 warnings（最终复核需重跑）。
2. 完成英文/中文 LaTeX/BibTeX 构建，检查 undefined citation/reference、Overfull/Underfull、PDF 匿名性和页数记录；已完成，页数不作为阻断。
3. 更新 `paper/release/build-manifest.yaml` 与受跟踪双语 PDF，核对 bytes/SHA-256、源 commit `620bf2b` 和构建日志；已完成。
4. `git diff --check`、Ruff、compileall 和 110 项测试已通过；canonical artifact check 仍按其历史 scope 单独复核，私有 raw predictions/cache 未纳入 Git 或匿名包。
5. 论文实质 commits `42ab734`、`620bf2b`、发布清单与 PDF commit `574d2e1` 已提交；本计划随后用独立中文文档 commit 归档。外部评审意见已按实际可审计输入落实；没有把 JESTR/GLMR、外部强 baseline 或 fresh loader 端到端评价误记为已完成。

第二阶段明确不做：未经统一数据与 checkpoint 的外部方法“复现”声称；把历史 alignment revision 混写为同配置随机重复；以及为达到正向结果而事后选择 seed、K 或模型变体。正式身份评价仅在第 13 节规定的官方数据快照核对完成后使用相应限定语。

## 13. 继续执行记录（2026-08-23）

用户继续要求按完整评审附件核验。此前第二阶段已完成无 forcing、relative/pointwise 对照和机制审计，但仍把官方候选来源核对与困难候选分层列为范围外。本次使用本地 Hugging Face 缓存中的 MassSpecGym 1.3.1 `MassSpecGym.tsv` 与 retrieval candidate JSON，核对候选集合和官方列表顺序，并在保存嵌入上重算排序；结果作为官方候选顺序交叉核验记录，不夸大为完整 loader 端到端评价、重新训练或外部方法复现。

本次继续执行还从已保存的逐 query 预测和候选 SMILES 计算了可复核的候选池大小、基础排序名次、top 候选 Morgan 相似度、基础分数间隔和谱峰数分层。外部 JESTR/GLMR、第二个 retriever、candidate-aware alignment 重训、跨数据集验证和端到端吞吐仍不在本地证据范围内。

## 14. 严格最低验收补充（2026-08-23）

逐条复核完整 ScholarGPT 意见后，确认前一阶段虽已完成无 rank、无 forcing、full-pool coarse-to-fine、谱图依赖损失、容量匹配 pointwise、机制消融和 seed-42 困难分层，但主表仍是 local exact-target-SMILES 协议，官方顺序核验只覆盖 seed 42，且没有把固定硬件前向审计拆分为完整候选池粗排/关系精排/吞吐证据。为避免把局部核验写成正式 evaluator，本阶段补做：

1. 依据 MassSpecGym 1.3.1 loader 的二维 InChIKey 标签和 retrieval JSON 顺序，对 relative/pointwise 三个 reranker seeds 及 Base 统一重算 R@1/R@5/R@20/R@40/MRR，并记录 tie-breaking 与多正例计数；
2. 从已保存的 seed-42 预测和模型前向审计中补出 full-pool coarse、top-40 relative branch、pointwise 和 Transformer 的参数/延迟/显存/吞吐分项；
3. 重新组织正文主表与实验问题，使 official-protocol 结果与 local-protocol 敏感性结果分开；若外部强 baseline、candidate-aware alignment 或同一代码版本的 alignment 重训仍无可审计输入，则保留为明确未完成项，不用本地模型替代。

本阶段不新增外部论文引用，不把缺少统一 checkpoint 的 JESTR/GLMR 写成复现结果，也不把三 reranker seeds 误称为 alignment 重复。页数仍不设 15 页硬上限，优先保证主结果清楚和证据闭合。

### 14.1 实际结果与偏差

- 官方兼容主表已完成，但严格意义上是“官方 retrieval JSON 顺序 + 已核验二维身份等价标签 + 保存 alignment embeddings”的 candidate/identity evaluation；没有重新运行 MassSpecGym loader 的谱图/分子编码，也没有重训 alignment。这是计划中明确保留的证据边界，不升级为端到端 official-loader equivalence。
- 全池身份审计复用已跟踪的 `relative_identity_eval.json`，两个候选池均为 17,556/17,556 positive queries、0 multiple-positive、0 collision；因此没有再次对约 700 万个候选逐个生成 InChIKey。中止了一个低效的全候选重复映射任务，未将其未完成输出纳入结果。
- 效率拆解已用 RTX 4090 完成；记录 coarse/relation/full、pointwise、参数/吞吐/显存，但正文仍不作部署效率优势声明。
- 外部强 baseline、candidate-aware alignment、统一 loader fresh re-encoding 和跨数据集验证没有可审计输入，按原计划保留为限制，没有扩大目标或伪造实验。
