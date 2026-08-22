# 论文修改计划：ScholarGPT 方法重构与实验复核

- 状态：`执行中`
- 创建日期：2026-08-23
- 最后更新：2026-08-23
- 负责人：Codex
- 关联意见：`docs/paper-change-plans/2026-08-21-ScholarGPT修改意见.md`、外部评审附件 `pasted-text-1.txt`
- 关联论文：`paper/main.tex` / `paper/main_cn.tex`
- 计划约束：优先完成评审明确要求的无捷径训练和候选相对判别核心；只把实际运行产生的结果写入论文，缺失原始数据/外部代码的项目保留为阻塞项，不伪造结论。

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
- `/data1/zyl/SpecEmbedding/processed/MassSpecGym/` 提供候选源及数据入口，但其是否足以生成无 forcing 全池 cache 需在运行时确认。
- 现有论文已明确 local exact-SMILES 与官方二维 InChIKey 协议边界；官方 evaluator、外部强 baseline、容量匹配、query bootstrap 只有在真实输入和运行完成后才能升级为主结果。

## 3. 修改范围

### 3.1 涉及文件与章节

- [ ] `prepare_rerank_cache.py`、`SpecEmbedding/data/datasets_rerank.py`：把无 forcing 设为新实验的强制配置；训练集只使用自然有标签 query，保留全池候选和 coverage 元数据。
- [ ] `SpecEmbedding/models_rerank.py`、`SpecEmbedding/utils/rerank.py`、`train_rerank.py`、`eval_rerank.py`：实现无 rank 的谱图条件相对候选 reranker、反对称 pair preference、谱图依赖损失和可审计配置。
- [ ] 新增 `analysis/transfer2026_scholargpt_method/`：保存配置、参数量、训练/评价结果、难度分层和效率测量；不覆盖既有 canonical 工件。
- [ ] `tests/`：增加模型等变性、反对称性、无 forcing 数据集、谱图依赖损失和参数统计测试。
- [ ] `paper/main.tex` / `paper/main_cn.tex`：在新实验完成后按真实结果重写标题、摘要、引言、方法、RQ 主结果、消融、限制和结论；中英文逐段同步。
- [ ] `README.md`、`reproducibility/`、`docs/project_memory_zh.md`：记录新命令、数据/检查点/匿名工件边界和未完成项目。
- [ ] `paper/release/`：源稿变化后重新生成双语发布 PDF 和构建清单。

### 3.2 明确不做的事项

- 不把现有 forced top-40 结果改名为无 forcing 新方法结果。
- 没有完整原始 candidate pool、外部模型或逐 query 预测时，不声称官方 evaluator、外部 baseline、bootstrap CI 或 full-pool 结果。
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

- [ ] 步骤 1：提交本计划并将状态设为 `执行中`。
- [ ] 步骤 2：确认无 forcing 256-cache、原始候选源和 GPU 运行入口；若失败记录阻塞。
- [ ] 步骤 3：实现无 rank 的 relative reranker、谱图依赖损失及无泄漏数据集。
- [ ] 步骤 4：单元测试形式性质、mask、loss 和训练配置；运行 Ruff/compileall。
- [ ] 步骤 5：通过 detached `tmux` 生成新 cache，训练 mass/formula 核心配置和必要对照。
- [ ] 步骤 6：评价主指标、coverage、难度/效率分析；保存 JSON/CSV/manifest。
- [ ] 步骤 7：按真实结果重写英文稿，随后同步中文稿、README 和复现说明。
- [ ] 步骤 8：完整双语构建、引用/匿名性/工件检查和测试。
- [ ] 步骤 9：使用中文 Conventional Commit 提交论文/代码实质修改。
- [ ] 步骤 10：回填本计划实际结果和 commit，将状态设为 `已执行`；用独立中文文档 commit 归档计划。

## 8. 风险、证据边界与待确认事项

- 新模型可能无法稳定优于 pointwise；这是评审验收条件，不能用叙事修补。
- 全池缓存和相对 pair 计算可能受显存/磁盘限制；先用小限额 smoke test，再提交后台长任务。
- 现有缓存路径包含历史机器绝对路径元数据；结果记录应使用仓库相对路径和哈希，不把私有路径写入论文或 release PDF。
- 若实现需要改变官方 evaluator、数据身份规则或引入新外部模型，计划应暂停为 `已偏离待确认`，先向用户报告。

## 9. 验证方案

- [ ] `conda run -n specembedding python -m pytest -q tests`
- [ ] `conda run -n specembedding python -m ruff check .`
- [ ] `python freeze_adma2026_artifacts.py --check`
- [ ] `python -m compileall` 覆盖新增脚本/模块。
- [ ] `bash paper/build_release.sh`，检查双稿 citation/reference、LaTeX 错误、匿名元数据和私有路径。
- [ ] `git diff --check`、工作区和 manifest/结果哈希一致。

## 10. 执行记录

2026-08-23：读取外部评审附件。确认其要求已超出上一份只做文字边界收紧的计划；新建本计划。发现本地存在历史 forced cache、alignment/reranker checkpoints 和候选数据入口，下一步先实现无 forcing/no-rank 核心并用小规模 GPU smoke test 验证。

## 11. 最终结果

- 完成日期：尚未完成
- 最终状态：`执行中`
- 验证结果：尚未完成
- 论文修改 commit：尚未提交
- 计划归档 commit：尚未生成
- 相对原计划的偏差：本计划主动扩大到方法与训练实现；无法由现有工件可靠完成的官方 evaluator、外部 baseline 和跨编码器项目保留为未完成事项。
