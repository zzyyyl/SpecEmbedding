# 论文修改计划：ScholarGPT 外部评审后的主线收紧与评价边界

- 状态：`未执行`
- 创建日期：2026-08-21
- 最后更新：2026-08-21
- 负责人：Codex
- 关联意见：`docs/paper-change-plans/2026-08-21-ScholarGPT修改意见.md`
- 关联论文：`paper/main.tex` / `paper/main_cn.tex`
- 关联实现与工件：`evaluation_protocol.py`、`eval_rerank.py`、`SpecEmbedding/data/datasets_rerank.py`、`analysis/transfer2026_*`、`paper/adma2026_artifact_manifest.json`
- 计划约束：先收紧创新主线和证据边界；只执行可由当前缓存/检查点和本地环境可靠完成的补充分析。缺失原始 MassSpecGym 数据、候选池或未保存预测时，不伪造 official evaluator、top-K、外部 baseline、容量匹配或 bootstrap 结果。

## 1. 修改目标与动机

将论文中心从“candidate-set Transformer 的结构创新”收紧为：在本地 exact-target-SMILES 单正例协议和固定 top-40 候选列表上，监督式、非生成式 second-stage reranking 能够修正固定 embedding retriever 的列表内排序；pointwise 已解释大部分收益，candidate self-attention 的额外收益仍是条件性、未稳定建立的分析问题。

同时回应评审提出的“reranker 学到了什么”“官方 evaluator 可比性”“K、容量和统计不确定性”问题：

1. 先检查现有缓存是否足以完成 official-compatible identity 的只读/评价分析；若可执行，新增结果必须与 local protocol 分表并明确 evaluator 版本。
2. 对 top-K sensitivity、parameter-matched pointwise、matched external retriever、query-level bootstrap 等需要原始候选池、重新训练或逐 query 预测的项目，先核验输入；缺失时只新增限制、未来工作和计划记录，不把静态事实写成实验结果。
3. 将 score/embedding 信息来源的已有消融准确解释为有限证据，不扩展成未完成的四级输入消融结论。

## 2. 当前证据与问题定位

- 当前最稳固证据：三种 reranker training seeds、三个历史 alignment checkpoints、两种候选池；两种 learned reranker 相对各自 base 的 12 个 aggregate cells 均改善 R@1/MRR。
- Transformer 相对 Pointwise 的 alignment-level 方向仅 2/3；candidate-set Transformer 参数量显著更高；不能归因于 self-attention 独立收益。
- no-rank-embedding 在 2 pools × 3 reranker training seeds 中同时改善两个指标，且训练期 positive forcing 存在 rank shortcut；这是重要负面发现。
- 现有 `evaluation_protocol.py` 只声明 reference evaluator 为二维 InChIKey/可能多正例，`eval_rerank.py` 和 cache dataset 当前按 exact SMILES 单正例计算；未保存 official-compatible query-level 结果。
- 当前工作区没有 `data/MassSpecGym/` 原始谱图和候选 pickle；虽然 artifact manifest 记录其 hash，不能据此生成 K=20/80/full-pool 或重新训练结果。
- 现有 rerank cache/checkpoint 记录 top-40 canonical 运行，可在不改变训练的前提下探索“从现有 top-40 预测中按二维身份重算指标”的可行性；若逐 query top-1 预测未保存，需运行只读 evaluator，运行结果必须单独标为新增评价。

## 3. 修改范围

### 3.1 涉及文件与章节

- [ ] `paper/main.tex` / `paper/main_cn.tex`：重写摘要、引言贡献、结果讨论和结论，突出 supervised reranking framework，弱化 Transformer 首要创新；集中 protocol caveat。
- [ ] `paper/main.tex` / `paper/main_cn.tex`：新增“reranker information source”分析段，严格区分已有 no-base-score/no-rank/no-interaction 消融与尚未完成的仅 score/仅 embedding 四级消融。
- [ ] `paper/main.tex` / `paper/main_cn.tex`：将 K=40、容量不匹配、official identity、外部 baseline、bootstrap 等未完成项目集中写入 limitations/future work，避免重复防御性说明。
- [ ] `README.md`、`evaluation_protocol.py`、`reproducibility/anonymous-supplement-README.md`：同步可执行性和数据/工件缺失边界；如新增 evaluator 分析，记录命令、身份实现和工件 hash。
- [ ] `analysis/transfer2026_scholargpt_review/`：保存仅由现有工件得到的 feasibility/identity 统计或明确 blocked 记录，不覆盖既有 canonical artifacts。
- [ ] `paper/release/`、`docs/project_memory_zh.md`：若论文正文变化，重新构建并更新受跟踪 PDF、manifest 和哈希。

### 3.2 明确不做的事项

- 不在缺少原始数据/候选池时声称 official evaluator、K=20/80/full-pool 或 candidate coverage 的新实验结果。
- 不在没有新训练授权和 parameter-matched checkpoint 时声称容量控制结论。
- 不复现 JESTR/GLMR 或引入未经核验的新 external baseline 数字。
- 不把已有三 seed SD 改写成显著性检验；只有实际生成逐 query 配对结果后才报告 bootstrap CI。
- 不新增未经一手来源核验的参考文献；评审列出的相关工作均已有 25-key 引用闭包。

## 4. 具体内容设计

### 4.1 主线与贡献

- 英文/中文摘要第一主张改为“direct supervised non-generative second-stage reranking improves fixed embedding retrieval under the stated local protocol”。
- contribution 1 改为 controlled empirical instantiation/audit，不暗示 reranking、residual、listwise 或 Transformer 本身首创。
- candidate-set Transformer 作为 secondary conditional comparison：保留 canonical 小幅差异、2/3 alignment direction 和 parameter confounding；不称 robust self-attention benefit。
- 将 no-rank-embedding 的全配对改善及 forcing shortcut 提升为重要分析发现；明确它削弱 rank prior 独立正收益主张。

### 4.2 Reranker 学习内容

- 仅使用现有结果写成有限证据：pointwise 已获得大部分 Transformer-to-base gain；现有 removal rows 表明 base-score pathway、interaction features、rank embedding 和 residual 的方向依赖 candidate/seed。
- 不把“仅 (z_s,z_m) / 仅 b_i / (z_s,z_m,b_i)”写成已完成实验。若新增 evaluator/统计无法提供该分解，明确列入 future controlled feature-source ablation。
- 解释当前 reranker 可能同时承担 nonlinear score calibration 与 embedding compatibility correction；这是机制假设/待验证问题，不写成已测量因果结论。

### 4.3 评价协议、外部可比性与新增分析

- 先静态核验 cache 中目标/候选 SMILES 是否可由 RDKit 生成二维 InChIKey；若能运行，新增 `local_exact_smiles` 与 `reference_2d_inchikey_multi_positive` 双 protocol 的 query-level Recall/MRR，并在 manifest 记录 evaluator commit、identity collision/multiple-positive counts。
- 若无法可靠运行，新增 feasibility report：明确原始数据、官方 loader 或逐 query predictions 缺失，正文只保留 local protocol，limitations 明确“official-compatible evaluation remains pending”。
- 不把 reported-only JESTR/GLMR 重新放回数值表；可保留机制背景和 matched baseline future work。

### 4.4 K、容量、bootstrap 与外部 baseline

- 对 K=20/80/full-pool、parameter-matched pointwise、matched external retriever、query-level paired bootstrap 分别做输入/工件可用性检查。
- 只有在实际完成并保存结果工件后，才加入表格和主文结论；否则在 limitations 中集中说明它们是下一阶段验证，不新增数字。
- 将“confidence interval/显著性未提供”从多处重复说明收束到 Metrics/Limitations 两处。

### 4.5 双语与引用

- 中英文摘要、贡献、讨论、结论逐段同步。
- 现有 25 条引用继续使用；不新增引用。若需引用 official loader，仅使用已在正文核验的 MassSpecGym commit。

## 5. 引用文献与真实性核验

| 引用键 | 文献 | 支持的论断 | 一手来源 | 核验状态 |
|---|---|---|---|---|
| `bushuiev2024massspecgym` | MassSpecGym, NeurIPS 2024 | 数据集、候选池和官方协议背景 | 论文/官方代码 commit | 已核验，现有 25-key 闭包 |
| `kalia2025jestr` | JESTR, Bioinformatics 2025 | 外部机制背景，不报告本项目复现数字 | 论文 | 已核验，现有引用 |
| `zhang2026glmr` | GLMR, AAAI 2026 | 外部机制背景，reported-only | 论文 | 已核验，现有引用 |
| `liu2026massspecgymwild` | MassSpecGym in the Wild, 2026 preprint | protocol/shortcut 风险背景 | 预印本 | 已核验，现有引用 |
| 其余现有 21 条 | 现有 references.bib | 相关工作、指标和限制 | 原有 DOI/出版社来源 | 不新增、不改元数据 |

## 6. 实验与计算边界

- 是否新增训练实验：默认否；若用户随后授权并提供原始数据/预算，parameter-matched 或 retriever-agnostic 训练必须另立计划。
- 是否新增评价运行：可选是；仅允许使用现有 cache/checkpoint 做 reference-identity feasibility/evaluation，结果必须与 local protocol 分离。
- 是否进行静态数据统计：是；核对 cache schema、SMILES→2D InChIKey 转换可行性、artifact provenance、现有消融分母和缺失输入。
- 是否只修改文字、公式或引用：是，除非 official-compatible 只读评价实际完成并产生可审计工件。

## 7. 分步执行清单

- [ ] 步骤 1：提交本计划；状态从 `未执行` 改为 `执行中`。
- [ ] 步骤 2：核验官方 identity、cache schema、预测可生成性、原始数据/候选池和 GPU/磁盘可用性。
- [ ] 步骤 3：完成可行的 identity feasibility/评价分析；不可行项目写入 blocked report，不手填数字。
- [ ] 步骤 4：按主线收紧英文稿并集中 protocol/未完成实验边界。
- [ ] 步骤 5：同步中文稿、README、evaluation protocol 和补充包说明。
- [ ] 步骤 6：静态核对引用、数字、结论边界和中英文一致性。
- [ ] 步骤 7：完整构建双语稿；如有新结果，更新 analysis manifest、release PDF 和 project memory。
- [ ] 步骤 8：运行测试、Ruff、artifact check、git diff check。
- [ ] 步骤 9：提交实质论文修改；如新增实验，单独提交分析工件和结果。
- [ ] 步骤 10：必要时送外部专家复核；回填结果并将计划归档为 `已执行` 或 `已偏离待确认`。

## 8. 风险、证据边界与待确认事项

- 当前原始 `data/MassSpecGym` 不在工作区，manifest 仅有 hash；若官方兼容评价需要原始标签或候选文件，不能仅凭 hash 复现。
- top-K sensitivity 需要 full candidate pools 或按 K 重新构造 cache；直接截取训练于 K=40 的 cache 不等价于 K 实验。
- parameter-matched pointwise、matched external retriever 和 query bootstrap 分别需要新 checkpoint、外部模型或逐 query predictions；缺失时必须保持 pending。
- 若新增分析改变论文科学结论、评价身份定义或主表数字，计划应暂停为 `已偏离待确认`，等待用户决定是否扩大实验范围。

## 9. 验证方案

- [ ] 运行 `conda run -n specembedding python -m pytest -q tests`、Ruff、`freeze_adma2026_artifacts.py --check`。
- [ ] 检查双稿引用键与 `references.bib` 25 条闭包，确保不新增未经核验文献。
- [ ] 检查 local/reference evaluator 标签和所有新增数字的来源；逐 query 结果若存在需保存 hash/manifest。
- [ ] 英文/中文 `bash paper/build_release.sh`，检查 fatal、undefined citation/reference、overfull/underfull、页数、匿名元数据和私有路径。
- [ ] `git diff --check`、工作区状态、PDF 与 build-manifest 哈希一致。

## 10. 执行记录

尚未执行。

## 11. 最终结果

- 完成日期：尚未完成
- 最终状态：`未执行`
- 验证结果：尚未验证
- 论文修改 commit：尚未提交
- 计划归档 commit：尚未提交
- 相对原计划的偏差：尚未记录
