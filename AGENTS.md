# SpecEmbedding 协作约定

本文件适用于整个仓库。若子目录存在更具体的 `AGENTS.md`，则在对应范围内同时遵守其补充规则。

## 项目上下文与证据边界

本仓库是一个 MS/MS-to-molecule retrieval 系统，主流程为：

```text
MS/MS 谱图
  -> Transformer 谱图编码器（SiameseModel）
  -> GINE 分子图编码器与谱图--分子双塔对齐（SpecMolAlignModel）
  -> 基础相似度候选检索
  -> 固定候选池内的非生成式监督 reranker
```

主要代码分工如下：

- `SpecEmbedding/models.py`：峰序列、sinusoidal m/z embedding 和 Transformer 谱图编码器；
- `SpecEmbedding/models_align.py`：GINE 分子编码器和跨模态对齐模型；
- `train.py` / `eval.py`：谱图表征学习和谱图到谱图检索；
- `train_align.py` / `eval_align.py`：谱图--分子对齐和候选检索；
- `prepare_rerank_cache.py`、`train_rerank.py`、`eval_rerank.py`：候选缓存、二阶段排序训练和评估；
- `SpecEmbedding/utils/`、`SpecEmbedding/trainer/`：公共运行时、模型加载、数据提供、评估和训练逻辑；入口脚本主要负责参数解析和流程编排。

项目记忆和历史分析中的数字、路径与结论必须服从以下证据优先级：当前代码与 `params.yaml`、原始实验日志和可校验工件、论文修改计划、`docs/project_memory_zh.md`、README。若发现冲突，应先核对代码和原始日志，不把记忆文档中的历史数字当作当前实验结果。

当前论文可由已有实验支持的核心主张是：在固定跨模态检索器已经召回的候选列表内，监督式、非生成式 residual learning-to-rank 可以改善 Recall/MRR。不得将整体 reranking 增益归因于 candidate self-attention，也不得据此声称完整端到端稳定性、统计显著性、SOTA 或一般化能力。

截至 2026-08-23，新的无 rank、无 positive forcing、最多 256 候选 coarse-to-fine relative reranker 已完成 pilot 和官方候选顺序兼容评价；relative 在主要 R@1/MRR 汇总中略低于容量匹配的 pointwise，因此不能声称关系模块具有独立普遍收益。JESTR/GLMR 仍是 reported-only、未在本仓库统一复现且不可直接比较的外部工作。

## 数据、候选和评价协议

- 主要数据集为 MassSpecGym；使用官方 structure-disjoint split 时，不能进一步表述为 scaffold-disjoint。
- `mass` 与 `formula` 是两种候选协议；`formula` 假设已知真实分子式，必须称为 formula-conditioned retrieval，而不是完全开放的未知分子鉴定。
- reranker 是 closed-library 方法，只能重排输入候选，不能生成候选库之外的新分子，也不能恢复未进入 base top-$K$ 的真实分子。
- 训练 cache 可以使用 `force_include_positive=true` 保证监督样本有正例，并保留真实分子的原始 base rank；验证和测试必须使用 `force_include_positive=false`，正例未进入 top-$K$ 时记为 miss，所有查询仍纳入指标，并报告 pre-retrieval recall upper bound。
- canonical overlap-clean 结果使用 top-40 候选、alignment seed 42 和 reranker seeds 42/43/44；其标准差只描述 reranker-level variation。alignment seeds 42/43/44 的跨 alignment 结果是三个内部描述性 estimates，不是端到端方差、置信区间或显著性检验。
- 官方 JSON 顺序核验若复用保存的 embedding，只能称 official-compatible candidate-order/identity audit；没有重新运行官方 loader、alignment 编码或重训时，不得称为完整 official-evaluator 端到端等价。
- 当前身份审计显示保存的 full-pool 候选对每个测试 query 有一个二维 InChIKey 等价正例、没有多正例或候选身份碰撞；这不改变“结果基于保存嵌入和既定 cache”的评价边界。
- 已发现并审计过跨 split 的完全输入重叠。train--test 三条输入剔除敏感性不改变主结论；train--val overlap-clean 同时重训了 alignment，因此新旧指标差异不能解释为删除 validation 查询的隔离因果效应。
- 不得把静态代码检查、候选集合审计或 forward implementation audit 写成新实验结果；效率审计也不是端到端部署 latency 结论。

详细实验表、工件路径和历史提交记录见 `docs/project_memory_zh.md` 及 `analysis/`，引用结果前应核对对应运行目录、candidate protocol、checkpoint、seed scope 和日志。

## 环境、配置和实现约定

- 首选 Conda 环境为 `specembedding`，维护环境定义为 `environment.yml`；已验证的主要运行环境为 Python 3.12、PyTorch 2.5.1/CUDA 12.1，主要实验硬件为 RTX 4090。运行测试或实验前先确认 `torch`、PyG、RDKit、matchms 等依赖已安装。
- 训练、评估和 rerank 超参数统一放在 `params.yaml`。命令行只保留路径、候选类型、设备、运行模式和调试 limit 等频繁变化的参数；缺失配置应尽早失败，不提供静默 fallback。
- checkpoint 必须保存构建 reranker 所需的完整模型配置，尤其是模型类型、维度、归一化方式和相关特征开关。
- 公共逻辑应放在 `SpecEmbedding/utils/runtime.py`、`utils/model.py`、`utils/align.py`、`utils/rerank.py`、`utils/providers.py` 或 `SpecEmbedding/trainer/`；不要从 `train.py`、`eval.py` 等入口脚本反向导入公共业务函数。
- 默认路径和实验参数不能替代对实际运行目录及日志的核验；设计文档中的预期数字不是实验结果，`docs/reranker_solution_zh.md` 中的旧设计默认值也可能不同于论文 canonical top-40 实验。

常用入口：

```bash
python run_pipeline.py --no-pretrain --device cuda:1
python run_rerank_pipeline.py massspecgym \
  --align_save_dir checkpoints_align/<run> \
  --candidate_type formula --topk 40 --device cuda:1
```

调试运行应使用 `--limit`，并确保输出目录带有 `_topkN`/`_limitN` 后缀，避免覆盖正式工件。长时间实验应在 detached `tmux` 会话中运行。

## 论文、发布与匿名化

- 论文大改动必须遵守下文的计划流程；英文 `paper/main.tex` 和中文 `paper/main_cn.tex` 的适用内容应保持一致。
- LaTeX 中间产物放在 Git 忽略的 `paper/build/`；正式发布或审计使用 `paper/build_release.sh`，并核对 `paper/release/` 中 PDF、构建工具链、页数和 SHA-256 manifest。
- 论文必须持续区分：继承原 SpecEmbedding 的峰序列 Transformer 架构；不把原谱图--谱图 SupCon/Tanimoto 目标或预训练 checkpoint 写成本文方法；本文主张集中在跨模态对齐后的第二阶段 residual learning-to-rank。
- JESTR/GLMR 的外部 reported 数字必须标注未复现、不可直接比较，不得用于未经协议匹配的性能排序。
- 匿名补充材料必须从无 `.git` 历史的独立归档构建，并扫描用户名、hostname、Git remote/author、绝对路径、邮箱/ORCID、GPU UUID、公开托管链接及敏感文件类型；`reproducibility/anonymous-allowlist.txt` 和确定性 ZIP builder 是现有流程的一部分。
- 若实际使用 AI 的范围扩大，须同步更新论文中的 AI assistance disclosure；作者需核验 AI 辅助内容并承担结果责任。

## 论文较大改动必须先制定计划

对论文进行较大改动前，必须先在 `docs/paper-change-plans/` 中创建一份详细、完整的 Markdown 计划书，然后按计划逐步执行。不得先修改论文、后补计划。

“较大改动”包括但不限于：

- 新增、删除、合并或重组章节、小节、表格、图或附录；
- 改变论文的贡献表述、创新边界、方法定义、评价协议、实验结论或局限性；
- 一次性加入多篇文献或重写研究背景、相关工作、讨论；
- 同时修改中英文稿件的多处内容；
- 任何可能影响论点与现有证据是否一致的修改。

拼写、标点、LaTeX 排版、单处措辞等不改变论点的小修可以不单独创建计划，但不得借“小修”名义实施成组的实质性修改。拿不准时，按较大改动处理。

## 计划书位置与命名

- 目录：`docs/paper-change-plans/`
- 文件名：`YYYY-MM-DD-简短主题.md`
- 新计划应从 `docs/paper-change-plans/TEMPLATE.md` 复制结构。
- 同一项工作只维护一份计划；执行过程中直接更新该文件，而不是另建互相矛盾的版本。

## 执行状态

每份计划书开头必须包含明确状态，且只能使用以下状态之一：

- `未执行`：计划已写完，尚未修改论文；
- `执行中`：正在按计划修改；
- `已执行`：计划内工作、核验和必要提交均已完成；
- `已偏离待确认`：发现计划过时、事实不符或需实质扩大/改变范围，已暂停并等待用户确认；
- `暂停`：因其它明确原因暂时停止。

开始修改论文前，将状态从 `未执行` 更新为 `执行中`。完成后，只有在逐项核验、编译或其它约定验证完成，并回填实际结果与 commit 信息后，才能标记为 `已执行`。

## 计划书必备内容

计划至少应包含：

1. 修改目标与动机；
2. 当前证据和问题定位；
3. 修改范围、涉及文件与明确不做的事项；
4. 各章节拟增加、删除或重写的具体内容；
5. 引用文献清单及逐条真实性核验结果；
6. 是否需要新实验、静态统计或仅做文字修改；
7. 分步骤执行清单和每一步的完成状态；
8. 风险、证据边界及可能需要用户确认的决策；
9. LaTeX、引用、双语一致性及 `git diff` 等验证方法；
10. 实际执行记录、相对原计划的偏差、最终验证结果和 commit。

## 引用与证据要求

- 优先核对论文原文、出版社页面、DOI、官方文档或官方代码等一手来源。
- 新增引用前必须核对题名、作者、发表年份、出版物和 DOI/稳定链接；不得凭记忆补写文献。
- 预印本必须明确标注为预印本，不得表述为已正式发表成果。
- 引文必须直接支持附近论断；若只支持部分内容，应缩小论断范围。
- 不得把静态代码推断写成实验结果，也不得补写未实际测量的效率、显著性、泛化性或因果结论。
- 若修改评价身份定义、候选池、数据划分或指标口径，必须先核对实现与官方协议，并在计划中记录差异。

## 按计划执行与偏差处理

- 按计划中的顺序逐步修改，并及时勾选步骤、记录证据与验证结果。
- 小幅且不改变目标、论点或证据边界的偏差，可直接更新计划并说明原因。
- 如果计划因新证据而过时，或偏差会改变论文结论、引用依据、评价协议、修改范围或需要新增实验，必须停止相关修改，将状态设为 `已偏离待确认`，并询问用户确认后再继续。
- 用户确认后，在同一计划中记录决定、日期和调整内容，再恢复为 `执行中`。

## 论文修改完成条件

- 英文稿与中文稿在适用范围内保持一致；若有意不同，计划中必须说明原因。
- 引用键、参考文献元数据、交叉引用和数学符号已经检查。
- 在环境允许时完成 LaTeX 编译，并记录页数、警告或失败原因。
- 检查 `git diff`，确保没有混入无关文件或覆盖用户已有修改。
- 所有 Git 提交必须使用 Conventional Commit 格式；`type` 和 `scope` 使用简洁英文标识，冒号后的变更说明使用中文，并准确反映实际范围。例如：`docs(paper-plan): 开始执行 ScholarGPT 修订`。
- 为避免计划文件自我引用 commit 的循环，先提交论文实质修改，再在计划中回填该论文 commit、将状态改为 `已执行`，并用单独的文档 commit 归档计划；计划归档 commit 无需在计划内引用自身。
- 最终向用户报告计划文件、执行状态、主要变更、验证结果和 commit。
