# SpecEmbedding 协作约定

本文件是仓库级工作规范，适用于整个仓库。若子目录新增更具体的 `AGENTS.md`，在对应范围内同时遵守其补充规则。`docs/` 记录当前项目状态，不能替代代码、配置、原始日志和可校验工件。

<!-- CODEGRAPH_START -->
## CodeGraph

In repositories indexed by CodeGraph (a `.codegraph/` directory exists at the repo root), reach for it BEFORE grep/find or reading files when you need to understand or locate code:

- **MCP tool** (when available): `codegraph_explore` answers most code questions in one call — the relevant symbols' verbatim source plus the call paths between them, including dynamic-dispatch hops grep can't follow. Name a file or symbol in the query to read its current line-numbered source. If it's listed but deferred, load it by name via tool search.
- **Shell** (always works): `codegraph explore "<symbol names or question>"` prints the same output.
<!-- CODEGRAPH_END -->

## 1. 项目结构与当前目标

本仓库是 MS/MS-to-molecule retrieval 系统：

```text
MS/MS 谱图
  -> Transformer 谱图编码器
  -> GINE 分子编码器与跨模态对齐
  -> base candidate retrieval
  -> 固定候选池内的非生成式监督 reranking
```

主要代码分工：

- `SpecEmbedding/models.py`：峰序列、m/z embedding 和谱图 Transformer；
- `SpecEmbedding/models_align.py`：GINE 分子编码器和谱图--分子双塔对齐；
- `train.py` / `eval.py`：谱图表征学习与谱图检索；
- `train_align.py` / `eval_align.py`：跨模态对齐与基础候选检索；
- `prepare_rerank_cache.py`、`train_rerank.py`、`eval_rerank.py`：候选缓存、reranker 训练和评价；
- `run_rerank_pipeline.py`：rerank 流程编排；
- `SpecEmbedding/utils/`、`SpecEmbedding/trainer/`：公共运行时、模型加载、候选、评价和训练逻辑。

当前论文主张限定为：在固定跨模态检索器已召回的候选列表内，监督式、非生成式 residual learning-to-rank 可以改善 Recall/MRR。不得把增益归因于 candidate self-attention 或 relative 模块的独立普遍收益，也不得据此声称 SOTA、统计显著性、完整端到端稳定性、部署 latency 或一般化能力。

## 2. 证据优先级与评价边界

遇到数字、路径、状态或结论冲突时，按以下顺序核对：

1. 当前代码、测试、`params.yaml` 和 checkpoint 配置；
2. 原始实验日志、运行状态和可校验工件；
3. release/validation manifest；
4. `docs/` 当前文档、论文计划和 README；
5. Git 历史中的已归档记录。

不要把文档中的预期参数、历史数字或静态代码推断写成新实验结果。引用实验结果时必须同时核对 candidate type、候选池、checkpoint、seed scope、cache 协议和日志。

当前 rerank 主线是无 rank、无正例 forcing 的 `relative` 模型：最多 256 个自然候选先做 pointwise coarse scoring，再在 coarse top-40 上做谱图条件的候选相对判别；`pointwise` 是容量匹配对照。旧的 rank-aware Transformer 和强制正例只用于显式的 legacy 复核。

数据与评价规则：

- 数据集为 MassSpecGym；官方 structure-disjoint split 不表述为 scaffold-disjoint；
- `mass` 是质量候选协议，`formula` 假设已知分子式，必须称为 formula-conditioned retrieval；
- reranker 是 closed-library 方法，只能重排输入候选，不能生成候选库外分子或救回未被 base top-$K$ 召回的真值；
- 评价必须区分 cache-local exact-target-SMILES 结果和独立的官方候选顺序/二维身份审计；没有 fresh loader 重编码或 alignment 重训时，后者只称 official-compatible candidate/identity audit，不能把其结论继承给未审计的新 cache；
- local exact-target-SMILES 只作为敏感性视图；正例 coverage、miss 和 forcing 语义必须以 cache 元数据为准；
- JESTR/GLMR 是 reported-only 外部背景，未在本仓库统一复现，不与本地结果直接排序比较；
- 静态检查、候选身份审计和 forward implementation audit 不是新实验；效率拆解也不是端到端部署 latency 结论。

## 3. 环境、配置与实现

- 首选 Conda 环境为 `specembedding`，环境定义为 `environment.yml`，开发依赖为 `requirements-dev.txt`；运行实验前确认 Python、PyTorch、PyG、RDKit 和 matchms 可用。
- 训练、评价和 rerank 超参数统一放在 `params.yaml`；缺失配置应尽早失败，不提供静默 fallback。
- 从 A05 及后续启动起，使用 `run_with_storage.py` 和显式外部存储根；本机为
  `/data1/${USER}/SpecEmbedding`。数据、缓存、下载缓存、临时数据与新训练工件不写入 `/home`。
  不修改当前 A04 的源码、环境或文件路径；仍被活跃任务引用的旧文件保留。细则见 `docs/storage_zh.md`。
- checkpoint 必须保存构建 reranker 所需的完整模型配置，包括模型类型、维度、归一化方式和特征开关。
- 公共逻辑放在 `SpecEmbedding/utils/` 或 `SpecEmbedding/trainer/`；入口脚本负责参数解析和流程编排，不从入口脚本反向导入公共业务函数。
- 正式 alignment、reranker、对照和消融必须使用完整训练划分中符合协议的全部可训练样本；不得自行限量、抽样或截取前 N 条。验证/测试不并入训练，运行前后核验实际样本数。
- 正式训练严格使用显式 `cuda:N`，启动前检查 GPU 空闲显存和利用率；繁忙或显存不足时等待，不回退到 CPU、不终止其他任务。CPU 仅用于 synthetic 测试、处理、审计和明确标注的非论文实现 smoke；中断的 CPU 训练工件隔离并排除。

基础检查：

```bash
conda activate specembedding
python -m pytest -q
ruff check .
python -m compileall -q .
```

运行入口见 [中文指南](docs/README_zh.md)。统一 rerank 入口默认 train/val/test 均不 forcing；复用旧 cache 时仍须核对元数据。只有复核 legacy 训练 cache 时才显式使用 `--force-include-positive`，且该开关不能作用于验证或测试。`--limit N` 仅用于非正式调试，结果不替代正式实验；`--dry-run` 只打印命令。输出目录必须带 `_topkN`/`_limitN` 后缀，长任务使用 detached `tmux`。

## 4. 文档维护

当前 canonical 文档：

- `docs/README_zh.md`：运行入口和文档地图；
- `docs/project_memory_zh.md`：当前工程、实验和证据边界摘要；
- `docs/reranker_solution_zh.md`：当前 reranker 实现说明；
- `model_architecture.md`：谱图塔、分子塔与对齐实现；
- `analysis/README.md`：实验系列、协议和冻结证据索引；
- `docs/paper-change-plans/`：计划规则、模板、在途计划与历史追溯；
- `paper/TRANSFER_2026_PLAN.md`：唯一投稿待办；`paper/BUILDING.md` 与 `reproducibility/README.md`：发布与复现入口。

历史文档在信息迁移到当前 canonical 文档后可以删除；Git 历史保留追溯能力。不要在多个文档中复制同一组实验表、哈希或状态；详细结果应指向 `analysis/`、原始日志或 manifest。

## 5. 论文重大修改与计划

修改论文前，先在 `docs/paper-change-plans/` 创建计划，再开始修改。以下情况按重大修改处理：

- 新增、删除、合并或重组章节、表格、图或附录；
- 改变贡献、方法定义、评价协议、实验结论或限制性；
- 成组修改中英文稿、相关工作或引用；
- 任何可能改变论点与证据是否一致的修改。

拼写、标点、单处措辞或不改变论点的排版小修可不建新计划；拿不准时按重大修改处理。一个工作只维护一份计划，直接更新该计划，不另建互相矛盾的版本。

计划文件必须：

1. 从 `docs/paper-change-plans/TEMPLATE.md` 复制并按 `YYYY-MM-DD-简短主题.md` 命名；
2. 在开头使用唯一状态：`未执行`、`执行中`、`已执行`、`已偏离待确认` 或 `暂停`；
3. 写明目标、证据问题、范围与不做事项、具体修改、引用核验、实验边界、步骤、风险、验证和执行记录；
4. 开始论文修改前设为 `执行中`，完成逐项验证、编译和 commit 回填后才能设为 `已执行`；
5. 若新证据改变结论、协议、范围或需要新增实验，暂停并设为 `已偏离待确认`，等待用户确认。

引用必须优先核对论文原文、出版社页面、DOI、官方文档或官方代码。不得凭记忆补写元数据，不得把预印本写成正式发表成果，不得把静态推断写成效率、显著性、泛化性或因果结论。

## 6. 论文发布与匿名化

- 英文 `paper/main.tex` 与中文 `paper/main_cn.tex` 的适用内容保持一致；论文必须区分继承的 SpecEmbedding 峰序列 Transformer 与本文第二阶段 residual learning-to-rank。
- LaTeX 中间产物放在 Git 忽略的 `paper/build/`；发布使用 `bash paper/build_release.sh`，核对 `paper/release/` 的 PDF、页数、工具链和 SHA-256 manifest。
- 论文 AI assistance disclosure 必须覆盖实际使用范围；作者需核验 AI 辅助内容并承担责任。
- 匿名补充材料使用无 `.git` 历史的独立 allowlist 归档；排除 Git 历史、数据、checkpoint、cache、日志、内部 manifest、用户名、hostname、remote/author、绝对路径、邮箱/ORCID、GPU UUID、公开托管链接和敏感文件类型。
- 构建后执行匿名扫描、解包测试、shell 语法、Ruff、compileall 和 synthetic CPU smoke；结果记录在 `reproducibility/anonymous-supplement-validation.yaml`。匿名 ZIP 是生成工件，默认位于 Git 忽略的 `dist/`。

## 7. 安全、编辑与提交

- 修改本地文件使用 `apply_patch`；先检查工作树，保留用户已有修改，不覆盖无关文件。
- 删除或覆盖文件前确认目标明确且属于用户请求；避免以宽泛路径执行递归破坏性操作。
- 完成修改后按风险运行相关测试、静态检查、`git diff --check`，并检查没有混入无关变更。
- Git 提交使用 Conventional Commit：`type` 和 `scope` 使用简洁英文，冒号后的说明使用中文，例如 `docs(cleanup): 精简项目文档`。
- 论文计划遵循“先提交论文实质修改，再回填计划并用独立文档提交归档”的顺序，避免计划自我引用 commit。
- 最终报告应说明修改文件、计划状态、主要变更、验证结果、未完成边界和 commit。
