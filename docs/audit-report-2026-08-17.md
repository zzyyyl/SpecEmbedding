# SpecEmbedding 项目与论文审计报告

审计日期：2026-08-17
审计基线：`df7dec6`（分支 `codex/transfer-2026-0831`）
审计范围：源码、训练/评估脚本、测试、实验 artifact、英文稿 `paper/main.tex`、中文稿 `paper/main_cn.tex`、引用与投稿待办。

修复跟踪状态：**本轮审计范围内通过，无 P0/P1 阻断项**。第三轮独立审计基于远端
`5a1fe38` 确认代码、复现、证据边界、匿名性和构建有效性通过；用户随后再次明确页数可以
延后，英文 17 页只作为独立投稿整理事项，不作 P0/P1 或本轮审计阻断。待完稿并确认转投
venue 规则后统一微调。下文第 1--6 节保留初次审计时点的原始判定，第 7 节以后记录修复
演进，避免把历史发现误读为当前状态。

## 1. 总体结论

当前仓库已经具备较完整的两阶段 MassSpecGym 实验流水线，核心 reranker 多种子结果、核心组件消融结果和 overlap-clean 处理均有结构化 artifact 支撑。论文目前对证据边界的表述总体谨慎：外部 JESTR/GLMR 数字被标为 `reported`，正文明确不声称严格 SOTA，也承认 self-attention 的独立收益尚未被 checkpoint-robust 证据确认。

但是，当前版本不应直接作为“第三方可复现发布包”或“已完成投稿材料”交付，原因是：

1. 默认工作区的 `pytest -q` 在收集阶段失败；`specembedding` conda 环境中甚至没有 `pytest` 命令，测试尚未形成可验证的通过记录。
2. 多个数据常量和配置仍写死 `/data1/...` 主机路径，README 的通用命令不能覆盖全部数据处理和旧版评估流程。
3. 英文/中文论文已经编译成功；仓库中 8 月 1 日遗留 PDF 为 15/14 页，而 8 月 17 日当前源稿重新编译为 16/15 页；`paper/ADMA2026_TODO.md` 与 `docs/project_memory_zh.md` 仍记录 11/10 页，投稿合规记录已过时。
4. 主结果使用本地 exact-SMILES 单正例协议，而 MassSpecGym 官方代码路径默认使用二维 InChIKey 等价并可能存在多个正例；论文已披露差异，但影响尚未量化，因此结果不能直接解释为官方协议等价结果。
5. 主要 reranker 结果固定 alignment seed 42，新增 alignment seeds 的 artifact 虽已存在，但论文主文没有相应的跨 alignment 聚合结果；当前结论应限定为固定 alignment checkpoint 下的 reranker 证据。

审计判定：**研究结论可在明确限定条件下使用；工程复现与投稿材料为有条件通过，发布前需完成 P0/P1 修复。**

## 2. 审计证据与验证范围

检查了以下内容：

- `train.py`、`train_align.py`、`eval.py`、`eval_align.py`；
- `prepare_rerank_cache.py`、`train_rerank.py`、`eval_rerank.py`、`run_rerank_multiseed.py`；
- `SpecEmbedding/data/`、`SpecEmbedding/trainer/`、`SpecEmbedding/utils/`；
- `tests/`、`analysis/` 及 `checkpoints_align/`、`checkpoints_rerank/` 中的 summary/status artifact；
- `paper/main.tex`、`paper/main_cn.tex`、`paper/references.bib`；
- `README.md`、`environment.yml`、`params.yaml`、`paper/ADMA2026_TODO.md`、`paper/TRANSFER_2026_PLAN.md`。

已执行或读取的验证：

- `python -m compileall ...`：通过，退出码 0；
- 8 月 1 日遗留 PDF：英文 `paper/build/main.pdf` 为 15 页，中文 `paper/build/main_cn.pdf` 为 14 页；8 月 17 日核心消融入稿后重新编译的当前源稿为英文 16 页、中文 15 页；
- LaTeX 日志未发现致命错误、未定义引用或未定义交叉引用；仅见 `fontspec` 字体和 `amsmath` 重定义 warning；
- 当前默认环境 `pytest -q`：收集阶段失败；缺少 `torch`、`SpecEmbedding` 根目录导入路径以及多个根目录分析模块；
- `conda run -n specembedding pytest -q`：失败，环境中没有 `pytest` 命令；
- 核对 canonical reranker aggregate：mass/formula 的 full Transformer 均为 3 个 seed，均值分别为 R@1 68.5824/74.5671、MRR 0.7260/0.7802；与论文受控消融表的均值一致；
- 核对 representative seed-42 行：论文主表的 68.74/74.75、72.87/78.22 等数字可在 `summary.csv` 找到；
- `git status`：审计开始时工作区无未提交变更。

## 3. 严重问题清单

### P0-1：测试与运行环境无法直接验证

位置：`tests/`、`environment.yml`、项目根目录导入方式。

现象：默认 Python 3.13 环境执行 `pytest -q` 时，测试收集失败。错误包括缺少 `torch`、`SpecEmbedding`、`analyze_alignment_multiseed` 和 `run_rerank_*` 根目录模块。项目声明的 conda 环境为 Python 3.12，但该环境没有安装 `pytest` 命令。

影响：当前不能声明测试通过，也不能把测试文件视为第三方可复现的验收入口。部分问题可能只是环境配置问题，但实际失败发生在收集阶段，尚未进入测试逻辑。

建议：

1. 将 `pytest`、项目本身和测试所需的运行时依赖纳入开发环境；
2. 在 README 给出唯一可复制的命令，例如 `python -m pytest -q`，并确保从仓库根目录可运行；
3. 在 CI 或冻结环境中记录完整版本和最终测试结果；
4. 修复测试对根目录脚本的导入依赖，或明确配置 `PYTHONPATH`/安装 package。

### P0-2：数据与候选身份协议未与官方评价协议统一

位置：`paper/main.tex:507-514`、`paper/main_cn.tex:410-419`，以及 `prepare_rerank_cache.py`、`eval_rerank.py` 的正例判定逻辑。

现象：当前评估按 exact target-SMILES 赋予单一正例；论文同时指出官方 MassSpecGym loader 默认使用二维 InChIKey 等价并可能标注多个正例。该差异会影响 Recall、MRR 和候选覆盖定义，但当前没有 identity-rule sensitivity 数字。

影响：本地结果可以作为“本地 exact-SMILES 协议下的结果”，但不能未经限定地称为官方指标复现，也不能与外部 baseline 做严格公平比较。

建议：

- 实现并运行 exact-SMILES 与二维 InChIKey/多正例两套评估；
- 报告受影响查询数量、候选正例数量和指标差异；
- 在主文、README 和结果文件中固定协议名称；
- 在完成前继续保留“local protocol / reported external”标签，不使用严格 SOTA 表述。

### P1-1：源码包含机器绝对路径，破坏可移植性

位置：`SpecEmbedding/const/gnps.py:4`、`SpecEmbedding/const/mona.py:4`、`SpecEmbedding/const/tsne_cluster.py:3-11`、`params.yaml`。

现象：旧数据处理和可视化常量曾写死私有机器数据根，`params.yaml` 也曾包含另一台机器的默认路径。README 的路径覆盖只覆盖部分新 CLI。

影响：在另一台机器上导入相关模块可能立即访问不存在的目录，且 `mkdir` 会在导入时执行；旧版 notebook 和部分数据处理流程无法按 README 独立运行。

建议：将所有数据根目录改为 CLI/config 参数或环境变量；导入模块时不得创建目录；为每个数据集提供最小下载、预处理和验证说明。

### P1-2：论文与项目状态文档的页数记录过时

位置：`paper/ADMA2026_TODO.md:76,160`、`docs/project_memory_zh.md:699-700`。

现象：8 月 1 日遗留构建产物显示英文 15 页、中文 14 页；8 月 17 日当前源稿重新编译为英文 16 页、中文 15 页。但文档仍写“11/10 页”，并将英文 11 页标记为已完成。

影响：投稿合规审计会产生错误结论；英文稿已经达到投稿限制上限，任何新增内容、字体变化或编译环境变化都可能超页。

建议：立即更新状态文档为实际页数，记录编译命令、模板和 warning；在最终投稿前确认 PDF 页数和元数据，而不是沿用旧记录。

### P1-3：主结果的 alignment 不确定性未纳入论文主结论

位置：`paper/main.tex:518-542,585-588,759-766`；artifact 位于 `checkpoints_align/transfer2026_*` 与 `checkpoints_rerank/transfer2026_*`。

现象：正文主结果固定 alignment seed 42，只对 reranker seeds 42/43/44 报告均值和标准差；论文也承认这不是端到端 alignment uncertainty。仓库中已有 seed 43/44 相关 artifact，但主表和主文没有跨 alignment 的分层汇总。

影响：结果稳定性结论的适用范围较窄。当前可以支持“固定基础 alignment 下 reranker 的初始化稳定性”，不能支持完整流水线的随机种子稳定性。

建议：若不补充完整聚合，必须在摘要、结论和图表标题中明确“fixed alignment checkpoint”；若要声称端到端稳定性，按 alignment seed 分层报告 base、pointwise、Transformer 和配对差异。

### P1-4：外部 baseline 未独立复现，跨论文表存在协议不可比风险

位置：`paper/main.tex:555-565,632-640`、`paper/main_cn.tex:450-457,516-520`、`paper/ADMA2026_TODO.md:150`。

现象：JESTR/GLMR 数字来自 GLMR 论文 Table 1，正文已标记 `reported`，但主表仍把它们与本地结果放在同一比较表中；候选池、身份规则、MCES 实现和 solver 可能不同。

影响：读者可能快速将表格理解为公平 benchmark。论文文字已做风险缓释，但表格视觉呈现仍有误读可能。

建议：将外部数字单独放入“cross-paper context”表，或在列标题和 caption 中显著标注“not reproduced / not directly comparable”；删除或弱化“higher/lower than GLMR”的结果叙述，保留为背景观察。

### P2-1：论文主表、受控均值和代表性 seed 的统计口径容易混淆

位置：`paper/main.tex:590-640`、`paper/main.tex:642-715`。

现象：主表 caption 称为 representative seed-42，受控表使用三种子均值；两者数字均有 artifact 支撑，但表格相邻且列名相似。`paper/ADMA2026_TODO.md` 中又同时保留 aggregate 表和历史 checkpoint 表。

影响：读者或后续维护者可能将代表性 seed 与均值混用，尤其是将 68.74 与 68.58 视为矛盾或重复结果。

建议：在主表方法名中加入 `(seed 42)`，在受控表加入 `mean ± sample SD over seeds 42–44`，并在 artifact 索引中为每个数字记录来源文件和行。

### P2-2：开发依赖和版本没有完全锁定

位置：`environment.yml`、`requirements-dev.txt`、`paper/main.tex:638-641`。

现象：环境文件使用未锁定的 `pytorch`、`rdkit`、`torch-geometric`、`matchms`、`myopic_mces` 等依赖；论文也承认没有完全锁定确切版本。

影响：重新训练、图构建、候选身份处理和 MCES 结果可能发生环境相关差异。

建议：提供 `conda list --explicit` 或 lockfile、pip freeze/hash、CUDA 驱动信息，并把运行 commit、params、候选文件和 cache SHA-256 写入每次实验 manifest。

### P2-3：文档中的旧流程与新流程并存

位置：`README.md`、`docs/README_zh.md`、`paper/ADMA2026_TODO.md`、`docs/project_memory_zh.md`。

现象：README 同时介绍早期 spectrum-to-spectrum `.npy` 评估、当前 alignment/candidate 评估和 reranker；部分路径与环境说明仍服务于旧流程。

影响：新用户无法判断哪个流程生成论文主结果，也容易使用不兼容的 checkpoint、candidate file 或指标协议。

建议：在 README 首部明确“论文主结果复现路径”，将 legacy workflow 单列并标明不用于当前论文主表；为每种结果给出 artifact 目录、commit 和预期数值。

## 4. 已确认的正面结果

- `python -m compileall` 对主要脚本、包、测试和分析脚本通过。
- canonical reranker 的 2 个候选协议 × 2 个模型变体 × 3 个 reranker seeds 均有结构化结果；核心消融 summary 有 36 条 seed-level 记录，聚合结果为 10 个单元。
- 论文对训练/验证重叠、exact-SMILES 协议、top-40 coverage 上界、MCES 的 thresholded myopic 语义和固定 alignment 限制均有明确披露。
- 论文结论已经避免将“去除 rank embedding 后变好”误写成原设计贡献，也没有把 self-attention 的小幅增益写成稳定的独立因果贡献。
- 英文和中文稿均有可用 PDF；当前日志没有发现致命 LaTeX 错误或未定义引用。
- Git 历史未发现已跟踪的模型权重、密钥文件或 `.env` 文件；但 artifact/log 脱敏仍需在公开打包前单独执行。

## 5. 修改优先级与验收标准

### 发布/投稿前 P0

1. 修复环境和测试入口，至少获得一次 `python -m pytest -q` 的明确结果；不能通过的测试必须分类说明。
2. 完成 exact-SMILES 与官方身份规则的敏感性评估，或把所有结果严格标注为本地协议结果。
3. 更新页数、匿名性和最终投稿合规记录，确认英文 PDF 不超过限制。

### P1

1. 消除绝对路径并整理主流程 README。
2. 明确 fixed-alignment 结论边界，决定是否纳入跨 alignment 聚合。
3. 将外部 baseline 与本地结果视觉分离，并强化不可比说明。
4. 锁定运行环境版本并生成可审计 manifest。

### P2

1. 统一主表和历史表的统计口径与来源标注。
2. 补充参数量、cache 构建、推理延迟和显存信息；若不补，保留当前 limitation。
3. 清理 legacy 文档和过时项目记忆。

## 6. 审计结论

在不扩大证据边界的前提下，论文最稳健的结论是：**在 MassSpecGym 的本地 exact-SMILES、top-40 候选协议和固定 alignment seed-42 checkpoint 下，监督式第二阶段 reranking 相比基础检索显著改善候选排序；pointwise 已取得大部分提升，Transformer 的额外收益较小且受 checkpoint 与模型容量影响。**

不能从当前仓库直接推出：官方评价协议下的等价指标、严格 SOTA、跨 alignment 的端到端稳定性、self-attention 或具体组件的独立因果贡献，以及与 GLMR/JESTR 的公平速度或准确率优越性。

## 7. 审计后修复记录

### 2026-08-17：P0-1 测试与运行环境

状态：**已修复并验证**。

修改内容：

- 在 `environment.yml` 和 `requirements-dev.txt` 中加入 `pytest>=8,<9`；
- 在 `pyproject.toml` 中固定 `tests/` 为测试目录，并将仓库根目录加入 pytest 的 Python 导入路径；
- 在 `README.md` 中补充 canonical Conda 环境的创建、同步和唯一验收命令；
- 未修改或跳过任何测试逻辑。

验收环境与结果：

- Python `3.12.13`；
- pytest `8.4.2`；
- PyTorch `2.5.1`；
- `conda run -n specembedding python -m pytest --collect-only -q`：收集 70 项，退出码 0；
- `conda run -n specembedding python -m pytest -q`：`70 passed, 5 warnings in 11.02s`，退出码 0；
- warning 为 1 条 `torch_geometric.distributed` 弃用提示和 4 条 PyTorch nested-tensor 提示，没有测试失败。

边界说明：默认 shell 使用未声明的 Python `3.13.9`，且未安装项目运行时依赖 PyTorch；其 `pytest --collect-only -q` 因缺少 `torch` 仍退出 2。加入 pytest 根路径配置后，原有 4 类仓库根模块导入错误已全部消失。项目的发布验收以 `environment.yml` 声明的 Python 3.12 `specembedding` 环境为准，不把不完整的系统 Python 解释为受支持环境。

### 2026-08-17：P0-2 评价身份协议

状态：**已按“严格限定本地协议”方案修复；未新增身份敏感性实验**。

修改内容：

- 新增单一结构化协议定义，固定本地 evaluator 为 exact target-SMILES 单正例，参考
  MassSpecGym loader 为二维 InChIKey 等价且可能多正例，并固定
  `official_evaluator_equivalent: false` 与“影响未量化”；
- canonical freeze、核心消融与跨 alignment 报告/manifest 均写入该协议字段；历史配置读取
  固定到 source commit 的 `params.yaml` blob，不再受当前可移植配置改写影响；
- 双语摘要、关键表格 caption、讨论、局限、结论和 README 均明确使用“本地协议”口径，
  不再留下官方 evaluator 等价或严格 SOTA 的解释空间；
- 相关协议工件提交为 `ae8efe1`、`0e02992`、`48316f2`，论文证据边界提交为
  `1c0dee2`。

边界说明：按用户“不增加更多目标、主要完成待办”的决定，本轮没有新增二维 InChIKey/
多正例评价运行。因此该修复关闭的是协议标注与结论边界问题，而不是量化两种身份规则的
数值差异；此差异继续作为明确限制保留。

### 2026-08-17：P1-1 路径可移植性与导入行为

状态：**已修复并验证**，工程提交为 `f5e6e78`。

修改内容：

- `params.yaml`、源码、shell/batch 脚本和 11 份 legacy notebook 中的私有机器根已改为
  repository-relative 配置或显式环境变量；tracked 文件私有根扫描为零命中；
- 配置优先级固定为显式 `load_config(path)`、`SPECEMBEDDING_CONFIG`、仓库默认
  `params.yaml`，已知相对路径相对于所选 YAML 目录解析；
- `SpecEmbedding.const` 只解析路径，不在导入时创建目录；需要写出的 legacy notebook
  单元在 `np.save` 前显式创建父目录；
- 包导入仅为 Numba/Matplotlib 设置可覆盖的可写 cache 路径默认值，不创建目录，也不全局
  禁用 Numba JIT；可执行入口只在进入 `main()` 后显式准备 cache 目录；
- `run_pipeline.py` 使用仓库根定位 Git、子脚本和 subprocess 工作目录，data/cache/
  tokenset/pretrained/candidate 参数均完整透传；`unique_seed_train.py` 的数据加载和训练置于
  main guard 内，且每个 seed 的 mean/std 恢复在 seed 循环内写出；
- 新增配置相对路径、任意 cwd、导入写入、运行时 cache、CLI 透传、notebook JSON/写前建目录、
  私有路径扫描和 seed 写出作用域测试。

### 2026-08-17：P1-2 至 P1-4 论文与投稿记录

状态：**已修复并验证**，论文提交为 `1c0dee2`。

- 当前源稿重新编译为英文 17 页、中文 16 页；8 月 1 日旧 build PDF 15/14 页仅作历史记录，
  11/10 页旧状态已删除。按用户决定，页数压缩延后到完稿后，不把当前页数误记为最终合规；
- 新增 alignment seeds 42/43/44 分层表。每个 alignment 内先聚合 reranker seeds 42--44，
  不把九次运行展平：两个监督模型相对对应 base 在 12/12 个 alignment--candidate--model
  聚合单元的 Recall@1/MRR 均改善；Transformer 相对 Pointwise 在两类候选和两个指标上
  均只有 2/3 alignment 为正；
- 摘要与结论区分 canonical alignment seed 42、固定 alignment 内 reranker 初始化统计和
  三个 alignment 的描述性敏感性；不声称一般端到端稳定性、显著性或组件因果性；
- JESTR/GLMR 移到独立 reported-only cross-paper context 表，显著标注未独立复现、不可
  直接比较；删除直接 higher/lower 和方法归因叙述。

### 2026-08-17：P2 环境锁与统计口径

状态：**已完成最小可审计闭环**。

- `environment.yml` 与 `requirements-dev.txt` 固定直接依赖版本；新增带每包 MD5 URL 的
  Linux x86-64 Conda 显式锁、完整 PyPI 层版本锁和可重复执行的锁验证脚本；
- `reproducibility/runtime-snapshot.yaml` 记录 source commit、Linux/驱动/GPU/CUDA/cuDNN、
  直接包版本、锁文件 SHA-256 和实际验证结果，不含用户名、hostname、GPU UUID 或本机绝对
  路径；
- 从显式 Conda 锁创建全新环境并安装 PyPI 锁后，完整测试为
  `86 passed, 5 warnings in 26.92s`，Ruff 全部通过；canonical `specembedding` 环境完整测试
  为 `86 passed, 5 warnings in 33.48s`；
- 代表性 seed 42、固定 alignment 的三 reranker-seed 均值和三个 alignment-level 描述性
  估计已在表标题、caption、正文和项目状态文档中分开。

环境边界：该快照是 2026-08-17 对当前源码的事后验收复现环境，不证明历史训练环境逐包
相同，也不证明重新训练可 bitwise 重现。PyPI 锁固定版本但未记录 wheel hash；Conda 显式锁
记录逐包 MD5。

### 2026-08-17：最终本地验收摘要

- `conda run -n specembedding python -m pytest -q`：`86 passed, 5 warnings`；
- 显式锁全新环境：`86 passed, 5 warnings`；Ruff 通过；
- `compileall`、shell 语法、`git diff --check`：通过；tracked 私有路径：零命中；
- 英文/中文 LaTeX：17/16 页，无 fatal error、undefined citation/reference 或
  Overfull/Underfull；仅保留已知 `amsmath` 与 Fandol `fontspec` warning；
- 未新增引用、训练、评价、效率数字、显著性检验或外部 baseline 复现。

修复后仍保留的研究限制：身份规则影响未量化；三个 alignment 只构成内部描述性证据；
组件审计仍只覆盖 canonical alignment seed 42；JESTR/GLMR 未独立复现且不可直接比较。
这些限制已在论文关键独立阅读位置自包含披露。当前状态可提交原审计会话进行再次审计。

### 2026-08-17：第二轮独立审计续修

状态：**两项 P1 已完成本地修复，等待第三轮独立复审**。

- `run_rerank_pipeline.py` 现以脚本位置确定 repository root；Git 查询和所有 subprocess
  固定在仓库根运行，三个子脚本使用绝对路径，用户输入的相对 data/candidate/output 路径
  按仓库根解析。新增测试从临时 cwd 执行完整 `--dry-run --mode all`，同时核验默认 run name、
  commit、子脚本路径和 subprocess cwd；工程与论文局限修订提交为 `e597d89`。
- 双语 limitation 现明确区分：2026-08-17 事后验收环境已经按精确版本锁重建验证，但历史
  训练没有逐包环境记录，故不声称 bitwise retraining equivalence。PyPI 锁仍仅固定版本、
  不含 wheel hash，这一供应链边界继续在 runtime snapshot 中披露。
- 使用 `paper/build_release.sh` 从干净的 tracked source commit
  `e597d89f45eb2f56bc8ad6332c49016a62496ad7` 强制执行完整 BibTeX/LaTeX 构建。英文 PDF 为
  17 页、476068 bytes、SHA-256 `62d09761172b8b04e7facc5e8090f7cfc4e868d29f85e18726b9bae0260b5034`；
  中文 PDF 为 16 页、421756 bytes、SHA-256
  `f985499fbdcd06723aa6b0517874317c09145bd0b8d42b6bd0a79a2c1c6d3bcb`。
- 两份 PDF 与 `paper/release/build-manifest.yaml` 受 Git 跟踪；manifest 固定源码 commit、
  Ubuntu/TeX 工具链、页数、字节数、哈希和检查结果。构建日志无 fatal/LaTeX error、undefined
  citation/reference 或 Overfull/Underfull；英文作者元数据为空，中文作者字段缺失，字符串扫描
  无私有路径命中。忽略的 `paper/build` 已同步，避免旧 15/14 页 PDF 被误认成当前稿。
- 当前 canonical 环境完整测试为 `88 passed, 5 warnings`；Ruff、compileall、shell 语法和
  `git diff --check` 均通过。本轮未新增训练、评价、引用、性能数字或研究结论。

页数压缩仍按用户决定留到完稿后处理，17 页英文稿不被误记为已经满足最终页数限制。

### 2026-08-17：第三轮独立审计结论与分级更正

审计基线：远端分支 `codex/transfer-2026-0831`，HEAD `5a1fe38`。审计未保留任何工作区变更。

第三轮初始判定为“有条件通过”，条件仅来自把英文 17 页列为投稿合规 P1；代码、复现、
证据边界、匿名性和构建有效性当时均已通过。用户随后明确页数可以延后且不阻塞审计，原审计
会话据此将该项降级为独立非阻断待办。**更正后的本轮判定为通过，无 P0/P1 阻断项**。
第二轮提出的任意 cwd rerank 总管线和当前双语 PDF 可审计留存均已独立核验关闭。

独立执行结果：

- 从临时 cwd 执行完整 `run_rerank_pipeline.py --mode all --dry-run` 返回 0；绝对子脚本、
  repository-root 输出与相对 data/candidate 路径均正确，专项测试 `2 passed`；
- `conda run -n specembedding python -m pytest -q tests`：`88 passed, 5 warnings in 32.74s`；
- Ruff、compileall、`bash -n`、`git diff --check`：通过；
- `freeze_adma2026_artifacts.py --check`：60 个 canonical artifacts 验证通过并匹配 manifest；
- tracked 双语 PDF 的页数、字节数、SHA-256、作者元数据与 build manifest 一致；文本扫描无
  `/data1`、`/home/` 或 `git@` 私有路径；重建日志无 fatal/LaTeX error、undefined
  citation/reference 或 Overfull/Underfull。

独立非阻断投稿整理事项：英文稿为 17 页；若转投 venue 仍适用 15 页总限制，则最终投稿前
需要压缩。目前不能称为“最终可投稿稿”。该事项已诚实保留为开放 TODO，并按用户明确决定
在完稿后处理，不作 P0/P1，也不阻断本轮计划归档。

继续保留但不构成本轮阻断的限制：PyPI 锁没有 wheel hashes；local exact-SMILES 与官方
多正例协议差异未量化；cross-alignment 仅有三个描述性单位；组件未跨 alignment；
Transformer 容量与候选交互混杂；JESTR/GLMR 未独立复现；无统一效率测量；top-40 闭集
coverage 限制仍在。
