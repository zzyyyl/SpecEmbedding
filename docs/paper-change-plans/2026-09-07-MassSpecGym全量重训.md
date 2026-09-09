# 阶段计划：MassSpecGym 分子检索 Top-k 优化与 SOTA 对标

- 状态：`执行中`
- 创建日期：2026-09-07
- 最后更新：2026-09-09
- 负责人：Codex / 作者核验
- 关联论文：`paper/main.tex` / `paper/main_cn.tex`；目前不改论文结果或结论
- 计划约束：本文件沿用仓库模板与原全量重训计划，作为本工作的唯一计划；不另建冲突版本。
  2026-09-09 用户授权单 baseline、互联网研究和结构/参数迭代，随后把目标更新为
  **提高分子检索 Top-k，MRR 为辅；各项评价指标在 MassSpecGym 上达到或接近现有 SOTA。**
  此授权替代原“固定 alignment、不重训”和“立即完成 12 组矩阵”安排；历史执行记录保留于第 10 节。
  允许参考论文下载至用户指定论文目录；SpecEmbedding 质谱侧预训练模型的结构不得修改。

## 1. 阶段目标与验收

### 1.1 目标定义

训练可在本机运行的 MS/MS-to-molecule retrieval 模型。主目标是官方结构划分上的
MassSpecGym v1.5 **Mass retrieval Top-1/5/10/20 整体提高，MRR 为辅助指标**。
质量候选主任务不使用真值分子式；Formula-conditioned retrieval 后续单独验证，不能替代 Mass 达标。
可以借鉴或超过复杂方法的成绩，但实现优先选择轻量、可训练、可复核的方案。

“改善现有 baseline”是进展，“达到或接近 SOTA”才是本阶段最终目标；二者不能混写。
也不能通过只与较弱双塔方法比较、改用容易的 split、过滤困难 query 或只报告最好的 k 来达标。

### 1.2 “达到或接近”的操作定义

先固定同协议参考集合及核验日期，对每个指标 j 取可比方法的最好值 S_j，保存具体方法、
checkpoint/版本、来源表格、分母和评价设置。各项的最好值可以来自不同方法，必须逐项署明来源。

以下为开始优化前设定的工程验收阈值，不是统计等价性定义：

- **达到**：本地最终审计结果的每项 Top-k 均不低于对应 S_k；MRR 同样单列检查。
- **接近**：每个 Top-k 的差距同时满足 S_k − R_k ≤ 2 个百分点、R_k ≥ 0.95 × S_k；
  MRR 以 0–1 原值表示，差距同时满足 S_MRR − MRR ≤ 0.01、MRR ≥ 0.95 × S_MRR。
  超过参考值自然满足；不能把百分数和原值混用。
- 先用 seed42 寻找可行方案。最终冻结的一个方案再做必要 seed 稳定性复核，报告均值、样本 SD
  和每个 seed；不提前展开模型 × 候选协议 × seed 笛卡尔积。单次最好成绩不充当最终稳定性证据。
- 表格缺少 Top-10 或 MRR 时，记为“未报告 / 待同协议复核”，不能用 Top-5/20 插值或补零。
  通过公开预测或经过审计的同协议复现补齐；未补齐则这些验收项保持未完成。
- 既有“本轮不计算 MCES”边界保留于优化循环；最终若要声称覆盖官方全部指标，必须先补齐
  MCES 等官方指标的实现、资源评估及比较记录，否则结论明确限定为 Top-k/MRR。
- “当前 SOTA”需在最终验收前刷新一次公开文献和结果版本，保留初始与最终快照。
  不能事后放宽阈值、排除更强且可比的方法来宣布完成；未达到就明确记录差距并继续调整。

## 2. 当前证据与瓶颈

- v1.5 r4 已完成一个全量 alignment-42、六个缓存，以及 Mass relative 三 seed 和
  pointwise seed42 的完整训练/测试。用户要求缩小实验范围后，已主动中断 pointwise seed43
  与后续矩阵，保留原始工件；部分 checkpoint 不作为完成结果。
- 现有新结果仍为 cache-local exact-target-SMILES；数字、权重来源、GPU 重编码证据和新旧
  差异只维护于[准确率排查](../../analysis/massspecgym_v15_regression_audit.md)及其原始工件。
- Base 排序在进入 reranker 前已经较弱。新缓存并未复用旧 alignment 权重或旧磁盘 TokenSet。
  旧候选格式捷径已量化，但尚不能把旧模型的全部性能差额归因于单一因素。
- r4 alignment 按每轮全部谱图训练，以验证对比损失选择 checkpoint；reranker 目前按验证 MRR
  选择。A01 完成后未替换 r4；A02 完整训练与审计后整体改善，已成为下一轮比较基线。
  后续按完整验证轨迹判断结构、
  loss、预训练或参数变化的价值，不能只观察训练 loss。
- 原模型、旧 query cap/top-40/forcing 结果只保留追溯，不能当成 v1.5 的可靠目标线。
  reranker 无法救回其候选池外的真值，必须同时记录 Base、coarse 与最终排序的 coverage。

## 3. 修改范围与固定边界

### 3.1 用户授权的可调部分

| 部分 | 允许探索 | 每轮需要固定/记录 |
| --- | --- | --- |
| SpecEmbedding 质谱预训练 | 使用或不用既有权重；冻结、分阶段解冻及小学习率迁移 | 预训练架构不改；权重 SHA、训练数据来源、验证/测试身份污染审计 |
| 下游质谱塔 | 小型 Transformer/DeepSets、池化、m/z 与中性丢失表示、可用仪器元数据分支 | 与预训练模型分开定义和版本化；不能改共享类而隐式改变预训练架构 |
| 分子塔 | 现有 GINE 的宽度/层数/池化、小型图模型、Morgan 指纹 + MLP 或轻量融合 | 规范构图、二维身份一致性、参数量和编码耗时 |
| 对齐训练 | 温度、学习率、权重衰减、batch、正则化、指纹辅助监督、自然候选对比学习 | 全部 train query；正负标签正确，训练候选不引入验证/测试监督 |
| rerank | 固定 baseline 后逐项改 residual、候选交互、loss、coarse 策略或融合 | 只重排自然输入候选，不用来源位置/rank 捷径，不 forcing，不重复跑有无 pointwise 矩阵 |
| 工程性能 | 数据加载、图/TokenSet/冻结 embedding 缓存、分块、经过验证的混合精度 | 缓存必须含源码/数据/模型指纹；模型或增强变化不能静默复用过期表征 |

### 3.2 不变的实验条件

- 默认 seed42；起点为 Mass + relative，优化胜者成为下一轮 incumbent，原始 baseline 永久保留。
  模型名可随被验证的 rerank 改进更新，但一次只跑一个候选方案，不枚举 pointwise 开关。
- v1.5 官方 train/val/test 划分，原始数量 194,119 / 19,429 / 17,556；
  固定验证排除 [7686,7687,7688,8464,8465,8466]，有效验证数 19,423。
  不更改划分为 random、Formula split 或其它更容易的拆分。
- 正式训练每轮覆盖全部符合监督协议的 query；无正例单列统计，完整评价保留它们。
  不人为限制训练条数，不用前 N 条成绩代替正式结果。
- 候选使用版本化的官方自然候选，当前最多 256；所有 split 无 forcing。
  无效图、身份重复、同分排序和二维多正例必须显式审计，不静默改变官方分母或候选集合。
- 正式训练/编码/推理显式 cuda:N，忙则等待；默认在物理 GPU 0/1 中择一，通过 UUID 映射 cuda:0。
  不进行两组并行抢卡，不终止其他任务。CPU 仅处理、审计及 synthetic smoke。
- 不修改固定运行 worktree，不覆盖旧 checkpoint/cache/log/status/锁；每次正式实验独立目录。
- 本阶段不顺带启动 NPLIB1 或 GLACIER 独立复现队列；论文与代码可作为研究资料。
  外部比较如果需要实际运行，先纳入本计划的具体对标条目并检查已有队列，不重复派发。
- 禁止利用测试真值选择结构/参数，或把真值分子式、候选原始位置、SMILES 格式特征当作捷径。
  预训练来源不清或存在污染的权重不能进入“同协议达标”验收，可保留隔离诊断记录。

## 4. 迭代方案

### 4.1 第一优先：评价与选优对齐

1. 在不更改模型结构和训练 loss 的前提下，建立完整验证集的自然候选检索，采用固定的官方
   二维身份、候选顺序和同分规则；同时保存 Top-1/5/10/20、MRR、coverage 和每 query 排名。
2. 用现有 seed42 工件建立固定 baseline 记录。保存 checkpoint SHA、配置、验证排名、
   train/val 数量和硬件开销。缓存审计与重编码不写成新训练。
3. 增加按实际验证检索指标保存 checkpoint 和训练轨迹的能力。起初以 Top-1、MRR 作选优顺序，
   同时保留其它 Top-k 的 Pareto 候选，避免提前丢弃整体更好的 checkpoint。
   所有 checkpoint 选择只用验证集；测试命令在优化专用入口中默认不执行。
4. 以单 seed 的完整训练验证选优实现。此时不同时修改 batch、两塔和损失，便于判断
   “选错 checkpoint”是否是瓶颈；对比损失下降但检索不升不能算改善。

### 4.2 依证据选择的优化顺序

| 优先级 | 具体尝试 | 要回答的问题与资源控制 |
| --- | --- | --- |
| A | 学习率/温度/训练时长与早停；逐个改变 | 是否欠训练、过拟合或对比目标尺度不合适；沿用当前架构 |
| B | 同架构 SpecEmbedding 预训练权重 + 小学习率迁移 | 在权重来源审计通过后，是否改善未见结构；不改预训练模型结构 |
| C | 自然候选对比目标、难负例；Morgan 指纹辅助头 | in-batch 区分能否转化为候选内区分；分块计算，不减少 query 覆盖 |
| D | 轻量质谱表示和分子表示改进 | 比较一个明确瓶颈，如谱峰池化、精细结构的图扰动、可用元数据、GINE 池化或指纹融合 |
| E | 一种轻量 rerank 改进或校准融合 | Base 改善后是否仍有可学习的候选内残差；逐项改变，保留完整候选上界 |
| F | 无收益后的方向调整 | 连续三个完整候选方案无有效改善则总结错误分布、重新研究并调整假设，不机械扩大网络或 seed |

表中是先后优先级，不是要一次派发的网格，也不保证每项必须训练。
优先试成本低且有证据支持的单项，只有当前结果支持时才组合两个已经独立有效的改动。
大规模外部预训练、生成式大模型和从头复制复杂 forward simulator 不是默认路线。

### 4.3 单次尝试的完整闭环

- 当次训练运行期间提前完成下一候选的实现、测试、CPU输入核验、实验卡和独立源码准备；
  当次完成审计后，按既定晋升规则确定唯一parent，回填最终配置并重新确认输入指纹后派发。
  这是用户2026-09-09明确要求的衔接方式；准备与训练可重叠，两个模型训练仍串行，
  不把未审计的中途成绩绑定为下一轮基线，也不绕过GPU门槛或单次派发保护。
- 训练前写实验卡：experiment_id、parent baseline、一个主要假设、唯一主要改动、固定参数、
  预期资源、评价协议、早停规则、输入/权重/源码指纹和排队命令。
- 首轮完整 epoch 测量参数量、峰值显存、CPU 内存、每秒 query、训练/验证/编码耗时；
  估算剩余时间并记录。以单张 24 GiB RTX 4090 可容纳为默认，优先降低模型复杂度、
  分块或 checkpointing。不得把梯度累积误称为增大了 in-batch 对比负例池。
  记录必须写明测量范围：当前实现的 CPU 峰值仅覆盖主进程生命周期，CUDA 峰值仅覆盖
  当前进程/设备的 PyTorch 分配器；未测量的 worker 或整机峰值不能推断补齐。
- 采用完整训练集；GPU OOM/NaN/丢样/审计失败则保留日志，诊断后创建新版本和新 run。
  不能覆盖旧目录、去掉保护或用无界自动重试掩盖问题。数值与加载等已授权范围内修复可自主推进。
- 根据全部验证 Top-k 的 Pareto 改善决定是否保留，不把小于约定容差的随机抖动称为提升。
  默认容差为 Top-k 0.2 个百分点、MRR 0.002；至少一项 Top-k 实际改善且其余不显著退步。
  存在 trade-off 时保留两者，按验证目标差距和成本选择下一条路线，不依赖测试决策。
- 为覆盖“各项接近 SOTA”，最终候选需逐项过第 1 节门槛；平均 Top-k 或单一加权分数不能代替。
  MRR 用于辅助取舍，不能补偿某个 Top-k 没有达标。
- 每轮把成功、失败、无收益、资源消耗和下一步理由写入同一实验索引；维护一个 incumbent，
  不反复重跑原始 baseline，不把排队/派发/半轮运行记作完成。
- 最终候选冻结前，在完整验证集记录谱图置换/恒定输入敏感性，区分谱图信息与候选分布先验。
  保持候选、分母与身份规则不变；这是候选捷径审计，不用测试样本选择控制设置或模型。

## 5. 文献与 SOTA 参考核验

核验日期：2026-09-09。以下是**公开报告的参考线索，不是本仓库统一复现后的排序**。
官方在线榜页面本次无法读取，已改查官方仓库 CSV；CSV 记录较旧，不能代表最新全领域 SOTA。

| 来源与版本 | 已核验的内容 | 如何使用 / 尚缺什么 |
| --- | --- | --- |
| [官方数据卡](https://huggingface.co/datasets/roman-bushuiev/MassSpecGym/blob/main/README.md) | v1.5 统一 SMILES 表示，提供版本化候选 | 固定输入及身份规则，不能继承 v1 缓存审计 |
| [官方 Mass 结果 CSV](https://raw.githubusercontent.com/pluskal-lab/MassSpecGym/main/results/retrieval.csv) | JESTR Top-1/5/20 = 15.13/36.75/60.32% | 历史参考，缺 Top-10/MRR、未标 v1.5，不作为最终 SOTA 线 |
| [MassSpecGym in the Wild，2606.19624v1](https://arxiv.org/html/2606.19624v1) | Table 2 的 MIST + 数据安全 MIST-CF：Mass Top-1/5/20 = 26.64/38.47/54.90%；Table 3 是 Formula bonus | 说明现代强对标及协议修正；不是所有方法的完整 Mass 榜，不能把 bonus 表混入主任务 |
| [MSAlign，2605.19752v1](https://arxiv.org/html/2605.19752v1) | Table 3：原 MCES split Mass Top-1/5/20 = 16.2/35.6/59.9%；53.8% 来自另设 Formula split | 候选对比与轻量投影的设计线索；v1.5、外部预训练及统一评价仍需核验 |
| [JESTR，2411.14464v1](https://arxiv.org/abs/2411.14464v1) | 双塔及候选正则化路线 | 保存预印本版本；正式发表元数据如入稿另核验，不能混用版本数字 |
| [GLMR，2511.06259v1](https://arxiv.org/html/2511.06259v1) | Table 1 报 Mass Top-1 64.172%、MRR 67.817% | reported-only；旧版表示、数据与评估实现审计后才判断是否进入 S_j，不把高分先当有效或先排除 |
| [GLACIER，2606.29161v1](https://arxiv.org/html/2606.29161v1) | Table 1：Mass w/ CF Top-1/5/20 = 69.95/86.47/93.52%；w/o CF Top-5 = 86.58% | 强参考；完整 query 覆盖、候选标准化和评价条件待对齐。不能因其方法复杂而从比较集合中排除 |
| [SpecBridge，2601.17204v3 撤回记录](https://arxiv.org/abs/2601.17204v3) | 作者于 2026-03-03 撤回，说明预处理/评价流程存在问题；v2 为历史全文 | 不将撤回成绩纳入有效 S_j；可研究其冻结分子表征的思路，但不能引用旧高分证明方法有效 |

**当前不得声称已找到并冻结所有指标的 SOTA。** 特别是 Top-10、MRR 缺少完整同协议对比，
GLACIER 附录不同配置的 Top-10 不能无说明拼到主表 checkpoint。

2026-09-09 补充查询分母核验：官方 `RetrievalSimulationDataset` 继承 simulation 筛选，
仅使用 `simulation_challenge` 行，不能与完整 `RetrievalDataset` 混用；见
[官方数据加载器](https://github.com/pluskal-lab/MassSpecGym/blob/main/massspecgym/data/datasets.py)。
对已校验 v1.5 TSV 重新计数，完整 test 为 17,556，simulation test 为 9,954。
下载的 GLACIER **Formula** 候选表覆盖 17,147 个官方 test ID（缺 409 个），包含全部 simulation
test ID；因此它也不是单纯的 simulation 子集。缺失原因和对论文指标的影响尚未确定，
不能据此断言 GLACIER Table 1 使用哪一个分母，更不能把 Formula 包的覆盖继承给缺失的 Mass 包。
源哈希、逐 query 差集、固定源码快照哈希与可重跑脚本保存在
`/data1/zyl/SpecEmbedding/audits/sota_protocol_20260909/receipt.json` 及相邻 `audit.py`。
此项仅为比较条件审计，没有运行 GLACIER 推理、修改外部复现任务或改变 A01 完整协议。
进一步只读核验了已固定的上游 `ed8311f`（不把独立复现分支的本地修改视为论文实现）：

- [GLACIER 调用脚本](https://github.com/coleygroup/ms-pred/blob/ed8311f22958cb37f055b663b5f56c5c77a2ee33/run_scripts/glacier/03_run_retrieval.py)
  的 MassSpecGym 项仍被注释，使用的 `msg` 路径/配置没有给出绑定论文训练的输入哈希；
  README 中的 v1.5 数据处理说明不能单独证明该 checkpoint 已使用 v1.5。
- [通用评价器](https://github.com/coleygroup/ms-pred/blob/ed8311f22958cb37f055b663b5f56c5c77a2ee33/src/ms_pred/retrieval/retrieval_benchmark.py)
  使用 NumPy 稳定排序，并会跳过部分缺预测、空真谱、无正例或真值映射失败的 query；
  其分母和同分规则不能直接继承为本计划的完整评价。
- README 另建议使用 TorchMetrics 入口；该固定版本的
  [对应文件](https://github.com/coleygroup/ms-pred/blob/ed8311f22958cb37f055b663b5f56c5c77a2ee33/src/ms_pred/retrieval/retrieval_benchmark_torchmetrics.py)
  在第 139 行有未闭合字典表达式，独立 `ast.parse` 已证实不能原样执行。没有导入、运行或修复外部文件。

上述发现描述可获得的固定源码，不证明论文实际用了哪个 revision、分母或排序，因而不据此
否定其报告成绩。源文件、哈希与语法核验保存在上述 `sota_protocol_20260909` 审计根的
`glacier_source_protocol/receipt.json` 和 `torchmetrics_addendum.json`；后续仍需逐 query 预测/覆盖
及输入指纹，才能完成同协议比较。官方审计论文中的匿名架构也不直接对应为某篇论文的审计结论。
以约 70% Top-1 的公开报告作为需要核验的强参考，不把 15%–16% 自动定为达标线。
正式参考向量 S_j 只在逐项核对数据版本、完整分母、候选处理、身份规则、外部数据和模型版本后冻结；
不排除生成式/forward 方法，只因协议不匹配而单列。无法核验的高分保留为 reported-only 并说明原因。

参考 PDF 已下载至 `/data1/zyl/papers/`：审计论文、MSAlign、JESTR、GLMR、GLACIER 共五篇。
清单为该目录的 `specembedding_retrieval_20260909_manifest.json` 和
`specembedding_retrieval_20260909_additional_manifest.json`；保存 arXiv 版本、下载 URL、SHA-256 和时间，
不将 PDF 纳入仓库或匿名补充材料。下载论文不是完成代码复现，也不自动启动 GLACIER 独立任务。
补充保存 SpecBridge v2 历史 PDF（文件名标注 `WITHDRAWN_historical`）及 v3 官方撤回页面，
清单为 `specembedding_retrieval_20260909_specbridge_manifest.json`；PDF 为 14 页，文件哈希已核验。
撤回是排除其旧成绩的明确一手依据，不外推为其它方法存在同样问题，也不改变尚待核验的强参考。
如采用文献方法，先核对原文与官方实现；引入论文时再逐条核验作者、出版物和 DOI。

## 6. 实验与计算边界

- 新训练及评价：用户已授权范围内的单方案持续优化。每次新结构/参数不用重复确认，但须有实验卡。
  预训练 SpecEmbedding 结构保持不变，所有正式训练全量、GPU-only。
- 新 CPU 处理/审计：允许，完整身份、候选、权重污染、缓存来源与数据覆盖核对。
  验证集足以决策的改动不做测试集 sweep；已看过的历史测试结果不能伪装成从未接触的盲测。
- 全测试与稳定性：在验证选择冻结后执行，再为最终一个方案复核 seeds；先后顺序不改成矩阵搜索。
  若最终测试不足目标，公开记录这次测试及不足，只依据验证诊断选下一改动，不拿测试错误样本调模型。
- 模型复杂度：默认一个实验占一张卡，记录真实可训练参数、峰值内存和端到端训练/评价时间；
  优先本机可行的小模型，不预设加大网络一定更好，不把 GPU 低利用率等同于无人使用。
- 论文：目前只规划，不改写中英文结果或贡献。达到目标且审计完成后，重新检查实际方法能支持哪些
  主张，按仓库计划、双语构建、匿名化和提交规则处理，不提前宣称 SOTA、显著性或一般化能力。

## 7. 分步执行清单

- [x] 核验原始工件、排查旧缓存/权重与候选表示偏差；记录已确认与未确认的原因。
- [x] 根据用户单 baseline 要求停止旧矩阵，保留 4 组完成结果与中断工件。
- [x] 收窄正式默认配置及入口为 Mass/relative/seed42，拒绝意外展开矩阵。
- [x] 初步检索一手来源，建立本计划和本地 PDF 资料库。
- [ ] 补齐各指标 SOTA 来源/协议/分母对照，冻结版本化参考向量和缺失项。
- [ ] 完成完整验证集官方候选/二维身份 evaluator；从现有 seed42 工件冻结 baseline 验证报告。
- [x] 实现检索指标选 checkpoint、Pareto 记录及优化分支不跑测试；测试覆盖全量/设备保护，正式运行待核验。
- [ ] 按第 4 节顺序启动一个实验，持续完成“训练—验证—审计—保留/舍弃—下一假设”闭环。
- [ ] 核验预训练来源，保证其结构保持不变，再决定是否启用权重。
- [ ] 验证指标接近目标后冻结一个方案，在完整测试集评价并完成新缓存/身份审计。
- [ ] 对最终方案进行必要 seed 稳定性复核；逐项判断 Top-k/MRR 与 S_j 的差距，不只报告最好 seed。
- [ ] 更新论文前复核目标是否达成与主张边界，同步双语稿、核对引用、构建发布工件。
- [ ] 提交论文实质修改，再独立回填计划；全部必要工作完成后才设为“已执行”。

## 8. 风险、停止与调整条件

- SOTA 是目标，不是已证实结果或保证；本机训练约束下未达标时保留差距，不降比较标准。
- 连续三个无收益的完整尝试触发方法诊断，不触发擅自限量、矩阵扩张或停止全部研究。
- 发现 NaN、输入变化、标签丢失、设备错误或缓存不匹配，停止对应 run 后修复；资源忙则后台等待。
- 已授权模型/参数搜索范围内可以自主调整；若需修改预训练架构、官方划分、数据泄漏约束，
  或改变论文结论/评价定义，暂停受影响部分并标为“已偏离待确认”，其余独立工作继续。
- 多 seed 在方案成熟后进行；不能把单 alignment/单 reranker 的成功写成整个矩阵或多数据集成功。
- 官方全部指标、完整同协议 SOTA 比较或最终稳定性任一未完成，阶段不得因预算、运行次数、
  工程测试通过或某一项局部改善而标为完成。

## 9. 验证方案

- [x] 本轮检索验证、alignment 与 GPU/数据审计专项：53 通过。
- [x] 全仓测试：239 通过、1 跳过、1 个既有内部路径审计失败。
  当前失败涉及 GLACIER manifest、GLACIER 交接、项目记忆和已提交的 v1.5 准确率审计文档；
  与本轮范围修改无关，不改写这些原始来源以隐藏失败。
- [x] Ruff、compileall、git diff --check；后续新增代码必须重新执行相应验证。
- [ ] evaluator 测试：二维多正例、候选 permutation、同分、无正例、无效图、完整分母；
  用小型 synthetic 例与独立官方实现核对，CPU 测试不作为正式实验。
- [ ] checkpoint 选优与恢复测试：保存指标/配置/权重一致，完整 train/val 数量与实际每轮一致。
- [ ] 预训练结构与来源：新旧结构签名、严格 state_dict 加载、tokenizer 语义、身份污染审计。
- [ ] 正式运行：独立干净源码、配置/数据/SHA 固定、CUDA UUID、GPU 门槛、阶段完成标志与进程一致。
- [ ] 最终结果：完整 Top-k/MRR、可比 S_j、每项差距、单 seed/均值/SD、完整审计，不遗漏失败项。
- [ ] 论文修改后执行双语编译、引用/匿名扫描、发布 manifest 和仓库要求的复现检查。

## 10. 执行记录

- 2026-09-09 21:13：D08三组完整编码与独立分数核验于21:03:50完成，入口退出、pane exit0，
  回执为`/data1/zyl/SpecEmbedding/audits/validation_batch_benchmark_20260909/receipt.json`，SHA-256
  `7bb8fc7e07f7a7788b42ee89ca90ee0f4395a5cb76ebd81f13b80afe0ef14d8c`。完整结果及限制只维护
  于优化索引D08；按事前采用条件选择1024，未改变训练batch128、学习率、分母或候选。
  A04最终配置、完整预检、固化决策与派发脚本保存于
  `/data1/zyl/SpecEmbedding/audits/optimization_a04_launch_20260909/`；`preflight.json` SHA-256
  `b8039f53dfd46bc5d991afb782e9e3e2b23c1391fd7d61c8a45b36c29af222fc`，`final_decision.json` SHA-256
  `79e7d8a7877e5389b44fe04982944601a03e0857e5d88551590b07e22f00c3f2`，`prepared.json` SHA-256
  `f0ba9bfad3b6f510befa7b93d805f0cb80d54b869b823cf966f4553c07d1733d`；实际runtime配置与此前
  `82c20dd`全量预检相比，仅验证分子batch由512改为1024，配置来源及指纹重新固定。
  A04于21:08:20单次派发，固定源码`82c20dd8b79fe88e91c14455202ad484d144db91`仍干净，
  运行根为`/data1/zyl/SpecEmbedding/experiments/massspecgym_v15_opt_a04_20260909_topk256/`，
  socket为`/tmp/specembedding-opt-a04-20260909-1010/tmux.sock`，session `opt`、pane `%0`，
  入口PID3360929。`launch_receipt.json` SHA-256为
  `133a88f43fbd8d404984b0608d78dd3f5d8839c5a67375f6a769c6b68b1c66b6`，原脚本及全部阶段命令
  已保存在run中；单次派发锁与已存在工件保护保留，没有旧任务重试。
  21:09:13完成完整数据导入，21:09:37完成验证索引导入，源/目标SHA、19,423条验证query、
  827,600个分子及6个图排除一致。训练候选元数据覆盖194,119条query、自然正例全部存在，
  固定输入回执和运行配置与最终预检一致；实际逐轮训练数仍须待训练产生后核验。
  21:10实测入口存活并等待GPU0/1；21:11:58通过持续空闲与最终复查，选中物理GPU0映射
  严格cuda:0开始baseline完整验证，21:13日志确认全量分子编码推进。训练阶段随后独立等待。
  此时尚未开始模型训练，没有A04 checkpoint或收益结论；按用户要求为A01/A03账本条目
  添加删除线，原始结果与未晋升原因保留。
- 2026-09-09 20:49：A03于20:40:48全部阶段完成，第27轮正常早停；原入口/训练进程退出，
  `opt/%0`为dead、exit0。使用原固定`c50ddf0`源码的独立CPU完成审计已通过，覆盖全部27轮
  分数/排名、实际query计数、最终选优/Pareto权重与资源记录。回执为
  `/data1/zyl/SpecEmbedding/audits/optimization_a03_20260909/receipt.json`，SHA-256
  `64163c203bc6ebf2796e4126c6b6fb313c145532db800e8187f9d0207fcc6479`；审计独立socket为
  `/tmp/specembedding-opt-a03-audit-20260909-1010/tmux.sock`，session `audit`、pane `%0`正常退出。
  选中epoch22及全部最终Pareto候选21/22/23均有实质退步，保留A02；指标只维护于优化索引A03。
  唯一决策保存于同目录`decision.json`，SHA-256
  `e76004e317c889cad20a1888eb901309e561fd48ba3f9a3e3406afee8a63af51`；A04沿用A02图增强开启，
  模型改动仅增加自然候选监督，最终验证batch仍待工程测量结果。
  已准备的D08测量于20:45单次派发，独立socket为
  `/tmp/specembedding-validation-batch-20260909-1010/tmux.sock`，session `benchmark`、pane `%0`，
  入口PID3339565；原manifest未修改，新增`launcher_receipt.json`、`dispatch.json`记录实际派发。
  20:47:11首次通过原持续空闲门槛，选中物理GPU0映射严格cuda:0，首组512完整编码正在运行；
  尚无三组测量完成回执或提速结论，不启动重复队列。首次操作的自动审批服务断连导致拒绝，
  只读确认没有任何派发工件后重新提交相同已授权操作，随后才发生上述首次实际派发。
- 2026-09-09 20:33：CPU验证索引复用实现`82c20dd8b79fe88e91c14455202ad484d144db91`已推送，
  全仓382通过、1跳过、1个相同既有路径审计失败，仓库Ruff/compileall/diff检查通过。
  确认A04运行目录不存在、无关联进程且worktree干净后，将尚未使用的A04源码从`67787bc`
  更新到此版本；正在训练的A03仍固定`c50ddf0`，未修改。真实全量CPU索引导入核验完成，
  回执位于`/data1/zyl/SpecEmbedding/audits/validation_index_import_20260909/receipt.json`，SHA-256
  `5f748b64d0005162bece9df9c906f3c281e5980db94f2c91dfdc98ca41362f4f`；源/目标索引及相邻回执
  完全一致，计数与耗时见优化索引D08。新的真实全量dry-run保存于
  `/tmp/specembedding_a04_index_import_preflight_20260909.json`，SHA-256
  `b8fd5d95cb0ba89f9404863109453806125e162cd441f125d0ffc04cc061145c`，阶段为import_v15、
  import_validation、baseline_validation、alignment42；相比旧预检，训练配置、候选来源和
  GPU命令相同。旧预检保留历史追溯，新预检仍暂用A02/图增强开启/验证batch512，最终决策后
  重新冻结。20:32的A03已完成25轮、仍在训练，A04及独立验证batch测量均未派发。
- 2026-09-09 20:09：按用户要求把下一轮准备前移到A03训练期间。候选监督正式集成
  `67787bc33266f89ca85c95e3438d221acf924e71`已提交推送，全仓366通过、1跳过、1个相同
  既有路径审计失败，Ruff/compileall/diff通过；18项新增检查包含实际小模型训练及完整采样重放。
  固定源码 `/data1/zyl/repos/SpecEmbedding-opt-a04-20260909` 干净，真实全量dry-run保存于
  `/tmp/specembedding_a04_candidate_preflight_20260909.json`，SHA-256
  `d2f81c3f5c3bee5e16ce7e34d13ad0c9e275e663dd40623b44a04fa93b95ce88`；原始query映射与D07
  独立完整映射一致。该预检暂以A02、验证batch512和图增强开启为基线设置；最终parent或
  已验证工程参数变化时重做。候选训练的运行目录尚未创建，实验卡见根目录`attempt_4.md`。
  D08完整验证batch测量脚本、三份配置和输入指纹已准备于
  `/data1/zyl/SpecEmbedding/audits/validation_batch_benchmark_20260909/`，manifest SHA-256
  `610b1ca962cf3e42b6412c0aa2d6e0f68b76bafc4fa1ac07dbe778d48fc82aca`。
  它要求A03完整终止后才可单次派发，每阶段独立等待GPU且三次固定同一卡；尚未派发，
  不把已准备的脚本当作完成测量。A03当前仍训练，原源码/配置/工件保持不变。
- 2026-09-09 19:36：按用户的低风险提速要求，用A02原始状态及A03前11轮资源文件核验单次
  尝试耗时，记录于优化索引D08；不把条件速度计算写成已实现收益。优先准备验证分子batch
  512→1024的单项工程测量，完整分母、候选、权重和验证频率固定；数值/性能采用条件已提前
  写明。等待A03完成后再执行，当前未派发，训练batch/学习率及固定运行源码不变。
  19:34确认A03入口和训练进程存活，19:35:51完成第12轮，最新中途指标已同步attempts.md；
  完整结果未定，A02仍为baseline。候选监督的正式入口/完成审计集成仍在开发，未派发A04。
- 2026-09-09：按用户新要求新增根目录`attempts.md`，每轮正式模型尝试以一句话记录改动、
  结果及基线决策，补齐A01/A02和仍在训练的A03；详细证据继续引用优化索引，必要时补充
  `attempt_<n>.md`。后续每轮同步更新，不把未运行的组件或静态检查列成已完成模型尝试。
- 2026-09-09 19:11：A03于18:43:40通过训练阶段自己的GPU等待，在物理GPU0/严格cuda:0
  启动alignment；入口PID3265298、训练PID3268007于19:08确认存活，固定`c50ddf0`源码干净。
  当时已完成5轮完整检索验证，逐轮覆盖完整train/val，首轮资源记录真实生成且计数/时长一致；
  指标和资源细节只维护于优化索引A03，尚无最终选择/收益结论，不提前切换实验。
  候选数据/损失/训练组件已提交并推送为`bee5b63`，新增23项测试通过；全仓348通过、
  1跳过、1个相同既有路径审计失败，Ruff/compileall/diff通过。多进程测试最初受沙箱本机socket
  限制而阻塞，取得明确权限异常后在主机CPU环境通过，未修改A03或回避多进程验证。
  真实全量谱图/token/候选映射CPU预检于19:09完成，详见优化索引D07；回执
  `/data1/zyl/SpecEmbedding/audits/candidate_dataset_binding_20260909/receipt.json`，SHA-256
  `ef3e5b24d68d85c688f3257c7dd31d21a4b3089dc4ec33b9387976b2c330d4d2`。
  同目录保存完整映射和可重跑`preflight.py`。正式配置/入口/完成审计集成和唯一实验卡仍待完成，
  未派发候选监督训练，不把组件、CPU预检或A03中途数据写成最终实验。
- 2026-09-09 18:40：A03 baseline验证于18:38:41通过连续120秒GPU门槛，选择物理GPU0。
  子进程PID3266643的UUID单卡显露、严格CUDA标志、显式cuda:0与独立运行配置均已核验，
  正在重新编码完整验证候选。alignment尚未启动，后续该阶段仍须重新等待GPU。
- 2026-09-09 18:38：A03于18:36:36完成全部CPU准备，验证索引SHA与A01/A02相同，
  baseline验证进入GPU等待；指定pane存活，入口PID3265298在运行，未创建第二个监测器。
  两张卡瞬时满足门槛仍执行连续120秒规则；训练阶段后续再次等待，等待不等于已完成训练。
  新候选工具的真实全量CPU采样预检完成，详细计数见优化索引D06；回执位于
  `/data1/zyl/SpecEmbedding/audits/train_candidate_sampling_20260909/receipt.json`，SHA-256
  `21f649c223450536c06931cb8951b1c959471e33aacef569f9bc63002f51c1ba`。
  同目录保存全部抽样索引、源位置、计数及`preflight.py`；全部query保留，未接入训练或改变A03。
- 2026-09-09 18:34：A02于18:09:37完成，27轮早停、进程退出、pane exit0；独立完成审计
  逐轮重算全部有效验证排名及指标通过。最终epoch22是唯一Pareto候选，符合预先声明的整体
  改善规则，已晋升为下一轮比较baseline；指标与解释边界集中于优化索引A02。
  回执 `/data1/zyl/SpecEmbedding/audits/optimization_a02_20260909/receipt.json`，SHA-256
  `deb8fa650502204387d6ab4b724073796f41344be33bb9fcb508985f05f19d5d`；相邻`decision.json`
  保存选中权重、原baseline、差值和A03范围，原始A02/r4工件均保留。
  A03已在固定`c50ddf0`重新完成真实输入预检，使用A02 epoch22作比较、mass_blocks/block32，
  只关闭分子图增强，继续随机初始化和完整train/val，不运行test/reranker。
  18:32:27单次启动；运行根
  `/data1/zyl/SpecEmbedding/experiments/massspecgym_v15_opt_a03_20260909_topk256`，独立socket
  `/tmp/specembedding-opt-a03-20260909-1010/tmux.sock`，session`opt`/pane`%0`，入口PID3265298。
  `launch_receipt.json`、`launch.py`、`parent_decision.json`和`inputs_and_commands.json`保存
  派发及输入指纹。18:34核验数据导入完成、CPU验证候选准备中；GPU阶段继续0/1择一逐次等待。
- 2026-09-09：全量自然训练候选读取与不同二维负例采样工具已提交推送为`01024f5`，
  23项新增测试通过；全仓325通过、1跳过、1个相同既有路径审计失败，Ruff/compileall/diff通过。
  工具尚未接入模型或训练入口，真实全量CPU采样预检进行中，详见优化索引D06。
  不修改固定A02/A03源码，不把输入工具或准备工件写成候选监督模型已完成。

- 2026-09-09 17:45：全量训练自然候选元数据准备及独立源文件复核完成，详见优化索引 D06。
  工件根为 `/data1/zyl/SpecEmbedding/audits/train_candidate_feasibility_20260909/`；
  `receipt.json` SHA-256 `a1f971378d667d8e44bd7f32165cb813638f044326de9f6354c7439d23ddfad4`，
  `verification.json` SHA-256 `5e6d81afb569ddc812680e13d5cd777ce6eb6d44fb21c57531ce3b694e6d3987`。
  全部194119条query及6350650条候选条目通过独立读回核验，正例/源位置/图排除一致；
  原始候选保留，无forcing，无训练或模型推理。重复二维负例、held-out无标签结构交集及
  存储规模已记录；该诊断元数据未被训练入口使用，不改变A02/A03配置。
  A03 固定源码 worktree 已创建为 `/data1/zyl/repos/SpecEmbedding-opt-a03-20260909`，
  detached `c50ddf04d2be4fedc007365e511a95f36e298c08`、干净，入口帮助及环境导入通过；
  与已测试实现一致，未创建训练运行或派发命令。正式parent/batching与新预检待A02完整审计。
  A02 原进程仍存活，已完成21轮完整检索验证，当前各指标与Pareto取舍只维护于优化索引。
- 2026-09-09 17:22：完整验证候选的 CPU 指纹可区分性诊断完成，未做模型推理。
  工件根 `/data1/zyl/SpecEmbedding/audits/fingerprint_collisions_20260909/` 保存全候选指纹载荷
  摘要、完整query计数、全部歧义对与可重跑脚本。`receipt.json` SHA-256 为
  `a2b8179564b99713d0a79bab4533b1214221f02bc67a024a593e8d90bb320f7e`；
  `verification.json` SHA-256 为 `705382c61e35bd12df06b12c328205ccd687383e6443a34a5fe707a67c8ada22`。
  独立重算两种位宽的全部19423条query，并重编码全部碰撞分子的二维身份与位向量；
  诊断结论、资源范围及后续意义见优化索引 D05。JESTR/MSAlign 的候选监督路线原文核验见 R01，
  未将其报告性能当作预期增益，未扩大训练队列或改变 A03。A02 第15轮各项验证指标均高于
  原固定基线，原入口/训练进程仍在运行；继续完整轨迹和审计后再作 incumbent 决策。
- 2026-09-09 17:01：逐轮资源记录与审计实现 `c50ddf0` 已提交并推送。53 项专项通过，
  全仓 302 通过、1 跳过、1 个相同既有路径审计失败；Ruff、全仓 compileall、diff 检查通过。
  记录早停末轮，合成对照验证开启前后 RNG/选中权重/早停一致，损坏记录审计拒绝通过。
  A01/A02 的固定源码未修改，缺失资源记录明确标为未测量；A03 最终配置和源码冻结后需
  重新执行相应预检，未提前派发。只读复查 A02 已完成11轮完整验证、继续训练，指标见优化索引。
  同时复核论文目录的五篇主要 PDF 及 SpecBridge 历史 PDF/撤稿页面，字节数与清单 SHA 均匹配。
- 2026-09-09：A03 单项关闭分子图增强实现已提交为 `8092912`，65 项专项通过，
  全仓 292 通过、1 跳过、1 个相同既有路径审计失败；Ruff、compileall、diff 检查通过。
  真实输入 dry-run 只改变 node/edge 扰动率，保留谱图增强、全量样本、网络与其它超参数；
  预检为 `/data1/zyl/SpecEmbedding/audits/graph_augmentation_20260909/preflight_8092912.json`。
  该预检暂用 r4/random，并未创建训练目录或队列；A02 完整审计后再选择唯一 parent/batching，
  不在中途停止 A02 或展开新矩阵。A02 截至16:34已完成5轮并继续训练，数字只维护于优化索引。
  同目录 `receipt.json`、`audit.py` 保存完整 train 的图扰动与多正例熵下界诊断，
  SHA-256 `7fbd08370376f1710f9a1cf037f6f00a62f7350574f7650ac542153a95571578`。
  该项是 CPU 输入诊断，不是新模型推理或增益结论；实验卡、统计与解释边界见索引 A03/D04。
  已核验旧 A01 selection 的增强快照与其固定运行配置一致，新完成审计不会以默认值替代缺失记录。

- 2026-09-09：实现第 4 节预定的完整验证谱图输入敏感性检查，入口为
  `alignment_validation.py --spectrum-control permuted|constant`，说明见 GPU/tmux 指南。
  仅在独立新输出中运行，不参与普通选优；固定 checkpoint 指纹、完整候选/分母保持不变，
  输出分数经过独立 CPU 重排。14 项新增测试通过，全仓 287 通过、1 跳过、1 个相同既有
  路径审计失败，Ruff、compileall、diff 检查通过。真实全索引 CPU 构造回执和脚本位于
  `/data1/zyl/SpecEmbedding/audits/spectrum_controls_20260909/preflight.json`、`preflight.py`。
  这不是正式控制推理，未插入 A02，详细指标与解释边界见优化索引 D03。
  A01/A02 的重复 r4 基线分数亦完成独立 CPU 核验，回执与脚本位于
  `/data1/zyl/SpecEmbedding/audits/baseline_repeat_20260909/receipt.json`、`audit.py`，差异见 D02。

- 2026-09-09 16:17：A02 前置基线验证于 16:10:39 完成，训练阶段重新等待 GPU，
  16:12:44 绑定物理 GPU 0 的 UUID 并以 `cuda:0` 启动 alignment。
  训练子进程 PID 3221482，父进程仍为 3219277；已核验子进程严格 CUDA 与本运行配置环境一致。
  原始日志确认 seed42、随机初始化、无 TokenSet cache、mass_blocks/batch128/block32，
  首轮完整训练/验证计数通过并完成检索排名，继续后续 epochs。源码仍固定 `fc42b27`，
  没有修改运行配置、增加并行模型或启动 test；首轮结果只维护于优化索引。

- 2026-09-09 16:01：A01 完整审计与基线决策后，单次启动 A02 质量邻近 batch 队列。
  干净 detached 源码为 `/data1/zyl/repos/SpecEmbedding-opt-a02-20260909/`，固定提交
  `fc42b271ba355306f8ba6fad604e81053da78904`；运行根为
  `/data1/zyl/SpecEmbedding/experiments/massspecgym_v15_opt_a02_20260909_topk256/`。
  独立 socket `/tmp/specembedding-opt-a02-20260909-1010/tmux.sock`，session `opt` / pane `%0`，
  入口 PID 3219277，派发时间 16:01:20。`launch_receipt.json` 绑定 A01 审计与决策哈希，
  `parent_decision.json` 保存保留 r4 的依据；`inputs_and_commands.json` 已经真实输入预检复核。
  16:01:31 数据导入完成，16:05:28 验证索引准备与核验完成，实测进入
  `baseline_validation/waiting_gpu`，尚无 alignment 训练目录。索引 SHA-256 与 A01 相同，
  query/候选/排除及自然正例覆盖一致；初始两卡均为 0% / 空闲 24,206 MiB，继续保持 120 秒检查。
  只增加 `--alignment-batching mass_blocks`，保持 seed42、随机初始化、batch128/block32，
  不从 r4/A01 继续训练；r4 仅作固定比较。四阶段与完整协议不变，各 GPU 阶段继续独立等待
  物理 GPU 0/1 择一并绑定逻辑 `cuda:0`，没有新增外部监测器或并行实验。

- 2026-09-09 16:00：A01 于 15:58:57 在第 19 轮按既定规则早停；外层四阶段 complete、
  指定 pane dead=1，入口 PID 3182732 与训练 PID 3192041 均已退出。独立完成审计通过，
  原始回执为 `/data1/zyl/SpecEmbedding/audits/optimization_a01_20260909/receipt.json`，
  SHA-256 `ab402240b49b86b3c296f40198051a0bc3c76460973791312e548b8013471354`；
  相邻 `decision.json` 记录保留 r4 和下一项 A02 的范围。选优、完整非支配候选及比较数字
  只维护于优化索引。全部 19 轮样本覆盖、保存分数/排名/指标、输入指纹与权重关联通过核验；
  未运行 test，没有将指标取舍写成整体改善或 SOTA 结果。计划继续 `执行中`，论文保持未修改。
  本轮只更新文档：结果表逐项对照完成回执、相对链接与 diff 检查通过，未重复运行无变更的训练测试。

- 2026-09-09 15:24：A01 第 11 轮完整验证刷新本次 Top-1，既定早停计数重置，训练继续；
  尚无最终选优与整体收益结论，A02 未派发。完成审计入口已提交并推送为 `0e2eaf3`，
  新入口对真实固定 baseline 的全部保存分数核验通过，原始指标与排名逐条一致。
  文献补充核验及 SpecBridge 历史 PDF/官方撤回记录保存完成，比较边界见第 5 节。

- 2026-09-09：新增独立 CPU 完成审计入口 `audit_alignment_optimization.py`，检查所有保存排名、
  指标与权重/选优关联，并报告完整 Top-k/MRR 的改善与退步；仅接收完整优化队列，输出不得覆盖。
  14 项 synthetic 审计测试通过，全仓 273 通过、1 跳过、1 个相同既有路径审计失败，静态检查通过。
  使用 A01 当前真实运行验证未完成保护，明确拒绝生成完成回执；没有停止或重启 A01。
  入口和证据边界见 GPU/tmux 指南及优化索引，A01 最终审计尚未执行。

- 2026-09-09：完成固定 baseline 的全部有效验证错误诊断；输入与输出哈希、可重跑脚本、
  每 query 排名和分组结果保存在 `/data1/zyl/SpecEmbedding/audits/validation_errors_20260909/`
  的 `receipt.json`、`audit.py` 和 `queries.tsv`，结论只维护于优化索引 D01。
  保存分数的独立 CPU 重排与原排名逐条一致，未进行新推理/训练或输入修复。
  质量硬过滤会丢失真值，未实施；元数据分组不能解释为因果收益，A02 仍保持单项 batch 改动。
  A01 第 5 轮完整训练/验证已完成，继续原定训练；最终还需比较同一轨迹的 loss 最优与检索最优轮，
  不将 r4/A01 的训练数值波动全部归因于 checkpoint 选择。

- A02 实现提交 `fc42b27` 已推送；真实输入 dry-run 核验 CPU 导入/验证索引/基线验证/单 alignment
  四阶段、GPU 0/1 池、batch128/block32 与其余固定配置，未创建训练运行目录或进程。
  原始预检位于 `mass_batching_20260909/preflight_fc42b27.json`（审计根见下一条记录）。
  A01 首轮完整训练及检索验证已于 14:41 完成，实际计数与每轮排名保存于原运行根，继续后续训练；
  当前没有 A01 最终选优结论，不依据早期 epoch 决定切换或停止。

- 2026-09-09 14:36:53：A01 在物理 GPU 0、UUID 绑定的 `cuda:0` 上启动 alignment，
  PID 3192041，cwd 仍为固定 `cba38a4` worktree。前置固定 baseline 于 14:34 完成，
  指标与耗时只维护于优化索引及运行根 `baseline_validation/metrics.json`，不把新验证基线
  写成模型提升。已核验进入完整检索验证，checkpoint 最终选择尚未完成。
- A02 质量邻近 batch 实现与完整 CPU 顺序检查通过，未派发；原始证据与实现 SHA 位于
  `/data1/zyl/SpecEmbedding/audits/mass_batching_20260909/receipt.json` 及相邻脚本。
  60 项专项通过；全仓 257 通过、1 跳过、1 个相同既有路径审计失败，之后新增两项队列完成
  审计测试通过。Ruff、compileall、diff 检查通过。未修改正在运行的 A01 源码、配置或工件。

- 2026-09-09：A01 等待期间完成完整 train 负例分布与旧预训练来源诊断，证据位于
  `/data1/zyl/SpecEmbedding/audits/alignment_inputs_20260909/receipt.json` 及相邻可重跑脚本。
  旧 TokenSet 的 train/val 谱图 token 与新原始划分的重新分词逐条多重集完全一致，计算二维
  身份与 test 无交集；但旧预训练缺原始指纹、按分子 key 重采样，选优还包含既定六条验证排除项。
  现有权重只通过当前谱图塔严格 state_dict 加载，不能据此证明完整历史配置或正式协议合规。
  暂不将其接入正式训练。下一项准备质量邻近 batch 组装，实验卡见优化索引；保持 A01 不变，
  待 A01 完整轨迹后再决定派发，不展开模型/seed 矩阵。

- 2026-09-09：补充 SOTA 查询分母审计，结果与边界见第 5 节；源 TSV 与已下载候选包哈希均
  与既有来源 manifest 匹配。该审计没有新增模型分数；最终同协议参考向量仍未冻结。

- 2026-09-09 14:12（北京时间）：A01 已完成数据导入及完整验证索引准备，状态实测为
  `baseline_validation/waiting_gpu`，CPU 准备子进程已退出。候选/query 数量与协议见
  [优化索引](../../analysis/massspecgym_optimization.md)，详细指纹见运行根的
  `validation/mass_val_topk256.json`；全部有效 query 保留自然正例，没有 forcing。
  14:07:18 单次启动，固定源码 `cba38a459fe438d29bc21bd19c54995f8d7e298e` 位于
  `/data1/zyl/repos/SpecEmbedding-opt-a01-20260909/`，detached worktree 保持不变。
  新运行根：`/data1/zyl/SpecEmbedding/experiments/massspecgym_v15_opt_a01_20260909_topk256/`；
  独立 socket：`/tmp/specembedding-opt-a01-20260909-1010/tmux.sock`，session `opt` / pane `%0`，
  入口 PID 3182732。`launch_receipt.json` 保存派发命令与固定提交；`inputs_and_commands.json`、
  `runtime_params.yaml`、`status.json`、`runner.log` 保存输入、配置、阶段和等待证据。
  启动前已核验目录/socket 未使用且没有同运行入口；未重发旧队列或删除任何锁。
  本队列依次执行 CPU 导入、CPU 验证索引、旧 checkpoint 完整验证、单 seed42 alignment，
  不包含 test 或 reranker。两 GPU 阶段分别等待物理 GPU 0/1，选中 UUID 映射为显式 `cuda:0`。
  14:12:02 的历史快照为 GPU 0 利用率 93% / 空闲 19,188 MiB，GPU 1 为 76% / 20,646 MiB，
  均未满足利用率门槛；后台继续等待，无新增外部监测器。排队不代表模型开始训练或性能改善。

- 2026-09-09 A01：已实现完整验证检索选优，实验卡见
  [单 baseline 索引](../../analysis/massspecgym_optimization.md)。主排名使用已核验的
  `torchmetrics 1.8.2` 逐 query CPU argsort，稳定排序作为同分敏感性视图；此前的 stable-only
  设计由官方实现核验细化，不更改候选身份或分母。官方代码来源为
  [retrieval/base.py](https://github.com/pluskal-lab/MassSpecGym/blob/main/massspecgym/models/retrieval/base.py)
  和 [torchmetrics hit_rate.py](https://github.com/Lightning-AI/torchmetrics/blob/v1.8.2/src/torchmetrics/functional/retrieval/hit_rate.py)。
  本机新增 torchmetrics1.8.2、lightning-utilities0.15.3；没有更改 PyTorch/CUDA。
  53 项专项通过，全仓 239 通过、1 跳过、1 个既有路径审计失败；实现未改两塔/预训练结构。

- 2026-09-09：用户先授权“Top-1 优先、MRR 为辅”，后将目标更新为 Top-k 整体及各项指标接近
  SOTA。已重写本计划的当前部分；原全量迁移过程保留如下。当前没有派发新的优化训练。
  已保存三篇核心论文并核对 PDF 可读性（28/25/7 页）；后续参考论文的来源与哈希由外部 manifest 维护。
  现有入口仍按旧损失/MRR 选 checkpoint，Top-k 选优属于下一实现项，不能据计划表述当成已完成。
- 2026-09-09：单 baseline 入口、共享范围/完成校验和测试提交为 `f92e56c`；未改固定 r4 源码。
  五篇 PDF 均已核验 SHA-256 和可解析性，页数依次为审计论文 28、MSAlign 25、JESTR 7、
  GLMR 13、GLACIER 22。这些是资料与工程验证，不是新增性能结果。

以下记录描述各时间点事实；旧的运行授权/矩阵设计不覆盖本计划前述最新范围。

- 2026-09-09 13:29：根据单 baseline 授权，核验并向 r4 专属进程组 3110729 发送 SIGINT，
  已确认全部成员退出、原独立 socket 的 v15/%0 pane dead=1。外层状态为 failed_or_interrupted，
  内层状态残留 running；保留原始文件，以 r4 根目录 `single_baseline_stop_receipt.json` 的
  进程核验、时间和状态 SHA-256 解释主动中断，不重启、覆盖或删除锁。固定源码 ad81b61 未改。
  入口与 params 改为仅允许 Mass/relative/seed42；全量、缓存审计、GPU 等待与不复用目录保护保留。

- **当前状态，2026-09-09 13:11（北京时间）**：用户要求排查新旧准确率落差，特别检查
  是否复用旧编码器或 `train_cache`。六 cache 指纹、四 test cache 全 query 一致性与新权重
  分散样本 GPU 重编码核验通过，未发现旧权重或 TokenSet 被复用。发现并量化旧候选表示的
  强烈格式捷径，详情与原始工件见[回归排查](../../analysis/massspecgym_v15_regression_audit.md)。
  该证据影响旧高分与论文论据的解释，故本计划设为 `已偏离待确认`，暂停论文结论更新；
  在用户确认解释范围及后续控制实验前，不新增重训或因果消融。已经授权的原 r4 队列没有
  改动或中断：alignment、六 cache 已完成，Mass relative 三 seed 和 pointwise seed42
  已完成训练及全测试，当前是 Mass pointwise seed43 训练。队列运行状态与本计划的论文
  解释暂停状态分别记录，不能将部分完成写为矩阵完成。
- 2026-09-08 23:58（北京时间，历史迁移记录）：按用户条件授权，确认 r3 仍在等待、
  没有 alignment 日志/目录或训练子进程后，于 23:55:48 中断旧入口 PID 2990241。
  旧状态记录 `failed_or_interrupted / KeyboardInterrupt()`，这是主动迁移取消，原日志、锁与工件保留。
  新 r4 固定源码为 `/data1/zyl/repos/SpecEmbedding-v15-fulltrain-20260908-r4/`，
  detached `ad81b61`、工作树干净；运行根为
  `/data1/zyl/SpecEmbedding/experiments/massspecgym_v15_fulltrain_20260908_r4_topk256/`。
  独立 socket 为 `/tmp/specembedding-v15-fulltrain-20260908-r4-1010/tmux.sock`，
  session `v15` / pane `%0`，入口 PID 3110729，于 23:55:52 启动。
  `import_v15` 已完成：r3 完整 CPU 数据经源指纹和输出哈希核验后复制到新目录，
  不复用旧训练状态、cache 或 checkpoint。输入与命令见 `inputs_and_commands.json`，
  主动停止证据见 `migration_receipt.json`，实时状态见 `status.json` / `runner.log`。
  实测进入 `alignment42/waiting_gpu`，同时等待物理 GPU 0、1，各卡独立保持原有门槛，
  单阶段择一绑定到逻辑 `cuda:0`；初始 GPU 0 为 0% / 24,206 MiB，GPU 1 为 91% / 4,505 MiB。
  连续 120 秒及输入指纹/最终设备复查后，于 23:58:03 选择物理 GPU 0，启动 alignment
  子进程 PID 3111216；`status.json` 为 `alignment42/running`，训练日志确认 RTX 4090、
  `cuda:0`、seed 42、`formal_fulltrain=True` 与原六条验证排除，开始完整 194,119 条训练谱图 tokenization。
  后续复查已进入第 1 轮训练（1,517 batches）；子进程环境只显露选中 GPU 0 的 UUID，
  严格 CUDA 标志及本运行配置路径一致，`nvidia-smi` 确认同一 PID 在该 UUID 上使用 2,300 MiB。
  没有新增外部监测器；后续各 GPU 阶段仍重新择卡，全部训练和评价尚未完成。
- 2026-09-08 晚间：按用户授权实现 `--gpus 0 1`（也支持更多编号）的逐阶段择卡。
  各卡独立计时、查询超时单卡重置、最终复查、UUID 固定、逐个派发；选中卡在子进程内
  显式绑定为 `cuda:0`，实际物理编号与 UUID 另记，不修改训练样本、模型超参数或评价协议。
  同时增加完整 CPU 准备数据的指纹核验与复制导入，支持迁移队列而不重跑已完成的审计。
  验证：全仓 227 通过、1 跳过、1 个相同的既有路径审计失败；Ruff、compileall、diff 检查通过。
  新测试覆盖独立计时、失败卡不阻塞其他卡、候选优先级、最终检查、UUID 变更、真实 CPU
  子进程的 UUID 环境、跨阶段重新择卡及导入失败阻止 GPU 阶段。
  实现提交 `ad81b61` 已推送；r3 固定源码与既有数据保持原样，实际迁移记录见上。
- 2026-09-08 00:38:30（北京时间，历史等待记录）：r3 的 `prepare_v15` 已完成，
  `data/MassSpecGym/dataset_manifest.json` 为 `complete`，入口已完成输出文件校验并进入
  `alignment42/waiting_gpu`。Mass/Formula 的完整条目数量、图资格统计及身份转换重试结果
  与前次独立 CPU 审计一致；两协议均保留源顺序、forcing=false，全部源列表保留 exact target，
  三个 split 数量与既定验证排除项一致。原始记录仍在本运行目录，不继承旧运行的完成状态。
  用户要求前台监测至开始等待 GPU；本轮监测没有新故障，无需修复或重启。
  初始等待日志为 GPU 1 利用率 31%、空闲 4,019 MiB，未满足门槛；预处理 PID 2990250
  已退出，入口 PID 2990241 与指定 pane 仍存活，尚无 alignment 日志。
  前台监测结束，原后台队列继续等待，不派发第二个任务。
- 2026-09-08 00:05（北京时间，r3 启动记录）：固定源码为
  `/data1/zyl/repos/SpecEmbedding-v15-fulltrain-20260908-r3/`，detached `15b1b5d`、工作树干净；
  Python 代码和 `params.yaml` 与已通过完整 CPU 验证的 `aecdfee` 一致，仅更新恢复状态文档。
  运行根为 `/data1/zyl/SpecEmbedding/experiments/massspecgym_v15_fulltrain_20260908_r3_topk256/`，
  独立 socket 为 `/tmp/specembedding-v15-fulltrain-20260908-r3-1010/tmux.sock`，
  session `v15` / pane `%0`，入口 PID 2990241，初始 CPU 预处理 PID 2990250。
  先保存并复核 `inputs_and_commands.json`，再单次启动；实测 `status.json` 为
  `prepare_v15/running`，数据 manifest 已记录 v1.5 与修复后的身份策略，尚无 alignment 日志。
  子进程已核验 UUID 全卡顺序、严格 `cuda:1` 环境及本运行 `runtime_params.yaml`；
  后续物理 GPU 1 门槛仍为利用率 ≤10%、空闲 ≥20,000 MiB、30 秒轮询并连续满足 120 秒，
  无最长等待时间。启动前读数为利用率 96%、空闲 4,019 MiB，不代表后续实时状态。
  本入口自带逐阶段等待，没有新增外部监测器；旧队列、审计目录和固定源码均保留。
  状态入口为运行根的 `status.json`、`runner.log`、`logs/prepare_v15.log` 和
  `data/MassSpecGym/dataset_manifest.json`；此启动记录不代表正式数据审计、训练或评价完成。
- 2026-09-08：用户明确恢复目标，计划设为 `执行中`。重新核验旧队列与独立 CPU 审计均已退出，
  完整审计 manifest 及其输出通过哈希复查；当前没有运行中的正式训练。
  继续使用已验证的 InChI 修复和现有正式入口，在新 worktree、新运行根和独立 socket 中启动。
  入口在新目录重新生成并审计完整数据，再等待物理 GPU 1，以 `cuda:1` 训练；不绕过旧工件保护，
  不把独立 CPU 验证目录伪装成完成了正式阶段的运行根。启动后的实测状态另行回填。
- 2026-09-07 22:53：修复 `aecdfee` 的独立完整 CPU 验证通过；正式队列继续暂停。
  固定源码为 `/data1/zyl/repos/SpecEmbedding-v15-inchi-audit-20260907/`（detached、干净），
  验证根为 `/data1/zyl/SpecEmbedding/audits/massspecgym_v15_inchi_fix_20260907_topk256/`。
  独立 socket `/tmp/specembedding-v15-inchi-audit-20260907/tmux.sock` 中仅运行
  `prepare_massspecgym_v15.py`，CUDA 显式不可见，没有训练或监测后续阶段；该进程现已退出。
  `verification_receipt.json` 保存固定提交与完整命令，`prepare.log` 为原始日志；
  `data/dataset_manifest.json` 与 `independent_verification.json` 均为 `complete`。
  两候选协议所有源列表审计完成；独立读取检查了全部 split 数量、文件哈希、候选内容/顺序、
  图排除与身份转换的源位置和身份结果。数量汇总只维护于[修复记录](../../analysis/massspecgym_v15_inchi_fix.md)。
  这是修复验证，不复用为自动恢复的正式队列，不改变旧运行工件或论文结论。
- 2026-09-07 22 时：用户要求停止任务并排查修复。核验三个指定 server：旧 v1 只剩空闲 shell，
  两个 v1.5 pane 均已退出，没有训练或监测进程；不删除锁、日志、失败工件或固定源码。
  已定位 RDKit 2026.03.1 的默认 InChI 与 sanitization 使用不同 Kekulé 搜索顺序。
  修复仅在身份转换的分子副本上重试搜索，保留标准 InChI 选项及全部候选；新增逐项记录和校验，
  仍失败则带候选上下文停止。详见[故障修复记录](../../analysis/massspecgym_v15_inchi_fix.md)。
  专项 18 项、全仓 214 项通过，1 跳过，1 个相同的既有路径审计失败；静态检查通过。
  后续只执行独立 CPU 验证；正式队列维持停止，修复完成不自动恢复。本计划状态为 `暂停`。
- 2026-09-07 21:54（北京时间，历史核验）：r2 队列已于 20:33:12 失败退出，
  顶层 `status.json` 与数据 manifest 均为 `failed_or_interrupted`；指定 socket 的 `v15 / %0`
  为 `dead=1 exit=1`，PID 2977661 已不存在。未生成 alignment 日志/目录或 rerank 状态。
  原始 traceback 指向 `audit_candidate_list` 的 `Chem.MolToInchiKey(mol)`：候选 SMILES
  已通过解析，生成身份时抛出 `KekulizeException`。这是身份转换失败，不能未经核验套用
  上次的无效图排除，也不能据此判定源分子无效或修改候选协议。
  最后持久化进度为 Mass 4,096/32,010 个列表、1,034,580 个条目（约 12.8% 列表）；
  子审计的 `running` 是失败前进度快照，不代表仍在运行。Formula 审计尚未开始。
  三个 split 文件已生成，数量为 194,119/19,429/17,556；完整数据审计未通过，不可用于正式训练。
  本次只读核验运行现场，保留所有失败工件、锁与固定源码，未重发或创建队列。
  下一步需定位触发候选并核验身份转换原因；若处理需要改变协议，按计划规则确认后执行。
- 2026-09-07 20:29（北京时间，历史启动记录）：修复提交 `d15510b` 已推送；新的固定源码为
  `/data1/zyl/repos/SpecEmbedding-v15-fulltrain-20260907-r2/`，detached HEAD、工作树干净。
  新运行根为 `/data1/zyl/SpecEmbedding/experiments/massspecgym_v15_fulltrain_20260907_r2_topk256/`；
  新 socket 为 `/tmp/specembedding-v15-fulltrain-20260907-r2-1010/tmux.sock`，session `v15` / pane `%0`，
  主进程 PID 2977661。已先保存并复核预检指纹，再启动；`status.json` 实测为 `prepare_v15/running`。
  候选审计通过后自动等待物理 GPU 1，以 UUID 顺序映射的 `cuda:1` 训练；其后仍逐阶段等待。
  不存在附加外部监测器。旧 v1 监测器和首个 v1.5 队列均已退出，没有向任何旧 pane 再派发。
  本目录的 `data/MassSpecGym/dataset_manifest.json`、`logs/prepare_v15.log` 记录 CPU 审计进度，
  `invalid_graph_mass.jsonl` / `invalid_graph_formula.jsonl` 位于数据目录，完整记录源列表的图排除。
  此状态不代表数据审计、alignment、十二组训练/测试、结果身份审计或论文更新已完成。
- 2026-09-07 20:17：首个 v1.5 队列在 CPU 候选审计阶段失败停止，未进入 GPU 阶段。
  三个 split 已生成并核验完整，失败原因是官方候选包含 RDKit 无法解析的过价 Si 分子；
  日志和原始失败目录均保留。现有 `MolSmilesDataset` / `mol_collate_fn` 本来就排除无法构图的分子，
  新审计不应把该已存在的编码资格规则误当成所有源候选均可解析的保证。
  修正审计为逐项记录无法构图条目、源目标和位置，保留原始候选内容/顺序；目标不可编码仍失败。
  缓存记录实际图排除和版本指纹，拒绝丢失 query 或正例。本次没有缩减训练数据，也没有更换官方候选库。
  必须以新 commit、新 worktree、新运行目录再次执行，不修改或自动重试 `6843e12` 失败队列。
- 修复验证：全仓 212 通过、1 跳过、1 个相同的既有路径审计失败；v1.5/正式队列专项 32 项通过，
  Ruff、compileall、diff 检查通过。额外修正旧 anonymous CPU smoke 的合成候选夹具：其入口显式
  no-forcing 却未提供正例，与验证断言矛盾；现在源夹具自然包含正例，未启用 forcing。
  完整 synthetic prepare → train → eval CPU smoke 已通过（6/4/4 条合成 query、1 epoch），
  anonymous supplement 相关测试 13 项通过。没有重建或发布匿名归档，也不是正式数据实验。
- 2026-09-07 20:15（北京时间）：v1.5 实现提交 `6843e12` 已推送，固定 detached 源码位于
  `/data1/zyl/repos/SpecEmbedding-v15-fulltrain-20260907/`，工作树干净；旧 `509fbeb` worktree 未修改。
  新运行根为 `/data1/zyl/SpecEmbedding/experiments/massspecgym_v15_fulltrain_20260907_topk256/`，
  已保存原始输入哈希、依赖版本、运行配置及全部阶段命令至 `inputs_and_commands.json`。
  独立 socket：`/tmp/specembedding-v15-fulltrain-20260907-1010/tmux.sock`，session `v15` / pane `%0`，
  主进程 PID 2964130，初始 CPU 预处理 PID 2964153。该入口自带 GPU 等待，不启动第二个外部监测器。
  `status.json` 已核验为 `prepare_v15/running`；源数据 manifest 已生成，完整候选审计尚未完成，
  未启动 GPU 训练。进度见 `runner.log`、`logs/prepare_v15.log`、`data/MassSpecGym/dataset_manifest.json`；
  后续 GPU 阶段使用物理 GPU 1、UUID 顺序映射的 `cuda:1`，保持既定利用率/显存及持续时间门槛。
  不复用或删除旧运行锁、不向旧训练 pane 派发、不自动重试失败工件。
- 2026-09-07 19:48（北京时间）：用户授权 v1.5 迁移及必要的 alignment 重训。
  核验旧监测器 PID 2853075、指定 socket 与命令后，仅向该监测器发送 SIGINT；
  `monitor.log` 于 19:48:40 记录停止。复查仅剩 `fulltrain` / `%0` 空白 bash，未派发训练，
  无 `runner.log`、`status.json`、缓存或 checkpoint。未删除锁、未改固定 worktree，未终止其他任务。
  本计划保持 `执行中`；旧排队状态不再代表当前 v1.5 进度。
- v1.5 实现：新增独立 CPU 数据/候选身份审计与整队入口；先审计所有源记录，再等待 GPU 训练。
  明确保存源候选的顺序和内容，只统计二维重复、不静默改写候选池。构图策略 opt-in，旧权重加载
  不继承新策略；新 alignment 保存构建配置、RDKit 版本、数据哈希与逐轮样本计数。
  官方源目标的二维 train/val/test 交集未发现；历史六条验证排除继续保留，不能把它们写成
  本次新发现的交集。完整候选审计、训练、测试结果仍待新队列执行。
- v1.5 专项 synthetic CPU 验证包含真实小模型 alignment 训练/保存/重载、全谱计数、丢尾批拒绝、
  非有限损失拒绝、规范图表示一致性、真实 spawn 进程池预处理及 CPU 审计失败阻断 GPU 阶段。
  这些测试不是正式训练或性能实验；临时工件不进入论文。最终全仓为 211 通过、1 跳过、
  1 个交接时已存在的内部路径审计失败（相同三个文件，未修改 GLACIER 记录）；新增 v1.5 专项
  15 项全部通过。Ruff、compileall、`git diff --check` 通过；固定源码 commit 在队列启动后回填。
- 2026-09-07：正式入口 `run_fulltrain_rerank.py` 及公共检查 `SpecEmbedding/utils/fulltrain.py`
  已提交并推送为 `509fbeb`。固定 detached 源码在 `/data1/zyl/repos/SpecEmbedding-fulltrain-20260907/`；
  已核验输入 train/val/test 为 194,119/19,429/17,556，alignment-42 checkpoint 与 selection 记录的
  SHA-256 相符。来源指纹和全部命令只维护于运行根目录的 `inputs_and_commands.json`。
- 2026-09-07：已启动后台自动等待流程，不需要前台会话存活；独立 tmux server socket 为
  `/tmp/specembedding-fulltrain-20260907-1010/tmux.sock`，训练 session `fulltrain` / pane `%0`，
  监测 session `gpu-monitor` / pane `%1`。监测物理 GPU 1，按 UUID 顺序显露所有 GPU，训练
  显式 `cuda:1`；利用率 ≤10%、空闲 ≥20,000 MiB，30 秒轮询、连续 120 秒，不设最长等待时间。
  初次核验利用率 87%、空闲 1,687 MiB，仍在等待，未发送训练命令；训练 pane 为专用空白 bash。
  从未向原有工作会话派发测试或训练命令。各阶段再次等待，失败停止，不自动重试或覆盖工件。
- 运行根目录：`/data1/zyl/SpecEmbedding/experiments/massspecgym_align42_fulltrain_20260907/`。
  `monitor.log` 是 GPU 等待/派发日志；进入正式入口后生成 `runner.log` 和 `status.json`。
  **SENT 仅表示命令派发，不代表训练启动/完成；新实验数字、三 seed 汇总与论文更新尚未完成。**
  查看监测器：`tmux -S /tmp/specembedding-fulltrain-20260907-1010/tmux.sock attach -t gpu-monitor`。
  如需取消等待，只在监测 session 中 Ctrl+C，不向训练 pane 发送中断。
- 独立 tmux 进程能跨终端/当前任务关闭继续运行，但不保证主机重启后恢复，也不是 GPU 资源调度器。
  低占用不证明设备无人使用，最终复查仍无法消除竞争；源码/输入指纹变化或新证据影响协议时停止核对。
- 2026-09-07：用户明确恢复本计划。GPU/tmux 监测器已实现，30 项专项测试（含独立 tmux
  server 的真实传递测试）通过，提交 `67dadfb` 已推送。将先补齐正式全量入口及校验，再固定
  源码、创建专用空白训练 pane，并由监测器等待派发；不向现有工作 pane 发送测试或训练命令。
  全仓测试发现既有路径审计失败（GLACIER 交接和项目记忆中的内部路径），其余 180 项通过、1 项跳过；
  不以修复该无关测试为由改动既有交接信息。
- 2026-09-07：用户要求继续全量重训；创建 `codex/massspecgym-fulltrain` 分支。
  主机环境已核验 PyTorch/PyG/RDKit/matchms 与 CUDA 可用；两张 RTX 4090 正在高负载，未启动新 GPU 计算。
- 2026-09-07：用户要求计划制定后暂时休息；计划标记为 `暂停`。尚未修改训练代码或配置，
  尚未创建运行源码 worktree、候选缓存、后台等待队列，未启动任何新训练或推理。
  只完成计划文档检查与提交；恢复执行需用户明确指示，不因 GPU 自动空闲而自行启动。

## 11. 最终结果

- 完成日期：尚未完成
- 最终状态：`执行中`；单 baseline 配置与计划已完成，模型优化和 SOTA 达标未完成
- 验证结果：候选监督正式集成、真实全量绑定及CPU验证索引导入预检通过；最新全仓382通过、1跳过、1个既有路径审计失败；静态检查通过
- 当前训练状态：r4旧矩阵停止；A01未替换r4；A02完成审计并晋升；A03完整审计通过但未晋升；A04已单次派发并完成CPU准备，21:13正在GPU0上做baseline完整验证
- 论文修改 commit：尚未提交；本轮不修改论文
- 计划归档 commit：无需在本文件中自我引用
- 相对原计划的偏差：用户已授权从立即跑 12 组矩阵改为单方案持续优化，成熟后再做稳定性与矩阵；
  目标升级为 Top-k/MRR 接近同协议现有 SOTA，预训练模型结构保持不变
