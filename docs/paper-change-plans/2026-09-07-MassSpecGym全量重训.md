# 阶段计划：MassSpecGym 分子检索 Top-k 优化与 SOTA 对标

- 状态：`执行中`
- 创建日期：2026-09-07
- 最后更新：2026-09-10
- 负责人：Codex / 作者核验
- 关联论文：`paper/main.tex` / `paper/main_cn.tex`；目前不改论文结果或结论
- 计划约束：本文件沿用仓库模板与原全量重训计划，作为本工作的唯一计划；不另建冲突版本。
  2026-09-09 用户授权单 baseline、互联网研究和结构/参数迭代，随后把目标更新为
  **提高分子检索 Top-k，MRR 为辅；各项评价指标在 MassSpecGym 上达到或接近现有 SOTA。**
  此授权替代原“固定 alignment、不重训”和“立即完成 12 组矩阵”安排；历史执行记录保留于第 10 节。
  允许参考论文下载至用户指定论文目录；SpecEmbedding 质谱侧预训练模型的结构不得修改。
  用户进一步明确：先使基础检索达到或接近同协议SOTA，再用rerank强化；达到该门槛前不开展
  rerank实验。常规进展只汇报模型改动、检索成绩、速度与问题，不重复汇报是否使用rerank。
  2026-09-10用户进一步固定核心方法：谱图塔和分子塔独立映射到同一嵌入空间，以对比学习
  训练并按相似度检索；可借鉴GLACIER及近期语言模型的注意力、残差等模块，保持本机可训。

## 1. 阶段目标与验收

### 1.1 目标定义

训练可在本机运行的 MS/MS-to-molecule retrieval 模型。主目标是官方结构划分上的
MassSpecGym v1.5 **Mass retrieval Top-1/5/10/20 整体提高，MRR 为辅助指标**。
质量候选主任务不使用真值分子式；Formula-conditioned retrieval 后续单独验证，不能替代 Mass 达标。
可以借鉴或超过复杂方法的成绩，但实现优先选择轻量、可训练、可复核的方案。

“改善现有 baseline”是进展，“达到或接近 SOTA”才是本阶段最终目标；二者不能混写。
也不能通过只与较弱双塔方法比较、改用容易的 split、过滤困难 query 或只报告最好的 k 来达标。
基础检索须先按第1.2节逐项核验同协议成绩，才进入后续rerank强化；单个指标改善或未经审计的
训练中途成绩不能触发切换。不能因目前先优化基础检索而排除可比且更强的多阶段方法参考线。

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
- r4阶段结果为cache-local exact-target-SMILES；数字、权重来源、GPU重编码证据和新旧
  差异维护于[准确率排查](../../analysis/massspecgym_v15_regression_audit.md)及其原始工件。
  A01起的优化采用完整v1.5自然候选、二维身份检索验证，完成结果见[优化索引](../../analysis/massspecgym_optimization.md)。
- Base 排序在进入 reranker 前已经较弱。新缓存并未复用旧 alignment 权重或旧磁盘 TokenSet。
  旧候选格式捷径已量化，但尚不能把旧模型的全部性能差额归因于单一因素。
- r4 alignment 按每轮全部谱图训练，以验证对比损失选择 checkpoint；reranker 目前按验证 MRR
  选择。A01完成后未替换r4，A02首次整体改善；A04在完整25轮训练与独立审计后进一步改善，
  epoch20已成为当前incumbent，A05基于它优化训练batch并应用固定验证图缓存。
  A05随后完成24轮训练及独立审计，全部Pareto候选均未通过改善门槛，继续保留A04；
  A06已确认继承A04的训练设置，输入对照结束后才最终绑定配置和派发。
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
| rerank | 基础检索达到或接近同协议SOTA后，逐项改 residual、候选交互、loss、coarse 策略或融合 | 先通过第1.2节门槛；只重排自然输入候选，不用来源位置/rank 捷径，不 forcing，不重复跑有无 pointwise 矩阵 |
| 工程性能 | 数据加载、图/TokenSet/冻结 embedding 缓存、分块、经过验证的混合精度 | 缓存必须含源码/数据/模型指纹；模型或增强变化不能静默复用过期表征 |

### 3.2 不变的实验条件

- 默认seed42、Mass基础检索，优化胜者成为下一轮incumbent，原始baseline永久保留。
  一次只跑一个候选方案；基础检索达到或接近同协议SOTA后才开展rerank强化，不枚举pointwise开关。
- 核心为独立双塔、共享维度的嵌入和对比学习；候选分子表示可以预编码，最终按嵌入相似度
  排序。塔内自注意力、学习查询池化或残差可优化，但不把跨谱图—候选联合编码或谱图生成
  作为本阶段核心检索器。辅助目标也须保留对比学习主目标，并单独记录其数据来源与权重。
- v1.5 官方 train/val/test 划分，原始数量 194,119 / 19,429 / 17,556；
  固定验证排除 [7686,7687,7688,8464,8465,8466]，有效验证数 19,423。
  不更改划分为 random、Formula split 或其它更容易的拆分。
- 正式训练每轮覆盖全部符合监督协议的 query；无正例单列统计，完整评价保留它们。
  不人为限制训练条数，不用前 N 条成绩代替正式结果。
- 候选使用版本化的官方自然候选，当前最多 256；所有 split 无 forcing。
  无效图、身份重复、同分排序和二维多正例必须显式审计，不静默改变官方分母或候选集合。
- MRR沿用与Top-k相同的CPU argsort首正例排名，分数≤0不删除正例，无正例贡献0并保留分母。
  这是既有实现的明确说明；不等同于TorchMetrics独立MRR函数的topk/正值过滤语义，详见D15。
- 正式训练/编码/推理显式 cuda:N，按用户授权的显存门槛等待、利用率仅记录；默认在物理
  GPU 0/1 中择一，通过 UUID 映射 cuda:0；当前batch128门槛为10,000 MiB/连续120秒。
  不进行两组并行抢卡，不终止其他任务。CPU 仅处理、审计及 synthetic smoke。
- 不修改固定运行 worktree，不覆盖旧 checkpoint/cache/log/status/锁；每次正式实验独立目录。
- 用户已授权失效缓存清理：构造规则、相关来源/版本或缓存实现变化时，新run停用旧缓存并
  构建新版本；确认没有训练、准备或审计进程仍在引用后，清除失效的固定输入载荷，保留来源、
  manifest、审计回执和清理记录。该授权不包含源数据、checkpoint和失败日志；有效缓存可跨轮复用。
- 本阶段不顺带启动 NPLIB1 或 GLACIER 独立复现队列；论文与代码可作为研究资料。
  外部比较如果需要实际运行，先纳入本计划的具体对标条目并检查已有队列，不重复派发。
- 禁止利用测试真值选择结构/参数，或把真值分子式、候选原始位置、SMILES 格式特征当作捷径。
  预训练来源不清或存在污染的权重不能进入“同协议达标”验收，可保留隔离诊断记录。

## 4. 迭代方案

后续结构研究按[研究索引R06](../../analysis/massspecgym_optimization.md#r06双塔框架下借鉴glacier与近期语言模型组件)
执行：先研究GLACIER的结构关系表示、共享节点编码和困难负例思想，以及QK归一化、
注意力输出门控、保持恒等通路的小残差分支；每轮只改变一个主要假设。先完成A06并审计，
再依据其是否晋升确定下一轮parent与实验卡；待选组件不等于已排队模型。全部改动限下游
独立构造版本，SpecEmbedding预训练共享结构不改，参数量/显存/整轮时间须实际记录。

用户要求后续每轮均提前写入`attempts.md`和`attempt_<n>.md`；启动前标注准备状态、唯一
假设、parent绑定条件、参数、缓存影响与验收标准，实际启动/完成后更新同一条目，不预填
收益。当前已登记[A07谱图差值残差](../../attempt_7.md)，A06完整审计后已绑定A04并单次
启动入口，实际状态见第10节；失败尝试的删除线及完整追溯要求继续适用。
下一轮[A08谱图Q/K归一化](../../attempt_8.md)已在A07等待期间预登记并开展实现验证；
parent仍待A07完整审计；随后已单次启动CPU衔接队列，状态和绑定见第10节，等待不代表
模型训练或检索结果已产生。

A08之后的[A09图与固定指纹残差](../../attempt_9.md)也已预登记；它只改变分子表示融合，
parent必须取A08完整审计后保留的唯一incumbent。当前仅组件准备，组合输入与正式审计
接入尚未完成，没有A09运行或队列，不改变A07/A08顺序和固定源码。

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
| E（达标后） | 一种轻量 rerank 改进或校准融合 | 基础检索先通过第1.2节同协议SOTA门槛，再检验候选内残差；逐项改变，保留完整候选上界 |
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
  增补保留每query实测母离子、移除全部碎片的输入对照，避免将母离子信息变化混写为碎片
  贡献；这些冻结模型的输入干预仍不是重新训练的消融，不作独立因果归因。

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
| [MS-MOLE，2602.16507v1](https://arxiv.org/html/2602.16507v1) | 指纹相似度目标与候选检索目标的取舍 | 设计线索；当前官方源码标签按指纹相等，未证明二维检索同协议，详见优化索引R02 |
| [FLARE，bioRxiv 2026.01.27.702086v1](https://pmc.ncbi.nlm.nih.gov/articles/PMC12873900/) | §4.2使用已知母体分子式与加合物构造峰表示 | Mass候选名称不代表无真值分子式输入；保留为额外输入条件下的参考，不直接纳入当前S_j |

**当前不得声称已找到并冻结所有指标的 SOTA。** 特别是 Top-10、MRR 缺少完整同协议对比。
2026-09-10已将GLACIER附录Table 10的Top-10按独立变体明确记录到
[版本化参考快照](../../analysis/sota_reference_snapshot_20260910.json)，不拼到主表checkpoint；
该文件的已报告最大值不是最终S_j。GLACIER的MRR未报告，不能因缺失就默认低于GLMR，
因为相同分母/排名定义下MRR≥Top-1；完整排名与协议证据仍待补齐，详见优化索引R04。

GLMR的[正式出版页](https://ojs.aaai.org/index.php/AAAI/article/view/37132)已核验：AAAI 2026，
40(2):1561–1569，DOI `10.1609/aaai.v40i2.37132`，发布日期2026-03-14；正式PDF已归档。
上述数字与正式版Table 1一致，发表状态不再仅为预印本；其同协议复现状态仍为reported-only，
模型阶段与尚缺的输入/评价证据见优化索引R03。

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
- [x] 旧队列曾收窄为Mass/relative/seed42；当前优化入口运行单seed基础检索，拒绝意外展开矩阵。
- [x] 初步检索一手来源，建立本计划和本地 PDF 资料库。
- [ ] 补齐各指标 SOTA 来源/协议/分母对照，冻结版本化参考向量和缺失项。
- [x] 完成完整验证集官方候选/二维身份 evaluator，并冻结seed42 baseline报告；A01–A03完成回执及
  A02 incumbent见第10节。2026-09-10重验A02/A03回执绑定的117项工件哈希与完成状态一致。
- [x] 实现检索指标选 checkpoint、Pareto 记录及优化分支不跑测试；测试覆盖全量/设备保护，
  已在A01–A03正式运行中核验，后续每次实验继续独立审计。
- [ ] 按第 4 节顺序启动一个实验，持续完成“训练—验证—审计—保留/舍弃—下一假设”闭环。
- [ ] 核验预训练来源，保证其结构保持不变，再决定是否启用权重。
- [ ] 验证指标接近目标后冻结一个方案，在完整测试集评价并完成新缓存/身份审计。
- [ ] 对最终方案进行必要 seed 稳定性复核；逐项判断 Top-k/MRR 与 S_j 的差距，不只报告最好 seed。
- [ ] 基础检索达到或接近同协议SOTA后，再以固定基础模型开展单方案rerank强化并独立评价。
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
- [x] evaluator 测试：二维多正例、候选 permutation、同分、无正例、无效图、完整分母；
  Top-k与独立TorchMetrics Hit Rate核对，MRR按共同排名定义并显式测试独立RR函数的差异，见D15。
  CPU合成检查不作为正式实验，完整真实分数的MRR敏感性与独立读回已完成，范围见D15。
- [x] checkpoint 选优与恢复测试：保存指标/配置/权重一致，完整 train/val 数量与实际每轮一致；
  A02/A03完整27轮回执与真实工件重验通过，后续每次实验仍须独立完成审计。
- [ ] 预训练结构与来源：新旧结构签名、严格 state_dict 加载、tokenizer 语义、身份污染审计。
- [ ] 正式运行：独立干净源码、配置/数据/SHA 固定、CUDA UUID、GPU 门槛、阶段完成标志与进程一致。
- [ ] 最终结果：完整 Top-k/MRR、可比 S_j、每项差距、单 seed/均值/SD、完整审计，不遗漏失败项。
- [ ] 论文修改后执行双语编译、引用/匿名扫描、发布 manifest 和仓库要求的复现检查。

## 10. 执行记录

- 2026-09-10：A07训练、A08等待期间预登记A09，并新增独立组件
  `SpecEmbedding/models_graph_fingerprint.py`、未激活的`graph_fingerprint_residual`配置和
  [说明](../graph_fingerprint_encoder_zh.md)。保留GINE及两塔/投影/对比路径，在分子表示上
  加固定Morgan特征残差；真实条件构造增加557,952参数，完整条件数集中于实验卡，
  不是parent绑定或GPU成本测量。22项合成检查通过，原优化器覆盖新参数；预训练
  `SpecEmbedding/models.py` SHA仍为`f34af906ff4f3f3db3beb1583cfd57ab4c020bb249087a2b8ff60fff513e8c32`。
  新类型尚未接通组合输入/正式审计，正式构造器继续拒绝，不派发A09。
  主机环境全仓612通过、1跳过、1个相同既有路径审计失败，Ruff、compileall和diff检查
  通过。首次沙箱运行在原有DataLoader多进程共享socket处被权限拒绝，124通过后中断；
  仅向核验过的本轮pytest PID3614237/starttime1451723370发送SIGINT，随后按原测试
  重跑，未跳过多进程检查或修改训练。中断操作首轮自动审批因服务连接中断被拒，补充
  只读进程归属核验后重试成功，未向A07/A08或其它进程发送信号。
  R07原文来源已核验并归档到`/data1/zyl/papers/`：
  `2205.03834v1_FP-GNN.pdf`为1,194,072字节，SHA为
  `0a3eb52c63610e3460a795f3e5675974b91a65c2841bd29b785477a658f72c55`；
  `1904.01561v5_Analyzing_Learned_Molecular_Representations.pdf`为2,359,573字节，SHA为
  `f95ac4b49d196bc1b63e00e75fa7dff4709bce1d6ed0967f8d6a496d35a23447`。
  相邻JSON保存确切arXiv版本、URL、出版来源与标题核验；后一PDF的arXiv水印在文本
  提取时插入两行标题之间，去掉精确水印行后完成核验，未修改PDF。没有执行外部代码
  或权重，两文的分子性质预测结果不当作本地检索证据。计划仍执行中，论文未修改。
  16:45再次核验A07原入口/训练/审计及A08等待进程均存活，A07前9轮完整计数及第7–9轮
  保存query顺序/排名重算通过，当前日志选优第9轮。原固定源码工作树均保持干净，
  具体中途指标只维护于A07实验卡；A04仍为incumbent，没有本轮完成审计或晋升结论。
- 2026-09-10 16:10：A08专用CPU衔接队列已于16:08:34单次启动，目录为
  `/data1/zyl/SpecEmbedding/audits/optimization_a08_successor_queue_20260910_r2`，
  socket为`/tmp/specembedding-opt-a08-20260910-1010/tmux.sock`，session `opt` / pane `%0`，
  PID3609809/starttime1451589300。独立读回实际命令、原PID/starttime、pane存活、完整外部
  存储环境及空CUDA可见列表均通过；A07原入口3590868、训练3596520、审计3592109及
  starttime均匹配且存活，没有A08最终绑定或模型目录，当前仅`waiting_parent`。
  `preparation_receipt.json` SHA为`3287caa5dcc6529c09ee6f5ba4d6ba27b624bede085d1799346bc6190b207a56`，
  `manifest.json` SHA为`5c8a67f6b2d8e5904da43feec653c2b721aaac18692ec0107ec80e8dbd8ea5d3`，
  `successor.py` SHA为`fcea0b2cf4832df037ceeb9cb2911316473145d62848341b2ea2c022d74b9523`，
  `runtime_verification.json` SHA为`c079af3d5bfac670300cf07a64651f455efba65702d8900da72ce042551ec257`。
  19项合成检查通过，含真正隔离的tmux传参/环境及拒绝重复启动；另有真实CPU等待探测、
  Ruff和归档文件指纹读回。首次准备目录`optimization_a08_successor_queue_20260910`因
  `queue.py`遮蔽标准库而在PyTorch导入时失败，尚未启动队列或模型；改名后新建r2，
  `previous_preparation.json`保留旧日志SHA与原因，不另计模型attempt，不修改A07固定源码。
  新队列要求A07三原进程结束、全阶段及独立审计完成、全部工件来源一致，再按预登记
  规则绑定parent并执行一次最终CPU预检；成功后同一PID exec正式入口，继续既定GPU0/1
  门槛。失败、歧义、已有目录或来源变化保留现场并停止，不重试。`runner_exec_requested`
  仅代表请求执行，正式训练状态须结合运行目录、阶段日志和进程核验；A08自己的独立
  完成审计仍须在实际运行来源生成后绑定，不能继承A07审计。本项不含后续其它模型。
  A07已于16:03:39完成第6轮，新增第5/6轮全量query顺序、计数及保存排名重算通过，
  早期结果/资源维护于实验卡。当前日志选优第5轮，正式selection需训练结束才生成，
  尚无本轮完成审计或晋升决定，A04仍为incumbent；阶段目标未完成，论文未修改。
- 2026-09-10 15:48：已归档A08仅CPU的最终parent绑定入口，目录为
  `/data1/zyl/SpecEmbedding/audits/optimization_a08_finalization_preparation_20260910`，
  `receipt.json` SHA为`662d0de1957fa21674d4eab5129f840db8dcc6ffc9ef5d0b14a88653c3cbb8a0`，
  `finalize.py` SHA为`c0986e33d3673cad38b0ea56c670c741583b2e7f9cb8b02e89383ab44ba62426`。
  21项合成检查和Ruff通过；归档包含测试、日志及实际只读等待观测，全部文件SHA读回通过。
  入口绑定原A07入口/训练/完成审计PID及starttime，完整终态、审计和全部工件SHA通过后
  才确定唯一parent；支持保留A04或晋升A07，歧义、失败和额外协议/命令变化均拒绝。
  它只运行CPU最终dry-run，尚未部署自动GPU衔接队列；真实检查确认三原进程仍存活，
  没有创建A08最终绑定或正式运行目录，不自动重试或覆盖工件。A07已完成前4轮全量
  训练与验证，第3/4轮完整query顺序和保存排名重算通过，中途结果集中于A07实验卡。
  同时完成D21已保存排名诊断，目录为
  `/data1/zyl/SpecEmbedding/audits/optimization_a06_rank_overlap_20260910`，
  `receipt.json` SHA为`38e8efda4f5bb0e1e768f8c1dcc2c737561bf042ec601681962dd011f949d79c`，
  `verification.json` SHA为`bbf405cc65cd1cf7c8ea02087bec36592475aeed361735b4b02bf0bed67ea7fa`。
  全部19,423条CSV与两份原始排名独立读回一致，标准库重算交集/配对排名及完整指标通过；
  没有新推理、融合权重拟合或test使用。交集数及解释边界仅维护于优化索引D21，不据此
  晋升A06或改变A08主要假设。计划保持执行中，本次未修改论文或重启正在运行的任务。
- 2026-09-10：A08组件与预登记提交`9c2839e`已推送，固定源码为
  `/data1/zyl/repos/SpecEmbedding-opt-a08-20260910`，干净detached HEAD。A04条件全量CPU
  预检于15:06:51通过，目录为`/data1/zyl/SpecEmbedding/audits/optimization_a08_conditional_a04_20260910`，
  `receipt.json` SHA为`23fdfbe4ba6ddb8ce90d52153e240b7da095baa4f5ee854d427ca8926be69b9c`，
  `preflight.json` SHA为`af83545023f5d54fa80ff3e38ea47a5d9b756beab2492857c23b0e2718ee3a8d`。
  完整输入、独立A04构造、全量计数及固定缓存来源通过；没有正式A08运行目录或GPU派发。
  A07仍需完整审计后才能决定是否保留差值分支并绑定最终parent，不能沿用条件分支作决定。
  A07原入口于14:48:33通过GPU0准入，14:49:22完成baseline完整重编码，14:51:48再次通过
  持续窗口和最终复查后开始alignment。15:13只读核验训练PID3596520/starttime1451128642，
  物理GPU0 UUID与status一致，映射显式cuda:0、严格CUDA标志及本轮runtime路径均正确；
  nvidia-smi也记录该PID位于所选UUID。原入口与CPU审计PID及两固定源码均核验无异常。
  前2轮全量train/val计数及保存指标已生成，完整中途Top-k/MRR与资源只维护于
  [A07实验卡](../../attempt_7.md)，没有最终成绩或晋升决定，不启动A08。
- 2026-09-10：14:35重新核验A07原入口PID3590868与CPU完成审计PID3592109及两专用pane
  存活，仍在等待显存准入，保持原队列。等待期间按R06预登记A08，仅添加谱图注意力
  每头特征维Q/K RMSNorm，完整继承最终parent的其余模型、训练与输入；实现见
  `SpecEmbedding/models_qk_norm.py`及[组件说明](../qk_norm_encoder_zh.md)。预训练共享类
  `SpecEmbedding/models.py`哈希保持不变，训练/eval/inference_mode均显式执行新路径，
  防止原生融合层绕过新增归一化；同seed原两塔权重与RNG保持，新参数使用常量初始化。
  125项相关CPU合成检查通过，包括实际训练入口两轮全量合成样本、候选重放及严格重载；
  全仓590通过、1跳过、1个相同既有路径审计失败，Ruff、compileall、diff检查通过。
  真实A04及含A07差值的模型构造可用，但不将未完成A07选为parent；固定源码、全量输入
  预检和最终绑定仍待，未创建第二个训练队列。参数量及模型范围集中于[A08实验卡](../../attempt_8.md)。
  参考原文再次核验：Qwen Team，*Qwen3 Technical Report*，arXiv:2505.09388v1，2025-05-14；
  第2节提出QK-Norm用于其训练稳定性，不证明本地质谱收益。原PDF已归档至
  `/data1/zyl/papers/2505.09388v1_Qwen3_Technical_Report.pdf`，779,424字节，SHA为
  `84a5e2b1fa04bb774bf12ae606d1e6d9dd2147ed2ebe78a4cfeb91ba380ffdd5`；相邻JSON记录来源
  `https://arxiv.org/pdf/2505.09388v1`、下载时间和PDF首屏标题核验，未下载权重或执行外部代码。
- 2026-09-10：A06 r3在第9轮正常早停，原入口于14:02:08完成全量检查，独立CPU审计队列
  于14:08:07完成全部保存排名、负例重放和严格权重重载；原入口退出0、原审计进程结束。
  完成审计目录为`/data1/zyl/SpecEmbedding/audits/optimization_a06_r3_20260910`，
  `receipt.json` SHA为`83ca88d066325bc5e425ffc2a0dbfa7e06ee5377243a8ed7dc95176337533b66`，
  `verification.json` SHA为`62dc1a97879802c923234691ae43948d4c1352390984251fafb155a128c6b951`，
  74项工件已重核。选中epoch4且为唯一Pareto候选，所有指标均低于A04，保留A04；完整指标与
  9轮资源统计仅维护于[A06实验卡](../../attempt_6.md)，未使用test作决定，attempts已加删除线。
  A07采用此前已登记的谱图差值残差，固定源码
  `/data1/zyl/repos/SpecEmbedding-opt-a07-20260910`，commit`36e1b0c`；实际固定源码全仓
  561通过、1跳过、1个相同既有路径审计失败。A04条件的全量预检于14:00:02完成，目录为
  `/data1/zyl/SpecEmbedding/audits/optimization_a07_conditional_a04_20260910`，
  `receipt.json` SHA为`698ebbc3c7b25957967909566df0c950697f9fd23f1f41b97718ab8579aecee6`。
  A06完成后另建`/data1/zyl/SpecEmbedding/audits/optimization_a07_finalization_20260910`，
  重核209项前序与条件工件、原进程终态及完整Pareto规则后绑定A04；`decision.json` SHA为
  `b9744c0e5642c7641e067c3bec9edabf5df0c94db6c88fd2112efddb4c7f1a30`。
  最终真实输入预检与条件预检仅配置路径不同，`preflight.json` SHA为
  `efe68c6177e0098631d898ec196be6e27123f4e945b7e787d9c80b8f0dd5c124`，
  `ready.json` SHA为`4d505002d38f3b8d5aebcc1a30b00037905b634a09ea21b8403efe41437cfa12`。
  14:20:38单次派发至新运行
  `/data1/zyl/SpecEmbedding/experiments/massspecgym_v15_opt_a07_20260910_topk256`，专用socket
  `/tmp/specembedding-opt-a07-20260910-1010/tmux.sock`，session`opt`/pane`%0`；原入口
  PID3590868/starttime1450941701。14:22:33完成数据与验证索引导入并进入双GPU等待，
  采用10,000 MiB/不限利用率、30秒轮询、连续120秒及最终复查。启动回执位于
  `/data1/zyl/SpecEmbedding/audits/optimization_a07_launch_20260910`，其中实际输入/配置/环境
  核验`runtime_verification.json` SHA为`6e74113c77767e1b49d8e65a53152cb968cdcb9a79e9a483dcf4a876a2caf9da`，
  实际runtime SHA为`37e6796605128cf07baac90c2ab0f4c664d50e8622b7862fc3f8243868ff2203`。
  独立完成审计准备目录为
  `/data1/zyl/SpecEmbedding/audits/optimization_a07_completion_queue_20260910`，13项生命周期
  检查通过；实际未完成审计探测退出1且未产生结果，23项固定输入指纹匹配。
  `manifest.json` SHA为`44238157ec51f238bcd697a482d92799317fb86262bd1010e4f1336b35797bc6`，
  `runner.py` SHA为`334bdfe5e61058c285e0e86ff927be125048f3fe299179cdcbf66c2f5be52694`。
  14:29:51在`/tmp/specembedding-a07-completion-20260910-1010/tmux.sock`单次启动，session
  `audit`/pane`%0`，PID3592109/starttime1450996996；`launch.json` SHA为
  `57f48c1ce452ae6cad0073c25d2d79bf51a0095b59c1b56796a7382fbedaa93f`。
  14:31核验两个原进程及专用pane存活，审计进程CUDA不可见、绑定正确runtime；仍为
  `waiting_parent`且`audit_started=false`，训练入口仍等待GPU。完成后预定输出为
  `/data1/zyl/SpecEmbedding/audits/optimization_a07_20260910/receipt.json`，尚未生成。
  不修改已运行源码，不重复创建GPU监测器；没有本轮模型成绩、晋升或论文更新。
- 2026-09-10：A06 r3已实测完成首轮全量训练194,119条与验证19,423条，整轮186.71秒；
  PyTorch峰值分配1,721,496,064字节、预留2,044,723,200字节（约1.60/1.90 GiB），资源文件
  位于当前运行的`alignment42_topk256/resources/stage2_epoch001.json`。这是当前单轮测量，
  不代表完整轨迹峰值、独占GPU吞吐或新模型晋升。训练期间已登记A07为下一轮候选，
  唯一改动为已实现的有符号母离子差值残差；parent须等A06完整审计后绑定，模型尚未派发。
  新增衔接公共检查：前序真实进程仍存活时等待；失败/不完整证据拒绝；重算全部Pareto
  候选的既有晋升规则并拒绝含糊选择；新配置仅加入指定下游特征和已授权GPU准入设置。
  新衔接检查与差值组件共38项合成CPU测试通过，尚不代表完整自动衔接入口已经部署。
  随后核验真实A04默认缓存/输出仍指向旧源码worktree；新配置构造增加显式外部存储模板
  改绑并拒绝越界路径，模型、训练、增强和tokenizer仍完整继承。扩展相关检查为70项通过，
  真实A05已审计四个Pareto候选经新决策检查仍保留A04；A04/A06条件模型CPU构造分别为
  9,121,481/9,932,673参数，未将运行中的A06选为parent，也未派发下一轮模型。
  用户要求将这种预登记方式用于以后所有实验，已写入AGENTS和本计划，准备与成绩分开标注。
- 2026-09-10：用户明确授权修改GPU门槛、忽略利用率且减少显存余量。当前batch128使用
  `min_free_mib=10000`、`max_utilization=100`；30秒轮询、连续120秒及派发前复查保持。
  A04全部25轮的PyTorch分配/预留峰值为3.67/8.23 GiB，A05 batch256为6.22/15.01 GiB；
  两列不相加，也不覆盖CUDA上下文等外部分配。新门槛比A04预留峰值多1.53 GiB，A06实际
  峰值尚未测量，改变模型/batch后重新评估；允许与高利用率任务共享GPU，不代表算力独占。
  13:12:06核验PID/starttime、专用pane和无GPU子进程后，停止r2原等待及CPU完成审计等待；
  原入口记录`KeyboardInterrupt()`，未创建baseline或训练输出，所有旧工件保留。停止回执位于
  `/data1/zyl/SpecEmbedding/audits/optimization_a06_resource_gate_change_20260910/stop_receipt.json`。
  新运行根为`/data1/zyl/SpecEmbedding/experiments/massspecgym_v15_opt_a06_20260910_r3_topk256`，
  继续使用干净固定源码`/data1/zyl/repos/SpecEmbedding-opt-a06-20260910-r2`及commit`65eecfc`，
  不改旧配置、不绕过已有工件保护。准备目录
  `/data1/zyl/SpecEmbedding/audits/optimization_a06_r3_preparation_20260910`保存完整CPU预检，
  逐项比较仅允许资源准入两参数和输出/配置路径改变，模型、训练设置、seed及数据/缓存不变；
  `ready.json`SHA为`272868cb0d2a57a6ceaca126108351e762d6a7b42ba0e056d9fdd259c434f51e`，
  `preflight.json`SHA为`adfa5639bd42ae3a1ab17576613aae8d27701cee139eef531b8e603735d0ee8c`。
  13:16:08经启动检查单次派发，专用socket`/tmp/specembedding-opt-a06-r3-20260910-1010/tmux.sock`，
  session`opt`/pane`%0`，初始PID3569147；启动回执在
  `/data1/zyl/SpecEmbedding/audits/optimization_a06_r3_launch_20260910`，派发不代表训练已启动。
  33项GPU池/正式队列测试通过，额外合成检查确认100%利用率、10,000 MiB可在120秒后通过，
  9,999 MiB拒绝且最终复查保留；Ruff及diff检查通过。本次不增加模型attempt，也无论文修改。
  实际入口PID3569147/starttime1450554637及外部存储环境、五份导入文件指纹、完整计数已
  独立核验；启动目录内`runtime_verification.json`SHA为
  `390f8f2062c254631295aaf08be5621126f149b2f9dd796a2f1b97a8032f20b9`，实际runtime SHA为
  `f65f7198dd5c1ef2afb507e6fd5f8e52708421f0035ea47ef695d5e6616b6ef7`。13:22:20最终复查物理
  GPU1空闲15,357 MiB、利用率88%后启动基线验证，13:23:24完成全部19,423条query和
  827,600个分子的重编码。13:26:24训练阶段再次通过独立等待和复查（利用率86%），实际
  子进程PID3570617、UUID为GPU1、严格cuda:0与新runtime配置均已核验；尚无新模型结果。
  13:21:23已单次启动专用完成审计队列，目录
  `/data1/zyl/SpecEmbedding/audits/optimization_a06_r3_completion_queue_20260910`；manifest SHA
  `6805768d927165485e9346aab56c3309134ed1e638b2cd0de28db5a952dca721`，launch SHA
  `2a9e3de86bedc672e341f663aedff20e08de536bf636736793d5eae0e7e03df3`。新绑定的13项生命周期
  检查通过，29项固定输入复核及真实未完成审计拒绝探针通过；专用socket
  `/tmp/specembedding-a06-r3-completion-20260910-1010/tmux.sock`，session`audit`/pane`%0`，
  原PID3569861/starttime1450586206已独立核验存活、纯CPU且绑定新runtime，状态为
  `waiting_parent/audit_started=false`；只在新入口成功退出且全部阶段完成后审计一次。
  未来审计输出为`/data1/zyl/SpecEmbedding/audits/optimization_a06_r3_20260910`，不自动晋升。
- 2026-09-10：准备A06完成审计时发现固定`d6226ef`入口在指纹模型训练结束后，会错误要求
  其`validation_graph_cache`等于GINE比较基线的图缓存；真实manifest的模型类型为fingerprint、
  baseline使用图缓存，而训练配置明确使用指纹缓存，故该检查会误报失败。尚无GPU阶段输出，
  核验原PID/starttime、等待阶段及目录后仅向原入口发送SIGINT，保留旧运行/锁/源码。
  准备以合成阶段执行路径复现并修复此检查，再用原A06源码加最小修复创建新固定版本及运行
  目录；模型、数据、候选、seed和训练超参数不变。刚准备的旧A06完成审计队列未启动，
  不向其派发，也不把预检或主动中断记为模型失败。此为执行错误修复，不改变实验协议或范围。
  原队列已于12:24:34记录`KeyboardInterrupt()`并退出，原PID3551413为Z，专用pane为dead；
  合成回归在修复前准确暴露指纹正常缓存被拒绝、错误图缓存被接受，修复后四种GINE/指纹
  路径符合预期。相关43项检查通过，修复`2e9ea2a`已推送。新的固定源码为
  `/data1/zyl/repos/SpecEmbedding-opt-a06-20260910-r2`，commit
  `65eecfce9c1522b8e86e6e92bb345d2c652799d0`，相对原d622仅结束检查及回归测试两文件不同；
  该实际运行版本全仓505通过、1跳过、1个相同既有路径审计失败，静态检查通过，工作树干净。
  新运行根为`/data1/zyl/SpecEmbedding/experiments/massspecgym_v15_opt_a06_20260910_r2_topk256`；
  完整CPU预检及原A06的`runtime_config`逐项比较通过，只允许源码commit和源码/配置/输出路径
  改变，未带入差值组件、改变参数或重建有效输入缓存。新准备目录
  `/data1/zyl/SpecEmbedding/audits/optimization_a06_r2_preparation_20260910`的`ready.json`SHA-256
  为`f53f21607a21ca7eb006e54fc25281ee804ced91aa712a93dffa7adad1d3696a`，`preflight.json`为
  `7fdadadecc9894a19f4f7875c73c63f6696dfc3863c776afe1c0cbd0fc2016db`。单次启动记录位于
  `/data1/zyl/SpecEmbedding/audits/optimization_a06_r2_launch_20260910`；12:36:24派发专用socket
  `/tmp/specembedding-opt-a06-r2-20260910-1010/tmux.sock`，session`opt`/pane`%0`，PID3563758、
  starttime1450316244。12:39:36已完成CPU导入并等待基线验证GPU；实际输入清单与新preflight
  逐字节一致，五份导入文件、完整计数及原PID/外部存储环境独立核验通过。启动目录内
  `runtime_verification.json`SHA-256为`49d567cf0fc171b444bb8999477a2900b03b122bafb3ac073214899f508a69ac`，
  runtime配置SHA仍为`29376c32ec777447d01f2347b50f37004ed08b803b409ea60ec5ece514ff2c34`；
  原worktree、失败前日志及单次派发锁均保留。暂无本轮GPU训练、模型结果或晋升决定。
  12:46:01另单次派发r2专用CPU完成审计队列，目录
  `/data1/zyl/SpecEmbedding/audits/optimization_a06_r2_completion_queue_20260910`；manifest SHA-256
  `b03e3ec25a428846d4b53a064c92cc5470d2b7d830efeddb06af4ae08ae28961`，runner SHA-256
  `4d7ac0a42656ea5c2b8394daf6573cf7702b033004d79b47faa549165af82f63`，launch SHA-256
  `b082c2cf470d6326816aebfee9e2bb5dcc5db36cfbd526e9e661c6f10118cf5e`。13项生命周期合成检查
  通过，29项固定输入重新核验，真实运行中审计入口退出1并明确拒绝不完整结果，未创建输出。
  专用socket为`/tmp/specembedding-a06-r2-completion-20260910-1010/tmux.sock`，session`audit`；
  原PID3565443、starttime1450373960于12:48:06独立核验存活、CUDA不可见、外部存储及
  `waiting_parent/audit_started=false`，回执`runtime_verification.json`SHA-256为
  `c4908afbc500601fe232712b759600ced75950da7c5cfe71918667f68b28068c`。它只在绑定的A06入口
  退出且全部阶段完成后运行一次CPU排名/全量计数/固定输入审计和严格权重读回；失败停留，
  不重试、不启动训练、不自动晋升。未来输出为`/data1/zyl/SpecEmbedding/audits/optimization_a06_r2_20260910`。
  旧`optimization_a06_completion_queue_20260910`准备目录没有派发，保留且不再使用。
- 2026-09-10：11:47只读核验A06原PID3551413仍存活且实际命令不变，继续等待基线验证GPU。
  利用等待时间准备独立下游谱图差值分支：保留有符号`precursor_mz - fragment_mz`，用小型
  Fourier/MLP残差加入峰表示，母离子角色与padding显式处理；不修改SpecEmbedding预训练
  共享类、现有tokenizer或增强。D18输入审计支持其输入范围，D17仅支持继续研究谱图信息，
  均不证明该结构有效。先完成模型构造、严格重载和合成训练验证；具体下一轮parent及是否
  使用此分支仍待A06完整结果，不提前绑定A07或启动第二个模型队列。
  随后已完成独立`models_precursor_delta.py`、GINE/指纹正式训练与严格重载接入及默认关闭的
  集中配置，设计见优化索引D20。CPU构造核对新增74,496个参数；原`models.py`SHA-256仍为
  `f34af906ff4f3f3db3beb1583cfd57ab4c020bb249087a2b8ff60fff513e8c32`，A06固定源码仍干净。
  主机合成CPU全仓测试539通过、1跳过、1个既有内部路径审计失败；失败涉及的四份历史记录
  在改动前HEAD已含对应路径，未修改其内容来绕过检查。Ruff、compileall、diff检查通过。
  先前沙箱专项检查96通过后因PyTorch张量共享的本地socket被禁止而出现worker超时；只中断
  该已核验的测试父进程获取栈，随后主机完整检查通过相应多进程测试，未改训练实现规避限制。
  12:07原A06 PID3551413仍存活并等待GPU，尚无模型阶段输出。组件准备不新增attempt成绩，
  本计划保持执行中，模型收益、GPU成本及后续正式预检仍待真实实验。
- 2026-09-10：A04输入对照的其余三个视图依次通过物理GPU0原空闲门槛，均使用严格cuda:0；
  四视图于10:59:02全部完成，原PID3479867及各子进程退出0，未重启队列。状态SHA-256
  `0c157ab68f5e7302c1d9170109da8e76e259caaa0660f8d48881d119aba5c054`；全部19,423条query的
  四份保存分数、主/稳定排名、指标及控制元数据再次独立复核通过，完整结果集中于优化索引D17。
  `/data1/zyl/SpecEmbedding/audits/spectrum_controls_a04_review_20260910/review.json` SHA-256
  `cf596e7fb818c609dc2f8d8ee4e77d236f777a63086e4e2c521a7d36694a00cf`，相邻`decision_a06.json`
  SHA-256 `d826f28535ac9b4fb9f4646b783421a9f1496bbbc791bfe182624759265549c6`；相邻`review.py`
  保留可重跑检查。正常输入优于干预输入，但常量/母离子视图也保留命中，不能将全部成绩
  归因于碎片或视为重训消融；该结果不改变A06假设、A04 parent及完整评价协议。
  原最终绑定入口的`--check-only`确认所有依赖完成及退出0；显式A04决定随后通过完整CPU
  dry-run，11:12:42生成`/data1/zyl/SpecEmbedding/audits/optimization_a06_finalization_20260910/ready.json`，
  SHA-256 `13ae34ddffabf6e23998679dff8f4c2171658110ff45289c5e55d416b943c4c7`。7份最终文件
  独立读回一致；`preflight.json` SHA-256 `62ece9b559f4cc06f1bbdd8010a6420c6fe3f6e18ef276668a530a85b09a2a7d`，
  配置SHA仍为`8f96d043b8c46c9c43652a8aada2f8363537abf6c77806619a1b260463f76dec`，固定源码仍是
  干净`d6226efb8492ce5595a953bc720ecc741c4f4867`，未修改在途worktree或模型架构方案。
  11:16:29以明确argv单次启动A06，独立socket为`/tmp/specembedding-opt-a06-20260910-1010/tmux.sock`，
  session`opt`/pane`%0`，PID3551413、starttime1449836778；启动脚本、排他`dispatch.json`和
  `launch.json`及stdout日志位于`/data1/zyl/SpecEmbedding/audits/optimization_a06_launch_20260910/`。
  运行根为`/data1/zyl/SpecEmbedding/experiments/massspecgym_v15_opt_a06_20260910_topk256/`；
  11:17:39实际进入CPU导入，11:19:42完成数据与验证索引。完整划分、有效验证和五项导入
  文件SHA通过，实际`inputs_and_commands.json`与最终preflight逐字节相同；运行配置SHA-256
  `29376c32ec777447d01f2347b50f37004ed08b803b409ea60ec5ece514ff2c34`。
  11:25:58独立核验原PID、真实命令、固定源码、配置及外部缓存/临时目录，启动审计目录中的
  `runtime_verification.json` SHA-256 `c03ac09e1dc5b32340f900960c00cccd87b8b0908f27fda783a5746314fafc0a`；
  相邻`verify_start.py`保存检查。实际CPU父进程的CUDA为空，等待GPU0/1原门槛，当前
  `baseline_validation/waiting_gpu`，尚未启动本轮模型训练。脚本Ruff/语法与只读启动检查通过；
  GPU设备、首轮实际覆盖/耗时和最终模型成绩仍待后续阶段核验。
- 2026-09-10：D19外部参考query覆盖审计于10:56:30完成，11:01:05独立pandas完整重读复核
  通过，两个CPU进程退出0。只读取已下载的两个Formula候选表和官方test元数据，不读取
  模型预测、不修改GLACIER独立复现或本地训练。工件根为
  `/data1/zyl/SpecEmbedding/audits/reference_query_coverage_20260910/`，`receipt.json` SHA-256
  `4e323563793acde438b44cdd2ff679950e7bc316104a2661b596afa4d3721030`，`verification.json` SHA-256
  `79bea8658d3de791adfd9c967f8af489564e25136191f555d692632fcb62b4f0`；相邻两份脚本及完整逐query表
  保留，4项输入/2项输出重验通过。两文件的query集合精确对应官方test仪器类型非空记录，
  具体覆盖数及边界集中于优化索引D19；它不是上游筛选因果证据，不能继承给缺失Mass包或
  论文Table 1分母，S_j和同协议资格仍待核验。原文及已固定上游源码只读检查未补齐这些绑定；
  未修复或运行外部文件，未依据外部test元数据调整模型。相关脚本Ruff/语法和文档检查通过。
- 2026-09-10：D18完整train/val母离子—碎片差值CPU输入审计于10:12:35完成，原PID3543224
  （starttime1449447742）已退出0；未增强的token与D14完整指纹一致，val逐项匹配固定索引，
  没有读取test或执行模型。独立复核因首次权限审核超时未启动，按工具允许重试一次后于
  10:26:32完成，9项输入、4项输出、完整query顺序和计数通过；未重启原数据审计或训练。
  工件根`/data1/zyl/SpecEmbedding/audits/precursor_delta_inputs_20260910/`，`receipt.json`
  SHA-256 `f14f30a4931f03ff60a3d8bca5c37f62d59b8aa22d1082deff22fa75db815057`，
  `verification.json` SHA-256 `d46d38cfa2191d4b2a0634496d9abf86802033486e738eef00e578ca3a706c0e`；
  相邻`audit.py`、`verify.py`和完整train/val逐query数组均保留。准备与原日志在
  `/data1/zyl/SpecEmbedding/audits/precursor_delta_preparation_20260910/`，专用socket为
  `/tmp/specembedding-precursor-delta-20260910-1010/tmux.sock`；该一次性CPU任务已结束，不重发。
  输入统计和设计限制仅维护于优化索引D18，尚未修改模型/缓存或指定A07。
  R05另归档DreaMS原文至`/data1/zyl/papers/DreaMS_s41587-025-02663-3.pdf`，SHA-256
  `84f8d3ff8df42b1709e201ce335fcf6f60e4479ed056c15ecbda9ba26c914e05`；官方源码子集位于
  `/data1/zyl/repos/DreaMS/`，绑定commit `dbec3a0b514a99e5056cfccde4559fda8cfe8129`，不是完整
  clone或论文运行版本绑定。下载manifest为`/data1/zyl/papers/specembedding_dreams_20260910_manifest.json`，
  SHA-256 `b1ff2e7027309faef2e8ffd7d4a33f72199d989773655fd7535c8c1ca79b9634`；PDF首页、PDF及
  三项源码/许可证指纹独立读回通过。编码差的线性偏置推断经合成数组核验，范围见R05；
  未执行外部模型、下载权重或更改SpecEmbedding预训练架构。10:24:41宿主机核验原对照
  PID3479867及指定tmux pane存活，`permuted/waiting_gpu`；GPU0/1利用率89%/83%、空闲
  显存6381/10431MiB，均未满足门槛。A06最终目录及模型运行目录仍不存在，未重复派发。
  10:30:44实际执行既有A06入口`--check-only`再次确认run/audit complete、controls running、
  原对照PID/starttime匹配及`dependencies_pending`，没有创建最终准备或运行目录。审计及
  复核脚本Ruff/语法、文档diff检查通过；此次只归档输入证据，不重跑全仓或宣称新模型验证。
- 2026-09-10：A05原入口于09:32:15完成全量24轮负例重放，四阶段均complete，原PID
  退出0；`status.json` SHA-256 `14e0df29a36ec1de152525ca81ce7c7427fde6cd9325425c2a1b0165c81cd04d`。
  既有独立完成审计队列于09:32:26单次启动PID3536765，实际进程的CUDA为空、OMP/MKL均为1、
  配置及缓存/临时目录均为外部存储；09:43:55完成且子进程退出0，原审计队列随后结束。
  `/data1/zyl/SpecEmbedding/audits/optimization_a05_20260910/receipt.json` SHA-256
  `e55b2a20c53d4f331ad10e420ea47f6a4173ad62f439fa2fea6be38edbcd5d45`，
  相邻`verification.json` SHA-256
  `f561fdc8831766a1a833cfb1234503de7f2a5ef2c98c3721b98b0b3ef8199b54`；完整24轮负例重放、
  baseline及各轮保存分数/排名、早停/Pareto/选中权重均通过，131项工件重验、严格CPU构造
  及重载通过，没有执行模型forward或test。09:49:56再次核验131项工件和审计源码，逐项
  比较选中epoch19及Pareto14/15/19/23后确认全部指标均退步且超过既定容差，决定保留A04。
  同目录`incumbent_decision.json` SHA-256
  `276abab7fa2da7014ea10ba97d5e18d882f2fb8ac204bbc42354955175bd88cc`，
  `review_decision.py` SHA-256 `67b0932efba2620b51ada0554737e23072f6404a98ef3132a0fb3b6fb0bbb8b9`；
  回执/脚本独立读回、Ruff/语法及文档检查通过。A05已加删除线，完整比较只维护于优化索引A05。
  A06确认继承A04的batch128、学习率1e-4及候选CE16/权重1，模型假设不变；真实
  `--check-only`返回run/audit complete、controls running及`dependencies_pending`，
  未创建最终准备或模型目录。09:51原输入对照队列仍等待GPU，最终配置绑定/派发未完成。
- 2026-09-10：A05于09:21:43在第24轮按既定patience5正常早停，09:21:45保存最终工件；
  原训练PID3455061已退出，原入口PID3454081继续逐轮重放全量自然负例，独立完成审计
  PID3491217仍等待入口结束。读回selection为best_epoch19/stop_epoch24，Pareto轮次
  14/15/19/23；24份资源记录均为train194,119/val19,423，耗时范围见A05实验卡。
  `alignment42_topk256/alignment_selection.json` SHA-256
  `b34d10dce7c3c264ab4d34dfa596b65fb85e0d378cf45274a359e46e14299c30`，
  `best_model_stage2.pth` SHA-256
  `9e1fe56207a140370bc684971640bfec9b7f6d1cd880f6d342b932723632d6d3`，文件与selection绑定
  一致。该检查不替代完整负例、分数及权重完成审计，A04仍为已确认incumbent；
  输入对照仍在等待GPU，A06未派发。已更新attempts及实验卡，暂不将A05标为最终失败或晋升。
- 2026-09-10：09:18宿主机只读核验A05原入口/训练PID及两个审计队列PID均存活；A05
  已完成23轮完整训练/验证，第24轮继续，选优仍为epoch19，patience4/5；未手动提前停止。
  正常输入对照已完成，其余视图等待GPU，独立完成审计队列仍等待A05原进程和运行完成。
  补核已归档MS-MOLE原文与干净固定源码，新增逐指标checkpoint选优的比较边界及默认
  分箱谱图MLP的静态参数计数，详见优化索引R02；原PDF哈希与下载记录一致。该路线
  仅作为未来设计依据，没有运行外部模型、改A05/A06配置或指定A07。文档检查通过，
  计划继续执行中；A06最终parent与派发仍待现有完成审计和整组输入对照。
- 2026-09-10：A04输入对照队列于08:45:06通过GPU1原连续空闲门槛及最终复查，正常
  输入阶段于08:46:20完成GPU推理与逐query分数审计，子进程PID3498552退出0；重新编码
  全部827,600个候选、评价全部19,423条有效验证query。指标在A04原容差内复现，差值
  见优化索引D17；两份工件SHA、完整保存排名重算及query顺序的独立读回通过。
  工件根为
  `/data1/zyl/SpecEmbedding/experiments/massspecgym_v15_a04_spectrum_controls_20260910_topk256/normal/`，
  `metrics.json` SHA-256 `23695e378295fde3544ac3ea05b81831b0810506c564085cf8b3e6b74807fe4e`，
  `baseline_epoch000.pt` SHA-256 `1d1d3c8fb2e2ae480bf299ddaa73454028fdf4f7eb1dcbe2fc7658c6f80e3a33`；
  checkpoint仍为已审计的A04 epoch20。这只证明正常输入复现，尚无谱图干预结果；队列
  已转入`permuted/waiting_gpu`，其后两个视图未执行，不能写作全部对照完成。
  现有入口与训练PID保持存活，A05于08:45:11完成第21轮并继续第22轮，完整train/val
  计数正确；选优仍为第19轮，patience2/5。A06仍待A05审计与整组对照结束，未派发。
- 2026-09-10：08:24:23完成A06最终配置绑定入口的准备和真实等待条件检查，源码仍为
  干净`d6226ef`；没有模型派发或新增后台队列。归档根为
  `/data1/zyl/SpecEmbedding/audits/optimization_a06_finalizer_preparation_20260910/`，`receipt.json`
  SHA-256 `0b0d726954ca2f840ac65875c565392e08562d03be4ce9d8a9d98fc9efbb6dd8`，`finalize.py`
  SHA-256 `2af4ce2434f974cc23c1d919d17a26b9b4b61900c6c85e593c7ba835f64f54b8`；脚本、32项
  CPU合成检查、准备脚本及真实调用输出已归档，7份归档文件与5项输入SHA独立读回通过。
  入口要求A05运行、独立完成审计及A04对照均完成，且四个绑定PID/starttime的原进程退出；
  再核验完整审计工件、对照分数工件及显式的parent/Pareto/对照复核决定，存在未处理的
  晋升候选则拒绝继续。真实A04配置完全复现既有条件准备，A05当前runtime仅用于检查
  batch256条件继承，不称为已选中的基线；自然负例预算和其它训练参数不变。
  `--check-only`退出0并返回`dependencies_pending`，指定不存在决定文件的调用退出2，
  均在创建正式目录前停止；Ruff/语法检查通过。这不等于最终预检通过，更不等于模型启动。
  最终决定完成后，入口仅生成
  `/data1/zyl/SpecEmbedding/audits/optimization_a06_finalization_20260910/`下的唯一配置与完整
  CPU dry-run回执；拟运行目录为
  `/data1/zyl/SpecEmbedding/experiments/massspecgym_v15_opt_a06_20260910_topk256/`，两者当前
  均不存在，实际GPU门槛、单次派发与现场核验仍为后续步骤。A05原训练继续，08:20:49
  完成第19轮并刷新当前选优，完整计数和保存排名重算通过，成绩见A05实验卡；计划保持执行中。
- 2026-09-10：07:59:49单次启动A05独立CPU完成审计等待队列；已复核PID3491217及
  专用pane存活，实际参数、CUDA不可见与外部缓存/临时路径符合准备命令。队列根为
  `/data1/zyl/SpecEmbedding/audits/optimization_a05_completion_queue_20260910/`，固定审计源码
  `/data1/zyl/repos/SpecEmbedding-spectrum-controls-20260910/`、commit `8f6971f`；原训练
  `a5aeb3b`源码未改。`manifest.json` SHA-256为
  `4abc7a6dbed7e0fdb441e463ec3f9ea53758d6ce13b751d55b8613395daf0782`，`runner.py` SHA-256为
  `24eac9f9b7dbd8f6567fefc951deb0c72598d7838550c5d936bda732236ff0af`，`launch.json` SHA-256为
  `665803e1005ee68976bac19a066b0f2c2219d1ee2de7d0f3745b31fb5c3deb8e`；脚本、命令与真实拒绝
  检查的原始输出均归档。独立socket为
  `/tmp/specembedding-a05-completion-20260910-1010/tmux.sock`、`audit/%0`；当前状态
  `waiting_parent`、`audit_started=false`，`dispatch.json`是等待入口的单次启动记录，不能
  写作完成审计。队列同时等待原runner3454081及trainer3455061（绑定各自starttime）结束，
  并要求原status和全部阶段complete；异常/不完整则保留现场并停止，不自动重试或派发训练。
  23项固定输入SHA及两处干净源码核验通过，13项CPU生命周期/解析合成测试通过，真实审计
  CLI对仍在运行的A05退出1且未创建输出；Ruff/语法检查通过。正式完成审计将输出至
  `/data1/zyl/SpecEmbedding/audits/optimization_a05_20260910/receipt.json`，逐轮检查全样本、
  自然负例、分数排名、选优和权重后，再重验全部工件SHA并严格CPU构造/重载选中模型，
  不执行模型forward或test，不自动晋升。当前原训练仍存活，已完成17轮、选优为第16轮，
  两轮新增保存排名及样本计数重算通过，指标见A05实验卡；A04诊断队列仍等待GPU。
  A06须等A05完成审计及A04诊断队列终态后再确定唯一配置并派发，计划保持执行中。
- 2026-09-10：07:10:44完成A05 GPU资源背景的有界只读采样，独立读回复核4份原始输出
  SHA及进程PGID/SID；GPU0上A05 PID3455061占13,912MiB，同卡另一进程组PID3473667
  占8,388MiB。该进程启动于06:12:07，不把启动时间当成连续GPU占用证据；单次pmon
  不能代表长时利用率。原始工件为
  `/data1/zyl/SpecEmbedding/audits/optimization_a05_gpu_context_20260910/`，`receipt.json`
  SHA-256 `3928218984233c63cdfb9acd393dd5dcf696c5781dfecc37fd28c988fdb89cc7`；包含脚本、
  GPU/计算进程列表、进程组信息、pmon及既有epoch13资源记录的SHA，采样脚本Ruff通过。
  这是资源背景观察，不是性能消融或减速定量测量；当前整卡利用率不归给A05，共享时段
  耗时不能作为独占卡batch性能结论，也不改写早先首轮测量。未停止或调整任何进程，
  A05继续全量训练，对照队列继续原GPU门槛；详见A05实验卡，计划保持执行中。
- 2026-09-10：A04 epoch20的正常/置换/常量/只保留母离子四种完整验证视图已绑定与预检，
  126项既有A04审计工件重验通过；固定源码仍为`8f6971f`，严格CPU构造及权重重载成功，
  没有CPU模型forward。原模型/训练/验证语义不变，仅将12项旧默认路径迁入外部根并添加
  storage配置；14项输入SHA与四条命令另行读回通过。准备根为
  `/data1/zyl/SpecEmbedding/audits/spectrum_controls_a04_preparation_20260910/`，`manifest.json`
  SHA-256 `e288274cbe2edd1305b1e898b6552607b85e011d3fc858171bef405b9e8a81fe`，入口脚本
  `runner.py` SHA-256 `ffb4da51f240b616a2409aa524192bcfdb91ffeb469386a23d1d979790be046d`。
  06:43:22已单次启动，`launch.json`与`dispatch.json`禁止删除或重发；独立socket为
  `/tmp/specembedding-a04-controls-20260910-1010/tmux.sock`、`controls/%0`、PID3479867。
  实际`status.json`首阶段为`normal/waiting_gpu`，PID与pane存活；GPU0/1当前均未满足门槛，
  尚未开始推理。输出将位于
  `/data1/zyl/SpecEmbedding/experiments/massspecgym_v15_a04_spectrum_controls_20260910_topk256/`。
  每阶段等待两卡择一的原120秒门槛后严格cuda:0；重新编码完整候选，正常视图先检查是否
  在原容差内复现A04，每阶段独立重算完整分数。脚本Ruff/语法通过，复用既有已测试的
  GPU池、验证及审计组件。此队列是固定checkpoint诊断，不是新模型训练；A06派发前需
  确认其终态，避免内部等待入口竞争。A05原训练PID3455061继续存活，06:40:34完成第11轮，
  当前选优仍为epoch9；具体指标见A05实验卡，计划保持执行中。
- 2026-09-10：补充D17保留母离子的碎片移除对照，实现提交`8f6971f`已推送，固定源码为
  `/data1/zyl/repos/SpecEmbedding-spectrum-controls-20260910/`、完整commit
  `8f6971f3c56b9d50ed5f29849ff17aa0422cd15c`；70项相关测试、Ruff、compileall和diff检查通过。
  06:15:32完成19,423条验证输入CPU重新分词/母离子逐条核验，随后独立读回全部保存张量并
  重验8项输入SHA；两进程均正常退出0。工件根为
  `/data1/zyl/SpecEmbedding/audits/precursor_control_inputs_20260910/`，完成回执SHA-256
  `ac06bacc5db1a0f4b09a5c2058712ed197ce610928671dbbe2dea060348fd81e`，独立验证回执SHA-256
  `6f19994a7c60532b1b5afac879e82bf92637e48bb1f0a7cebde708e4f14afc7a`；原始脚本、输入指纹和
  `controlled_inputs.npz`均保存在该目录。候选/标签/源位置、原序列及训练RNG不变；这只是
  输入敏感性准备，没有模型forward、GPU派发或test评价。A05/A06固定源码仍干净且未改。
  同次复查A05原PID3455061及指定pane存活，06:15:30完成第9轮并继续训练；当前最佳为
  epoch9，完整指标见A05实验卡，前9轮资源记录全部覆盖train194,119/val19,423。
- 2026-09-10：05:34完成固定指纹下16/64/255自然负例预算的全量CPU输入诊断，05:35独立
  读回通过；每档194,119条训练query，逐项核验源位置、不同二维负例及实际输入位值，详见
  优化索引D16。原始根为`/data1/zyl/SpecEmbedding/audits/fingerprint_negative_budget_20260910/`，
  完成回执SHA-256 `65c9f07a237155c52dfc918504acc7804f38a3c092f0f9e5b1362be26670ef17`，
  独立读回回执SHA-256 `8fca1c103b6544141eaec3a0d0bd58e276cc9f7ba626419f762f268004bba5eb`；
  重新核对20项输入和9项输出、25,046个target及三档全量query计数。单次CPU派发记录为
  相邻`fingerprint_negative_budget_20260910.launch.json`，专用socket
  `/tmp/specembedding-fingerprint-negative-budget-20260910-1010/tmux.sock`、`audit/%0`、PID3464183
  已终态；`/proc`确认退出码0。两份诊断脚本Ruff/语法通过，源码仍为干净`d6226ef`。
  此项没有模型forward、GPU派发或test评价，A05原训练进程仍存活，A06保持16负例配置；
  未来是否增加监督预算仍待A06结果和正式GPU成本测量，未创建新的模型尝试或宣称检索收益。
  05:38:34另记录A05第6轮完整验证刷新当前选优，Top-1为11.06%、MRR0.2218；前6轮资源均
  记录完整train194,119/val19,423，正式训练继续，详情见A05实验卡。
- 2026-09-10：05:03:33完成A06继承已晋升A04的全量CPU条件预检，源码仍为干净`d6226ef`，
  当前A05继续运行。新回执为`/data1/zyl/SpecEmbedding/audits/optimization_a06_conditional_a04_20260910/receipt.json`，
  SHA-256 `7c6a8a120de15294333ad620979220e1d5f332692dd76b596a931b8e84c35cd7`。
  预检前重新逐文件核对A04完成审计绑定的126项工件；预检后核对全部30项输入及6项准备文件，
  条件配置完整继承A04的batch128/block32、学习率1e-4与CE16/权重1，质谱塔和谱图增强不变，
  分子塔改为预先设计的固定Morgan位向量MLP、图扰动显式置零。完整指纹输入为训练5,711,753、
  验证827,600个分子，对应train194,119及有效val19,423条query；未创建模型运行目录或派发GPU。
  准备脚本Ruff/语法与真实完整dry-run通过；A06最终parent/batch仍等A05完成审计再冻结。
  同次现场核验A05原PID3454081/3455061及专用pane存活；05:01:44第3轮资源/完整验证快照已保存，
  早期验证Top-1为7.46%、MRR0.1709，详见A05实验卡，不作为最终模型收益或test结果。
- 2026-09-10：A04于03:39结束第25轮并按patience5早停，选中epoch20；原入口于03:50:03
  完成全部负例重放，PID3360929为exit0的终态，训练子进程已退出。独立完成审计于04:14确认
  正常退出，25轮完整样本/负例/排名/配置/权重与126项工件绑定通过；原始结果表见优化索引A04。
  当前有效完成回执为`/data1/zyl/SpecEmbedding/audits/optimization_a04_20260910_r2/receipt.json`，
  SHA-256 `39938d47a1d8e4f8cdd2a541e3252899f9ddf090c78a61b63683c28973206da3`。
  首次审计在相邻无`_r2`目录因把A04旧runtime用于新存储入口而在导入阶段退出1，日志及
  `failure.json`保留；修正仅是新审计导入配置的存储路径，模型/训练/增强/tokenizer保持A04
  语义，实际审计仍读取并核验原runtime SHA。没有修改A04、放宽路径保护或覆盖失败目录。
  独立审计socket为`/tmp/specembedding-a04-audit-r2-20260910-1010/tmux.sock`，
  `audit/%0`、PID3452883已dead/exit0；训练和审计的结束不混作SOTA目标完成。
- 2026-09-10：A05先完成A04条件parent完整预检，记录位于
  `/data1/zyl/SpecEmbedding/audits/optimization_a05_conditional_a04_20260910/`，回执SHA-256
  `ce27bf32377d34438c38f339fa4f4e0220ce8b5451a5af6b669727121ffe1482`；A04审计后复核全部工件，
  明确晋升A04 epoch20并冻结唯一A05配置，最终完整dry-run通过。最终绑定根为
  `/data1/zyl/SpecEmbedding/audits/optimization_a05_finalization_20260910/`；`ready.json` SHA-256
  `145375356c25919d7a1136d8dbb2e91b89b1fe0521591d88a29def604ac3cf9f`，保存唯一决策、参数、
  命令及派发记录。比较checkpoint SHA-256为
  `1cf7db2be2d6df8360dc6938f41b60f4e38f39eb776f7d6a69337d7e6d649661`；A05仍从随机初始化训练。
  04:18:19单次启动至`/data1/zyl/SpecEmbedding/experiments/massspecgym_v15_opt_a05_20260910_topk256/`，
  固定干净源码`/data1/zyl/repos/SpecEmbedding-opt-a05-storage-r2-20260910/`，commit `a5aeb3b`；
  专用socket `/tmp/specembedding-opt-a05-20260910-1010/tmux.sock`，`opt/%0`、入口PID3454081。
  完整导入与baseline验证完成后，04:25:36重新通过GPU门槛，在物理GPU0映射的严格cuda:0
  开始训练；PID3455061实际环境、运行配置SHA、外部缓存/临时路径及OMP/MKL=1已核验。
  首轮于04:38:01完成，实际train194,119、val19,423，759个batch且最后71条；完整观察query
  排列和文件SHA独立读回通过。早期指标与资源见A05实验卡，不称为新模型完成结果。
  runtime SHA-256为`fdabcea3762906f8ef39d8eec75fc3b6b18f18efda66a886437689b76f5fa9c1`；
  batch256、学习率1e-4、自然负例最多16/CE权重1及完整数据不变，所有新数据与缓存使用外部项目根。
- 2026-09-10：A05基线已用固定图缓存重新编码全部827,600个候选，全部19,423条验证query
  与A04同一checkpoint的选中轮保存分数独立对比通过；最大有效分数差4.18e-7，98条排名
  变化，各项指标差异均低于预设容差。记录位于
  `/data1/zyl/SpecEmbedding/audits/optimization_a05_baseline_cache_20260910/receipt.json`，SHA-256
  `b41e7d3c54b314e21f7a93a47fba9150ec031de8b566a02898eba5906c4345fa`。
  此次完整检索计时34.36秒，A04选中轮229.21秒；GPU/时段不同，不将全部比值归因缓存，
  首轮资源回执已记录整轮682.62秒、训练631.47秒、检索验证45.99秒；相较A04首轮整轮
  耗时短约15.5%，但训练阶段未加快，不能据此提前判定最终效率或模型收益。临时预检/对比
  脚本的Ruff和语法检查通过，固定源码未改。
- 2026-09-10：核对阶段验收清单，A02/A03完整回执及其117项工件逐文件哈希复查通过；
  纠正评价实现与checkpoint检查仍未勾选的过时状态，不改变当前实验或最终SOTA验收状态。
  新增独立MRR参考核对发现同分排序与非正分数处理的语义差异，说明及回归检查见D15。
  `/data1/zyl/SpecEmbedding/audits/mrr_semantics_20260910/`冻结100份完整验证快照和来源指纹，
  独立CPU诊断于03:20:42正常退出，`receipt.json` SHA-256为
  `4ccdb58f4cb5abeb3136186d899751fa167d41305ec689cb4c01b96d734e2186`；108项来源指纹和所有
  逐query排名工件的独立读回通过，`verification.json` SHA-256为
  `c8e40f3ab7e5390f448c49e064af3b58a20ad55c95998907c1d736417a1ded6d`。
  评价回归21项通过，其余相关61项通过；Ruff、语法与diff检查通过。保留现有共同排名定义，
  未改原始分数、选优记录、训练源码或A05草案；A04已完成24轮，正式完成审计仍待训练结束。
- 2026-09-10 02:26：完整train/val元数据及A04固定epoch20保存分数CPU审计完成，详情见D14；
  专用socket `/tmp/specembedding-query-metadata-20260910-1010/tmux.sock`，`metadata/%0`已dead，
  原PID3439063内核退出码0。审计根`/data1/zyl/SpecEmbedding/audits/query_metadata_a04_e020_20260910/`
  保存脚本、完整query元数据和来源/输入哈希；`receipt.json` SHA-256为
  `61a93ebcfaf4b4728683b0a968505fbdee98e464b295827805703636722429cf`。
  审计源码为既有干净`d6226ef`，没有修改A04、缓存构造或候选/身份定义，未执行GPU推理。
  完整保存表的独立读回复核与峰截断分组也已通过，`verification.json` SHA-256为
  `2df2dc7b0d22c8555f91129561ad3c052a1ebee84bea7397f501899b7c38eec5`；脚本Ruff/语法检查通过。
  这只覆盖固定epoch20与baseline，不替代完整轨迹和自然负例重放；A04原训练继续，未晋升。
  强参考逐指标来源和未完成资格条件另固定到版本化JSON，原文核对见R04，S_j仍未冻结。
- 2026-09-10：A06正式接入固定源码为`/data1/zyl/repos/SpecEmbedding-opt-a06-20260910/`，
  detached且干净，commit `d6226efb8492ce5595a953bc720ecc741c4f4867`；完整A02条件CPU
  dry-run通过，准备根`/data1/zyl/SpecEmbedding/audits/optimization_a06_preparation_20260910/`
  保存脚本、配置、实际存储入口命令、原始输出及预检；`receipt.json` SHA-256为
  `1854fcc5b1201d47a7cf0b52152fedf4e483733591d457b6951de896f96105c6`。
  该条件使用已审计A02的batch128/无候选CE，仅更换分子表示并显式关闭图增强；完整train/val
  固定位向量、GINE baseline自己的构造配置及共同输入协议核验通过。没有创建运行目录或
  执行GPU推理/训练。A05仍为下一次正式实验，A06最终parent与配置等待A05完成审计。
- 2026-09-10 02:00：A04原训练PID及专用pane存活，已完成18轮，当前最佳epoch17；
  完整原始验证成绩和资源见A04实验卡，中途独立审计仍固定到epoch4，没有test评价或最终晋升。
  A06正式接入`d6226efb8492ce5595a953bc720ecc741c4f4867`已提交，新增27项检查通过，
  全仓501通过、1跳过、1个相同既有路径审计失败（原4文件）；Ruff、compileall、diff通过。
  接入范围及严格加载/完整输入审计边界见A06卡与D12，尚无正式模型结果。
- 2026-09-10 01:27：实际队列子进程检查发现原入口覆盖Numba/Matplotlib外部缓存路径，
  `8eaf2c8`改为保留已传入环境；从第一版A05存储源码独立派生新worktree，未修改A04。
  当前A05固定源码为`/data1/zyl/repos/SpecEmbedding-opt-a05-storage-r2-20260910/`，
  detached且干净，commit `a5aeb3b4828093d3ac9b6e32cf63597183b562aa`；存储/队列相关29项通过。
  当前有效准备根为`/data1/zyl/SpecEmbedding/audits/optimization_a05_preparation_20260910_storage_r2/`，
  `ready.json` SHA-256为`45c514ba390668f0d2ac21fe66893197dd7fc5a1c34075103f7e2423e53ccc83`。
  两份条件草案的非路径设置保持一致，实际入口完整A02条件dry-run通过，旧回执/命令均保留。
  此回执替代第一版存储准备用于后续启动，仍须A04完整审计、唯一parent与最终预检；未派发。
- 2026-09-10 01:02：A04已于00:57:10完成第14轮完整训练/验证，当前选优仍为epoch11；
  当前指标均来自19,423条有效验证query，A04尚未运行test。逐轮变化与资源更新于A04实验卡，
  原训练、配置和源码不变，完成审计与最终晋升仍待结束后执行。
- 2026-09-10 01:00：用户要求当前轮次不改，下一轮及以后将数据、缓存、下载缓存、临时
  数据与工件写入`/data1/zyl/SpecEmbedding/`。存储实现`fdd99f1`已提交；从原A05的`881cdfa`
  独立派生并仅应用存储改动，固定源码为`/data1/zyl/repos/SpecEmbedding-opt-a05-storage-20260910/`，
  detached且干净，commit `af9979d58c06ef5a9c495a5649faa70e7100efdb`。
  新的有效准备记录在`/data1/zyl/SpecEmbedding/audits/optimization_a05_preparation_20260910_storage/`；
  `ready.json` SHA-256为`34660b0d50bb42d45bac44a3e638ac9a51d32b048a95ac939cdef0e2c0bfed10`，
  `drafts.json`、两份条件配置、原件复制回执与实际入口的完整A02条件dry-run均由该回执绑定。
  新准备取代09日旧草案用于后续派发，旧工件保留；两份草案全部非路径设置一致，A05的batch256、
  学习率、候选监督条件与验证图缓存不变。原始v1/v1.5副本分别位于项目根`raw/MassSpecGym/v1/`
  与`v1.5/`，逐文件SHA-256与原件一致；A04仍引用的home原件及旧源码不删除、不移动。
  新计划运行目录为`/data1/zyl/SpecEmbedding/experiments/massspecgym_v15_opt_a05_20260910_topk256/`，
  当前不存在，未启动训练或GPU等待任务。仍须A04完整审计、唯一parent决策与最终完整预检，
  再单次派发；不能把条件dry-run当作最终parent已经冻结。
  全仓473通过、1跳过、1个相同既有路径审计失败（原4个文件）；下一轮固定源码相关37项通过，
  主工作树存储/路径/匿名归档相关31项通过；Ruff、compileall、diff检查通过。
- 2026-09-10 00:33：`55f1afa`固定于`/data1/zyl/repos/SpecEmbedding-fingerprint-model-20260910/`，
  完整真实CPU模型输入预检完成，详细数量、采样比对和计时范围见D12。未运行模型forward或GPU。
  原始输入、脚本、日志与完成状态保存于`/data1/zyl/SpecEmbedding/audits/fingerprint_model_inputs_20260910/`；
  `inputs.json` SHA-256为`5e2c2fdc1cfdb4912acd25ab001aa37634929daea3ddd3c974f55bbc8e751c06`，
  `receipt.json` SHA-256为`bf4e75a40da10cbf7fcf66a11c14587afe654fcf61e627c24a5e782e4e2b28bf`。
  专用socket `/tmp/specembedding-fingerprint-model-20260910-1010/tmux.sock`的prepare/%0已dead，
  PID3411744只剩已终止僵尸条目，状态和回执均complete；未重启或据此派发新训练。
- 2026-09-10：用户要求构造参数等变化后清除失效缓存，已记录跨轮复用和失效清理规则；
  仅当实际来源/构造指纹变化时停用并重建，确认没有活跃引用后清理旧载荷并保留审计记录。
  当前固定图和Morgan输入仍有效，没有因新增分子MLP或更换epoch而删除缓存。
- 2026-09-10 00:20：再次核验A04原pane/训练PID存活；00:08:45完成第十一轮，继续第十二轮，
  运行源码与配置保持不变。新增A06固定指纹分子塔、配对/自然候选读取与完整检索验证组件，
  原谱图模型未修改；详细主假设与正式启动前要求见A06实验卡及D12。A05仍是下一次正式实验。
  21项新增合成检查通过，全仓464通过、1跳过、1个相同既有路径审计失败（原4个文件），
  Ruff、compileall与diff检查通过。尚未接入正式CLI、运行真实模型或宣称性能收益；
  后续先冻结源码，用完整真实输入进行CPU映射/读取预检，再完成正式配置与审计绑定。
- 2026-09-09：用户明确进一步固定顺序为“基础检索先达到或接近SOTA，再用rerank加强”，
  已同步目标、可调范围、迭代顺序与执行清单；采用既定同协议逐项验收，不因单次局部改善提前切换。
  常规进度不再重复汇报是否使用rerank；只在阶段切换或需要决策时说明相关变化。
  本次更新不改变A04正在运行的模型、A05准备方案、输入/选优协议或最终比较标准。
- 2026-09-09 23:46：核验A04专用pane仍存活，训练日志推进第十轮；第九轮的完整原始验证
  成绩及资源更新于A04实验卡。尚未决定晋升，没有创建A05运行目录或改动两个固定worktree。
  `93abdd8`的完整Morgan输入准备于23:30:39单次派发，23:41:59完成，独立CPU pane
  `fingerprint-inputs/%5`退出0，socket沿用`/tmp/specembedding-validation-graphs-20260909-1010/tmux.sock`；
  此任务只处理输入，不派发模型或GPU阶段。固定源码为
  `/data1/zyl/repos/SpecEmbedding-fingerprints-20260909/`，缓存根为
  `/data1/zyl/SpecEmbedding/fingerprint_cache/massspecgym_v15_morgan_r2_2048_20260909/`，
  两个子目录为`train_topk256/`与`validation_topk256/`；其`preparation.json` SHA-256分别为
  `f10f65b68e1a5048cbb90edb64c972c19000880f8723641fb4d16b3ae6570e11`与
  `6de90c93cf76fc730f60c17023aba2c4365b4f3d6b8abf2c736978e86a6691a7`。
  完整数量/字节及构建、独立逐位审计耗时只维护于D11；来源/全部命令/原始日志保存于
  `/data1/zyl/SpecEmbedding/audits/fingerprint_preparation_20260909/`，其中
  `inputs.json` SHA-256为`3adf74b950e9398296fec62b140c4d50cf3aa5f8994bfbacdd8dfa5007ecbea6`。
  完成后公共读取器重新核验全部候选来源与文件SHA，`completion_verification.json` SHA-256
  为`43b9f129ff956f424df50d643be7b7f18a9022e3cc19523a6616eb42218193ff`，相邻保存复核脚本。
  正式GLMR论文另归档为`/data1/zyl/papers/AAAI_2026_40_2_1561_GLMR.pdf`，SHA-256为
  `73160f5d07478e43242418dd751164d5ac40afe43a4a1cb2bcd9128386a45e17`；下载与出版信息清单为
  `/data1/zyl/papers/specembedding_retrieval_20260909_glmr_publication_manifest.json`，SHA-256
  `41bb6db3335c0f97b4a4d527f66b3a04085861ba913baf3996b562aeaf31521d`，原文边界见R03。
  新实现32项专项通过，全仓443通过、1跳过、1个相同既有路径审计失败；Ruff、compileall及
  diff检查通过，后续完成复核脚本另经静态检查。这里只完成输入与资料准备，计划仍执行中。
- 2026-09-09 23:13：A04完成第七轮、开始第八轮；原日志Top-1为10.17%、MRR为0.2151，
  属于训练中途进展。开始准备后续轻量分子表示的固定Morgan输入：使用已审计完整train候选
  和validation候选，各自保留源索引顺序，radius2/2048位、无手性，packed存储并独立重算
  每个分子的全部位值；绑定数据/索引/构造参数/源码/软件版本，不以指纹相等定义正例。
  先实现公共读写及CPU准备入口、覆盖真实多进程与篡改拒绝检查，再冻结源码构建完整输入。
  此项不训练模型、不使用test、不改SpecEmbedding预训练结构，也不改变A04或A05固定源码；
  指纹塔的正式实验卡/模型集成与性能验证仍在后续，不能把输入准备写成模型优化完成。
- 2026-09-09 23:00：完成D10全量训练流缓存模拟及短时真实资源观察，保留A05原训练缓存容量，
  继续使用D09验证图磁盘缓存；A04正在第七轮，固定源码和运行配置未改，未派发A05。
  缓存模拟于22:57:39完成，重放实际epoch5全部query/负例并核对原训练样本流SHA，回执为
  `/data1/zyl/SpecEmbedding/audits/training_graph_cache_replay_20260909/receipt.json`，SHA-256
  `ff5f7df2b325313719b3465db2b350dcc9a2e36e3976cced9694499af996af8e`；原PID3393096已退出，
  不从tmux session消失推断失败或重新派发。真实训练资源快照为
  `/data1/zyl/SpecEmbedding/audits/a04_live_resources_20260909/snapshot.json`，SHA-256
  `f0330c5ca823f089a0a073eac840a3d0b9853510946da42c38ea7fdcccae34d5`，相邻脚本可复核。
  原始观察仅约3.3秒，不能据此分离CPU与其他GPU任务对整轮速度的因果贡献。
  两个独立CPU/只读脚本Ruff及编译通过，完整重放自检、文件前后SHA和原记录一致；本轮只改文档。
  首次宿主机批量只读检查因自动审批连接中断被拒绝；拆为明确只读检查后通过，进程存活及
  GPU遥测已取得，没有遗留审批阻塞或改动其他任务。计划继续执行中，论文与最终SOTA验收未完成。
- 2026-09-09 22:44：A04原进程继续第六轮，前五轮原始成绩/资源见A04卡；没有改变固定源码、
  运行配置或派发A05。独立CPU中途审计于22:34:57完成，固定baseline与epoch1–4的完整保存
  分数、query顺序和输入指纹，原进程内核退出码0；不包含模型forward、最终选优或完整负例重放。
  回执为`/data1/zyl/SpecEmbedding/audits/optimization_a04_interim_e004_20260909/receipt.json`，
  SHA-256 `c930e92ea2159bdeb571e6a3c43b5feb56c3e47229b4c176cd37ab886b4e0e1a`；相邻`audit.py`
  与`paired_ranks_and_top1.npz`保留可复核结果。该审计未覆盖随后完成的epoch5，不决定晋升。
  MS-MOLE原文下载为`/data1/zyl/papers/2602.16507v1_MS_MOLE.pdf`，SHA-256
  `c298ff38ce146a885bf6bb769ab9ac03758eb2e233b8ad5a25b2af3576d17ed1`；官方代码只读快照为
  `/data1/zyl/repos/ms-mole/`的`f81558c4cb37ca2f3d4300eff8e3785eea53c9e3`，下载清单
  `/data1/zyl/papers/specembedding_retrieval_20260909_msmole_manifest.json` SHA-256为
  `862d312cb6a1102fb2f91c58db74d38cfc01779940c59cec94488859e2c8761a`。
  新增源码标签与FLARE原文输入条件核验见R02，不据此断言论文实际评价受影响的程度；
  没有安装依赖、运行外部模型或提前替换A05主假设。计划继续执行中，论文未修改。
- 2026-09-09 22:16：完整CPU图加载三组测量及张量指纹核验已完成，结果集中于D09；
  `/data1/zyl/SpecEmbedding/audits/validation_graph_loading_20260909/receipt.json` SHA-256为
  `24c03117b8a1d0e0fc01f232177dc0ca439516c95ddea48365dd1ea428ee834b`。没有模型/GPU计算，
  不把CPU图加载改善外推为整轮训练速度。A05准备汇总与后续必需步骤保存于准备根目录
  `ready.json`，SHA-256 `cf07a911dcbdda36b3499878f3358b0bb6301bff7df5b198815e7ead6087baf8`；
  A02条件parent的完整预检SHA-256为`c676a1af54b1b63118328c6b2020ae5727fa2817cd429da0abe8454ca54a0f52`。
  状态为准备完成、等待A04完整审计；没有自动衔接监测器或A05训练进程。
- 2026-09-09 22:08：固定图缓存实现`881cdfa2fc75d3f7b4f00977cc1b4bff145ba332`已提交推送，
  新独立源码`/data1/zyl/repos/SpecEmbedding-opt-a05-20260909/`保持干净；A04源码未修改。
  缓存准备于21:57:21单次派发，仅CPU，于22:01:19完整构建及独立逐图审计结束、pane exit0。
  缓存位于`/data1/zyl/SpecEmbedding/graph_cache/massspecgym_v15_validation_20260909_topk256/`；
  `manifest.json` SHA-256 `75f6f4e2c4fdc5fab30f4d423a6fc02d868941c15fd76b5488e8f9ec1c69cde4`，
  `audit.json` SHA-256 `4630182c4f48b800cd3d3cbf84489b60efe5b06c6547d42762ca85f301e9c813`，
  `preparation.json` SHA-256 `74d4b55b2176faf7b4c56dd1c0aefd213fc21c07961ee00870aa94f8c312e8bc`。
  启动来源与原始日志保存于`/data1/zyl/SpecEmbedding/audits/validation_graph_cache_20260909/`。
  专用CPU server为`/tmp/specembedding-validation-graphs-20260909-1010/tmux.sock`；`prepare/%0`
  已退出；同server内`a05-inputs/%1`完成CPU配置/采样准备并exit0；`graph-loading/%2`仅测量
  完整CPU图DataLoader，结果保存在`/data1/zyl/SpecEmbedding/audits/validation_graph_loading_20260909/`，
  仍进行中，没有模型/GPU计算，也不是另一个训练或GPU监测器。
  A05配置草案、完整100轮batch采样核验及命令保存于
  `/data1/zyl/SpecEmbedding/audits/optimization_a05_preparation_20260909/`；`receipt.json` SHA-256
  `1b12db9411b49d94193f2a45c5c6bed59ff62125b8b2669eb672d0da932f455e`，`drafts.json` SHA-256
  `dc62d2bfcf3fbf5a2b157a497f58a4210c9f94d2c5950262b0cc331b9f5adbab`。A02条件parent的完整
  dry-run保存为`preflight_from_a02.json`，新图缓存全指纹核验通过；未创建A05运行目录。
  两份配置只是待A04完整审计后择一的草案，不能视为两个已排队方案；A04当前完成3轮，
  最终parent、真实GPU运行和性能结果仍待后续。
- 2026-09-09 21:54：用户明确要求A04训练期间设计下一轮并增加缓存，从下一轮应用。
  无损固定验证图缓存、CPU完整构建/独立逐图张量审计、正式入口及完成审计接入已实现；
  新增29项检查通过，全仓411通过、1跳过、1个相同既有路径审计失败，静态检查通过。
  详细机制与证据见优化索引D09；真实缓存将在独立源码冻结后构建，尚未宣称提速。
  A05实验卡已准备：仅训练batch128→256，学习率保持1e-4；缓存另记工程改动；parent等待
  A04完整审计后确定。A04仍使用原`82c20dd`，在GPU1上完成前两轮全量训练/验证，进程正常；
  两轮资源和中途成绩见A04卡。未停止/修改A04，没有派发A05或新GPU任务。
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
- 验证结果：A09独立组件22项检查及真实条件构造通过，主机全仓612通过、1跳过、1个相同既有路径审计失败，Ruff、compileall及diff检查通过；A08已有组件/条件预检、最终配置绑定和单次衔接检查保留；A07前9轮完整计数及新增排名重算已核验，原训练与两个等待队列存活
- 当前训练状态（2026-09-10 16:45核验）：旧矩阵停止；A04 epoch20为当前incumbent；A05完整24轮和A06完整9轮及独立审计均未晋升；A07前9轮完整训练与验证正常，尚未完成审计或晋升；A08的CPU衔接队列等待前序完整审计，最终parent未绑定、模型未派发；A09已预登记并完成独立组件准备，组合输入/正式审计接入待做；模型优化和SOTA达标未完成
- 论文修改 commit：尚未提交；本轮不修改论文
- 计划归档 commit：无需在本文件中自我引用
- 相对原计划的偏差：用户已授权从立即跑 12 组矩阵改为单方案持续优化，成熟后再做稳定性与矩阵；
  目标升级为 Top-k/MRR 接近同协议现有 SOTA，预训练模型结构保持不变
