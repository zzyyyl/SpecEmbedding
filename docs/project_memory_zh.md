# SpecEmbedding 项目记忆

最后更新：2026-09-09。本文是内部状态摘要，不放入匿名补充包；事实以代码、配置和可校验工件为准。
运行入口见 [文档地图](README_zh.md)，模型细节见 [reranker 说明](reranker_solution_zh.md)。

## 当前实现与证据

当前默认实现为无 rank 的 `relative`：最多 256 个自然候选先做 pointwise coarse scoring，
再对 coarse top-40 计算谱图条件的反对称偏好；`pointwise` 为容量匹配对照。
统一 pipeline 默认所有 split 不 forcing。该默认值不代表历史实验配置。

[实验索引](../reproducibility/mentor2026_experiment_index.json) 中的 36 个主运行和 30 个
relative 消融均显式请求最多 20,000 条训练 query、train-k=40。它们复用旧 top-40 缓存，
训练含正例 forcing，验证/测试不 forcing；不能称为新版全量、全池 no-forcing 实验。
另有使用 pre-overlap-clean alignment 的 5,000-query/top-256 pilot 和独立官方候选顺序/二维
身份审计。各系列的来源与边界集中在 [分析索引](../analysis/README.md)，不混用其结果。

已有结果支持固定候选池内监督式 residual learning-to-rank 的收益，但 relative 相对
pointwise 的方向随候选协议、指标与 checkpoint 改变，独立普遍优势尚未建立。
保存 embedding 的候选/身份审计不是 fresh official-loader 重编码；所有评价名称须与工件匹配。
JESTR-style cosine 只是本地保存嵌入上的重实现打分控制，JESTR/GLMR 官方模型未统一复现。
其余论文主张约束见 [AGENTS.md](../AGENTS.md)。

## 在途任务

- [MassSpecGym 全量重训](paper-change-plans/2026-09-07-MassSpecGym全量重训.md)：**执行中，已获准转为单 baseline 优化；旧矩阵停止，论文更新暂停**。
  当前 processed 数据已核验为 v1。用户已授权采用 v1.5，并在审计证明必要时重训 alignment。
  旧监测器于 19:48 停止，未派发训练，原锁、日志和固定源码保留。
  旧构图的芳香性/键类型依赖 SMILES 表示，迁移需规范构图并重训一个 alignment-42；
  正式 alignment 已增加每轮全谱遍历与数量校验。首个 v1.5 队列于 20:17 在候选审计阶段停止：
  官方源含 RDKit 无法构图的候选，未启动 GPU，失败工件及 `6843e12` 固定源码保留。
  修正为显式记录现有候选编码器的无效图排除（保留源候选顺序/内容，正例和 query 不得丢失）；
  修复 `d15510b` 已推送并冻结新 worktree；r2 队列于 20:29 启动，但于 20:33 再次失败退出。
  本次候选已通过 SMILES 解析，随后生成 InChIKey 时触发 `KekulizeException`，不能直接归为
  既有无效图排除。21:54 核验进程已退出、未进入 GPU 阶段，日志与失败工件保留，未重启。
  用户随后要求停止任务、排查并修复。已定位 RDKit 解析与 InChI 转换的 Kekulé 搜索顺序差异；
  修复 `aecdfee` 仅转换分子副本并保留身份转换记录，不改候选或编码器图。
  完整 CPU 数据审计与独立文件比对已通过，验证进程已退出；详情见
  [修复记录](../analysis/massspecgym_v15_inchi_fix.md)。未启动新训练。
  2026-09-08 用户明确要求继续推进，r3 正式队列已于 00:05 在新独立 tmux 中启动，
  固定源码为 `15b1b5d`，与已验证修复的 Python 代码及配置一致；CPU 准备与审计已于
  00:38:30 完成，实测进入 `alignment42/waiting_gpu`，预处理子进程已退出。
  随后按用户授权实现多 GPU 择一等待（`ad81b61`）：各卡独立计时、固定 UUID、最终复查，
  子进程将选中物理卡映射为显式 `cuda:0`；每个 GPU 阶段重新选择，支持两个以上 GPU。
  23:55:48 再次确认尚未训练后主动中断 r3 等待入口，保留旧源码、日志、锁与工件。
  新固定 r4 队列于 23:55:52 启动，已核验并复制导入完整 CPU 数据，实测同时等待 GPU 0/1。
  23:58:03 经持续空闲窗口与最终复查选中物理 GPU 0，alignment-42 子进程已启动；日志确认
  严格 `cuda:0`、正式全量模式和全部 194,119 条训练谱图，复查已进入第 1 轮训练，
  GPU 进程记录与 UUID 绑定一致。此单个 alignment-42 完成后再运行 12 个全量 reranker 组合；
  不能称为三 alignment 矩阵。
  2026-09-09 复查：alignment、六缓存已完成；Mass relative 三 seed、pointwise seed42 的
  全量训练和完整测试完成，pointwise seed43 正在训练。结果仍为 cache-local exact-target-SMILES。
  用户要求排查准确率落差，已核对新旧实际权重哈希、六 cache 指纹、四 test cache 全 query
  标签/来源/点积及分散样本 GPU 重编码，未发现旧权重或磁盘 TokenSet 缓存复用。
  旧候选真值与干扰项的芳香性表示存在强格式捷径，已用完整候选上的无谱图解析规则量化；
  不能将旧高分直接作为新版应恢复的可靠基准，旧模型依赖程度仍需控制实验。
  详见[准确率排查](../analysis/massspecgym_v15_regression_audit.md)。该证据影响旧结果解释，
  此前计划设为待确认；随后用户授权先固定单 baseline 优化，并把目标更新为分子检索
  **Top-k 整体提高、MRR 为辅，各指标达到或接近同协议现有 SOTA**。已核验后中断 r4 的
  pointwise seed43 及后续矩阵，4 组完整结果与全部
  工件保留，seed43 的部分 checkpoint 不作为完成结果。外层状态为中断，内层仍残留 running；
  已确认进程组退出、专用 pane dead，不能据残留状态重启。下一轮固定 Mass/relative/seed42，
  保持 v1.5、全量、无 forcing 和双 GPU 择一等待；入口已收窄范围。
  A01 已实现完整验证候选重编码与检索选优，Top-1/MRR 选 checkpoint 并保存 Top-k 的 Pareto
  候选；优化分支不运行 test 或 reranker。固定源码 `cba38a4` 的 A01 于 2026-09-09 15:58:57
  在第 19 轮早停，四阶段完成、进程退出，独立 CPU 完成审计通过。选中 epoch14，Top-1 略升，
  其余 Top-k/MRR 相对 r4 下降；两个非支配候选均未通过整体改善标准，保留 r4 为 incumbent。
  同一 A01 轨迹内检索选优优于最低 loss 轮，但不能据此对跨运行差异作因果归因。
  独立完成审计入口已实现，可逐轮重算保存分数与排名、核对权重/选优和完整覆盖，
  按预设容差列出各 Top-k/MRR 改善及退步；完成回执、基线决策与运行位置见同一阶段计划第 10 节。
  主排名匹配 torchmetrics1.8.2 逐 query CPU argsort，稳定排序另列敏感性视图；
  实验卡与最新测试见[优化索引](../analysis/massspecgym_optimization.md)。调参仅看验证集，矩阵后置。
  用户要求每轮尝试的修改和结果用一句话写入根目录[attempts.md](../attempts.md)，已补齐A01–A03；
  进行中的条目标明核验时间与未完成状态，必要的单次详细说明使用`attempt_<n>.md`并链接。
  等待期间已实现 A02 质量邻近 batch：保持 batch128、模型/loss 不变，全谱每轮一次，
  记录顺序/质量指纹与唯一覆盖；完整 CPU 核验表明邻近质量负例显著增加，但身份多样性减少。
  实现 `fc42b27` 的干净固定源码与真实输入 dry-run 已核验；A01 完成审计和决策后，A02 于
  16:01:20 单次派发，16:05:28 完成 CPU 准备，验证索引哈希与 A01 相同；16:12:44 在物理
  GPU 0 上启动 alignment；18:09:37在第27轮正常早停，完整逐轮样本/分数/权重审计通过。
  A02 epoch22相对r4的各项Top-k/MRR均超过预设改善容差，已成为下一轮比较baseline。
  A02 随机初始化，r4 checkpoint仅用于比较；这是单seed完整验证改善，不是test/SOTA结论。
  两次固定基线验证的微小数值差异已独立核验，远小于决策容差，见优化索引 D02。
  计划要求的完整验证谱图置换/常量检查已实现并通过合成测试与全索引 CPU 预检，见 D03；
  尚无正式控制推理，输入构造检查不称为模型结果。
  完整 train CPU 诊断已区分多正例损失的熵下界与图扰动影响，见优化索引 D04；不凭 A02
  较高的训练 loss 判失败。A03 仅关闭分子图增强的开关已实现、测试并完成真数据 dry-run，
  谱图增强保持不变；最终固定A02 epoch22作比较、mass_blocks/block32，随机初始化。
  A03于18:32:27单次派发，18:36:36完成CPU准备；18:38:41通过持续空闲检查，在物理GPU0
  的严格cuda:0上开始baseline完整重编码验证，实际子进程UUID映射已确认。
  验证索引与A01/A02哈希相同，18:41:36验证完成，训练阶段重新等待；18:43:40启动alignment。
  19:08核验已完成5轮完整检索验证，入口/训练PID存活，逐轮样本计数和资源记录生成正常。
  首轮耗时/吞吐/参数/内存实测与中途指标只维护于优化索引A03，尚无最终收益结论。
  正式 alignment 的逐轮资源记录与完成审计已实现，保存耗时、实际吞吐、参数量、CUDA
  分配峰值和主进程峰值 RSS；不含 worker 内存或其它 GPU 进程，不回填 A01/A02 未记录的峰值。
  实现与验证位置见优化索引A03；A02源码未修改，A03已冻结包含该实现的新源码。
  完整验证候选的 CPU Morgan 指纹可区分性诊断及独立复核已完成，见优化索引 D05；
  2048/4096 位的完全碰撞相同，不能据此推断训练效果相同。候选对比目标的原文核验见 R01，
  尚未派发指纹塔或新损失实验，也未生成可复用的训练指纹特征缓存。
  全量训练自然候选的紧凑元数据与独立源文件复核已完成，见优化索引 D06；所有query均有
  自然正例，重复二维负例和无标签候选结构与held-out目标的交集已记录，尚未用于正式训练。
  全量候选读取/不同二维负例采样工具`01024f5`通过新增23项测试及真实全量CPU采样预检；
  每条query保留，同身份正例排除、源位置、无重复负例和RNG均核验通过，详细统计见D06。
  诊断预算不是正式超参数，未改变A03。后续数据/损失/训练组件`bee5b63`已完成，详见D07；
  它逐query补充自然候选监督，正负图策略一致，不改变预训练架构或检索打分。
  新增23项测试及真实全量原始谱图/token/候选映射核验通过；尚未接入正式配置/CLI/完成审计，
  不派发候选监督实验，等待A03完整结果后固定唯一实验卡。
  A03独立源码为`c50ddf0`，正式预检与派发指纹已保存。
  固定 baseline 的完整验证错误诊断已保存，分组不能证明元数据特征的因果收益；额外的质量
  硬过滤会丢失真值，未实施。诊断结论与后续假设集中于优化索引 D01，未改变 A01/A02 配置。
  旧预训练权重的现存缓存与新划分 token 可逐条匹配，
  未见 test 二维身份交集，但缺历史指纹并使用过未排除六条记录的 validation，暂不直接接入正式训练。
  用户授权互联网研究及预训练权重使用、下游质谱塔/分子塔/rerank 结构与参数的逐项迭代，
  要求模型不过度复杂、能在本机训练；SpecEmbedding 质谱侧预训练模型结构不得改动。
  参考论文放入用户指定的论文目录；对标来源、可比性、近似达标阈值、下载位置与执行路线
  统一见上述计划。SOTA 目标未完成，不因当前单 seed 或局部指标改善而宣布达标。
  运行路径、独立 socket、状态入口及验证记录集中于计划。
- [NPLIB1 增强](paper-change-plans/2026-09-05-NPLIB1跨数据集增强.md)：计划状态为执行中，
  数据处理、重叠审计与实现 pilot 已记录，正式 GPU 矩阵尚未完成。该数据与 MassSpecGym
  有明显训练身份重叠，定位为第二套 in-domain 派生协议；去重 zero-shot 仅作小样本敏感性分析。
  中断的 CPU alignment 工件已隔离，不用于论文。
- [GLACIER 交接](glacier_reproduction_handoff_zh.md)：代码、权重与开放输入已下载；尚未
  安装独立环境或完成推理。来源与缺口见 [manifest](../analysis/glacier_reproduction_manifest.json)。

这些任务的状态不因文档整理而自动恢复或完成。正式训练必须全量且 GPU-only，详见仓库规则。

## 存储与发布

用户约定的外部工件路径：代码位于 `/data1/zyl/repos/<repo_name>/`，模型数据、权重及推理
工件位于 `/data1/zyl/<model_name>/`；例如 GLACIER 分别为 `ms-pred/` 与 `GLACIER/`。
本项目外部数据卷为 `/data1/zyl/SpecEmbedding/`。下载前记录来源、版本/commit、校验值和日期，
不把数据或外部仓库复制进当前工作树。

双语稿、参考文献和 PDF 在 `paper/`；页数、源版本和 SHA-256 只维护于
[release manifest](../paper/release/build-manifest.yaml)。现有匿名 ZIP 是已跟踪的历史交接
工件，其验收仅适用于 [验证记录](../reproducibility/anonymous-supplement-validation.yaml)
指定的源码版本，不代表最新代码。后续发布须重建并独立验收，细节见
[复现说明](../reproducibility/README.md)。目标 venue、作者/COI 和最终联合匿名检查仍见
[投稿待办](../paper/TRANSFER_2026_PLAN.md)。
