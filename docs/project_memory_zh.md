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
  保持 v1.5、全量、无 forcing 和双 GPU 择一等待；入口已收窄范围，尚未派发新训练。
  先针对 alignment 的真实验证检索选优做单项改进，当前仍按验证对比损失选 alignment；
  Top-k 检索选优实现尚待完成，不能写成已实现或已提升。调参仅看验证集，矩阵后置。
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
