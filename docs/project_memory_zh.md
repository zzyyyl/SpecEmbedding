# SpecEmbedding 项目记忆

最后更新：2026-09-07。本文是内部状态摘要，不放入匿名补充包；事实以代码、配置和可校验工件为准。
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

- [MassSpecGym 全量重训](paper-change-plans/2026-09-07-MassSpecGym全量重训.md)：**执行中，v1.5 迁移**。
  当前 processed 数据已核验为 v1。用户已授权采用 v1.5，并在审计证明必要时重训 alignment。
  旧监测器于 19:48 停止，未派发训练，原锁、日志和固定源码保留。
  旧构图的芳香性/键类型依赖 SMILES 表示，迁移需规范构图并重训一个 alignment-42；
  正式 alignment 已增加每轮全谱遍历与数量校验。首个 v1.5 队列于 20:17 在候选审计阶段停止：
  官方源含 RDKit 无法构图的候选，未启动 GPU，失败工件及 `6843e12` 固定源码保留。
  修正为显式记录现有候选编码器的无效图排除（保留源候选顺序/内容，正例和 query 不得丢失）；
  修复通过验证后使用新目录和新源码重新执行。完整 CPU 审计通过后才等待 GPU，重训一个
  alignment-42，再运行 12 个全量 reranker 组合；不能称为三 alignment 矩阵。
  运行路径、独立 socket、状态入口及验证记录集中于计划。尚无新 GPU 训练结果。
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
