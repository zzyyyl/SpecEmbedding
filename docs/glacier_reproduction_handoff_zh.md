# GLACIER checkpoint 推理复现交接

状态：**独立环境、完整 checkpoint 推理和四项评价已完成；与论文及本项目的完整同协议对标待完成**。
2026-09-11 已只读核验外部任务于 2026-09-09 06:45:18 UTC 完成，并独立复核保存分数。
不要根据旧交接的“尚未运行”状态重复安装环境或派发推理。

当前完成证据、补算 Top-10/MRR 及排序/身份边界见
[完整分数对标审计](../analysis/glacier_reference_bridge_20260911.md)，来源与回执位置集中于
[阶段计划第10节](paper-change-plans/2026-09-07-MassSpecGym全量重训.md)。
`analysis/glacier_reproduction_manifest.json` 保留首次下载来源，不能替代后续完成工件。

## 已准备资源

- 官方仓库：`/data1/zyl/repos/ms-pred/`，上游 commit
  `ed8311f22958cb37f055b663b5f56c5c77a2ee33`，当前干净的独立复现源码为
  `c0cb5bc0c81ca13c8390754bf23ae8c205f6e584`；
- GLACIER 对比微调 checkpoint：`/data1/zyl/GLACIER/checkpoints/best.ckpt`；
- MassSpecGym 1.5 TSV、MGF 和官方 mass/formula 候选 JSON：
  `/data1/zyl/GLACIER/data/`；
- ms-pred 作者提供的 formula 候选 TSV：
  `/data1/zyl/GLACIER/data/msg/retrieval/`；
- 原始下载包保留在 `/data1/zyl/GLACIER/downloads/`。

checkpoint 的只读检查显示：epoch 23、global step 74352、约 15.1M state
elements，`contr_weight=1.0`、`contr_threshold=0.5`、entropy contrastive loss，
因此它是带对比目标的 GLACIER 权重，而不是未微调版本。

## 已完成范围与剩余边界

独立 CUDA 环境、完整谱图准备、Mass/Formula 官方候选表、43 个预测分块及完整 test
评价均已有工件；348 项完成清单文件重新计算 SHA 一致。此次只读取现有复现，未改动
外部源码、模型、候选或日志；所有派生分数审计保存在 SpecEmbedding 外部审计目录。

后续先核对已有完成回执和原始工件。需要补齐的是公开 checkpoint 与论文最优结果的
绑定、CE 填补来源、候选规范化和身份、查询分母及排序规则的完整对齐；不能因为已有
Top-10/MRR 就称为论文 SOTA 的统一复现。目标身份重算已确认五条 query 与 TSV 键不同，
没有修改标签；全部候选身份的再次重算仍未完成。原始和派生结果分别保留。

本地优化继续使用完整有效验证集决定模型，外部 test 参考不用于选择本地超参数。
先对齐候选协议和评价器再决定可比范围；此前强参考快照与 SOTA 门槛保持不变。

## 当前不能执行的外部基线

GLMR 暂未发现作者公开的官方代码和训练 checkpoint。ChemFormer 只是其初始化权重，不能替代
GLMR 的预检索、cross-fusion 和生成解码器。因此 GLMR 继续保持 reported-only；除非作者
提供工件，不将自行重实现数字写成官方复现。
