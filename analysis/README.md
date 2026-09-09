# 实验证据索引

本目录包含不同阶段的实验与审计。下列范围以原始命令、cache/checkpoint 和 manifest 为准；
目录名中的 full、formal、canonical 或报告生成时的“当前”不代表新版全训练集协议。

| 系列 | 协议与用途 | 来源 |
| --- | --- | --- |
| v1.5 全量迁移 | 原矩阵已按用户要求中断，已有部分全量结果；转为单 baseline 优化，论文更新暂停 | [执行计划](../docs/paper-change-plans/2026-09-07-MassSpecGym全量重训.md)、[准确率排查](massspecgym_v15_regression_audit.md)、[身份转换修复](massspecgym_v15_inchi_fix.md) |
| Mentor 主矩阵与 relative 消融 | 36 个主运行、30 个消融；alignment 42--44，reranker 42--44；训练 cap=20,000、top-40，训练 forcing=true，val/test=false | [运行/哈希索引](../reproducibility/mentor2026_experiment_index.json)、[主矩阵](mentor2026_experiments/alignment_summary/manifest.json)、[消融](mentor2026_experiments/ablation_summary/manifest.json) |
| 无 forcing 方法 pilot | pre-overlap-clean alignment；5,000 query、3 epochs、top-256；先单 seed，再三 reranker seeds | [单 seed](transfer2026_scholargpt_method/pilot-results.md)、[第二阶段报告](transfer2026_scholargpt_review/second_stage_report.md)、[manifest](transfer2026_scholargpt_review/second_stage_manifest.json) |
| 旧 Transformer 跨 alignment | 固定旧 top-40 协议，Pointwise/Transformer 描述性对比 | [报告](transfer2026_alignment_multiseed/report.md)、[manifest](transfer2026_alignment_multiseed/analysis_manifest.json) |
| 旧 Transformer 组件消融 | alignment-42、五种组件移除，不能混作 relative 消融 | [报告](transfer2026_core_ablations/report.md)、[manifest](transfer2026_core_ablations/analysis_manifest.json) |
| top-40 身份审计 | overlap-clean alignment-42 保存 cache；exact SMILES 与二维身份的单 checkpoint 核对 | [报告](transfer2026_scholargpt_review/report.md)、[身份统计](transfer2026_scholargpt_review/identity_inventory_report.md) |
| 外部资源与新数据 | GLACIER 下载不等于推理完成；NPLIB1 数据审计不等于正式模型结果 | [GLACIER manifest](glacier_reproduction_manifest.json)、[NPLIB1 来源](../reproducibility/nplib1-data-sources.yaml) |

## 读取限制

- 2026-09-07 只读复核了 Mentor 索引的 66 条训练命令及 18 个唯一 cache：全部 top-40，
  6 个 train cache forcing=true、12 个 val/test cache=false。测试每池 17,556 query。
  全量重训已获准迁移至 v1.5，按 [执行计划](../docs/paper-change-plans/2026-09-07-MassSpecGym全量重训.md) 管理。
- 官方候选 JSON 顺序与二维身份的 full-pool 评价属于独立 pilot 的保存嵌入审计；
  不能继承给其它 alignment、top-K 或新缓存。主矩阵入口使用 local exact-target-SMILES 标签。
- manifest 冻结的报告保留原始字节，供数字与哈希追溯。旧报告称身份差异“未量化”时，是生成
  时点的边界；后续 alignment-42 top-40 审计另有报告，不据此声称其余 checkpoint 完成同样审计。
- 两种 reranker 相对各自 Base 的改善不证明 relative 的独立优势；配对方向依赖候选池、
  指标和 checkpoint。seed 汇总、query bootstrap 均按各自统计单位解释。
- 旧 Transformer 消融的六组配对是两个候选池 × 三个 reranker seeds，全部固定在 alignment-42，
  不是六次独立 alignment 重复。该说明集中于此，生成报告保持其冻结 manifest 对应的版本。
- JESTR-style cosine 是使用本地对齐嵌入的打分控制；GLMR 为 reported-only。
  cutoff 是评价截断，forward-only 测量不是部署 latency，synthetic smoke 不是科学结果。

最新执行状态与开放事项见 [项目记忆](../docs/project_memory_zh.md) 和
[投稿待办](../paper/TRANSFER_2026_PLAN.md)；本索引不复制各报告的实验表格。
