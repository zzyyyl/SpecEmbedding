# 第二阶段 pilot 的证据缺口

本文件只描述 2026-08-23 pilot 的范围；已完成内容见
[second_stage_report.md](second_stage_report.md)，当前工作安排见
[投稿待办](../../paper/TRANSFER_2026_PLAN.md)。

以下工作未由该 pilot 完成：

- 独立外部 retriever 和统一候选构造下的 baseline 复现；reported-only JESTR/GLMR 不能替代。
- fresh official-loader 谱图/分子重编码；既有候选顺序与二维身份审计只使用保存嵌入。
- candidate-aware alignment 重训和独立 alignment 层级推断。
- 不同候选规模的分别重训及端到端部署成本；已完成的 cutoff 为评价截断，效率为 forward-only。

三 reranker seeds、query bootstrap、困难度分层和机制控制已在后续报告中记录，不再列为待完成。
上述边界不预设 relative 优于 pointwise，也不构成新的实验结果。
