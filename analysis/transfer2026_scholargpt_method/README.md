# ScholarGPT 方法重构 pilot

本目录保存 2026-08-23 方法重构的历史单 seed pilot，不能作为当前正式运行指南。

- [pilot-results.md](pilot-results.md)：5,000-query、3-epoch、seed-42、top-256、no-forcing
  实验的配置、缓存/模型指纹与结果；使用 pre-overlap-clean alignment。
- [第二阶段复核](../transfer2026_scholargpt_review/second_stage_report.md)：后续三 reranker
  seeds、官方候选顺序/二维身份、逐 query、困难分层和 forward 审计。
- [证据总索引](../README.md)：与旧 top-40 矩阵及待完成全量训练的区别。
- [当前实现与命令](../../docs/reranker_solution_zh.md)：正式实验不得限量，不能直接复用 pilot 命令。

初期“尚无 bootstrap、困难分层和效率记录”的待办已被第二阶段工件替代，后续未完成工作统一见
[投稿待办](../../paper/TRANSFER_2026_PLAN.md)。
