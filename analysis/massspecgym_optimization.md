# MassSpecGym 单 baseline 优化实验索引

目标、比较协议和验收条件见[阶段计划](../docs/paper-change-plans/2026-09-07-MassSpecGym全量重训.md)。
本索引记录优化决策，不替代外部运行目录中的配置、状态、原始日志及 SHA-256 工件。

## A01：按完整验证检索选择 alignment checkpoint

- 状态：实现和 synthetic 验证完成，正式队列待派发；性能收益未知。
- 固定起点：v1.5 r4 alignment seed42，原按验证对比损失选中 epoch15；来源见
  [准确率排查](massspecgym_v15_regression_audit.md)。不重新执行旧 12 组矩阵。
- 假设：验证对比损失最优的 epoch 未必具有最好的候选检索排名；直接观察完整验证检索
  可以改善选优，并区分 checkpoint 选择问题与表示/训练目标问题。
- 主要改变：每个完整 epoch 后，重新编码全部 Mass 验证候选与 query，按 Top-1、MRR
  顺序选 checkpoint，同时保存 Top-1/5/10/20/MRR 的非支配候选文件和完整轨迹。
- 固定条件：seed42、随机初始化、原有两塔结构/损失/增强/AdamW/学习率/温度/batch128；
  每轮 194,119 个 train query，19,423 个有效 val query，最多 100 epochs、patience5。
  验证索引和 DataLoader 使用独立 RNG，不应额外消耗训练随机数流。
- 评价基线：使用同一验证索引，对 r4 原 checkpoint 做一次 fresh encoding；避免把旧 fp16
  exact-SMILES cache 指标直接与新 float32、二维多正例指标比较。索引保留源顺序与重复条目，
  只沿用已审计的无效图排除，所有无正例 query 保留分母。
- 同分规则：主指标对应 `torchmetrics==1.8.2` 的逐 query CPU `argsort(descending=True)`，
  padding 先移除；稳定排序另存敏感性视图。MRR 按同一排名扩展，不声称它是官方已报告指标。
- 资源：单卡显式 CUDA，物理 GPU 0/1 择一等待；分子编码 batch512、谱图 batch128、CPU workers4。
  现有 val cache 约 82.8 万个唯一分子，新索引的最终数量以实际审计为准。逐轮记录验证耗时，
  先实测再决定是否需要图缓存；不为省时减少 query 或候选。
- 队列阶段：导入已审计数据 → CPU 验证索引 → 固定 checkpoint 验证 → 单 seed alignment。
  该队列不包含 test 推理或 reranker，完成只表示 A01 候选完成。
- 保留规则：依据完整验证 Top-k/MRR 的改善和代价决定是否成为新 incumbent；若只有选优
  改善而表示能力仍不足，下一项按计划考虑参数、预训练来源及候选对比目标，不转向 test 调参。
- 验证：53 项专项通过；全仓 239 通过、1 跳过、1 个既有内部路径审计失败；Ruff、compileall、
  diff 检查通过。新增指标测试使用真实 `torchmetrics 1.8.2`，覆盖同分、多正例和空正例。

原始运行地址、源码 commit 和后续结果在派发后回填；当前没有新的性能结论。
