# 评审建议中尚未闭合的项目

以下事项不是由静态代码检查或历史工件自动完成的，当前不应写成论文结果：

1. 官方 MassSpecGym evaluator 的完整主表：需要与官方 loader 完全一致的身份、候选和多正例标签；现有 cache-only 二维 InChIKey 审计只能作为独立 identity audit。
2. JESTR/GLMR 或其它外部强 baseline 的 matched reproduction：当前没有统一 candidate handling、checkpoint 和 evaluator 的可审计运行。
3. 同一 commit、同一配置下的 3--5 个 alignment 重新训练：历史 alignment checkpoints 来自不同 revision，不能改称严格随机重复。
4. 逐 query paired bootstrap、结构相似度/基础分数间隔/峰数等困难度分层：除非保存新模型逐 query 预测，否则只记录为后续工作。
5. 完整 latency/吞吐/显存曲线：单次 GPU smoke 只能证明模型可运行；正式效率表需固定硬件、batch、精度和 warm-up 协议。

新模型若没有稳定优于容量相近的 pointwise 基线，论文主线必须回退为“监督式二阶段重排的条件性结果”，不能通过改写摘要来宣称相对候选机制已被证明。

