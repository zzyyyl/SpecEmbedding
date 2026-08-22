# ScholarGPT 第二阶段方法复核记录

## 运行边界

- alignment：同一 pre-overlap-clean `d4c1f70` checkpoint；质量与分子式候选池分别训练。
- reranker：无 rank、无 positive forcing，完整缓存池最多 256 个候选；relative 分支只处理 coarse top-40。
- pilot：每个候选池使用 5,000 个训练 query、3 个 epoch、reranker training seeds 42/43/44；seed 不是独立 alignment 重复。
- 评价：17,556 条测试 query；K=20/40/80 是同一 256-candidate cache 上的 evaluation-only cutoff，不是 K-specific retraining。
- 代码/硬件：训练实现提交 `488a04a`，逐 query 评估工具提交 `8ce2655`，NVIDIA RTX 4090。

## 三 seed 主对照

均值 ± 样本 SD；Base 在同一 alignment/cache 上不随 reranker seed 变化。

| pool / model | R@1 | R@5 | R@20 | R@40 | MRR |
|---|---:|---:|---:|---:|---:|
| mass base | 43.78 | 61.10 | 75.55 | 82.15 | 52.19 |
| mass pointwise | 51.37 ± 0.54 | 67.34 ± 0.39 | 80.71 ± 0.55 | 86.03 ± 0.61 | 59.03 ± 0.44 |
| mass relative | 50.98 ± 0.87 | 67.06 ± 0.67 | 80.49 ± 0.55 | 86.13 ± 0.70 | 58.71 ± 0.76 |
| formula base | 58.49 | 71.50 | 81.81 | 87.30 | 64.79 |
| formula pointwise | 66.65 ± 1.30 | 76.76 ± 0.87 | 85.33 ± 0.42 | 89.43 ± 0.29 | 71.66 ± 1.04 |
| formula relative | 66.32 ± 0.46 | 76.29 ± 0.23 | 85.08 ± 0.07 | 89.31 ± 0.06 | 71.29 ± 0.33 |

结论边界：两种监督 reranker 均超过 matched base；relative 没有在主 R@1/MRR 汇总中超过容量匹配 pointwise，因此不主张 relative branch 的独立普遍收益。

## Query-level paired analysis

seed-42 保存了四组逐 query prediction JSON（mass/formula × pointwise/relative），每组 17,556 行，包含 base/rerank rank、candidate count 和 K 截断指标。mass relative 相对 base 的 paired bootstrap（2,000 次，seed 20260823）为 R@1 差 `+6.30` 个百分点，95% CI `[+5.69,+6.93]`；MRR 差 `+0.0576`，95% CI `[+0.0530,+0.0625]`。这是 query-level descriptive uncertainty，不替代独立 alignment 或 retraining inference。

## 2D identity audit

`relative_identity_eval.json` 对新 relative full-pool seed-42 test cache 应用 MassSpecGym 1.3.1 的 2D InChIKey prefix transform。两个候选池均为 17,556/17,556 query，multiple-positive query=0、candidate identity collision=0；local exact-SMILES 与 reference 2D identity 的 base/rerank 指标完全相同。该结果是 cache-level identity audit，不是官方 loader 的完整重跑，也不覆盖未保存的外部候选池。

## 固定硬件前向审计

RTX 4090、batch size 64、同一 test cache 的 forward-only 测量：relative mass 0.083 ms/query、302 MB peak allocation；pointwise mass 0.030 ms/query、244 MB；formula relative/pointwise 分别 0.076/0.029 ms/query、302/244 MB。这些是实现审计，不构成端到端吞吐或效率优势声明。

## 信息来源与机制消融（seed 42）

以下均为同一 5,000-query/3-epoch pilot 的单 seed R@1/MRR；relative full 参照为 mass `50.08/.5795`、formula `66.80/.7161`。

| variant | mass R@1 / MRR | formula R@1 / MRR |
|---|---:|---:|
| score only | 43.78 / .5219 | 58.49 / .6479 |
| embedding only, no relative module | 50.07 / .5808 | 66.13 / .7110 |
| embedding + base, no relative module | 50.07 / .5798 | 66.28 / .7141 |
| candidate only | 45.00 / .5245 | 52.73 / .5960 |
| no spectrum conditioning | 50.76 / .5829 | 64.25 / .6925 |
| no molecular relation | 50.94 / .5889 | 66.15 / .7112 |
| no antisymmetric pairing | 50.75 / .5846 | 66.11 / .7109 |

解释边界：base-score-only 没有带来提升，embedding 分支贡献了主要 pilot 增益；相对关系及其子组件在两个候选池方向不一致，不能作独立因果收益或组件排序结论。

## 工件

- 逐 query：`pilot_predictions/`；聚合：`multiseed_summary.json`。
- 2D identity：`relative_identity_eval.json`。
- 训练 checkpoint 和 cache 位于本机/内部路径，未随匿名补充包发布。
