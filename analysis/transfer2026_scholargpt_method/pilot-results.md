# ScholarGPT 方法重构：无 forcing full-pool pilot

## 运行边界

- 运行日期：2026-08-23
- 代码/配置提交：`d28106bffb55e2d068e94bf03e4d5ac308bcd7e3`
- `params.yaml` SHA-256：`54a1314f99799c67f90526d77b686960725b143477e74cd5e554d7ce9ee7dd4a`
- 硬件：NVIDIA GeForce RTX 4090，`cuda:0`
- 训练：seed 42、5,000 条训练 query、3 epochs、batch size 16、K=256、relation top-40；不是多 seed 或全训练集主结果。
- 标签：训练和评价均不执行 positive forcing；mass/formula 训练 cache 日志均记录 `force_include_positive: False`、194,119 条 valid query、`labeled query fraction: 1.0000`。
- 协议：MassSpecGym supplied candidate files、本地 exact-target-SMILES single-positive；不是官方二维 InChIKey/multi-positive evaluator。

## 文件指纹

| 工件 | 相对路径 | bytes | SHA-256 |
|---|---|---:|---|
| mass train cache | `rerank_cache/scholargpt_relative_fullpool_mass/massspecgym_mass_train.pt` | 7,743,083,510 | `a7c3760a4d8f27c1ed9a119dddb2d4d0510b23afb616e723fc675f178605ba82` |
| formula train cache | `rerank_cache/scholargpt_relative_fullpool_formula/massspecgym_formula_train.pt` | 6,295,631,879 | `f0a5841a9820ec1950c67fb6cae118649a0ca82fc49bac0d85886568e213a910` |
| mass relative checkpoint | `checkpoints_rerank/scholargpt_relative_fullpool_experiments/mass/relative/best_reranker.pth` | 5,953,224 | `820073481a64a208e6335566f959030411c594f0f7951200263de95b164ebc41` |
| mass pointwise checkpoint | `checkpoints_rerank/scholargpt_relative_fullpool_experiments/mass/pointwise/best_reranker.pth` | 5,921,688 | `2c8793d665014932d8643ec46ae9db6874c3eb7aa14e82484e7c34421c939c8e` |
| formula relative checkpoint | `checkpoints_rerank/scholargpt_relative_fullpool_experiments/formula/relative/best_reranker.pth` | 5,953,288 | `886d59da213cc6193136d601246747532177f526a88782d937bcd5eb59cbc3f5` |
| formula pointwise checkpoint | `checkpoints_rerank/scholargpt_relative_fullpool_experiments/formula/pointwise/best_reranker.pth` | 5,921,688 | `df12d0dea1ca3315768ed96dd4d229a74b7b7875b712c662935252a834379273` |

## 测试结果

所有数值为百分比，除 MRR 外；完整池行使用 `--max-candidates 256`，K=40 行使用评价时无 forcing 截断。

| pool / evaluation | method | U | R@1 | R@5 | R@10 | R@20 | MRR |
|---|---|---:|---:|---:|---:|---:|---:|
| mass / K=256 | base | 100.0000 | 43.7799 | 61.0959 | 68.4324 | 75.5468 | 0.5219 |
| mass / K=256 | relative | 100.0000 | 50.0797 | 66.4160 | 73.5076 | 80.9125 | 0.5795 |
| mass / K=256 | pointwise | 100.0000 | 51.6348 | 67.7945 | 74.3905 | 81.2657 | 0.5935 |
| mass / K=40 | relative | 82.1542 | 50.5183 | 66.4958 | 72.8241 | 78.5145 | 0.5784 |
| mass / K=256, shuffled spectrum | relative | 100.0000 | 45.0501 | 62.8275 | 70.8704 | 78.5657 | 0.5358 |
| formula / K=256 | base | 100.0000 | 58.4871 | 71.4855 | 76.8569 | 81.8068 | 0.6479 |
| formula / K=256 | relative | 100.0000 | 66.7977 | 76.4696 | 81.3568 | 85.1390 | 0.7161 |
| formula / K=256 | pointwise | 100.0000 | 68.0508 | 77.7227 | 82.1713 | 85.7997 | 0.7282 |
| formula / K=40 | relative | 87.3035 | 66.3762 | 75.6038 | 80.2233 | 83.7605 | 0.7085 |
| formula / K=256, shuffled spectrum | relative | 100.0000 | 63.7902 | 73.3880 | 78.4005 | 82.6213 | 0.6860 |

## 解释边界

Relative improves its matched base in this pilot, but is below the capacity-matched pointwise control by 1.55 mass R@1 points / 0.0139 MRR and 1.25 formula R@1 points / 0.0121 MRR. The shuffled-spectrum audit leaves residual performance, so candidate and base-score pathways remain material. These are descriptive pilot observations, not evidence of a causal or generally stable relative-module gain. No MCES, official evaluator, external baseline, multi-seed aggregate, difficulty stratification, or efficiency claim is made from this artifact.
