# v1.5 与旧版准确率差异排查

核验日期：2026-09-09。对象为旧 overlap-clean alignment-42 / Mentor 主矩阵与新 r4 v1.5
全量队列。本文记录实现与保存工件审计，不是新增正式训练，也不是完整官方评价。

## 已确认的来源与结果

旧 Mass Base Top-1 为 47.4596%，relative 三 seed 为 57.4277/57.6612/58.5384%；
新 Mass Base Top-1 为 5.7018%，relative 为 9.1023/8.7321/8.9941%。
以上均逐条核对原始 `eval_rerank.log`，分母均为 17,556。下降在 reranker 之前已出现，
不能将它归因于 reranker batch 32→8、训练 query 增加或去除训练 forcing。
这些变化可能影响第二阶段，但没有直接参与 Base 打分。

旧 test cache 是 Base 排序后的 top-40，新 test cache 是最多 256 个自然候选。
对固定分数和相同候选集合，保留 top-40 或 top-256 不会改变 Base 的第一名；
因此截断大小本身也不能解释该 Base 落差。新旧分子表示和 alignment 同时改变，不能据此
把全部落差分解为某一个因素的因果贡献。

## 未发现旧权重或旧 TokenSet 被复用

- 旧 alignment 权重实际 SHA-256：`ad5d1eb76805c51563349f259a4b4c935336064b6171a4e650472b78aeeaa06f`。
- 新 alignment 权重实际 SHA-256：`d58b6c0a073486dcd07558bce841ca5fc0afc297fc4b8393639e586eaa35f9a3`。
- 六个新 cache 的 checkpoint 指纹全部为新值，数据版本均为 1.5、图策略均为
  `rdkit_sanitized`、top-K=256、limit=0、forcing=false；实际文件 SHA-256 全部匹配阶段审计记录。
- 六份 prepare 日志均指向新 checkpoint 和本轮数据目录；加载函数会校验权重哈希、
  模型配置、图策略与 RDKit 版本，再 `strict=True` 加载该文件并进入 eval 模式。
- 旧 `alignment_selection.json` 确实记录了 `train_cache/tokenset_massspecgym.pkl`。
  新记录是 `tokenset_cache=null`；`--formal-fulltrain` 明确拒绝 TokenSet 和旧预训练权重，
  走 `classified_full_spectra`，从本轮 train/val pickle 重新 tokenize，并在内存中按二维身份分组。
  `get_classified_data` 仅在非正式兼容分支调用，正式分支没有默认 cache fallback。
  各 epoch 复用进程内 token 数据，不是每轮重新 tokenize。
- 新 r4 导入的是 r3 已完成审计的 **v1.5 数据文件**，未导入其训练状态或 embedding。
- 新 checkpoint 的 fresh GPU 编码检查通过：固定分散选取 43 条测试谱图及其真值/首位候选
  共 76 个不同缓存分子，覆盖 511/512、1023/1024 批边界；从源 pickle 重新 tokenize 和构图。
  与缓存相比，谱图向量最大绝对误差 `5.96e-8`，分子向量为 `5.95e-5`（缓存 FP16）；
  最小 cosine 均大于 `0.9999998`。这是部分样本的编码实现核验，不能写成全量 fresh loader 评价。
  审计先等待物理 GPU 1 满足既定 120 秒门槛，再将其 UUID 显露为 `cuda:0`；审计进程已退出。

## 对四份 test cache 的独立检查

对旧/新 × Mass/Formula，每份遍历全部 17,556 query：

- label 对应的候选字符串、谱图索引、候选来源检查均无错误；test forcing 标志均为 false。
- 从保存的谱图和分子 embedding 独立重算点积，与保存分数的最大绝对误差不超过 `3e-7`。
- 新缓存所有列表均等于源候选的字符串集合（按既定图资格排除后）；旧 cache 只是 top-40 子集。
- 新旧测试目标的二维身份顺序哈希一致；在各自缓存分数首位上，exact-target-SMILES 与
  二维 InChIKey 判断的命中 query 数完全相同。它排除了此次 Base Top-1 差距由标签字符串
  等价判断引起，不能替代全部 reranker 排序的官方顺序/多正例审计。
- 审计 JSON 的 `base_metrics` 使用严格大于真值分数的计数，属于并列情况下的乐观名次；
  `checks.exact_top1` 使用缓存顺序下的首个最大值。二者不可混同，正式结果引用原始评价日志。

## 已量化的旧版表示偏差

旧测试目标 SMILES 的芳香性小写标记出现率是 **0/17,556**，新版本为
**14,422/17,556（82.15%）**。旧候选池中非真值候选使用芳香性标记的比例很高：
按 query 内干扰候选比例再取平均，Mass 约 90.03%，Formula 约 82.88%。
这表示旧版真值与干扰分子存在可识别的输入格式差异，并不表示旧真值化学结构没有芳香环。

为量化该偏差，对完整原始候选列表做一个不使用谱图、不使用模型、不使用 Base top-K 的
解析审计规则：优先选择 **不含芳香性小写原子标记** 的 SMILES；有多个时均匀选择，
没有时从完整列表均匀选择。报告均匀选择的精确期望命中率，无随机抽样或拟合。

| 源候选协议 | 旧 v1 | 新 v1.5 | 完整候选均匀随机的期望 |
| --- | ---: | ---: | ---: |
| Mass | 15.35% | 0.61% | 0.40% |
| Formula-conditioned | 64.42% | 3.36% | 3.17% |

这是候选表示泄露目标身份线索的直接证据，**不是证明旧模型取得的每个命中都依赖该线索**。
模型依赖路径也确实存在：旧 `smiles_to_graph` 使用 `sanitize=False`，显式编码
`GetIsAromatic()`、键类型及共轭性。因此图模型没有自动消除字符串写法差异。
新流程统一 v1.5 SMILES，并在训练和缓存编码中都使用 sanitized graph，移除了这条明显的捷径。

官方 v1.5 [数据说明](https://huggingface.co/datasets/roman-bushuiev/MassSpecGym/blob/main/README.md)
指出候选沿用原分子并统一 RDKit 表示；相关原始研究
[MassSpecGym in the Wild，§5.1](https://arxiv.org/html/2606.19624v1#S5.SS1)
专门识别了真值与干扰候选的 canonicalization 差异。外部结论与本地审计机制吻合，
但其模型数字不用于本项目排序或宣称本项目已完成相同的因果消融。

## 解释与尚未完成的范围

目前最强的解释是：旧评价含可利用的表示捷径；新版消除了该捷径，同时重训了 alignment，
暴露出较弱的实际排序能力。不能把旧高分直接视为新版应恢复的可信基准，也不能以恢复旧
构图或旧缓存作为“修复”。新版低分本身仍需要改进。

尚未隔离的因素包括 alignment 从分子抽谱到逐谱训练的权重变化、二维正例分组、
验证批次和选择方式；这些因素与分子表示一起发生变化，现有证据不能定量归因。
对旧模型依赖程度的规范表示消融、更多 alignment 训练对照属于后续实验，需先确认计划。
完整新缓存官方候选身份/顺序审计以及三 seed pointwise/relative 比较仍未完成。

本次没有修改固定运行源码、停止或重复派发正式队列，也没有改写论文。

## 原始工件与复核

独立审计目录：`/data1/zyl/SpecEmbedding/audits/v15_regression_20260909/`。
`receipt.json` 记录固定源码 commit、执行说明、脚本和以下工件的 SHA-256：

- `audit_v15_regression.py`、`specembedding_v15_regression_audit.json/.log`：四 cache 全 query
  的索引、分数、来源、首位二维身份检查，以及完整原始候选上的格式规则解析审计。
- `specembedding_v15_six_cache_fingerprints.json`：六 cache 的内容指纹和协议断言。
- `audit_v15_fresh_encoding.py`、`launch_v15_encoding_audit.py`、
  `specembedding_v15_fresh_encoding.json`、`specembedding_v15_encoding_gpu_selection.json`、
  `specembedding_v15_encoding_audit.log`：分散样本重新编码、设备选择和误差断言。

审计代码保留在该内部目录，绝对输入路径只适用于本机；不并入匿名补充包。没有改变训练代码，
没有重跑全仓测试；本次验证为上述实际审计与文档 `git diff --check`。
