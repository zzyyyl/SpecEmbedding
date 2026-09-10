# 训练自然负例的结构相似度缓存与混合采样

这是[A10预登记方案](../attempt_10.md)的实现入口。新策略保持双塔、损失及每query负例预算，
只改变自然训练候选的选择分布；实现或输入预检不等于检索成绩。

## CPU准备

`prepare_training_similarity.py`只接受完整训练候选索引、既有固定Morgan缓存和新输出目录，
没有limit、GPU训练、覆盖或自动续跑选项。训练源重新完整核验后，按精确目标SMILES计算
每个源候选的位交集/并集；同一二维负例组对全部源条目取精确分数均值，降序排列，同分
按二维身份键处理。全部同身份正例排除，指纹相等不改变二维身份标签。

通过外部存储入口运行，路径替换为已核验的实际输入：

```bash
spec_storage_root=/data1/${USER}/SpecEmbedding
python run_with_storage.py --storage-root "$spec_storage_root" -- \
  python prepare_training_similarity.py \
  --data-path /path/to/prepared/MassSpecGym \
  --training-candidates /path/to/candidate_training_metadata_topk256.pkl \
  --fingerprint-cache /path/to/verified/train_fingerprints \
  --output "$spec_storage_root/training_candidate_cache/new_version_topk256"
```

`params.yaml`中的`training_candidate_sampling`是未激活的A10预设，提供指纹半径/位数和
采样参数。CPU准备生成`intersection.npy`、`union.npy`、`group_order.npy`及来源清单。
独立审计用解包位向量重算全部交并集，另行分组并验证完整精确排序；只有完整audit和
所有文件指纹一致时才能加载。已有目录拒绝重建，失败时保留现场再排查。

## 正式训练启用

先完成前序审计并绑定最终parent配置，保留其完整`candidate_supervision`字段，仅新增：

```yaml
train:
  align:
    candidate_supervision:
      enabled: true
      negative_count: 16
      loss_weight: 1.0
      pool_cache_size: 128
      graph_cache_size: 10000
      sampling:
        type: tanimoto_mixed
        near_count: 8
        near_pool_size: 32
        fingerprint_radius: 2
        fingerprint_bits: 2048
        cache_directory: /path/to/verified/training_candidate_cache
```

继承参数以最终parent为准。正式入口沿用显式`--alignment-training-candidates`及完整验证
路径，从配置读取新策略；不新增隐式CLI覆盖。旧配置没有`sampling`字段时仍为原schema1
均匀采样，抽样结果保持不变；新增字段采用schema2，未知类型、缺项、未激活监督、缓存
不匹配或向新策略传入旧索引都拒绝，不能静默回到均匀采样。

新策略先从相似度前min(32,N)组无放回抽min(8,B)组，再从全部剩余组补满B=min(16,N)，
打乱所选组后在各组内均匀选择一个原源条目。全部组和剩余组按二维键排序，组内按原源
位置排列，使用私有`SeedSequence(seed, epoch, raw_query_index)`；不消耗全局增强RNG。
小池、空负例query均保留。每轮保存真实query/源候选摘要，完成审计另行重建分组和抽样
过程、重放每一条query，不调用训练采样器来证明其自身正确。

## 来源、复用与失效

缓存绑定完整元数据、目标/分子字符串、二维身份、候选和query行映射、指纹来源与构造
参数、精确排序规则及实现SHA。训练输入/运行清单同时登记相似度文件和其指纹源载荷。
构造或来源变化使对应缓存失效，拒绝读取；有效的原图/指纹缓存不会仅因新采样策略失效。
修改预算或相似池大小时重新生成实验输入回执；完整结构排序本身与这些抽样参数无关。

训练worker只映射小型组顺序数组，按需维护有界组缓存；不加载完整指纹载荷来重新计算
相似度，不缓存模型嵌入。缓存文件保留在外部项目存储根；活跃任务引用的旧目录不能删除
或原地更新。正式训练继续按既定GPU准入、全量计数和独立完成审计规则执行。
