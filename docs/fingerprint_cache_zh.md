# 完整候选 Morgan 指纹输入

这是后续轻量分子表示实验的CPU输入准备工具。当前A04/A05仍使用GINE；本入口不训练或
评价模型，也不自动给alignment或reranker切换输入。SpecEmbedding质谱预训练结构不变。

`params.yaml`的`molecule_fingerprints`固定radius、位宽、CPU进程数和chunk大小。当前为
radius2、2048位、8个进程；不含手性、计数模拟或模型embedding。使用标准化构图语义的
已审计自然候选，保留全部SMILES条目及原索引顺序。不会按指纹相同合并候选或生成正例标签。
有限指纹可以对应不同二维结构，正式实验仍须使用候选索引中的二维身份与完整query分母。

先设置以下变量为已有可信输入与新的输出目录，输出名保留`_topk256`。在`specembedding`
环境中运行，长任务放入明确指定socket的detached tmux；启动前检查已有任务，避免重复执行。

```bash
CUDA_VISIBLE_DEVICES='' python prepare_candidate_fingerprints.py \
  --data-path "$SPEC_DATA" \
  --training-candidates "$TRAIN_CANDIDATES" \
  --output "$TRAIN_FP_OUTPUT"

CUDA_VISIBLE_DEVICES='' python prepare_candidate_fingerprints.py \
  --data-path "$SPEC_DATA" \
  --validation-index "$VALIDATION_INDEX" \
  --output "$VAL_FP_OUTPUT"
```

训练输入是完整且已独立核验的自然候选metadata，不是旧rerank缓存；验证输入是完整验证
candidate/identity索引及其相邻回执。入口不支持test、limit或断点续写。已有输出直接拒绝，
失败时保留部分文件，排查后使用新目录。

每个分子2048位packed为256字节，固定存储于只读映射的`fingerprints.bin`。读取器只将请求
的行展开为float32，保留请求顺序及重复行；返回值可修改而不污染底层缓存，worker重启后
重新打开映射。全部候选不需同时展开为float32，也不需要在GPU保留整库固定特征。

`manifest.json`绑定完整输入顺序、候选索引与数据manifest、参数、NumPy/RDKit版本、构造
源码及字节指纹；`audit.json`记录重新打开缓存后，使用另一RDKit Morgan接口重新计算所有
分子的位值并逐位比较。`preparation.json`仅在来源前后核验完成后写入，记录完整来源和范围。
审计和预检不是模型训练或速度收益；正式模型接入还需把缓存回执绑定到训练、逐轮验证及
完成审计，并核验两塔梯度与完整训练样本。当前输入准备不能替代这些后续步骤。

轻量分子塔的组件及后续集成要求见[A06实验卡](../attempt_6.md)。分子MLP配置集中于
`params.yaml`的`fingerprint_encoder`，模型及两侧投影每轮训练，只有Morgan输入是固定的。
读取器沿用原二维query标签与负例采样，完整检索验证每轮重编码分子，不复用旧embedding。
固定指纹要求节点/边图增强参数显式设为0，谱图增强仍走原实现。组件尚未接入正式CLI，
不能仅添加该配置就视为已切换模型；正式parent/新模型的分别加载和输入绑定必须先完成。

## 跨轮复用与失效清理

epoch、batch、学习率或模型权重变化不使固定Morgan输入失效；模型embedding仍需重新编码。
候选索引/顺序、数据来源、指纹radius/位宽/构造选项、相关缓存实现或RDKit/NumPy版本变化时，
旧缓存不能继续给新运行使用，必须重新构建并审计新版本目录。

用户已要求清理失效缓存：先核对训练、准备与审计进程的实际引用，确认没有仍在读取旧缓存的
任务，再删除该失效版本的二进制载荷；保留来源、manifest、审计回执及明确的清理记录。
不删除源数据、权重或失败日志，也不覆盖原有回执来伪装成新缓存。正在使用的版本先保留，
待相关任务退出后再清理；不得因新实验更换模型而清除构造规则未变、仍然有效的固定输入。
