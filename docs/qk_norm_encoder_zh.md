# 下游谱图Q/K归一化

该组件供[A08](../attempt_8.md)准备使用，默认不启用，尚无正式检索成绩。它保持双塔共享
嵌入及对比学习，只在谱图注意力的Q/K投影后按每头特征维做RMSNorm；V、输出投影、原
LayerNorm、残差、前馈网络和池化均继承，不改`SpecEmbedding/models.py`预训练结构。

对每层的Q和K分别计算`x / sqrt(mean(x²) + eps) * weight`，`eps=1e-6`，缩放向量初值为1。
同层各头共享一个head_dim向量，Q与K不共享；512维、16头、4层总计新增256参数。注意力
仍用原`1/sqrt(head_dim)`缩放，不增添文本位置编码或因果掩码。此参数预算不是显存或提速
实测；Q/K幅值变化是否改善峰关系学习，只能由完整验证结果决定。

实现位于`SpecEmbedding/models_qk_norm.py`，由现有下游谱图工厂显式安装到新实例。
每个`QKNormEncoderLayer`继承原层的模块与state_dict键，新增Q/K缩放参数；构造不再随机
初始化QKV、FFN等权重，因此相同seed下除新增常量参数外，保留两塔原有初始权重和RNG。
新层使用显式前向和PyTorch SDPA，eval和inference_mode也必须经过归一化，不能由原生
TransformerEncoderLayer融合快路径跳过。保留原attention/FFN dropout训练语义，推理关闭。

配置参数集中在`params.yaml`的`qk_norm_encoder`；准备新运行时显式复制到
`model.spec_encoder.qk_norm`。只添加根段不会激活模型。与差值分支可独立或同时构造，
同时构造时保留parent已有差值配置，不能借配置继承引入额外未登记改动。旧配置仍构造
原模型；带Q/K归一化的checkpoint须保存完整配置并严格重载，缺失新参数即失败。

谱图只接受完整双向注意力与padding，拒绝因果或额外attention mask；padding支持bool
及框架转换后的0/负无穷表示，空谱或全padding拒绝。SDPA的布尔语义为True允许参与，
与key padding的True排除相反，组件显式转换；padding特征在层间置零。正式输入仍需按
既有tokenizer、有效峰、完整query及候选协议审计，不能用组件检查代替数据检查。

缓存输入构造没有改变；原token、图、候选索引和固定指纹按原来源核验后可复用，学习
embedding仍随权重重编码。正式入口要求baseline按自己的selection独立构造，不能用候选
Q/K配置加载旧A04权重。新运行继续使用外部存储，当前A07固定源码与队列均不受开发改动影响。
