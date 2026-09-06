# GLACIER checkpoint 推理复现交接

状态：代码、checkpoint 和开放原始输入已下载并校验；尚未安装独立环境，尚未运行推理或训练。

详细来源、大小、SHA-256、checkpoint 元数据和缺口以
`analysis/glacier_reproduction_manifest.json` 为准。本文只给出后续会话的执行边界。

## 已准备资源

- 官方仓库：`/data1/zyl/repos/ms-pred/`，固定 commit
  `ed8311f22958cb37f055b663b5f56c5c77a2ee33`；
- GLACIER 对比微调 checkpoint：`/data1/zyl/GLACIER/checkpoints/best.ckpt`；
- MassSpecGym 1.5 TSV、MGF 和官方 mass/formula 候选 JSON：
  `/data1/zyl/GLACIER/data/`；
- ms-pred 作者提供的 formula 候选 TSV：
  `/data1/zyl/GLACIER/data/msg/retrieval/`；
- 原始下载包保留在 `/data1/zyl/GLACIER/downloads/`。

checkpoint 的只读检查显示：epoch 23、global step 74352、约 15.1M state
elements，`contr_weight=1.0`、`contr_threshold=0.5`、entropy contrastive loss，
因此它是带对比目标的 GLACIER 权重，而不是未微调版本。

## 后续会话先完成的工作

1. 阅读官方 README 和本交接清单，在 `ms-pred` 内建立独立 CUDA 12.4 环境；不要改动
   SpecEmbedding 的 `specembedding` 环境。
2. 将外部数据通过显式软链接接入 `ms-pred/data/spec_datasets/msg/`，不要复制到本仓库，
   也不要把数据加入 Git。
3. 从已校验的 MassSpecGym 1.5 mass JSON 生成
   `cands_df_test_mass_256.tsv`。官方 Dropbox 候选包虽然标为 full/deduplicated，实际只含
   formula TSV 和两个 ICEBERG checkpoint，不能把它误当作 mass 候选已经齐备。
4. 从已校验的 TSV 或 MGF 构建 `spec_files.hdf5`，并核对 231,104 条数据、test query
   数量、二维 InChIKey 规则、候选覆盖和候选截断。
5. 先用很小的候选子集做 GPU checkpoint smoke，再运行完整 mass/formula 推理。训练和正式
   推理都必须显式使用 GPU；若 GPU 高负载或显存不足则等待，不能回退到 CPU。
6. 不要原样运行 `run_scripts/glacier/03_run_retrieval.py`：当前脚本启用的是 NIST20，
   MassSpecGym 配置仍被注释。先复制为本地配置并固定 checkpoint、候选与输出路径。
7. 只有在候选协议和 evaluator 与目标比较表完全匹配后，才可把结果与 SpecEmbedding 并列。

## 当前不能执行的外部基线

GLMR 暂未发现作者公开的官方代码和训练 checkpoint。ChemFormer 只是其初始化权重，不能替代
GLMR 的预检索、cross-fusion 和生成解码器。因此 GLMR 继续保持 reported-only；除非作者
提供工件，不将自行重实现数字写成官方复现。
