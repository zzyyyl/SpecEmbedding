## SpecEmbedding

SpecEmbedding 是一个专注于 MS/MS 光谱嵌入的深度学习模型，结合了正弦位置编码与监督对比学习策略，旨在提升化合物鉴定与结构相似性检索的性能。

模型的训练与测试基于经过 MSBERT 团队初步清洗的 GNPS、MoNA 和 MTBLS1572 数据集。在此基础上，我们进一步去除了一些格式错误和无效的 SMILES 序列，以提升整体数据质量。所有清洗后的数据及相关处理脚本现已发布至 [figshare](https://doi.org/10.6084/m9.figshare.28876751.v2)，包括用于模型评估的 10 次查询集与参考集随机划分结果。

为扩展到其他整理后的谱图数据，我们还在 MassBank 和 MassSpecGym 上进行了评价；相应结果
与协议见所链接工件。增加数据集覆盖本身不构成一般性的跨域稳健或泛化保证。

为确保实验评估的公正性，我们严格保留了 MSBERT 设定的训练集划分，仅在测试集上进行了多次随机划分，并基于这些划分结果报告了 hit@k 的平均值与标准差。

此外，关于超参数搜索空间、模型训练细节、消融实验结果，以及各模型在不同任务下的完整评估指标，也一并公开存储在 [figshare](https://doi.org/10.6084/m9.figshare.28876751.v2) 中，供他人复现与参考。

更多关于评估的结果以及细节可参照 [SpecEmbedding-Comparison](https://github.com/sword-nan/SpecEmbedding-Comparison)

### 1. 环境配置

2026-08-17 完成验收的事后复现环境为 Ubuntu 20.04、Python 3.12.13、
PyTorch 2.5.1、CUDA runtime 12.1 与 cuDNN 9.1。该记录是复现/验收快照，
不证明历史训练运行逐包使用完全相同的环境。`environment.yml` 是持续维护的环境
recipe；论文的版本化 manifests 另行冻结数据、候选集、checkpoint、参数与分析工件。

在仓库根目录创建 canonical Conda 环境：

```bash
conda env create -f environment.yml
conda activate specembedding
```

若已有开发环境，可按受跟踪 recipe 同步。该命令用于更新工作环境，不是历史精确 lock：

```bash
conda env update -n specembedding -f environment.yml --prune
conda activate specembedding
```

> ⚠️ 注意：在 Windows 系统上运行时，如发现计算余弦相似度矩阵出现异常结果，请将 @njit 装饰器注释掉（该装饰器来自 numba），即可恢复正常。

### 1.1 开发与测试工具

Pytest 是仓库验收测试的统一入口，已加入 `environment.yml`。Ruff 用于 lint 与 import
排序，保留在 `requirements-dev.txt` 中，不作为运行时依赖。

需要时安装开发依赖：

```bash
python -m pip install -r requirements-dev.txt
```

从仓库根目录运行完整测试：

```bash
python -m pytest -q
```

运行 Ruff 检查：

```bash
ruff check .
```

应用 Ruff 的安全自动修复：

```bash
ruff check . --fix
```

### 2. 分子图特征

在质谱-分子对齐任务中，SpecEmbedding 使用图同构网络 (GINE) 对分子结构进行编码。为了捕捉质谱 (MS) 碎片预测所需的关键化学细节，我们采用了一套精简的原子和键特征：

#### 原子特征 (Node)
- **原子序数 (Atomic Number)**：原子的身份（B, C, N, O, F, Si, P, S, Cl, Br, I 或其他）。
- **形式电荷 (Formal Charge)**：电荷状态，决定电离和碎片化过程。
- **度 (Degree)**：相邻原子数，反映连接性。
- **总氢原子数 (Total Hydrogens)**：隐式和显式氢原子数。
- **芳香性 (Aromaticity)**：指示原子是否属于芳香系统。
- **环成员 (Ring Membership)**：原子是否属于环状结构。
- **环尺寸 (3-6)**：针对 3, 4, 5, 6 元环的特定标识（对碎片断裂能至关重要）。

#### 键特征 (Edge)
- **键型 (Bond Type)**：单键、双键、三键或芳香键。
- **共轭 (Conjugation)**：化学键是否属于共轭系统。
- **是否在环内 (In Ring)**：化学键是否在环内。

### 3. 示例展示

#### 3.1 计算 query 和 reference 的余弦相似度矩阵
```python
import sys
sys.path.append("..")

from SpecEmbedding.utils.model import embedding, cosine_similarity, load_tanimoto_supcon_aug_model
from SpecEmbedding.utils.clean import read_raw_spectra
from SpecEmbedding.trainer.trainer import ModelTester
from SpecEmbedding.data.tokenizer import Tokenizer

# Load query and reference spectra
q = read_raw_spectra("./q.msp")
r = read_raw_spectra("./r.msp")

# Initialize tokenizer and device
tokenizer = Tokenizer(100, True)
device = "cpu"

# Define the SiameseModel architecture
model = load_tanimoto_supcon_aug_model(device)

# Initialize the ModelTester
tester = ModelTester(model, device, True)

# Generate embeddings for query and reference spectra
q, _ = embedding(tester, tokenizer, 512, q, True)
r, _ = embedding(tester, tokenizer, 512, r, True)

# Compute the cosine similarity matrix
cosine_scores = cosine_similarity(q, r)
```

#### 3.2 计算 Top1 的候选化合物

```python
import sys
sys.path.append("..")

from SpecEmbedding.utils.model import embedding, cosine_similarity, load_tanimoto_supcon_aug_model, top_k_indices
from SpecEmbedding.utils.clean import read_raw_spectra
from SpecEmbedding.trainer.trainer import ModelTester
from SpecEmbedding.data.tokenizer import Tokenizer

# Disable progress bar for simplicity
show_progress_bar = False

# Load query and reference spectra
q_spectra = read_raw_spectra("./q.msp")
r_spectra = read_raw_spectra("./r.msp")

# Initialize tokenizer and device
tokenizer = Tokenizer(100, True)
device = "cpu"

# Define the SiameseModel architecture
model = load_tanimoto_supcon_aug_model(device)

# Initialize the ModelTester
tester = ModelTester(model, device, show_progress_bar)

# Generate embeddings for query and reference spectra
q, _ = embedding(tester, tokenizer, 512, q_spectra, show_progress_bar)
r, _ = embedding(tester, tokenizer, 512, r_spectra, show_progress_bar)

# Compute the cosine similarity matrix
cosine_scores = cosine_similarity(q, r)

# Retrieve the indices of the top-1 candidates
indices = top_k_indices(cosine_scores, 1)
for i, index in enumerate(indices[:, 0]):
    print(f"The {i + 1}-th spectra with SMILES {q_spectra[i].get('smiles')} most similar compound is {r_spectra[index].get('smiles')}")
```

此外，我们提供了完整的训练脚本与测试脚本。

用户可以通过阅读以下两个 Jupyter Notebook 文件了解详细的训练流程与评估方法：

[训练脚本](../demo/train_model.ipynb)
[评估脚本](../hit_metric/GNPS&MoNA&MTBLS1572.ipynb)

### 4. 命令行流程

默认路径和超参数由 `params.yaml` 配置。配置文件选择优先级为：显式
`load_config(path)` 参数、`SPECEMBEDDING_CONFIG` 环境变量、仓库默认 `params.yaml`。
相对路径统一相对于所选 YAML 所在目录解析，不依赖 shell 当前目录；命令行路径参数优先于
已加载配置。

默认处理数据根为 `data/processed`，其下按 `MassSpecGym`、`MassBank`、`GNPS`、
`MoNA` 或 `NPLIB1` 分目录，每个目录通常包含 `train.pkl`、`val.pkl` 和 `test.pkl`；
MassSpecGym 检索还使用 `candidates_mass.pkl` 与 `candidates_formula.pkl`。这些文件由
`src/data/base.py` 加载。

早期谱图到谱图 notebooks 属于归档流程，不是论文主结果流水线。其私有机器路径已替换为
仓库内 `data/legacy` 相对路径；准备好 figshare/raw 输入后，从 notebook 所在目录运行。
legacy Python 常量还可通过 `SPECEMBEDDING_LEGACY_DATA_ROOT`、
`SPECEMBEDDING_MSBERT_ROOT` 与 `SPECEMBEDDING_TSNE_CLUSTER_DIR` 重定位。

#### 4.1 论文主结果流程与协议

转投稿件的当前结果来自谱图--分子 alignment、full-pool candidate cache 与 rank-free
relative/pointwise reranking，而不是早期的谱图到谱图 notebook 评价。主评价使用 MassSpecGym
1.3.1 retrieval JSON 候选顺序、二维 InChIKey 身份规则和保存的 alignment embeddings；这是
official-compatible candidate/identity audit，不是重新运行 official loader、重新编码或重训
alignment。local exact-target-SMILES 结果仅作为敏感性视图。

当前方法使用最多 256 个供给候选，先做 pointwise coarse scoring，再在 coarse top-40 上做
spectrum-conditioned relative refinement；训练、验证和测试均默认不强制插入正例。Pointwise
是容量匹配对照。三种 reranker seeds 和官方候选顺序结果见
`analysis/transfer2026_scholargpt_review/`；relative 在主要 R@1/MRR 汇总中略低于 pointwise，
因此不声称候选关系模块具有独立普遍收益。论文中的 JESTR/GLMR 为 reported-only，未在本项目
统一复现，也不可与本地结果直接比较。

下列命令还包括仓库的通用流程与 legacy 流程；若候选、身份、划分、cache 和 checkpoint
协议没有与记录工件完全匹配，其输出不得替代论文表格。

每个划分文件都是 pickle 文件，内容为 `list[matchms.Spectrum]`。每条 `Spectrum` 至少需要支持：

```python
spectrum.get("smiles")
spectrum.get("precursor_mz")
spectrum.peaks.to_numpy  # shape: [num_peaks, 2]，两列分别是 mz 和 intensity
```

预训练谱图编码器：

```bash
python train.py \
  --dataset_type massspecgym \
  --data_path /path/to/processed \
  --save_dir ./checkpoints
```

训练谱图-分子跨模态对齐模型：

```bash
python train_align.py \
  --dataset_type massspecgym \
  --data_path /path/to/processed \
  --save_dir ./checkpoints_align/run \
  --pretrained_spec ./checkpoints/model.ckpt
```

如果不传 `--pretrained_spec`，或指定文件不存在，对齐训练会从随机初始化的谱图编码器开始。`--graph_cache_size` 控制每个 DataLoader worker 内的分子图惰性缓存：`0` 表示关闭缓存，`-1` 表示不限制缓存大小。

在预先生成的 query/reference `.npy` 文件上评估谱图到谱图检索：

```bash
python eval.py \
  --checkpoint ./checkpoints/model.ckpt \
  --data_dir /path/to/replicated_splits \
  --loss_type custom
```

使用候选集评估谱图到分子检索：

```bash
python eval_align.py \
  --checkpoint ./checkpoints_align/run/final_aligned_model.pth \
  --dataset_type massspecgym \
  --data_path /path/to/processed \
  --candidate_type mass
```

默认情况下，`eval_align.py` 会根据 `--candidate_type` 从所选数据集目录加载 `candidates_mass.pkl` 或 `candidates_formula.pkl`。也可以直接指定自定义候选集文件：

```bash
python eval_align.py \
  --checkpoint ./checkpoints_align/run/final_aligned_model.pth \
  --dataset_type massspecgym \
  --data_path /path/to/processed \
  --candidate_path /path/to/custom_candidates.pkl
```

传入 `--candidate_path` 时，它会覆盖 `--candidate_type`。自定义候选集 pickle 必须是：

```python
dict[str, list[str]]
```

其中 key 是 `test.pkl` 中 query 或真实分子的 SMILES，value 是该 query 对应的候选分子 SMILES 列表。评估时会先按 test split 中存在的 SMILES 过滤候选集。分子 embedding 默认存储在 CPU；只有确认完整候选 embedding 矩阵能放入显存时，才建议使用 `--mol_embedding_storage cuda`。`--candidate_chunk_size 0` 会根据可用 CUDA 显存自动估计候选分块大小，`--no-mces` 可关闭 MCES 计算。

#### 4.2 当前 rerank 流程

当前论文方法使用最多 256 个自然候选，训练、验证和测试均不强制插入正例：

```bash
python prepare_rerank_cache.py \
  --checkpoint ./checkpoints_align/run/best_model_stage2.pth \
  --dataset_type massspecgym \
  --candidate_type mass \
  --split train \
  --topk 256 \
  --no-force_include_positive \
  --save_path ./rerank_cache/run/mass_train.pt

python train_rerank.py \
  --train_cache ./rerank_cache/run/mass_train.pt \
  --val_cache ./rerank_cache/run/mass_val.pt \
  --model_type relative \
  --train-k 256 \
  --relation-top-k 40 \
  --save_dir ./checkpoints_rerank/run

python eval_rerank.py \
  --cache ./rerank_cache/run/mass_test.pt \
  --checkpoint ./checkpoints_rerank/run/best_reranker.pth \
  --max-candidates 256 \
  --save_dir ./checkpoints_rerank/run
```

也可以使用 `run_rerank_pipeline.py` 编排 cache、训练和测试；它默认 no-forcing，并支持
`--model_type relative`。只有需要复核历史训练协议时，才显式传入
`--force-include-positive`；该开关只作用于 train cache，验证和测试始终不插入正例。

### 5. Web 服务

我们也为用户提供了一个 web 服务，每个人都可以通过访问网址 [SpecEmbedding](https://huggingface.co/spaces/xp113280/SpecEmbedding) 使用。
