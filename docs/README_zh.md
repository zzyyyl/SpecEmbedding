# SpecEmbedding 运行指南与文档地图

流程为谱图 Transformer 与 GINE 分子塔对齐、基础候选召回、固定候选池内的监督式非生成重排序。
旧谱图到谱图流程仍保留在 notebook 中；当前研究状态见 [项目记忆](project_memory_zh.md)。

## 环境与数据

从仓库根目录执行：

```bash
conda env create -f environment.yml
conda activate specembedding
python -m pip install -r requirements-dev.txt
python -m pytest -q
ruff check .
python -m compileall -q .
```

[params.yaml](../params.yaml) 集中配置路径和超参数；配置选择顺序为显式 `load_config(path)`、
`SPECEMBEDDING_CONFIG`、仓库 YAML。设置`storage.root`或`SPECEMBEDDING_STORAGE_ROOT`后，
配置相对路径按外部存储根目录解析；没有存储设置的历史配置仍按YAML所在目录解析。
CLI路径参数优先，正式队列通过外部存储入口启动时还会检查输出是否在指定根目录内。
[复现说明](../reproducibility/README.md) 提供验收环境锁和工件边界。

默认数据目录为外部存储根下的`processed/<Dataset>/`。`train.pkl`、`val.pkl`、`test.pkl` 包含
`list[matchms.Spectrum]`，每条须有 `smiles`、`precursor_mz` 和两列 m/z/intensity 谱峰。
候选 pickle 为 `dict[str, list[str]]`，将目标 SMILES 映射到候选 SMILES 列表。

- MassSpecGym：`candidates_mass.pkl` / `candidates_formula.pkl`；后者假设已知分子式。
- NPLIB1：`candidates_supplied.pkl` 只含 test retrieval 候选；全 split 的
  `candidates_formula.pkl` 来自固定分子库重建。默认 `supplied` 不能直接用于全流程训练，
  训练须显式选择 `formula`，遵守 [NPLIB1 计划](paper-change-plans/2026-09-05-NPLIB1跨数据集增强.md)。
- `--candidate_path` 可覆盖 provider 候选文件；不能通过逐 query 插入真值补齐缺失候选。

## 运行入口

先提供真实数据、可用 GPU 和独立输出目录。正式训练使用完整可训练划分、显式 `cuda:N`；
长任务使用 detached `tmux`，其余约束见 [AGENTS.md](../AGENTS.md)。

从下一轮起，通过`run_with_storage.py`启动正式任务；本机启动配置使用
`/data1/${USER}/SpecEmbedding`，可显式指定`--storage-root`或环境变量
`SPECEMBEDDING_STORAGE_ROOT`；通用YAML不绑定本机路径，入口缺少根目录时拒绝启动。
训练数据、输入缓存、下载缓存、临时数据及新checkpoint
放在该根目录下，拒绝home目录和越界符号链接。该入口在导入训练/下载依赖前设置
Hugging Face、Torch、Numba、CUDA等缓存变量，并传给所有子进程；不修改`HOME`。
仅影响新进程，当前训练及它引用的旧文件不移动、不删除。具体规则见[存储说明](storage_zh.md)。

需要等 GPU 空闲后向已有训练 pane 单次派发命令时，使用
[GPU/tmux 监测器](gpu_tmux_monitor_zh.md)；它不会自动恢复暂停的实验计划，也不是资源调度器。

```bash
export SPECEMBEDDING_STORAGE_ROOT="/data1/${USER}/SpecEmbedding"
python run_with_storage.py -- python "$PWD/train_align.py" --dataset_type massspecgym \
  --data_path processed --save_dir checkpoints_align/my_run --device cuda:1

python run_with_storage.py -- python "$PWD/eval_align.py" --dataset_type massspecgym \
  --checkpoint checkpoints_align/my_run/best_model_stage2.pth \
  --candidate_type mass --device cuda:1 --no-mces

python run_with_storage.py -- python "$PWD/run_rerank_pipeline.py" massspecgym \
  --align_save_dir checkpoints_align/my_run --candidate_type mass \
  --pre_top_k 256 --model_type relative --device cuda:1
```

对齐示例从零训练谱图塔；复用预训练塔时显式提供 `--pretrained_spec` 并核验加载日志。
rerank 默认 train/val/test 均不 forcing：`relative` 对最多 256 个自然候选逐点打分，再在
coarse top-40 上做关系精排；`pointwise` 为容量匹配对照。分阶段命令与元数据见
[reranker 说明](reranker_solution_zh.md)。

`--dry-run` 只打印命令，`--limit N` 仅用于非正式调试并使用 `_limitN` 输出目录。代码默认值
不能代替历史工件协议；已发布的限量 top-40 矩阵与 top-256 no-forcing pilot 的区别见
[证据索引](../analysis/README.md)。

## 文档地图

| 文档 | 维护内容 |
| --- | --- |
| [项目记忆](project_memory_zh.md) | 当前任务、运行约束入口、证据缺口 |
| [编码器结构](../model_architecture.md) | Tokenizer、谱图塔、GINE 与对齐损失 |
| [reranker 实现](reranker_solution_zh.md) | 第二阶段模型、cache、训练与评价接口 |
| [GPU/tmux 监测器](gpu_tmux_monitor_zh.md) | 空闲门禁、单次派发、CUDA 映射、锁与安全边界 |
| [候选指纹输入](fingerprint_cache_zh.md) | 完整Morgan特征的CPU构建、逐位审计和按需读取；尚未自动接入模型 |
| [分析索引](../analysis/README.md) | 实验系列、协议、报告和 manifest |
| [论文计划](paper-change-plans/README.md) | 在途计划、模板和历史追溯 |
| [投稿待办](../paper/TRANSFER_2026_PLAN.md) | 科学证据与最终提交检查 |
| [论文构建](../paper/BUILDING.md) | 双语 PDF、页数和 release manifest |
| [复现说明](../reproducibility/README.md) | 环境锁、匿名包与历史验收 |
| [GLACIER 交接](glacier_reproduction_handoff_zh.md) | 已下载外部工件、缺口与后续推理步骤 |

代码按阶段分布在 `SpecEmbedding/models*.py`、`train*.py`、`eval*.py`；公共逻辑位于
`SpecEmbedding/utils/` 和 `SpecEmbedding/trainer/`。修改规范见仓库 AGENTS。
