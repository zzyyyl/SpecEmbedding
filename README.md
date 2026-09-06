# SpecEmbedding

[中文运行指南](docs/README_zh.md)

SpecEmbedding supports MS/MS spectral embedding and spectrum-to-molecule retrieval.
The current pipeline aligns a peak-sequence Transformer with a GINE molecular
encoder, retrieves candidates, and learns to rerank a fixed list without generating
new molecules. The rank-free `relative` model has a pointwise coarse branch and a
spectrum-conditioned candidate-pair branch; `pointwise` is its capacity control.

## Setup

From the repository root:

```bash
conda env create -f environment.yml
conda activate specembedding
python -m pip install -r requirements-dev.txt
python -m pytest -q
ruff check .
python -m compileall -q .
```

For an existing environment, use `conda env update -n specembedding -f environment.yml --prune`.
The [environment recipe](environment.yml) and [validation locks](reproducibility/README.md)
describe the maintained runtime and post-hoc acceptance environment, respectively;
they do not prove that every historical training run used identical packages.

## Data and configuration

[params.yaml](params.yaml) holds paths and hyperparameters. Configuration selection
is explicit `load_config(path)`, then `SPECEMBEDDING_CONFIG`, then the repository
YAML. Relative configuration paths resolve from that YAML; CLI path arguments
override configuration values.

Processed datasets live under `data/processed/<Dataset>/`. Each split pickle contains
`list[matchms.Spectrum]` with `smiles`, `precursor_mz`, and peak arrays of shape
`[num_peaks, 2]` (m/z, intensity). MassSpecGym also requires `candidates_mass.pkl`
and `candidates_formula.pkl`: each maps a target SMILES to a list of candidate SMILES.
`--candidate_path` overrides the provider's candidate file. Alignment evaluation
filters mapping **keys** to test targets; candidate molecules need not themselves
occur as test targets.

NPLIB1 distinguishes test-only `supplied` candidates from reconstructed
`formula` candidates for all splits. Its derived validation split and overlap
restrictions are recorded in the [active plan](docs/paper-change-plans/2026-09-05-NPLIB1跨数据集增强.md).

## Retrieval workflow

These examples require the processed data and an available, explicitly selected GPU.
Use complete eligible training splits for formal experiments and detached `tmux`
for long runs; see [repository rules](AGENTS.md).

```bash
python train_align.py --dataset_type massspecgym \
  --data_path data/processed --save_dir checkpoints_align/my_run --device cuda:1

python eval_align.py --dataset_type massspecgym \
  --checkpoint checkpoints_align/my_run/best_model_stage2.pth \
  --candidate_type mass --device cuda:1 --no-mces

python run_rerank_pipeline.py massspecgym \
  --align_save_dir checkpoints_align/my_run --candidate_type mass \
  --pre_top_k 256 --model_type relative --device cuda:1
```

The rerank pipeline defaults to no positive forcing in all splits. `formula`
assumes a known molecular formula. Reranking cannot recover a target absent from
the input list. `--dry-run` previews commands; `--limit N` is for nonformal debugging
only. Model and cache details are in the [reranker guide](docs/reranker_solution_zh.md).

Code defaults are distinct from historical experimental protocols. The 36-run
pointwise/relative matrix and 30 ablations used a 20,000-query cap and top-40
caches, including training positive forcing. The separate top-256 no-forcing pilot
and saved-embedding identity audits do not validate that matrix as a full-training,
official-loader experiment. See the [evidence index](analysis/README.md) and
[paused full-training plan](docs/paper-change-plans/2026-09-07-MassSpecGym全量重训.md).
The evidence supports supervised reranking within the tested pools; it does not
establish an independent general advantage for candidate interactions.

## Documentation and legacy resources

- [Documentation map and Chinese workflows](docs/README_zh.md)
- [Encoder architecture](model_architecture.md) and [current project state](docs/project_memory_zh.md)
- [Submission checklist](paper/TRANSFER_2026_PLAN.md) and [paper build instructions](paper/BUILDING.md)
- Legacy spectrum-to-spectrum recipes: [training](demo/train_model.ipynb),
  [search](demo/search.ipynb), and [evaluation](hit_metric/GNPS&MoNA&MTBLS1572.ipynb).
  Run notebooks from their own directories after supplying data under `data/legacy`;
  Python legacy paths can be relocated with `SPECEMBEDDING_LEGACY_DATA_ROOT`,
  `SPECEMBEDDING_MSBERT_ROOT`, and `SPECEMBEDDING_TSNE_CLUSTER_DIR`.
- Earlier spectral-embedding resources: [data and experiment archive](https://doi.org/10.6084/m9.figshare.28876751.v2),
  [metric comparison repository](https://github.com/sword-nan/SpecEmbedding-Comparison),
  and [demo space](https://huggingface.co/spaces/xp113280/SpecEmbedding).
  These resources belong to the earlier workflow and do not replace the current paper's evidence.
