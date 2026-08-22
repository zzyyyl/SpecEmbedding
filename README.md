## SpecEmbedding

Chinese Version: [SpecEmbedding](./docs/README_zh.md)

SpecEmbedding is a deep learning model designed specifically for MS/MS spectral embedding. It combines sinusoidal positional encoding with a supervised contrastive learning framework to improve performance in compound identification and structural similarity retrieval tasks.

The model was trained and evaluated on the GNPS, MoNA, and MTBLS1572 datasets, initially preprocessed by the MSBERT team. To further improve data quality, we removed entries with malformed or invalid SMILES strings. All cleaned data, along with the preprocessing scripts and 10-fold query/reference splits used for evaluation, are available on [figshare](https://doi.org/10.6084/m9.figshare.28876751.v2).

To extend the evaluation to additional curated spectral data, we also tested
MassBank and MassSpecGym. The corresponding results and protocols are reported
in the linked artifacts; this broader coverage is not, by itself, a general
out-of-domain robustness guarantee.

To ensure a fair and reproducible evaluation, we strictly retained the original training set split used by MSBERT and only applied random splitting to the test sets. Final results are reported as the average and standard deviation across the 10 splits.

Details on the hyperparameter search space, training procedure, ablation studies, and full benchmark results across multiple tasks are also available on [figshare](https://doi.org/10.6084/m9.figshare.28876751.v2) for reproducibility and further research.

More details about Evaluation Metrics can be found in [SpecEmbedding-Comparison](https://github.com/sword-nan/SpecEmbedding-Comparison)

### 1. Environment

The post-hoc acceptance environment validated on 2026-08-17 used Ubuntu 20.04,
Python 3.12.13, PyTorch 2.5.1 with CUDA runtime 12.1, and cuDNN 9.1. This is a
reproduction/validation snapshot, not proof that every historical training run
used an identical package set. `environment.yml` is the maintained environment
recipe; the versioned paper manifests separately freeze data, candidates,
checkpoints, parameters, and analysis artifacts.

Create the canonical Conda environment from the repository root:

```bash
conda env create -f environment.yml
conda activate specembedding
```

For an existing development environment, synchronize it with the checked-in
recipe before running tests or experiments. This updates a working environment;
it is not an exact historical lock:

```bash
conda env update -n specembedding -f environment.yml --prune
conda activate specembedding
```

> ⚠️ Note for Windows Users: When running on Windows, you may encounter numerical errors during cosine similarity computation. This is caused by @njit decorators from the numba library. You can fix it by commenting out all @njit decorators in the code.

### 1.1 Development and Test Tools

Pytest is the repository acceptance-test entry point and is included in
`environment.yml`. Ruff is configured for linting and import sorting, but it is
kept out of the runtime environment. For an existing environment, install or
refresh both development tools with:

```bash
python -m pip install -r requirements-dev.txt
```

From the repository root, run the complete test suite with the environment's
Python interpreter:

```bash
python -m pytest -q
```

Run Ruff checks:

```bash
ruff check .
```

Apply safe automatic fixes:

```bash
ruff check . --fix
```

### 2. Molecular Graph Features

For spectrum-molecule alignment tasks, SpecEmbedding uses a Graph Isomorphism Network (GINE) to encode molecular structures. To capture the chemical nuances critical for Mass Spectrometry (MS) fragmentation prediction, we employ a streamlined set of atom and bond features:

#### Atom Features (Node)
- **Atomic Number**: Identity of the atom (B, C, N, O, F, Si, P, S, Cl, Br, I, or others).
- **Formal Charge**: Electrical state, which dictates ionization and fragmentation.
- **Degree**: Number of neighboring atoms, reflecting connectivity.
- **Total Hydrogens**: Number of implicit and explicit hydrogens.
- **Aromaticity**: Indicates if an atom is part of an aromatic system.
- **Ring Membership**: Whether the atom is part of a cyclic structure.
- **Ring Sizes (3-6)**: Specific flags for 3, 4, 5, and 6-membered rings (critical for fragmentation energy).

#### Bond Features (Edge)
- **Bond Type**: Single, Double, Triple, or Aromatic.
- **Conjugation**: Whether the bond is part of a conjugated system.
- **In Ring**: Whether the bond is part of a ring.

### 3. Demo

#### 3.1 Compute Cosine Similarity Matrix Between Query and Reference Spectra

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

#### 3.2 Get the Top-1 Most Similar Compound for Each Query Spectrum

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

In addition, we provide full training and evaluation scripts.
Users can refer to the following Jupyter notebooks for details on model training and performance evaluation:

[Model Training Script](./demo/train_model.ipynb)
[Model Model Evaluation and Metrics](./hit_metric/GNPS&MoNA&MTBLS1572.ipynb)

### 4. Command-line workflows

The default paths and hyperparameters are defined in `params.yaml`. Configuration
selection follows: explicit `load_config(path)` argument, then the
`SPECEMBEDDING_CONFIG` environment variable, then the repository `params.yaml`.
Relative paths are resolved from the selected YAML file, so configuration does
not depend on the shell's working directory. Command-line path arguments override
the loaded configuration.

By default, processed data is stored under `data/processed`. It contains one
subdirectory per dataset, for example `MassSpecGym`, `MassBank`, `GNPS`, `MoNA`,
or `NPLIB1`. Each directory should contain `train.pkl`, `val.pkl`, and `test.pkl`;
MassSpecGym retrieval additionally uses `candidates_mass.pkl` and
`candidates_formula.pkl`. Each split is loaded through `src/data/base.py`.

Older spectrum-to-spectrum notebooks are archival recipes rather than the paper
result workflow. Their private machine paths were replaced with repository-local
`data/legacy` paths; run them from their notebook directory after placing the
figshare/raw inputs there. Legacy Python constants can instead be relocated with
`SPECEMBEDDING_LEGACY_DATA_ROOT`, `SPECEMBEDDING_MSBERT_ROOT`,
and `SPECEMBEDDING_TSNE_CLUSTER_DIR`.

#### 4.1 Paper-result workflow and protocol

The transfer-paper workflow uses spectrum--molecule alignment followed by a
rank-free, no-forcing full-pool reranker: pointwise coarse scoring over at most
256 supplied candidates, then a spectrum-conditioned relative branch on the
coarse top-40. A capacity-matched pointwise control and the earlier rank-aware
Transformer audit are kept separate. Recall and MRR use MassSpecGym-supplied
candidate files with a local exact-target-SMILES single-positive rule. The
reference loader defaults to two-dimensional InChIKey equivalence and may yield
multiple positives. A MassSpecGym 1.3.1 transform audit of the saved 17,556-query
mass/formula test caches found no multiple-positive query or candidate identity
collision, and the seed-42 relative metrics were unchanged; this remains a
cache-level audit rather than a full official-loader rerun.

The no-forcing pilot uses pre-overlap-clean alignment checkpoint `d4c1f70`,
5,000 training queries, three epochs, and reranker seeds 42--44. Both relative
and capacity-matched pointwise improve the base in both pools; pointwise is
slightly higher at the main R@1/MRR summaries, so no independent relative-module
gain is claimed. Query-level predictions, K=20/40/80 evaluation-only sensitivity,
bootstrap summaries, mechanism controls, and RTX 4090 forward audits are
recorded in `analysis/transfer2026_scholargpt_review/second_stage_report.md` and
`second_stage_manifest.json`. JESTR and GLMR remain reported-only, not
independently reproduced, and not directly comparable mechanism background; no
external numeric values are displayed.

The new method entry points are `prepare_rerank_cache.py` with
`--topk 256 --no-force_include_positive`, `train_rerank.py --model_type
relative --train-k 256 --relation-top-k 40`, and `eval_rerank.py` with
`--max-candidates 256` or `40`; `--shuffle-spectrum` runs the dependency audit.
Long real-data runs should be launched in a detached `tmux` session. These
commands require the excluded MassSpecGym data and alignment checkpoint.

The following commands describe the repository's broader and legacy workflows;
their outputs should not be substituted for the paper tables without matching
the recorded candidate, identity, split, cache, and checkpoint protocol. The
anonymous supplement is a source-inspection and synthetic smoke package only;
it does not include the real spectra, candidate pools, checkpoints, caches, or
analysis logs needed to reproduce the reported metrics.

Each split file is a pickle file containing `list[matchms.Spectrum]`. Each `Spectrum` must provide:

```python
spectrum.get("smiles")
spectrum.get("precursor_mz")
spectrum.peaks.to_numpy  # shape: [num_peaks, 2], columns are mz and intensity
```

Pre-train the spectrum encoder:

```bash
python train.py \
  --dataset_type massspecgym \
  --data_path /path/to/processed \
  --save_dir ./checkpoints
```

Train the spectrum-molecule alignment model:

```bash
python train_align.py \
  --dataset_type massspecgym \
  --data_path /path/to/processed \
  --save_dir ./checkpoints_align/run \
  --pretrained_spec ./checkpoints/model.ckpt
```

If `--pretrained_spec` is omitted or the file is missing, the spectrum encoder is trained from scratch during alignment. `--graph_cache_size` controls the lazy molecule graph cache per DataLoader worker: `0` disables it and `-1` makes it unlimited.

Evaluate spectrum-to-spectrum retrieval on replicated query/reference `.npy` files:

```bash
python eval.py \
  --checkpoint ./checkpoints/model.ckpt \
  --data_dir /path/to/replicated_splits \
  --loss_type custom
```

Evaluate spectrum-to-molecule retrieval with candidate sets:

```bash
python eval_align.py \
  --checkpoint ./checkpoints_align/run/final_aligned_model.pth \
  --dataset_type massspecgym \
  --data_path /path/to/processed \
  --candidate_type mass
```

By default, `eval_align.py` loads `candidates_mass.pkl` or `candidates_formula.pkl` from the selected processed dataset directory according to `--candidate_type`. A custom candidate file can be used instead:

```bash
python eval_align.py \
  --checkpoint ./checkpoints_align/run/final_aligned_model.pth \
  --dataset_type massspecgym \
  --data_path /path/to/processed \
  --candidate_path /path/to/custom_candidates.pkl
```

When `--candidate_path` is provided, it overrides `--candidate_type`. The custom candidate pickle must contain:

```python
dict[str, list[str]]
```

where each key is a query or ground-truth SMILES string from `test.pkl`, and each value is the list of candidate molecule SMILES strings for that query. During evaluation, candidate entries are filtered to SMILES present in the test split. Molecule embeddings are stored on CPU by default; use `--mol_embedding_storage cuda` only when the full candidate embedding matrix fits in GPU memory. `--candidate_chunk_size 0` lets the script choose a chunk size from available CUDA memory, and `--no-mces` disables MCES calculation.

### 5. Web Service

An online web interface is available for demonstration and public use: [SpecEmbedding](https://huggingface.co/spaces/xp113280/SpecEmbedding)
