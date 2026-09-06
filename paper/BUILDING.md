# Paper release build

Run from the repository root (or invoke the script by its absolute path):

```bash
bash paper/build_release.sh
```

The script forces complete BibTeX/LaTeX rebuilds of `main.tex` and
`main_cn.tex`, rejects fatal errors, undefined citations/references, and
Overfull/Underfull boxes, and requires `pdfinfo` for page counting. It writes
the auditable PDFs to:

- `paper/release/specembedding-adma2026-en.pdf`
- `paper/release/specembedding-adma2026-zh.pdf`

It also synchronizes the ignored `paper/build/main*.pdf` files so that the
legacy local build directory does not retain stale PDFs.

The script prints page counts and hashes; it does not update the YAML manifest.
After a release, record the committed manuscript source, toolchain, page counts,
hashes and verification in [build-manifest.yaml](release/build-manifest.yaml).
An old manifest must not be presented as verification of newly built PDFs.

For local checks that keep all intermediate files in ignored `paper/build/`, run
from `paper/`:

```bash
latexmk -gg -pdf -outdir=build -interaction=nonstopmode -halt-on-error main.tex
latexmk -gg -xelatex -outdir=build -interaction=nonstopmode -halt-on-error main_cn.tex
```

The release script currently builds in `paper/` before copying PDFs to `build/`
and `release/`; its auxiliary files are ignored. Known non-blocking messages are
the English `amsmath` accent warning and Chinese Fandol `fontspec` warnings.
Final submission checks are in the [checklist](TRANSFER_2026_PLAN.md).
