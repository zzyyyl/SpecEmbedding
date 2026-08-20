# Paper release build

Run the release build from any working directory:

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

The verified 2026-08-20 toolchain uses TeX Live 2019/Debian, `latexmk` 4.67,
pdfLaTeX for English, XeLaTeX for Chinese, BibTeX 0.99d, and the TeX Live Fandol
fonts supplied with `ctex`. The known non-blocking messages are the English
`amsmath` math-accent redefinition warning and Chinese Fandol `fontspec`
warnings. See `paper/release/build-manifest.yaml` for the source commit, page
counts, file hashes, and exact verification record of the tracked PDFs.
