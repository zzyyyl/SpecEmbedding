#!/usr/bin/env bash
set -euo pipefail

paper_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${paper_dir}"

latexmk -gg -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -gg -xelatex -interaction=nonstopmode -halt-on-error main_cn.tex

problem_pattern='Undefined control sequence|LaTeX Error|LaTeX Warning: (Citation|Reference).*undefined|There were undefined references|Overfull|Underfull|Fatal error'
if grep -En "${problem_pattern}" main.log main_cn.log; then
  printf 'Release build rejected because a fatal, undefined-reference, or box warning was found.\n' >&2
  exit 1
fi

english_pages="$(pdfinfo main.pdf | awk '/^Pages:/ {print $2}')"
chinese_pages="$(pdfinfo main_cn.pdf | awk '/^Pages:/ {print $2}')"

mkdir -p release build
install -m 0644 main.pdf release/specembedding-adma2026-en.pdf
install -m 0644 main_cn.pdf release/specembedding-adma2026-zh.pdf

# Keep the ignored legacy build location synchronized so local users do not
# accidentally inspect the stale August 1 PDFs.
install -m 0644 main.pdf build/main.pdf
install -m 0644 main_cn.pdf build/main_cn.pdf

printf 'English pages: %s\n' "${english_pages}"
printf 'Chinese pages: %s\n' "${chinese_pages}"
sha256sum release/specembedding-adma2026-en.pdf release/specembedding-adma2026-zh.pdf
