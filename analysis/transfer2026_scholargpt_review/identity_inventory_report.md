# Identity inventory (cache-only audit)

This audit uses the saved canonical top-40 test caches only. It converts the
stored candidate and target SMILES to the first block of a 2D InChIKey and
counts identity matches within each cached list. It does not load the official
MassSpecGym evaluator and must not be read as an official-evaluator rerun.

| cache | queries | reference-positive queries | multiple-positive queries | extra positive candidates | local/base R@1 | reference/base R@1 | reference upper bound |
|---|---:|---:|---:|---:|---:|---:|---:|
| mass | 17,556 | 14,726 | 0 | 0 | 47.4596% | 47.4596% | 83.8802% |
| formula | 17,556 | 15,670 | 0 | 0 | 63.1009% | 63.1009% | 89.2572% |

Within these top-40 caches, no query has multiple positive candidates under
the tested 2D identity prefix, and no base top-1 identity changes. The lower
reference-positive count reflects targets whose equivalent identity is absent
from the stored top-40 list; it is therefore a coverage observation, not proof
of equivalence to the official loader or full candidate pool.

Source: `identity_inventory.py` and `identity_inventory.json`, generated from
the two tracked canonical cache paths. Full reranker identity metrics, if
completed, are stored separately in `official_identity_eval.json`.
