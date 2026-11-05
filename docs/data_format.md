# Triple data format

Files are UTF-8 TSV with exactly three non-empty columns:

```text
HEAD<TAB>RELATION<TAB>TAIL
```

Blank lines and lines beginning with `#` are ignored. Exact duplicate triples are rejected.

## Transductive schema

The training split defines the entity and relation vocabularies. Every entity and relation used
in development or test must occur in training. Unknown identifiers produce an error instead of
being silently mapped to an unknown token.

## Splitting guidance

Random triple splitting can leak inverse or near-equivalent facts. Prefer official benchmark
splits or domain-aware splits that account for reciprocal relations, temporal boundaries, data
sources, and duplicate facts. The included splitter only provides schema coverage; it cannot
guarantee a scientifically appropriate experimental design.

## Filtered evaluation

For a query such as `(h, r, ?)`, multiple tails may be correct. Filtered evaluation removes
other known true tails before ranking the target. Training triples and the evaluated file are
included automatically. Pass validation or other true triples with `--filter-data` when needed.
