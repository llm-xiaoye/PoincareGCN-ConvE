# Using your own knowledge graph

1. Export binary triples using stable entity and relation identifiers.
2. Remove duplicates and decide how inverse/symmetric relations should be handled.
3. Create leakage-aware train, development, and test splits.
4. Verify development/test schema coverage with `poincare-validate` and training metadata.
5. Choose dimensions and a ConvE grid whose area equals twice the GCN dimension.
6. Train and select the checkpoint on development MRR.
7. Freeze the configuration and evaluate once on the held-out test split.
8. Review high-ranked novel links before importing them into an operational graph.

For a credible report, record provenance, licenses, entity/relation counts, triple counts,
split policy, reciprocal relation handling, filtering set, curvature, dimensions, optimizer,
seed, epochs, early stopping, hardware, and filtered MRR/Hits@K.

Dataset paths are runtime arguments. The repository ignores artifacts, weights, and output
directories, so proprietary graph data need not be committed.
