# Architecture

SWIFT-C consolidates four research codebases under a single Python distribution.

| Public module | Responsibility | Original component |
|---|---|---|
| `swiftc.clustering` | Embeddings, fusion, clustering, synthetic scenarios, numerical baselines, cached results | WeakLabel EAC research package |
| `swiftc.discrete` | Discrete textual agreement and must-link supervision | Discrete real-world controls |
| `swiftc.selection` | Ranking and selecting methods based on synthetic results | Experiment selector |
| `swiftc.retrieval` | Evaluating existing cluster assignments as recommendation sets | Video recommendation evaluator |

The numerical clustering, similarity, optimization, and evaluation algorithms
are retained from their original implementations. Integration changes concern
Python imports, package namespaces, CLI entry points, dataset naming, input
configuration, documentation, and orchestration.

The command line dispatcher is `src/swiftc/cli.py`. It delegates directly to
the original implementations instead of creating a second algorithm layer.

The library supports arbitrary item descriptions and label suggestions using a
JSON dataset mapping. Real-world data may have any number of suggestion
sources. Synthetic generation currently uses three simulated sources because
the controlled scenario generator was defined for that setting.

The article workflow in `paper/reproduce.py` is separate. It only invokes the
same installed public commands and records execution provenance. No model or
numerical algorithm is copied into `paper/`.

The CSVs and mapping files in `data/` and `configs/` are bundled for research
reproducibility. For other applications, users supply their own data and map
its columns to the expected input roles.
