# Reproduction of the article experiments

This directory is separate from the SWIFT-C library. It defines the fixed
experimental protocol for the manuscript and invokes the library commands.
Clustering and retrieval algorithms are not duplicated here.

The main program `paper/reproduce.py` can execute each stage independently.
It records commands, dependency versions, dataset checksums, and stage completion
in `results/paper/reproduction_manifest.json`. The same workspace is protected
against silently changing the dataset, mapping, or paper protocol.

It uses `paper/protocol.json`, which contains the evaluation configurations,
generator seeds, and model choices. The output directory is `results/paper/`.

## Commands

To inspect all commands without executing any model, run

```bash
python paper/reproduce.py --stage all --dry-run
```

To run the controlled synthetic evaluation, use

```bash
python paper/reproduce.py --stage synthetic
```

To generate embeddings and cluster the real data, use

```bash
python paper/reproduce.py --stage realworld --device cuda --dec-device cuda
```

After real-world clustering completes, run the discrete comparison, the
cluster-restricted video retrieval evaluation, and table preparation.

```bash
python paper/reproduce.py --stage discrete
python paper/reproduce.py --stage retrieval
python paper/reproduce.py --stage tables
```

The synthetic-based selector is optional. It is not required to run the
main article protocol.

```bash
python paper/reproduce.py --stage selection
```

For a sequential run of the full article protocol, use

```bash
python paper/reproduce.py --stage all --device cuda --dec-device cuda
```

The synthetic and real experiments are computationally expensive. They
support resuming from existing results and caches. The script does not
download the video files, and the encoder is downloaded only when needed.

## Experimental scope

Synthetic experiments include 29 scenario families. The four internal
sanity-check scenarios are intentionally excluded from publication tables.
The real-world experiment uses 10,000 videos, automatically generated
descriptions, and suggestions from three LLMs. It does not make new LLM
calls. Reference categories are used only for post-fit evaluation.

The retrieval evaluation ranks other videos within each assigned cluster
using cosine similarity between the frozen video-description embeddings.
It does not benchmark global retrieval or claim improvement in the ranker.

## Output

`synthetic/runs.csv` contains the synthetic clustering results.
`realworld/runs.csv` contains real-world clustering results.
`discrete/metrics.csv` contains the discrete baseline comparison.
`retrieval/ranking_summary.csv` contains Precision@k, Recall@k, and mAP@k.
`tables/` contains combined CSV tables assembled from completed runs.

The reported results in the manuscript reflect the evaluated runs. Running
additional methods or changing the protocol may produce a wider table than
the one printed in the article. The generated CSV files remain the primary
source for exact reproduction and comparisons.
