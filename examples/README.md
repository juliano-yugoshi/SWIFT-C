# Applying SWIFT-C to another dataset

SWIFT-C accepts a CSV or ZIP containing item descriptions and independently
generated natural-language weak labels. A reference category column is
optional for fitting and can be used for evaluation when available.

`custom_dataset.csv` is a small illustrative dataset with six themes and
two weak-label sources. It is not part of the paper experiments.

```bash
swiftc realworld \
  --input examples/custom_dataset.csv \
  --mapping configs/example_mapping.json \
  --encoder hashing \
  --profile quick \
  --methods input,labels_only,fusion \
  --clusters 3 \
  --seeds 42 \
  --sources all \
  --sample 0 \
  --output results/example
```

The hashing encoder is for interface and pipeline tests only. Use a
semantic sentence encoder for meaningful research results.

```bash
swiftc realworld \
  --input examples/custom_dataset.csv \
  --mapping configs/example_mapping.json \
  --encoder Qwen/Qwen3-Embedding-0.6B \
  --profile full \
  --methods input,labels_only,fusion,spectral_fusion \
  --clusters 3 \
  --seeds 42 \
  --sources all \
  --output results/example_qwen
```

For a new dataset with different columns, create a mapping with
`swiftc configure --input PATH --output new_mapping.json`.
