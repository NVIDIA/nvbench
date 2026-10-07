# `nvbench-normalize-jsonbin`

`nvbench-normalize-jsonbin` repairs legacy NVBench JSON results whose jsonbin
sidecar filenames were recorded relative to the launch directory instead of
the JSON file directory.

Preview changes without modifying the result:

```bash
nvbench-normalize-jsonbin --dry-run result.json
```

Write a normalized copy, or update the result in place with a backup:

```bash
nvbench-normalize-jsonbin --output normalized.json result.json
nvbench-normalize-jsonbin --in-place result.json
```

The tool resolves sample-time and sample-frequency sidecars relative to the
JSON file or launch directory. A matching file under `--sidecar-root` takes
precedence; otherwise, missing or ambiguous sidecars are errors. Paths in an
`--output` copy are relative to that output file, and the copy is written even
when the input already has normalized paths. `--output` cannot name the input;
use `--in-place` to update it with a backup.
