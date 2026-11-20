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

The tool resolves relative sample-time and sample-frequency sidecar filenames
only against `--sidecar-root`, which defaults to the input JSON's directory.
For legacy results, pass the original launch directory explicitly. Absolute
filenames are used directly; a missing sidecar is an error and is not searched
for elsewhere. References are rewritten relative to the output JSON's
directory using resolved physical locations, so symlink-based paths may change
even when they already resolve correctly. An `--output` copy is written even
when no references change. `--output` cannot name the input; use `--in-place`
to update it with a backup.
