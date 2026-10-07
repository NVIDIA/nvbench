# JSON Output

NVBench writes benchmark results as a JSON document when you use the
`--json` option. Use `--jsonbin` when you also need the raw sample times and
sample frequencies for post-processing.

```console
./benchmark --json results.json
./benchmark --jsonbin results-with-samples.json
```

Both options write the same JSON structure. The `--jsonbin` option additionally
writes binary sidecar files next to the JSON document.

## Document Structure

The top-level JSON object contains these keys:

| Key | Description |
| --- | --- |
| `meta` | Command-line arguments and NVBench/JSON format versions. |
| `devices` | Device properties used by the benchmark run. |
| `benchmarks` | Benchmark definitions and their measured states. |

The JSON format is versioned at `meta.version.json`. Consumers should inspect
this value before parsing the document. The current format version is `1.0.0`.

Each item in `benchmarks` contains the benchmark `name`, numeric `index`,
execution settings, selected `devices`, `axes`, and `states`. Each state
contains its axis values, summaries, and an `is_skipped` flag. Skipped states
also include `skip_reason`.

Summary values are represented by a `tag` and optional `name`, `description`,
`hint`, and `data` fields. Entries in `data` use this structure:

```json
{
  "name": "mean",
  "type": "float64",
  "value": "1.234"
}
```

Integer and floating-point summary values are encoded as strings so consumers
do not lose precision when parsing large values.

## Binary Sidecar Files

With `--jsonbin`, NVBench creates these directories next to the JSON file:

* `<json-file-name>-bin/` contains sample-time files.
* `<json-file-name>-freqs-bin/` contains sample-frequency files.

For example, `results-with-samples.json` produces:

```text
results-with-samples.json
results-with-samples.json-bin/0.bin
results-with-samples.json-freqs-bin/0.bin
```

The JSON summary that produced a sidecar contains a `filename` relative to the
JSON file's directory and a `size` with the number of stored samples. Resolve
the filename relative to the JSON file rather than the current working
directory.

Each binary file stores `size` little-endian IEEE 754 32-bit floating-point
values. Use the summary's `filename` and `size` to associate a file with its
state; do not infer that association from the numeric filename alone.

If a sidecar cannot be written, NVBench keeps the summary metadata but omits
`filename` and `size` and reports a warning. Consumers should treat such bulk
data as unavailable instead of assuming that an absent file is empty.
