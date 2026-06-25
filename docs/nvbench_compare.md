# NVBench Compare

`nvbench-compare` compares two NVBench JSON outputs and classifies matching
benchmark states as `SAME`, `FAST`, `SLOW`, `AMBG`, or `????`.

NVBench treats benchmark performance data as describing a timing interval over
which measured timings varied. The interval is not intended as a precise
statistical confidence interval; it is an intuitive representation of the
observed timing range used to decide whether two benchmark results are clearly
separated, clearly compatible, or ambiguous.

The comparison is intentionally conservative. It reports `FAST` or `SLOW` only
when the timing intervals have a clear gap and the gap is confirmed in cycle
space when clock information is available. Ambiguous cases stay `AMBG`
instead of forcing a pass or regression.

## Common Invocations

Compare two JSON files:

```bash
nvbench-compare reference.json compare.json
```

Limit the comparison to one benchmark:

```bash
nvbench-compare --benchmark copy_type_sweep reference.json compare.json
```

Limit the comparison to one benchmark and one axis value:

```bash
nvbench-compare \
  --benchmark copy_type_sweep \
  --axis T=F32 \
  reference.json compare.json
```

Choose a table display mode. The default `intervals` mode shows timing centers
with compact intervals and the status. `legacy` shows the older time/noise/diff
columns. `explain` adds explicit low/center/high interval endpoints and decision
reason codes:

```bash
nvbench-compare --display intervals reference.json compare.json
nvbench-compare --display legacy reference.json compare.json
nvbench-compare --display explain reference.json compare.json
```

Plot the comparison summary, or plot timings along a positive numeric axis. Add
`--dark` to the summary plot when it should use a dark theme:

```bash
nvbench-compare --plot --dark reference.json compare.json
nvbench-compare --plot-along "Elements{io}" reference.json compare.json
```

Generate Python code with bulk sample/frequency filenames for every displayed
row:

```bash
nvbench-compare --bulk-debug-python /path/to/output.py reference.json compare.json
```

Compare selected devices. Device filters are paired by position, so this
compares reference device `0` against compare device `1`:

```bash
nvbench-compare \
  --reference-devices 0 \
  --compare-devices 1 \
  reference.json compare.json
```

Disable ANSI color codes. In this mode, status values are prefixed with emoji
markers so copied output still carries the status category:

```bash
nvbench-compare --no-color reference.json compare.json
```

Use a built-in comparison preset:

```bash
nvbench-compare --preset permissive reference.json compare.json
```

Use custom settings from TOML:

```bash
nvbench-compare --config compare.toml reference.json compare.json
```

Use a CLI preset as the base while preserving explicit TOML overrides:

```bash
nvbench-compare --config compare.toml --preset strict reference.json compare.json
```

Print the effective default configuration:

```bash
nvbench-compare --dump-config
```

Print the effective configuration for a built-in preset:

```bash
nvbench-compare --preset permissive --dump-config
```

## Matching Inputs

`nvbench-compare` matches benchmark states by benchmark name, device pairing,
axis filters, and state occurrence order within each device section.

Device sections must match unless `--ignore-devices` is specified or explicit
device filters are used:

```bash
nvbench-compare \
  --ignore-devices \
  reference.json compare.json
```

```bash
nvbench-compare \
  --reference-devices 0 \
  --compare-devices 1 \
  reference.json compare.json
```

The device filter value may be `all`, one non-negative integer device id, or a
comma-separated list of non-negative integer ids. Filtered reference and compare
device lists must have the same length; devices are paired by position.

Benchmark and axis filters follow NVBench CLI scoping:

```bash
nvbench-compare -b copy_type_sweep -a T=F32 reference.json compare.json
```

For integer axes displayed with NVBench `pow2` formatting, filter by exponent
with `NAME[pow2]=EXP`. For example, an axis value displayed as `2^20` is
selected with:

```bash
nvbench-compare -b base -a "Elements{io}[pow2]=20" reference.json compare.json
```

`-a` / `--axis` applies to the most recent `-b` / `--benchmark`, or to all
benchmarks if it appears before any benchmark filter.

## Timing Data Used

For each matched state, `nvbench-compare` extracts GPU timing summaries emitted
by NVBench cold measurements:

- `min`
- `max`
- `mean`
- `stdev/absolute`
- `stdev/relative`
- `q1`
- `median`
- `q3`
- `iqr/absolute`
- `iqr/relative`
- `sm_clock_rate/mean`

When JSON output is generated with the NVBench `--jsonbin` option,
sample-time and sample-frequency binary data are loaded lazily and used for
bulk-data confirmation.

Missing or empty bulk data are treated as unavailable. Bulk files that are
present and non-empty but fail lazy loading or validation are treated as
unusable evidence and reported as warnings.

## Bulk Debug Python Output

`--bulk-debug-python /path/to/output.py` writes a Python script to the specified
file. The generated script contains a `bulk_rows` list. Each entry corresponds
to one row that `nvbench-compare` prints in its display tables after all
benchmark, axis, device, and threshold filters are applied.

Use `stdout` instead of a file path to print the generated Python code:

```bash
nvbench-compare --bulk-debug-python stdout reference.json compare.json
```

Generated bulk-debug Python is enclosed in comment markers:

```python
# NVB-BULK-BEGIN
...
# NVB-BULK-END
```

Because the markers are valid Python comments, the generated helpers can be
filtered directly into the standard Python REPL. This example uses process
substitution, which requires a shell such as Bash, Zsh, or Ksh:

```bash
python -i <(
  nvbench-compare --bulk-debug-python stdout reference.json compare.json \
    | sed -n '/^# NVB-BULK-BEGIN$/,/^# NVB-BULK-END$/p'
)
```

IPython does not reliably accept process-substitution paths as startup files.
For IPython, write the generated code to a temporary file directly:

```bash
tmp=$(mktemp "${TMPDIR:-/tmp}/nvbench-bulk.XXXXXX")
nvbench-compare --bulk-debug-python "$tmp" reference.json compare.json
ipython -i "$tmp"
rm -f "$tmp"
```

Each `bulk_rows` entry includes:

- `row_index`: zero-based index among displayed comparison rows
- `table_row_index`: zero-based index within the displayed table for a device
  section
- `benchmark`
- `reference_json` and `compare_json`
- `reference_device_id` and `compare_device_id`
- `state_key`
- `occurrence` and `occurrence_count`, which disambiguate duplicate states
- `axis_values`
- `status`, `reason`, and `reason_message`
- sample and frequency filenames and counts for reference and compare data

The generated script also defines `load_bulk_data(row)`, which reads the
float32 sample and frequency files for a selected row.

Select the first displayed row:

```python
row = bulk_rows[0]
arrays = load_bulk_data(row)
```

Select the second ambiguous row:

```python
ambiguous = [row for row in bulk_rows if row["status"] == "AMBG"]
row = ambiguous[1]
arrays = load_bulk_data(row)
```

If `-b` and `-a` narrow the report to one comparison of interest, the desired
entry is usually available positionally as `bulk_rows[0]`. If duplicate states
remain after filtering, use `occurrence` to distinguish them.

## Time Estimates And Intervals

`nvbench-compare` prefers robust timing summaries when both sides provide them:

- center: `median`
- relative dispersion: `iqr/relative`, or `iqr/absolute` / `median`
- interval: `[min, q3]`

If robust summaries are not available on both sides, it falls back to classical
summaries:

- center: `mean`
- relative dispersion: `stdev/relative`, or `stdev/absolute` / `mean`
- interval: `[max(min, mean - stdev), min(max, mean + stdev)]`

Centers and interval endpoints must be positive and finite. States with unusable
centers are not compared.

Rows with `????` status could not form a valid timing comparison input. This
status is emitted for skipped benchmark states, missing GPU timing summaries,
or timing centers that are missing, non-finite, or non-positive. These rows are
included in the total match count so data-collection issues remain visible.

## Decision Tree

The comparison logic starts from `AMBG` and upgrades only when enough
evidence is available.

### 1. Check For A Clear Gap

The reference and compare intervals are checked for non-overlap.

`FAST` is possible when the compare interval is entirely below the reference
interval:

```text
cmp.upper < ref.lower
(ref.lower - cmp.upper) / cmp.upper >= clear_gap.relative
```

`SLOW` is possible when the compare interval is entirely above the reference
interval:

```text
cmp.lower > ref.upper
(cmp.lower - ref.upper) / ref.upper >= clear_gap.relative
```

These ratios are algebraically equivalent to checking a log-scale relative gap,
but avoid evaluating logarithms for every row.

### 2. Confirm Clear Gap In Cycle Space

If sample times and frequencies are available, `nvbench-compare` computes:

```text
cycles = sample_time * sample_frequency
```

It then builds cycle intervals from the bulk cycle samples and requires the
cycle interval comparison to agree with the timing interval comparison. A timing
gap that is not confirmed by bulk cycle intervals is `AMBG`.

If bulk data are missing or empty, `nvbench-compare` falls back to summary
clock-rate confirmation using `sm_clock_rate/mean`. If non-empty bulk data are
present but fail lazy loading or validation, the clear-gap decision remains
`AMBG` instead of falling back. If the clock-rate summary is missing or invalid,
the clear-gap decision also remains `AMBG`.

### 3. Check Bulk-Data Compatibility For SAME

When there is no clear gap and bulk sample/frequency data are available,
`nvbench-compare` compares both time samples and cycle samples using symmetric
nearest-neighbor coverage in log space.

For each unique value in one run, the nearest unique value in the other run is
found. A value is covered when the nearest-neighbor log distance is within:

```text
log(1 + same.center_relative)
```

Both directions must pass:

- sample-weight coverage must be at least `bulk.sample_coverage`
- unique-support coverage must be at least `bulk.support_coverage`

Sample-weight coverage uses occurrence counts. Unique-support coverage treats
each retained unique value equally.

### 4. Fall Back To Summary SAME

If bulk data are unavailable, summary data can still support `SAME` when all of
the following are true:

- both relative dispersion values are finite
- `max(ref_noise, cmp_noise) <= same.relative_dispersion_ceiling`
- centers are close:

```text
abs(ref.center - cmp.center) / min(ref.center, cmp.center)
  <= same.center_relative
```

- intervals overlap strongly enough:

```text
overlap_fraction >= same.overlap_fraction
```

If `sm_clock_rate/mean` is available on both sides, the same check must also be
confirmed in summary cycle space. If clock-rate summaries are unavailable, the
summary timing decision can still report `SAME`.

### 5. Otherwise Report AMBG

If none of the clear-gap or same-result paths has enough evidence,
`nvbench-compare` reports `AMBG` and records a reason in the summary.

## What To Do With AMBG Results

`AMBG` does not mean a benchmark improved or regressed. It means
`nvbench-compare` did not find enough evidence to classify the result as
`SAME`, `FAST`, or `SLOW`.

Useful next steps are:

- Re-run both measurements and collect JSON with bulk sample data:

```bash
./benchmark --jsonbin reference.json
./benchmark --jsonbin compare.json
nvbench-compare reference.json compare.json
```

Here `./benchmark` is the NVBench-instrumented executable or Python script that
uses `cuda.bench`.

- Use `--display explain` to inspect the interval, noise, and decision reason
  for each compared state.
- Use `--bulk-debug-python /path/to/output.py` to generate Python code that
  identifies sample and frequency files for every displayed row.
- If cold-start effects are expected, adjust cold warmup controls such as
  `--cold-warmup-runs` and `--cold-max-warmup-walltime`.
- Try a different stopping criterion when the default does not collect useful
  evidence. For example, use `--stopping-criterion entropy`, or use
  `--stopping-criterion sample-count` with an explicit `--target-samples`
  value.
- After collecting stable data, use `--dump-config` as a starting point for a
  TOML config if the default comparison thresholds are not appropriate for the
  benchmark or machine.

## Configuration

Configuration files are TOML. The current format version is `1`.

```toml
version = 1

[preset]
name = "default"

[clear_gap]
relative = 0.005

[same]
center_relative = 0.005
overlap_fraction = 0.5
relative_dispersion_ceiling = 0.02

[bulk]
sample_coverage = 0.97
support_coverage = 0.8

[bulk.rare_support]
sample_fraction = 0.001
max_removed_sample_fraction = 0.01
```

The parser is strict. Unknown top-level tables, unknown keys, wrong nesting,
unsupported versions, invalid types, non-finite values, and out-of-range values
are rejected.

TOML parsing is lazy. Python 3.11 and newer use the standard-library
`tomllib`; Python 3.10 requires the optional `tomli` package only when
`--config` is used.

## Preset And Config Precedence

Preset resolution is:

1. Use `default` when neither TOML nor CLI selects a preset.
2. Use `[preset] name = "..."` from TOML as the base preset when present.
3. Use `--preset ...` as the base preset when present, overriding the TOML
   preset selection.
4. Apply explicit TOML threshold values over whichever base preset was selected.

For example, with this config:

```toml
version = 1

[preset]
name = "permissive"

[bulk]
sample_coverage = 0.99
```

This command uses the `permissive` preset as the base and overrides only
`bulk.sample_coverage`:

```bash
nvbench-compare --config compare.toml reference.json compare.json
```

This command uses the `strict` preset as the base, but still overrides
`bulk.sample_coverage` from TOML:

```bash
nvbench-compare --config compare.toml --preset strict reference.json compare.json
```

## Built-In Presets

Built-in presets are available through `--preset`. To inspect the exact values
for the default configuration, run:

```bash
nvbench-compare --dump-config
```

To inspect a named preset, combine `--preset` with `--dump-config`:

```bash
nvbench-compare --preset strict --dump-config
nvbench-compare --preset permissive --dump-config
```

This avoids duplicating preset values in documentation and keeps the displayed
configuration tied to the installed `nvbench-compare` version.

## Configuration Keys

### `clear_gap.relative`

Valid range: `>= 0`

Minimum relative gap required before a non-overlapping timing interval can be
classified as `FAST` or `SLOW`.

### `same.center_relative`

Valid range: `>= 0`

Maximum relative center difference for summary `SAME` decisions. This value is
also used as the log-space tolerance for bulk nearest-neighbor coverage:

```text
log(1 + same.center_relative)
```

### `same.overlap_fraction`

Valid range: `0 <= value <= 1`

Minimum interval overlap fraction required for summary `SAME` decisions. The
overlap is measured relative to the narrower interval.

### `same.relative_dispersion_ceiling`

Valid range: `>= 0`

Maximum allowed relative dispersion for summary `SAME` decisions.

### `bulk.sample_coverage`

Valid range: `0 <= value <= 1`

Minimum sample-weight coverage for bulk `SAME` decisions. This check uses
counts of repeated sample values, so common values carry more weight.

### `bulk.support_coverage`

Valid range: `0 <= value <= 1`

Minimum unique-support coverage for bulk `SAME` decisions. This check treats
each retained unique value equally.

### `bulk.rare_support.sample_fraction`

Valid range: `0 <= value <= 1`

Unique values with count below:

```text
max(2, ceil(sample_fraction * total_sample_count))
```

are considered rare for support coverage.

This filter only affects unique-support coverage. Sample-weight coverage always
uses all samples.

### `bulk.rare_support.max_removed_sample_fraction`

Valid range: `0 <= value <= 1`

Maximum sample mass that may be removed from unique-support coverage by the rare
value filter. If filtering would remove more sample mass than this, remove every
unique value, or operate on an all-unique dataset, support coverage falls back
to the full unique support.
