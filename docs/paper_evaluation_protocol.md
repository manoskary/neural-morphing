# Paper Evaluation Protocol

This document maps the paper-strengthening checklist to repository commands.
Run commands from the repository root. On this Windows workspace, prefer
`.venv\Scripts\python.exe`; on Unix-like systems use your active `python`.

## 1. Immutable Manifests

Build the required CSV manifests and evaluation JSON manifests:

```powershell
.\.venv\Scripts\python.exe tools\build_dataset_manifests.py `
  --palette-dir C:\path\to\palette_freesound `
  --source-dir C:\path\to\source_lofi_drums `
  --reference-dir C:\path\to\reference_lofi_drums `
  --metadata-csv C:\path\to\clip_metadata.csv `
  --out-dir data\manifests `
  --test-pairs 96 `
  --legacy-count 32 `
  --strict
```

The metadata CSV can provide `clip_id`, `path`, `license`, `source_url`,
`source_ref`, `crop_start`, `crop_end`, `tempo_bpm`, and `tempo_bucket`.
The default license filter is exact and derivative/commercial-compatible:
`CC0-1.0`, `CC-BY-3.0`, `CC-BY-4.0`, `CC-BY-SA-3.0`, `CC-BY-SA-4.0`.

## 2. Main 96-Pair DAC Result

Run DAC only with the four main conditions:

```powershell
.\.venv\Scripts\python.exe tools\paper_eval_protocol.py run `
  --manifest data\manifests\eval_manifest_test.json `
  --skip-prepare `
  --skip-data-audit `
  --output-root artifacts\paper\dac_full96 `
  --codecs dac `
  --ablations greedy_full_layer,greedy_rvq_group,beam_full_layer,beam_rvq_group `
  --clip-limit 96 `
  --seeds 1234
```

Render paper artifacts:

```powershell
.\.venv\Scripts\python.exe tools\paper_experiment_suite.py dac-main `
  --run-dirs artifacts\paper\dac_full96\seed_1234
```

Outputs:
`results/dac_main_full96.csv`,
`tables/dac_main_full96.tex`,
`figures/dac_metric_distributions.pdf`.

## 3. Sequence Baselines

`greedy` is emission-only; `greedy_smooth` uses the continuity transition
term; `beam` is bounded-width search; `viterbi` is exact over retained Top-K.

```powershell
.\.venv\Scripts\python.exe tools\evaluate_morphing.py evaluate `
  --manifest data\manifests\eval_manifest_test.json `
  --output-dir artifacts\paper\sequence `
  --codecs dac `
  --ablations greedy_rvq_group,greedy_smooth_rvq_group,beam_rvq_group,viterbi_rvq_group `
  --runner-cmd ".\.venv\Scripts\python.exe tools\run_morph_ablation.py --codec {codec} --palette-manifest {palette_manifest} --source {source} --output-wav {output_wav} --tokens-npy {tokens_npy} --match-indices-npy {match_indices_npy} --latency-json {latency_json} --diagnostics-json {diagnostics_json} --matcher {matcher} --swap {swap} --temperature {temperature} --threshold {threshold} --continuity {continuity} --rvq-focus {rvq_focus} --unit {unit} --stride {stride} --top-k {top_k} --seed {seed}"

.\.venv\Scripts\python.exe tools\paper_experiment_suite.py sequence `
  --run-dirs artifacts\paper\sequence
```

Outputs:
`results/sequence_optimizer_comparison.csv`,
`tables/sequence_optimizer_comparison.tex`,
`figures/beam_viterbi_pareto.pdf`.

## 4. RVQ Band and Threshold Diagnostics

Run band policies with the Beam matcher:

```powershell
.\.venv\Scripts\python.exe tools\evaluate_morphing.py evaluate `
  --manifest data\manifests\eval_manifest_dev.json `
  --output-dir artifacts\paper\rvq_bands `
  --codecs dac `
  --ablations beam_identity,beam_coarse_gated,beam_coarse_forced,beam_middle_only,beam_fine_only,beam_middle_fine,beam_rvq_group,beam_full_layer,beam_full_layer_forced `
  --runner-cmd ".\.venv\Scripts\python.exe tools\run_morph_ablation.py --codec {codec} --palette-manifest {palette_manifest} --source {source} --output-wav {output_wav} --tokens-npy {tokens_npy} --match-indices-npy {match_indices_npy} --latency-json {latency_json} --diagnostics-json {diagnostics_json} --matcher {matcher} --swap {swap} --temperature {temperature} --threshold {threshold} --continuity {continuity} --rvq-focus {rvq_focus} --unit {unit} --stride {stride} --top-k {top_k} --seed {seed}"

.\.venv\Scripts\python.exe tools\paper_experiment_suite.py rvq-diagnostics `
  --run-dirs artifacts\paper\rvq_bands
```

Outputs:
`results/rvq_threshold_sweep.csv`,
`figures/emission_distribution_by_band.pdf`,
`figures/coarse_gate_activation_curve.pdf`.

## 5. Rho and Grain/Hop Sweeps

For rho, repeat the evaluation at
`0.00,0.15,0.30,0.50,0.70,0.85,1.00` using `--rvq-focus` or `--rho`,
then summarize all run directories:

```powershell
.\.venv\Scripts\python.exe tools\paper_experiment_suite.py rho-sweep `
  --run-dirs artifacts\paper\rho_*
```

For grain/hop, repeat with at least `--unit/--stride` pairs `3/1`, `7/2`,
and `11/4`, then summarize:

```powershell
.\.venv\Scripts\python.exe tools\paper_experiment_suite.py grain-hop `
  --run-dirs artifacts\paper\grainhop_*
```

## 6. Realtime and Portability

Chunk parity sweep:

```powershell
.\.venv\Scripts\python.exe tools\paper_eval_protocol.py parity-sweep `
  --palette-manifest artifacts\paper\dac_full96\seed_1234\palette_train.txt `
  --manifest data\manifests\eval_manifest_test.json `
  --output-dir artifacts\paper\chunk_parity `
  --codec dac `
  --chunk-sizes 2048,4096,8192,16384,32768
```

SpectroStream confirmation uses the same main and sequence commands with
`--codecs spectrostream`. Treat CoDiCodec-style non-RVQ adapters as a separate
backend adaptation stress test, not as native RVQ portability evidence.

## 7. Listening and Claims

After main/sequence runs:

```powershell
.\.venv\Scripts\python.exe tools\paper_experiment_suite.py listening-manifest `
  --run-dirs artifacts\paper\sequence `
  --conditions greedy_rvq_group,beam_rvq_group,viterbi_rvq_group `
  --sets 12

.\.venv\Scripts\python.exe tools\paper_experiment_suite.py claim-table
```

Outputs:
`listening/stimuli_manifest.csv` and `tables/claim_to_evidence.tex`.
