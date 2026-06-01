# Paper Experiments

This paper draft is wired to deterministic evaluation subsets and generated LaTeX assets.

## Subsets

Subset generation is deterministic and driven by:

```bash
.venv-paper-eval/bin/python tools/paper_assets.py select-subsets \
  --manifest artifacts/validation/datasets/paper_eval_v1/paper_manifest.json \
  --output-dir artifacts/validation/datasets/paper_eval_v1/paper_subsets \
  --parity-count 12 \
  --dac-main-count 32 \
  --spectro-count 16 \
  --spectro-palette-count 32 \
  --spectro-parity-count 4
```

Outputs:

- `artifacts/validation/datasets/paper_eval_v1/paper_subsets/paper_manifest_dac_main32.json`
- `artifacts/validation/datasets/paper_eval_v1/paper_subsets/paper_manifest_spectro_appendix16.json`
- `artifacts/validation/datasets/paper_eval_v1/paper_subsets/dac_parity_sources_12.txt`
- `artifacts/validation/datasets/paper_eval_v1/paper_subsets/spectro_parity_sources_4.txt`
- `artifacts/validation/datasets/paper_eval_v1/paper_subsets/subset_selection.json`

## Experiment Roots

- DAC main study: `artifacts/validation/paper_eval_dac_main32`
- SpectroStream appendix: `artifacts/validation/paper_eval_spectro_appendix16`

Each root contains per-seed run directories:

- `seed_1234`
- `seed_2234`
- `seed_3234`

Aggregated reports are written to:

- `paper_reports/aggregated_summary.json`
- `paper_reports/paper_report.md`
- `paper_reports/table_quality_structure.csv`
- `paper_reports/table_system_health.csv`
- `paper_reports/table_significance.csv`
- `paper_reports/figure_latency_quality_pareto.csv`

Parity sweeps are written to:

- `parity_sweep/parity_sweep_summary.json`
- `parity_sweep/parity_curve.csv`
- `parity_sweep/parity_per_clip.csv`

## Paper Asset Export

Generated LaTeX fragments for `paper/paper.tex` come from:

```bash
.venv-paper-eval/bin/python tools/paper_assets.py export \
  --dac-root artifacts/validation/paper_eval_dac_main32 \
  --dac-parity-root artifacts/validation/paper_eval_dac_main32 \
  --spectro-root artifacts/validation/paper_eval_spectro_appendix16 \
  --spectro-parity-root artifacts/validation/paper_eval_spectro_appendix16 \
  --output-dir paper/generated
```

Main generated files:

- `paper/generated/dac_main_results_table.tex`
- `paper/generated/dac_health_table.tex`
- `paper/generated/dac_significance_table.tex`
- `paper/generated/dac_parity_table.tex`
- `paper/generated/spectro_appendix_results_table.tex`
- `paper/generated/spectro_significance_table.tex`
- `paper/generated/spectro_parity_table.tex`

## End-to-End Wrapper

To rerun the full paper workflow:

```bash
paper/run_results.sh
```

Useful environment variables:

- `RUN_DAC=0`
- `RUN_SPECTRO=0`
- `RUN_PARITY=0`
- `RUN_BUILD=0`
- `DAC_GPU_IDS=0,1,2`

## TeX Build

The paper build step expects `pdflatex` and `bibtex` in `PATH`:

```bash
paper/build.sh
```

If TeX tools are unavailable, `paper/run_results.sh` skips the PDF build and still exports the tables.
