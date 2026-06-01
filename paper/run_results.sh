#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PAPER_PY="${PAPER_PY:-$ROOT_DIR/.venv-paper-eval/bin/python}"
SPECTRO_PY="${SPECTRO_PY:-$ROOT_DIR/.venv-spectrostream/bin/python}"

DATASET_ROOT="${DATASET_ROOT:-$ROOT_DIR/artifacts/validation/datasets/paper_eval_v1}"
SUBSET_DIR="${SUBSET_DIR:-$DATASET_ROOT/paper_subsets}"
DAC_ROOT="${DAC_ROOT:-$ROOT_DIR/artifacts/validation/paper_eval_dac_main32}"
SPECTRO_ROOT="${SPECTRO_ROOT:-$ROOT_DIR/artifacts/validation/paper_eval_spectro_appendix16}"
GENERATED_DIR="${GENERATED_DIR:-$ROOT_DIR/paper/generated}"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/artifacts/validation/paper_logs}"

RUN_DAC="${RUN_DAC:-1}"
RUN_SPECTRO="${RUN_SPECTRO:-1}"
RUN_PARITY="${RUN_PARITY:-1}"
RUN_BUILD="${RUN_BUILD:-1}"

DAC_GPU_IDS="${DAC_GPU_IDS:-0,1,2}"
DAC_SEEDS=(1234 2234 3234)

if [[ ! -x "$PAPER_PY" ]]; then
  echo "Missing paper eval python: $PAPER_PY" >&2
  exit 1
fi

if [[ ! -x "$SPECTRO_PY" ]]; then
  echo "Missing SpectroStream python: $SPECTRO_PY" >&2
  exit 1
fi

mkdir -p "$LOG_ROOT" "$GENERATED_DIR"

run_cmd() {
  local log_file="$1"
  shift
  mkdir -p "$(dirname "$log_file")"
  echo "\$ $*" | tee "$log_file"
  "$@" 2>&1 | tee -a "$log_file"
}

echo "[1/6] Regenerating deterministic paper subsets"
run_cmd "$LOG_ROOT/select_subsets.log" \
  "$PAPER_PY" tools/paper_assets.py select-subsets \
  --manifest "$DATASET_ROOT/paper_manifest.json" \
  --output-dir "$SUBSET_DIR" \
  --parity-count 12 \
  --dac-main-count 32 \
  --spectro-count 16 \
  --spectro-palette-count 32 \
  --spectro-parity-count 4

IFS=',' read -r -a gpu_ids <<< "$DAC_GPU_IDS"
if [[ "${#gpu_ids[@]}" -ne "${#DAC_SEEDS[@]}" ]]; then
  echo "DAC_GPU_IDS must provide exactly ${#DAC_SEEDS[@]} GPU ids, got ${#gpu_ids[@]}." >&2
  exit 1
fi

if [[ "$RUN_DAC" == "1" ]]; then
  echo "[2/6] Running DAC main subset across ${#DAC_SEEDS[@]} seeds"
  dac_runner="NEURAL_MORPHING_USE_TORCH_CUDA=1 ./.venv-paper-eval/bin/python tools/run_morph_ablation.py --codec {codec} --palette-manifest {palette_manifest} --source {source} --output-wav {output_wav} --tokens-npy {tokens_npy} --match-indices-npy {match_indices_npy} --latency-json {latency_json} --matcher {matcher} --swap {swap} --temperature {temperature} --threshold {threshold} --continuity {continuity} --rvq-focus {rvq_focus} --unit {unit} --stride {stride} --top-k {top_k} --palette-cache-dir artifacts/validation/palette_cache/dac256 --seed {seed}"
  dac_pids=()
  for idx in "${!DAC_SEEDS[@]}"; do
    seed="${DAC_SEEDS[$idx]}"
    gpu="${gpu_ids[$idx]}"
    log_file="$LOG_ROOT/dac_seed_${seed}.log"
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      run_cmd "$log_file" \
        "$PAPER_PY" tools/paper_eval_protocol.py run \
        --manifest "$SUBSET_DIR/paper_manifest_dac_main32.json" \
        --output-root "$DAC_ROOT" \
        --seeds "$seed" \
        --codecs dac \
        --skip-prepare \
        --skip-data-audit \
        --determinism-runs 1 \
        --bootstrap 200 \
        --no-aggregate \
        --runner-cmd "$dac_runner"
    ) &
    dac_pids+=("$!")
  done
  for pid in "${dac_pids[@]}"; do
    wait "$pid"
  done

  run_cmd "$LOG_ROOT/dac_aggregate.log" \
    "$PAPER_PY" tools/paper_eval_protocol.py aggregate \
    --run-dirs "$DAC_ROOT/seed_1234" "$DAC_ROOT/seed_2234" "$DAC_ROOT/seed_3234" \
    --output-dir "$DAC_ROOT/paper_reports" \
    --bootstrap 200
fi

if [[ "$RUN_SPECTRO" == "1" ]]; then
  echo "[3/6] Running SpectroStream appendix subset across ${#DAC_SEEDS[@]} seeds"
  spectro_runner="MAGENTA_RT_DEFAULT_ASSET_SOURCE=hf NEURAL_MORPHING_USE_TORCH_CUDA=0 NEURAL_MORPHING_SPECTROSTREAM_USE_GPU=0 ./.venv-spectrostream/bin/python tools/run_morph_ablation.py --codec {codec} --palette-manifest {palette_manifest} --source {source} --output-wav {output_wav} --tokens-npy {tokens_npy} --match-indices-npy {match_indices_npy} --latency-json {latency_json} --matcher {matcher} --swap {swap} --temperature {temperature} --threshold {threshold} --continuity {continuity} --rvq-focus {rvq_focus} --unit {unit} --stride {stride} --top-k {top_k} --palette-cache-dir artifacts/validation/palette_cache/spectrostream32 --seed {seed}"
  spectro_pids=()
  for seed in "${DAC_SEEDS[@]}"; do
    log_file="$LOG_ROOT/spectro_seed_${seed}.log"
    (
      export MAGENTA_RT_DEFAULT_ASSET_SOURCE=hf
      export NEURAL_MORPHING_USE_TORCH_CUDA=0
      export NEURAL_MORPHING_SPECTROSTREAM_USE_GPU=0
      run_cmd "$log_file" \
        "$PAPER_PY" tools/paper_eval_protocol.py run \
        --manifest "$SUBSET_DIR/paper_manifest_spectro_appendix16.json" \
        --output-root "$SPECTRO_ROOT" \
        --seeds "$seed" \
        --codecs spectrostream \
        --skip-prepare \
        --skip-data-audit \
        --determinism-runs 1 \
        --bootstrap 200 \
        --no-aggregate \
        --runner-cmd "$spectro_runner"
    ) &
    spectro_pids+=("$!")
  done
  for pid in "${spectro_pids[@]}"; do
    wait "$pid"
  done

  run_cmd "$LOG_ROOT/spectro_aggregate.log" \
    "$PAPER_PY" tools/paper_eval_protocol.py aggregate \
    --run-dirs "$SPECTRO_ROOT/seed_1234" "$SPECTRO_ROOT/seed_2234" "$SPECTRO_ROOT/seed_3234" \
    --output-dir "$SPECTRO_ROOT/paper_reports" \
    --bootstrap 200
fi

if [[ "$RUN_PARITY" == "1" ]]; then
  echo "[4/6] Running parity sweeps"
  run_cmd "$LOG_ROOT/dac_parity.log" \
    "$PAPER_PY" tools/paper_eval_protocol.py parity-sweep \
    --palette-manifest "$DAC_ROOT/seed_1234/palette_train.txt" \
    --source-list "$SUBSET_DIR/dac_parity_sources_12.txt" \
    --output-dir "$DAC_ROOT/parity_sweep" \
    --codec dac \
    --chunk-sizes 8192,16384,32768

  run_cmd "$LOG_ROOT/spectro_parity.log" \
    "$PAPER_PY" tools/paper_eval_protocol.py parity-sweep \
    --palette-manifest "$SPECTRO_ROOT/seed_1234/palette_train.txt" \
    --source-list "$SUBSET_DIR/spectro_parity_sources_4.txt" \
    --output-dir "$SPECTRO_ROOT/parity_sweep" \
    --python-bin "$SPECTRO_PY" \
    --codec spectrostream \
    --chunk-sizes 8192,16384,32768
fi

echo "[5/6] Exporting LaTeX paper assets"
run_cmd "$LOG_ROOT/export_paper_assets.log" \
  "$PAPER_PY" tools/paper_assets.py export \
  --dac-root "$DAC_ROOT" \
  --dac-parity-root "$DAC_ROOT" \
  --spectro-root "$SPECTRO_ROOT" \
  --spectro-parity-root "$SPECTRO_ROOT" \
  --output-dir "$GENERATED_DIR"

if [[ "$RUN_BUILD" == "1" ]]; then
  echo "[6/6] Building paper PDF if TeX tools are present"
  if command -v pdflatex >/dev/null 2>&1 && command -v bibtex >/dev/null 2>&1; then
    run_cmd "$LOG_ROOT/build_paper.log" "$ROOT_DIR/paper/build.sh"
  else
    echo "Skipping paper build because pdflatex/bibtex are not installed."
  fi
fi

echo "Paper results workflow complete."
