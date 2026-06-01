#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/manos/codes/neural-morphing}
PY=${PY:-"$ROOT/.venv-paper-eval/bin/python"}
GPU=${GPU:-0}
MEMORY_FRACTION=${NEURAL_MORPHING_CUDA_MEMORY_FRACTION:-0.5}
MANIFEST=${MANIFEST:-"$ROOT/data/manifests/eval_manifest_test.json"}
PALETTE=${PALETTE:-"$ROOT/artifacts/paper/dac_full96_greedy_full_layer/seed_1234/palette_train.txt"}
CACHE=${CACHE:-"$ROOT/artifacts/paper/palette_cache/dac247_shared"}
WARM=${WARM:-"$ROOT/artifacts/paper/cache_warmup/dac247_shared"}
LOGROOT=${LOGROOT:-"$ROOT/artifacts/paper/logs"}
CONDITIONS=("$@")
if [ "${#CONDITIONS[@]}" -eq 0 ]; then
  CONDITIONS=(beam_rvq_group greedy_rvq_group beam_full_layer greedy_full_layer)
fi

mkdir -p "$CACHE" "$WARM" "$LOGROOT"
cd "$ROOT"

stamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

FIRST_SOURCE=$("$PY" - <<'PY'
import json
m = json.load(open("/home/manos/codes/neural-morphing/data/manifests/eval_manifest_test.json"))
print(m["source_eval"][0]["source"])
PY
)

echo "[$(stamp)] starting DAC lane on GPU $GPU with memory fraction $MEMORY_FRACTION"
echo "[$(stamp)] warming shared palette cache at $CACHE"
CUDA_VISIBLE_DEVICES="$GPU" \
NEURAL_MORPHING_USE_TORCH_CUDA=1 \
NEURAL_MORPHING_CUDA_MEMORY_FRACTION="$MEMORY_FRACTION" \
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
"$PY" tools/run_morph_ablation.py \
  --codec dac \
  --palette-manifest "$PALETTE" \
  --source "$FIRST_SOURCE" \
  --output-wav "$WARM/output.wav" \
  --tokens-npy "$WARM/tokens.npy" \
  --match-indices-npy "$WARM/match_indices.npy" \
  --latency-json "$WARM/latency.json" \
  --diagnostics-json "$WARM/diagnostics.json" \
  --matcher beam \
  --swap rvq_group \
  --temperature 0.47 \
  --threshold 0.55 \
  --continuity 0.93 \
  --rvq-focus 0.3 \
  --unit 7 \
  --stride 2 \
  --top-k 7 \
  --seed 1234 \
  --palette-cache-dir "$CACHE" > "$LOGROOT/cache_warmup_dac247_single_gpu.log" 2>&1

RUNNER_DAC="CUDA_VISIBLE_DEVICES=$GPU NEURAL_MORPHING_USE_TORCH_CUDA=1 NEURAL_MORPHING_CUDA_MEMORY_FRACTION=$MEMORY_FRACTION PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True} $PY tools/run_morph_ablation.py --codec {codec} --palette-manifest {palette_manifest} --source {source} --output-wav {output_wav} --tokens-npy {tokens_npy} --match-indices-npy {match_indices_npy} --latency-json {latency_json} --diagnostics-json {diagnostics_json} --matcher {matcher} --swap {swap} --temperature {temperature} --threshold {threshold} --continuity {continuity} --rvq-focus {rvq_focus} --unit {unit} --stride {stride} --top-k {top_k} --seed {seed} --palette-cache-dir $CACHE"

for cond in "${CONDITIONS[@]}"; do
  out="$ROOT/artifacts/paper/dac_full96_$cond"
  log="$LOGROOT/${cond}_single_gpu.log"
  echo "[$(stamp)] resuming $cond on GPU $GPU" | tee -a "$log"
  "$PY" tools/paper_eval_protocol.py run \
    --manifest "$MANIFEST" \
    --skip-prepare \
    --skip-data-audit \
    --output-root "$out" \
    --codecs dac \
    --ablations "$cond" \
    --clip-limit 96 \
    --seeds 1234 \
    --determinism-runs 0 \
    --bootstrap 2000 \
    --resume-existing \
    --no-aggregate \
    --runner-cmd "$RUNNER_DAC" >> "$log" 2>&1
  echo "[$(stamp)] completed $cond" | tee -a "$log"
done

echo "[$(stamp)] DAC lane finished"
