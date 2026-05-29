# neural-morphing
A JUCE-based VST3/AU insert effect that morphs incoming audio into a palette of target sounds using latent representations from the Descript Audio Codec (DAC).

## Project Layout

- `Source/` – JUCE plugin sources (processor, editor, workers, backend interface).
- `tools/export_dac.py` – utility to export DAC encoder/decoder artefacts for the native backend.
- `python_project_idea.py` – RVQ-aware Python prototype (continuity + Top-K mixing) for rapid iteration.
- `tools/evaluate_morphing.py` – deterministic evaluation harness (4 ablations + objective metrics).
- `tools/run_morph_ablation.py` – single-run ablation worker that emits audio/tokens/match indices/latency.
- `tools/build_dataset_manifests.py` – immutable dataset manifest builder with hashes, license filters, and eval-pair CSVs.
- `tools/paper_experiment_suite.py` – paper artifact builder for DAC, sequence, RVQ, rho, grain/hop, runtime, and listening summaries.
- `requirements.txt` – Python dependencies for the export tool and prototype notebooks.

## Terminology

- **Target files (palette)** – the sound profiles the input should morph toward (usually a set of short clips).
- **Source input** – audio being morphed (DAW track audio or, in standalone, a loaded source file).

## Runtime Flow

**DAW insert**
1. Load target files to build the palette (DAC tokens + latent vectors).
2. The DAW feeds track audio to the plugin; each block is encoded, matched with RVQ-aware grain descriptors + continuity-constrained beam search, mixed via Top-K token voting, decoded, then blended with dry.
3. If you change palette/parameters, the track is reprocessed by the DAW (bounce/re-render) rather than the plugin mutating already-rendered audio.

**Standalone**
1. Load target files to build the palette.
2. Load a source audio file (standalone UI) to drive morphing.
3. Playback loops continuously while processing through the active mode/backend policy.

## UI Sketch

```
[Add Target Files] [Clear Palette] [Rebuild]
[Load Source Audio] [Clear Source]  Source: <filename>   (standalone only)
Status: <palette status>                         Progress: <percent>
Backend: Native / Python Bridge
Processing Mode: Quality Parity / Live Realtime
Backend Policy: Native Preferred (+ Bridge Fallback) / Bridge Only / Native Only
Bridge Codec: DAC / SpectroStream
Knobs: Temperature | Threshold | Continuity | RVQ Focus | Unit | Stride | Similarity | Envelope | Dry/Wet | Output
```

## Performance Notes

- Lightweight cache stores recently decoded morph blocks to avoid recomputing repeats.
- Temporal smoothing is applied to the morphed output (controlled by the Envelope slider; 0 disables smoothing).
- Palette grain descriptors are built with the current Unit/Stride; change them and rebuild the palette.

## Prerequisites

- **JUCE** checked out as a submodule (`git submodule update --init --recursive`).
- **CMake ≥ 3.22** and a C++17 compiler.
- Platform SDKs required by JUCE.
  - Linux (Debian/Ubuntu): `sudo apt install libxrandr-dev libxinerama-dev libxcursor-dev libxrender-dev libfreetype6-dev libfontconfig1-dev libgl1-mesa-dev libglu1-mesa-dev libgtk-3-dev libwebkit2gtk-4.1-dev libcurl4-openssl-dev`
- Optional: ONNX Runtime development package if you plan to build the native DAC backend (`onnxruntime-dev` on Ubuntu, or use the prebuilt SDK).

### Python environment

Create a virtual environment and install the Python requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This provides the tooling needed for DAC export (`tools/export_dac.py`) and for experimenting with the RVQ-aware Python morphing prototype.

## Building the Plugin

```bash
cmake -S . -B build -DNEURAL_MORPHING_ENABLE_ONNX=ON
cmake --build build --config Release
```

If ONNX Runtime is not installed, either install it or pass `-DNEURAL_MORPHING_ENABLE_ONNX=OFF` to build against the lightweight stub backend. The generated VST3/AU artefacts will be placed in `build/NeuralMorphing_artefacts/`.

After a successful build you can install the plugin:

- macOS: `cmake --install build --config Release`
- Windows: `cmake --install build --config Release`
- Linux: copy the produced VST3 bundle manually to your plugin directory.

## DAC Export Workflow

1. Install the Hugging Face `transformers`, `torch`, and DAC dependencies (`pip install -r requirements.txt`).
2. Export the encoder/decoder and embeddings:
   ```bash
   python tools/export_dac.py --output /path/to/dac_export --model descript/dac_44khz
   ```
3. Point the plugin at the export directory by setting the environment variable `NEURAL_MORPHING_MODEL_DIR=/path/to/dac_export` before launching your DAW.
4. Ensure the CMake build is configured with ONNX Runtime available (`-DNEURAL_MORPHING_ENABLE_ONNX=ON`).

If the environment variable is absent or ONNX Runtime is unavailable, the plugin automatically falls back to the stub backend for development.

## Python Bridge: Multi-Codec + Realtime PCM

The HTTP bridge now supports:
- Codec switching (`dac`, `spectrostream`) via `POST /codec`
- Capability handshake via `GET /capabilities`
- Realtime-friendly in-memory endpoints:
  - `POST /encode_pcm`
  - `POST /decode_pcm`
- Backward-compatible endpoints:
  - `GET /health`
  - `POST /encode`
  - `POST /tokens_to_vectors`
  - `POST /decode`

Environment variables (bridge):
- `NEURAL_MORPHING_BRIDGE_CODEC` (default: `dac`)
- `BRIDGE_WARM_START` (`0`/`1`, default: `0`)
- `BRIDGE_ENABLE_STREAM_SESSIONS` (`0`/`1`, default: `0`, enables optional `/session/*` API)
- `BRIDGE_DEVICE` (`auto`/`cpu`/`cuda`, default: `auto`)
- `DAC_MODEL_NAME` (default: `descript/dac_44khz`)

Environment variables (plugin realtime tuning):
- `NEURAL_MORPHING_RT_UPDATE_MS` (override morph update interval in ms)
- `NEURAL_MORPHING_RT_ENCODE_WINDOW_MS` (override rolling encode window in ms; useful for heavier codecs such as SpectroStream)
- `NEURAL_MORPHING_QUALITY_LOOKAHEAD_MS` (quality-parity lookahead, default `1000`)
- `NEURAL_MORPHING_QUALITY_SUPERFRAME` (quality-parity superframe samples, default `32768`)
- `NEURAL_MORPHING_QUALITY_HOP` (quality-parity hop samples, default `8192`)
- `NEURAL_MORPHING_QUALITY_CROSSFADE` (quality-parity overlap/crossfade samples, default `4096`)
- `NEURAL_MORPHING_LIVE_CANDIDATES` (live mode candidate cap, default `48`)
- `NEURAL_MORPHING_LIVE_BEAM` (live mode beam width cap, default `6`)

For SpectroStream support, install Magenta RT and dependencies in your bridge environment:
```bash
pip install magenta_rt
```

## Evaluation Harness

Current tuned defaults (used by app/standalone + Python demo):
- DAC: `temperature=0.47`, `threshold=0.55`, `continuity=0.93`, `rvq_focus=0.30`, `unit=7`, `stride=2`, `top_k=7`
- SpectroStream: `temperature=0.4315`, `threshold=0.2431`, `continuity=0.7888`, `rvq_focus=0.3461`, `unit=2`, `stride=2`, `top_k=8`
- Both codecs: `matcher=beam`, `swap=full_layer`
- `tools/evaluate_morphing.py evaluate` and `tools/run_morph_ablation.py` use these tuned defaults per codec when parameter flags are omitted; any provided flag is treated as a global override.

Prepare a deterministic manifest:
```bash
python tools/evaluate_morphing.py prepare \
  --palette-dir /path/to/palette_train \
  --source-dir /path/to/source_eval \
  --reference-dir /path/to/original_refs \
  --output /tmp/neural_morph_manifest.json \
  --seed 1234
```

Run the 4-ablation matrix and compute metrics:
```bash
python tools/evaluate_morphing.py evaluate \
  --manifest /tmp/neural_morph_manifest.json \
  --output-dir /tmp/neural_morph_eval \
  --envelope-corr-gate 0.90 \
  --clipping-gate-fraction 0.0001 \
  --runner-cmd "./.venv/bin/python tools/run_morph_ablation.py --codec {codec} --palette-manifest {palette_manifest} --source {source} --output-wav {output_wav} --tokens-npy {tokens_npy} --match-indices-npy {match_indices_npy} --latency-json {latency_json} --matcher {matcher} --swap {swap} --temperature {temperature} --threshold {threshold} --continuity {continuity} --rvq-focus {rvq_focus} --unit {unit} --stride {stride} --top-k {top_k} --seed {seed}"
```

Reports are written to `.../reports/` with:
- per-clip CSV
- summary JSON (`quality_metrics`, `structure_metrics`, `system_health`, latency, significance)
- reproducibility JSON
- ranked presets JSON (gated by system-health checks)

Paper protocol automation (`audit-data`, `run`, `aggregate`, `parity-sweep`):
```bash
python tools/paper_eval_protocol.py run \
  --palette-dir /data/palette_train \
  --source-dir /data/source_eval \
  --reference-dir /data/original_refs \
  --output-root artifacts/validation/paper_eval \
  --seeds 1234,2234,3234
```

Aggregate existing run directories into paper-ready tables/figures:
```bash
python tools/paper_eval_protocol.py aggregate \
  --run-dirs artifacts/validation/paper_eval/seed_1234 artifacts/validation/paper_eval/seed_2234 artifacts/validation/paper_eval/seed_3234 \
  --output-dir artifacts/validation/paper_eval/paper_reports
```

Realtime parity sweep across chunk sizes:
```bash
python tools/paper_eval_protocol.py parity-sweep \
  --palette-manifest artifacts/validation/paper_eval/seed_1234/palette_train.txt \
  --manifest artifacts/validation/paper_eval/paper_manifest.json \
  --output-dir artifacts/validation/paper_eval/parity_sweep \
  --codec dac \
  --chunk-sizes 8192,16384,32768
```

Framework validation report:
```bash
python tools/evaluate_morphing.py validate \
  --manifest /tmp/neural_morph_manifest.json \
  --output-dir /tmp/neural_morph_eval \
  --report-json /tmp/neural_morph_eval/reports/validation.json
```

Random-search presets:
```bash
python tools/evaluate_morphing.py search \
  --manifest /tmp/neural_morph_manifest.json \
  --output-dir /tmp/neural_morph_search \
  --envelope-corr-gate 0.90 \
  --clipping-gate-fraction 0.0001 \
  --runner-cmd "./.venv/bin/python tools/run_morph_ablation.py --codec {codec} --palette-manifest {palette_manifest} --source {source} --output-wav {output_wav} --tokens-npy {tokens_npy} --match-indices-npy {match_indices_npy} --latency-json {latency_json} --matcher {matcher} --swap {swap} --temperature {temperature} --threshold {threshold} --continuity {continuity} --rvq-focus {rvq_focus} --unit {unit} --stride {stride} --top-k {top_k} --seed {seed}"
```

Realtime parity comparison (Python full-context vs chunked proxy, optional plugin render):
```bash
python tools/compare_realtime_parity.py \
  --palette-manifest /tmp/palette_train.txt \
  --source /path/to/source.wav \
  --output-dir /tmp/neural_morph_parity \
  --codec dac \
  --matcher beam \
  --swap full_layer \
  --chunk-samples 32768
```

Interactive dashboard:
```bash
python tools/eval_dashboard.py
```

Paper-strengthening protocol:
```bash
python tools/build_dataset_manifests.py --palette-dir /data/palette --source-dir /data/source --reference-dir /data/reference --metadata-csv /data/metadata.csv --out-dir data/manifests --test-pairs 96 --strict
python tools/paper_eval_protocol.py run --manifest data/manifests/eval_manifest_test.json --skip-prepare --skip-data-audit --output-root artifacts/paper/dac_full96 --codecs dac --ablations greedy_full_layer,greedy_rvq_group,beam_full_layer,beam_rvq_group --clip-limit 96 --seeds 1234
python tools/paper_experiment_suite.py dac-main --run-dirs artifacts/paper/dac_full96/seed_1234
```

See `docs/paper_evaluation_protocol.md` for the full DAC, sequence, RVQ-band, rho, grain/hop, runtime, SpectroStream, and listening-test artifact map.

## Roadmap (Short)

- Palette file metadata + persistent state so sessions restore target lists.
- Incremental palette builds (append without full rebuild) with per-file progress.
- Standalone source recorder (in addition to file upload).

## Novelty Ideas (Later)

- Palette blending: expand Top-K mixing to latent barycenters (disabled by default; can sound washed out).
- Latent morph automation: sweep across palette clusters using MIDI/automation.
- Semantic tags for palette files and tag-driven matching curves.

## TODO

- [x] Scaffold JUCE VST3/AU project with parameter set, workers, and lock-free FIFOs.
- [x] Provide DAC export tool and ONNX Runtime backend stub integration.
- [ ] Replace linear `PaletteIndex` with HNSWlib ANN implementation and metadata catalogue.
- [x] Implement RVQ-aware grain matching with continuity + Top-K mixing in the processor path.
- [ ] Integrate the matching path into `MatchWorker` for async background matching.
- [ ] Integrate onset detector refinements (spectral flux, look-ahead) tuned for drum/percussive sources.
- [ ] Add persistent plugin state for palette file lists and backend settings.
- [ ] Build production-ready UI (file browser, progress meters, error messaging, parameter grouping).
- [ ] Add automated tests/CI scripts for export tooling and backend loading.
