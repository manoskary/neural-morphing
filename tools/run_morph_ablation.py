#!/usr/bin/env python3
"""Run one morphing ablation job and emit artifacts for evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from python_project_idea import LatentGranularSynthesis


def _read_palette_manifest(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Palette manifest not found: {path}")
    files = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        files.append(line)
    if not files:
        raise ValueError("Palette manifest is empty")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single morph ablation and export debug artifacts.")
    parser.add_argument("--palette-manifest", required=True, help="Text file with one palette file path per line")
    parser.add_argument("--source", required=True, help="Source/target clip to morph")
    parser.add_argument("--output-wav", required=True, help="Output audio path")
    parser.add_argument("--tokens-npy", required=True, help="Output morphed tokens (.npy)")
    parser.add_argument("--match-indices-npy", required=True, help="Output matched index path (.npy)")
    parser.add_argument("--latency-json", required=True, help="Output latency JSON path")
    parser.add_argument("--model", default="descript/dac_44khz", help="DAC model name/path")
    parser.add_argument("--matcher", choices=["greedy", "beam"], default="beam")
    parser.add_argument("--swap", choices=["full_layer", "rvq_group"], default="rvq_group")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--continuity", type=float, default=0.3)
    parser.add_argument("--rvq-focus", type=float, default=0.5)
    parser.add_argument("--unit", type=int, default=2)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    palette_manifest = Path(args.palette_manifest)
    palette_files = _read_palette_manifest(palette_manifest)

    synth = LatentGranularSynthesis(model_name=args.model)
    synth.set_temperature(args.temperature, args.threshold)
    synth.set_matching(args.continuity, args.rvq_focus)
    synth.set_unit(args.unit, args.stride)
    synth.set_topk(args.top_k)
    synth.set_ablation(args.matcher, args.swap)

    result = synth.build_dataset(palette_files, aug_checkbox=False)
    msg = str(result.get("message", ""))
    if "Done!" not in msg and "Codebook grains" not in msg:
        raise RuntimeError(f"Palette build failed: {msg}")

    sr, audio, debug = synth.morph_audio(args.source, return_debug=True)
    out_wav = Path(args.output_wav)
    out_tokens = Path(args.tokens_npy)
    out_match = Path(args.match_indices_npy)
    out_latency = Path(args.latency_json)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    out_tokens.parent.mkdir(parents=True, exist_ok=True)
    out_match.parent.mkdir(parents=True, exist_ok=True)
    out_latency.parent.mkdir(parents=True, exist_ok=True)

    audio_f32 = np.asarray(audio, dtype=np.float32) / 32767.0
    sf.write(out_wav, audio_f32, sr)
    np.save(out_tokens, np.asarray(debug["tokens"], dtype=np.int32))
    np.save(out_match, np.asarray(debug["match_indices"], dtype=np.int32))

    timings = debug.get("timings", {})
    payload = {
        "encode_ms": float(timings.get("encode_ms", 0.0)),
        "decode_ms": float(timings.get("decode_ms", 0.0)),
        "total_ms": float(timings.get("total_ms", 0.0)),
        "audio_seconds": float(len(audio) / max(sr, 1)),
    }
    out_latency.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
