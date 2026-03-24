#!/usr/bin/env python3
"""Run one morphing ablation job and emit artifacts for evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TUNED_CODEC_PARAMS = {
    "dac": {
        "temperature": 0.47,
        "threshold": 0.55,
        "continuity": 0.93,
        "rvq_focus": 0.30,
        "unit": 7,
        "stride": 2,
        "top_k": 7,
    },
    "spectrostream": {
        "temperature": 0.4315336855083648,
        "threshold": 0.24313963041725395,
        "continuity": 0.7887727172362835,
        "rvq_focus": 0.3460889655971231,
        "unit": 2,
        "stride": 2,
        "top_k": 8,
    },
}


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


def _codec_defaults(codec_id: str) -> dict:
    codec = (codec_id or "dac").strip().lower()
    return dict(TUNED_CODEC_PARAMS.get(codec, TUNED_CODEC_PARAMS["dac"]))


def _resolve_runtime_params(codec_id: str, args: argparse.Namespace) -> dict:
    defaults = _codec_defaults(codec_id)
    return {
        "temperature": float(args.temperature) if args.temperature is not None else float(defaults["temperature"]),
        "threshold": float(args.threshold) if args.threshold is not None else float(defaults["threshold"]),
        "continuity": float(args.continuity) if args.continuity is not None else float(defaults["continuity"]),
        "rvq_focus": float(args.rvq_focus) if args.rvq_focus is not None else float(defaults["rvq_focus"]),
        "unit": int(args.unit) if args.unit is not None else int(defaults["unit"]),
        "stride": int(args.stride) if args.stride is not None else int(defaults["stride"]),
        "top_k": int(args.top_k) if args.top_k is not None else int(defaults["top_k"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single morph ablation and export debug artifacts.")
    parser.add_argument("--palette-manifest", required=True, help="Text file with one palette file path per line")
    parser.add_argument("--source", required=True, help="Source/target clip to morph")
    parser.add_argument("--output-wav", required=True, help="Output audio path")
    parser.add_argument("--tokens-npy", required=True, help="Output morphed tokens (.npy)")
    parser.add_argument("--match-indices-npy", required=True, help="Output matched index path (.npy)")
    parser.add_argument("--latency-json", required=True, help="Output latency JSON path")
    parser.add_argument("--model", default="descript/dac_44khz", help="DAC model name/path")
    parser.add_argument("--codec", choices=["dac", "spectrostream"], default="dac")
    parser.add_argument("--matcher", choices=["greedy", "beam"], default="beam")
    parser.add_argument("--swap", choices=["full_layer", "rvq_group"], default="full_layer")
    parser.add_argument("--seed", type=int, default=1234, help="Deterministic seed")
    parser.add_argument("--temperature", type=float, default=None, help="Override (default: tuned per codec)")
    parser.add_argument("--threshold", type=float, default=None, help="Override (default: tuned per codec)")
    parser.add_argument("--continuity", type=float, default=None, help="Override (default: tuned per codec)")
    parser.add_argument("--rvq-focus", type=float, default=None, help="Override (default: tuned per codec)")
    parser.add_argument("--unit", type=int, default=None, help="Override (default: tuned per codec)")
    parser.add_argument("--stride", type=int, default=None, help="Override (default: tuned per codec)")
    parser.add_argument("--top-k", type=int, default=None, help="Override (default: tuned per codec)")
    args = parser.parse_args()
    runtime_params = _resolve_runtime_params(args.codec, args)

    use_torch_cuda = str(os.getenv("NEURAL_MORPHING_USE_TORCH_CUDA", "0")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if not use_torch_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if args.codec == "spectrostream":
        use_gpu = str(os.getenv("NEURAL_MORPHING_SPECTROSTREAM_USE_GPU", "0")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if not use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.environ["JAX_PLATFORM_NAME"] = "cpu"
            os.environ["JAX_PLATFORMS"] = "cpu"
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    from python_project_idea import LatentGranularSynthesis

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    palette_manifest = Path(args.palette_manifest)
    palette_files = _read_palette_manifest(palette_manifest)

    synth = LatentGranularSynthesis(model_name=args.model)
    synth.set_codec(args.codec)
    synth.set_temperature(runtime_params["temperature"], runtime_params["threshold"])
    synth.set_matching(runtime_params["continuity"], runtime_params["rvq_focus"])
    synth.set_unit(runtime_params["unit"], runtime_params["stride"])
    synth.set_topk(runtime_params["top_k"])
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
    tokens_arr = np.asarray(debug["tokens"], dtype=np.int32)
    match_arr = np.asarray(debug["match_indices"], dtype=np.int32)
    np.save(out_tokens, tokens_arr)
    np.save(out_match, match_arr)

    timings = debug.get("timings", {})
    source_info = sf.info(args.source)
    expected_output_samples = int(round((source_info.frames / max(source_info.samplerate, 1)) * sr))
    output_samples = int(audio_f32.shape[0]) if audio_f32.ndim > 0 else 0
    output_channels = int(audio_f32.shape[1]) if audio_f32.ndim > 1 else 1
    duration_drift_samples = int(output_samples - expected_output_samples)
    duration_drift_ms = float(1000.0 * duration_drift_samples / max(sr, 1))
    token_layout_valid = bool(tokens_arr.ndim == 2 and tokens_arr.shape[0] > 0 and tokens_arr.shape[1] > 0)

    hasher = hashlib.sha256()
    hasher.update(tokens_arr.tobytes(order="C"))
    hasher.update(match_arr.tobytes(order="C"))
    determinism_hash = hasher.hexdigest()

    encode_ms = float(timings.get("encode_ms", 0.0))
    decode_ms = float(timings.get("decode_ms", 0.0))
    total_ms = float(timings.get("total_ms", 0.0))
    audio_seconds = float(len(audio) / max(sr, 1))
    end_to_end_rtf = ((encode_ms + decode_ms) / 1000.0) / audio_seconds if audio_seconds > 0 else float("nan")
    payload = {
        "codec": args.codec,
        "matcher": args.matcher,
        "swap": args.swap,
        "seed": int(args.seed),
        "runtime_params": runtime_params,
        "encode_ok": bool(encode_ms > 0.0),
        "decode_ok": bool(decode_ms > 0.0),
        "token_layout_valid": token_layout_valid,
        "tokens_shape": list(tokens_arr.shape),
        "match_indices_len": int(match_arr.reshape(-1).shape[0]),
        "source_channels": int(source_info.channels),
        "output_channels": output_channels,
        "expected_output_samples": expected_output_samples,
        "output_samples": output_samples,
        "duration_drift_samples": duration_drift_samples,
        "duration_drift_ms": duration_drift_ms,
        "determinism_hash": determinism_hash,
        "failure_count": 0,
        "retry_count": 0,
        "encode_ms": encode_ms,
        "decode_ms": decode_ms,
        "total_ms": total_ms,
        "audio_seconds": audio_seconds,
        "end_to_end_rtf": end_to_end_rtf,
    }
    out_latency.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
