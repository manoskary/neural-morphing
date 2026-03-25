#!/usr/bin/env python3
"""Compare quality-parity (full-file) vs chunked realtime-proxy morphing outputs.

This tool is designed to quantify the gap between:
1) Python full-context morph output (reference),
2) Chunked sequential proxy output (standalone/VST-like scheduling),
3) Optional plugin render WAV (if provided).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python_project_idea import LatentGranularSynthesis  # noqa: E402
from tools.evaluate_morphing import (  # noqa: E402
    metric_boundary_phase_jump,
    metric_chromagram_difference,
    metric_envelope_correlation,
    metric_log_spectral_distance,
    metric_spectral_convergence,
    metric_waveform_discontinuity,
)

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


def _codec_defaults(codec_id: str) -> dict:
    return dict(TUNED_CODEC_PARAMS.get((codec_id or "dac").strip().lower(), TUNED_CODEC_PARAMS["dac"]))


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


def _read_manifest(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Palette manifest not found: {path}")
    files = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not files:
        raise ValueError("Palette manifest is empty")
    return files


def _as_float_audio(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio)
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32) / 32767.0
    else:
        arr = arr.astype(np.float32)
    return np.clip(arr, -1.0, 1.0)


def _save_audio(path: Path, samples: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), _as_float_audio(samples), sr)


def _load_source(path: Path, sr: int, stereo: bool) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=sr, mono=not stereo)
    arr = np.asarray(y, dtype=np.float32)
    if stereo:
        if arr.ndim == 1:
            arr = np.repeat(arr[:, np.newaxis], 2, axis=1)
        elif arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
            arr = arr.T
        if arr.shape[1] == 1:
            arr = np.repeat(arr, 2, axis=1)
    return arr


def _boundary_samples(total_samples: int, chunk_samples: int) -> np.ndarray:
    if chunk_samples <= 0:
        return np.zeros(0, dtype=np.int64)
    idx = list(range(chunk_samples, total_samples, chunk_samples))
    return np.asarray(idx, dtype=np.int64)


def _compute_metrics(name: str, a: np.ndarray, b: np.ndarray, sr: int, boundaries: np.ndarray) -> dict:
    return {
        "pair": name,
        "spectral_convergence": float(metric_spectral_convergence(a, b, sr)),
        "log_spectral_distance": float(metric_log_spectral_distance(a, b, sr)),
        "envelope_correlation": float(metric_envelope_correlation(a, b)),
        "chroma_difference": float(metric_chromagram_difference(a, b, sr)),
        "waveform_discontinuity_db": float(metric_waveform_discontinuity(a, sr, boundaries)),
        "boundary_phase_jump": float(metric_boundary_phase_jump(a, sr, boundaries)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare realtime-proxy morph output against Python full-context output.")
    parser.add_argument("--palette-manifest", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="descript/dac_44khz")
    parser.add_argument("--codec", choices=["dac", "spectrostream"], default="dac")
    parser.add_argument("--matcher", choices=["beam", "greedy"], default="beam")
    parser.add_argument("--swap", choices=["full_layer", "rvq_group"], default="full_layer")
    parser.add_argument("--temperature", type=float, default=None, help="Override (default: tuned per codec)")
    parser.add_argument("--threshold", type=float, default=None, help="Override (default: tuned per codec)")
    parser.add_argument("--continuity", type=float, default=None, help="Override (default: tuned per codec)")
    parser.add_argument("--rvq-focus", dest="rvq_focus", type=float, default=None, help="Override (default: tuned per codec)")
    parser.add_argument("--unit", type=int, default=None, help="Override (default: tuned per codec)")
    parser.add_argument("--stride", type=int, default=None, help="Override (default: tuned per codec)")
    parser.add_argument("--top-k", dest="top_k", type=int, default=None, help="Override (default: tuned per codec)")
    parser.add_argument("--chunk-samples", type=int, default=32768)
    parser.add_argument("--plugin-render-wav", default="", help="Optional exported standalone/VST render WAV for direct comparison.")
    args = parser.parse_args()
    runtime_params = _resolve_runtime_params(args.codec, args)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    palette_files = _read_manifest(Path(args.palette_manifest))

    synth = LatentGranularSynthesis(model_name=args.model)
    synth.set_codec(args.codec)
    synth.set_temperature(runtime_params["temperature"], runtime_params["threshold"])
    synth.set_matching(runtime_params["continuity"], runtime_params["rvq_focus"])
    synth.set_unit(runtime_params["unit"], runtime_params["stride"])
    synth.set_topk(runtime_params["top_k"])
    synth.set_ablation(args.matcher, args.swap)

    build = synth.build_dataset(palette_files, aug_checkbox=False)
    if "Done!" not in str(build.get("message", "")) and "Codebook grains" not in str(build.get("message", "")):
        raise RuntimeError(f"Palette build failed: {build}")

    source_path = Path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    sr_full, full_audio, full_debug = synth.morph_audio(str(source_path), return_debug=True)
    full_audio = _as_float_audio(full_audio)
    _save_audio(out_dir / "python_full.wav", full_audio, sr_full)
    np.save(out_dir / "python_full_tokens.npy", np.asarray(full_debug.get("tokens", np.zeros((1, 1), dtype=np.int32)), dtype=np.int32))
    np.save(out_dir / "python_full_match_indices.npy", np.asarray(full_debug.get("match_indices", np.zeros(0, dtype=np.int32)), dtype=np.int32))

    source_arr = _load_source(source_path, sr_full, stereo=synth.required_input_channels > 1)
    chunk_samples = max(128, int(args.chunk_samples))

    proxy_parts: list[np.ndarray] = []
    temp_root = Path(tempfile.mkdtemp(prefix="nm_rt_proxy_", dir=str(out_dir)))
    try:
        for i in range(0, source_arr.shape[0], chunk_samples):
            chunk = source_arr[i : i + chunk_samples]
            if chunk.shape[0] <= 0:
                continue
            chunk_path = temp_root / f"chunk_{i:09d}.wav"
            sf.write(str(chunk_path), _as_float_audio(chunk), sr_full)
            sr_chunk, wet_chunk = synth.morph_audio(str(chunk_path))
            if sr_chunk != sr_full:
                raise RuntimeError(f"Unexpected sample-rate mismatch in chunk morph: {sr_chunk} vs {sr_full}")
            proxy_parts.append(_as_float_audio(wet_chunk))
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

    if not proxy_parts:
        raise RuntimeError("Chunked realtime proxy produced no output.")

    if proxy_parts[0].ndim == 1:
        proxy_audio = np.concatenate([p.reshape(-1) for p in proxy_parts], axis=0)
    else:
        proxy_audio = np.concatenate([p if p.ndim == 2 else p[:, np.newaxis] for p in proxy_parts], axis=0)

    _save_audio(out_dir / "realtime_proxy.wav", proxy_audio, sr_full)

    n = min(full_audio.shape[0], proxy_audio.shape[0])
    full_aligned = full_audio[:n]
    proxy_aligned = proxy_audio[:n]
    boundaries = _boundary_samples(n, chunk_samples)

    result = {
        "config": {
            "codec": args.codec,
            "matcher": args.matcher,
            "swap": args.swap,
            "temperature": runtime_params["temperature"],
            "threshold": runtime_params["threshold"],
            "continuity": runtime_params["continuity"],
            "rvq_focus": runtime_params["rvq_focus"],
            "unit": runtime_params["unit"],
            "stride": runtime_params["stride"],
            "top_k": runtime_params["top_k"],
            "chunk_samples": chunk_samples,
        },
        "paths": {
            "python_full_wav": str((out_dir / "python_full.wav").resolve()),
            "realtime_proxy_wav": str((out_dir / "realtime_proxy.wav").resolve()),
            "plugin_render_wav": str(Path(args.plugin_render_wav).resolve()) if args.plugin_render_wav else "",
        },
        "metrics": {
            "proxy_vs_full": _compute_metrics("proxy_vs_full", proxy_aligned, full_aligned, sr_full, boundaries)
        },
    }

    if args.plugin_render_wav:
        plugin_path = Path(args.plugin_render_wav)
        if plugin_path.exists():
            plugin_audio, plugin_sr = sf.read(str(plugin_path), always_2d=False)
            plugin_audio = _as_float_audio(plugin_audio)
            if int(plugin_sr) != int(sr_full):
                if plugin_audio.ndim == 1:
                    plugin_audio = librosa.resample(plugin_audio, orig_sr=int(plugin_sr), target_sr=sr_full)
                else:
                    chans = [librosa.resample(plugin_audio[:, ch], orig_sr=int(plugin_sr), target_sr=sr_full) for ch in range(plugin_audio.shape[1])]
                    min_len = min(len(ch) for ch in chans)
                    plugin_audio = np.stack([ch[:min_len] for ch in chans], axis=1)

            pn = min(plugin_audio.shape[0], full_audio.shape[0], proxy_audio.shape[0])
            result["metrics"]["plugin_vs_full"] = _compute_metrics(
                "plugin_vs_full",
                plugin_audio[:pn],
                full_audio[:pn],
                sr_full,
                _boundary_samples(pn, chunk_samples),
            )
            result["metrics"]["plugin_vs_proxy"] = _compute_metrics(
                "plugin_vs_proxy",
                plugin_audio[:pn],
                proxy_audio[:pn],
                sr_full,
                _boundary_samples(pn, chunk_samples),
            )
        else:
            result["metrics"]["plugin_status"] = {"status": "missing_file", "path": str(plugin_path)}

    output_json = out_dir / "parity_summary.json"
    output_json.write_text(json.dumps(result, indent=2))
    print(f"Wrote parity summary: {output_json}")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
