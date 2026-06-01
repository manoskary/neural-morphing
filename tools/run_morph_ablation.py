#!/usr/bin/env python3
"""Run one morphing ablation job and emit artifacts for evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
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


def _palette_cache_key(codec: str, model: str, runtime_params: dict, palette_files: list[str]) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"neural-morphing-palette-cache-v1\0")
    for part in (codec, model, runtime_params["unit"], runtime_params["stride"]):
        hasher.update(str(part).encode("utf-8"))
        hasher.update(b"\0")
    for raw in palette_files:
        path = Path(raw).expanduser().resolve()
        hasher.update(str(path).encode("utf-8"))
        try:
            stat = path.stat()
        except OSError:
            hasher.update(b":missing")
            continue
        hasher.update(f":{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def _load_palette_cache(synth, cache_file: Path, meta_file: Path, cache_key: str, palette_files: list[str]) -> bool:
    if not cache_file.exists() or not meta_file.exists():
        return False
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if meta.get("cache_key") != cache_key:
            return False
        with np.load(cache_file) as data:
            palette_codes = np.array(data["palette_codes"], copy=True)
            palette_desc_full = np.array(data["palette_desc_full"], copy=True)
            group_descs = [np.array(data[f"palette_desc_group_{idx}"], copy=True) for idx in range(3)]
            file_ids = np.array(data["palette_file_ids"], copy=True)
            frame_indices = np.array(data["palette_frame_indices"], copy=True)
    except Exception as exc:
        print(f"Palette cache ignored ({cache_file}): {exc}")
        return False

    synth.files = [str(Path(p).expanduser().resolve()) for p in palette_files]
    synth.last_aug = False
    synth.prev_best_index = None
    synth.rvq_groups = [list(map(int, group)) for group in meta.get("rvq_groups", [])]
    if synth.codebook_embeddings is None:
        synth.codebook_embeddings = synth._load_codebook_embeddings()
    synth.palette_codes = torch.from_numpy(palette_codes.astype(np.int16, copy=False))
    synth.palette_desc_full = torch.from_numpy(palette_desc_full.astype(np.float32, copy=False))
    synth.palette_desc_groups = [torch.from_numpy(arr.astype(np.float32, copy=False)) for arr in group_descs]
    synth.palette_file_ids = file_ids.astype(np.int32, copy=False)
    synth.palette_frame_indices = frame_indices.astype(np.int64, copy=False)
    return bool(synth.palette_codes.numel() > 0)


def _save_palette_cache(
    synth,
    cache_file: Path,
    meta_file: Path,
    cache_key: str,
    palette_files: list[str],
    runtime_params: dict,
) -> None:
    if synth.palette_codes is None or synth.palette_desc_full is None or synth.palette_desc_groups is None:
        return
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "palette_codes": synth.palette_codes.cpu().numpy(),
        "palette_desc_full": synth.palette_desc_full.cpu().numpy(),
        "palette_file_ids": np.asarray(synth.palette_file_ids, dtype=np.int32),
        "palette_frame_indices": np.asarray(synth.palette_frame_indices, dtype=np.int64),
    }
    for idx in range(3):
        if idx < len(synth.palette_desc_groups):
            arrays[f"palette_desc_group_{idx}"] = synth.palette_desc_groups[idx].cpu().numpy()
        else:
            arrays[f"palette_desc_group_{idx}"] = np.empty((synth.palette_codes.shape[0], 0), dtype=np.float32)

    with tempfile.NamedTemporaryFile(prefix=cache_file.stem + ".", suffix=".npz", dir=str(cache_file.parent), delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        np.savez(tmp_path, **arrays)
        os.replace(tmp_path, cache_file)
        meta = {
            "cache_key": cache_key,
            "codec": getattr(synth, "codec_id", ""),
            "model": getattr(synth, "model_name", ""),
            "unit": int(runtime_params["unit"]),
            "stride": int(runtime_params["stride"]),
            "rvq_groups": [list(map(int, group)) for group in (synth.rvq_groups or [])],
            "palette_files": [str(Path(p).expanduser().resolve()) for p in palette_files],
            "num_grains": int(synth.palette_codes.shape[0]),
        }
        meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


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


def _apply_cuda_memory_fraction() -> float | None:
    raw = os.getenv("NEURAL_MORPHING_CUDA_MEMORY_FRACTION", "").strip()
    if not raw:
        return None
    fraction = float(raw)
    if not (0.0 < fraction <= 1.0):
        raise ValueError("NEURAL_MORPHING_CUDA_MEMORY_FRACTION must be in (0, 1].")
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction, device=0)
        return fraction
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single morph ablation and export debug artifacts.")
    parser.add_argument("--palette-manifest", required=True, help="Text file with one palette file path per line")
    parser.add_argument("--source", required=True, help="Source/target clip to morph")
    parser.add_argument("--output-wav", required=True, help="Output audio path")
    parser.add_argument("--tokens-npy", required=True, help="Output morphed tokens (.npy)")
    parser.add_argument("--match-indices-npy", required=True, help="Output matched index path (.npy)")
    parser.add_argument("--latency-json", required=True, help="Output latency JSON path")
    parser.add_argument("--diagnostics-json", default="", help="Optional output path for sequence/RVQ diagnostics JSON")
    parser.add_argument("--model", default="descript/dac_44khz", help="DAC model name/path")
    parser.add_argument("--codec", choices=["dac", "spectrostream"], default="dac")
    parser.add_argument("--matcher", choices=["greedy", "greedy_smooth", "beam", "viterbi"], default="beam")
    parser.add_argument(
        "--swap",
        choices=[
            "identity",
            "coarse_gated",
            "coarse_forced",
            "middle_only",
            "fine_only",
            "middle_fine",
            "rvq_group",
            "rvq_group_current",
            "full_layer",
            "full_layer_gated",
            "full_layer_forced",
        ],
        default="full_layer",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Deterministic seed")
    parser.add_argument("--temperature", type=float, default=None, help="Override (default: tuned per codec)")
    parser.add_argument("--threshold", type=float, default=None, help="Override (default: tuned per codec)")
    parser.add_argument("--continuity", type=float, default=None, help="Override (default: tuned per codec)")
    parser.add_argument("--rvq-focus", type=float, default=None, help="Override (default: tuned per codec)")
    parser.add_argument("--rho", dest="rvq_focus", type=float, default=None, help="Alias for --rvq-focus")
    parser.add_argument("--unit", type=int, default=None, help="Override (default: tuned per codec)")
    parser.add_argument("--stride", type=int, default=None, help="Override (default: tuned per codec)")
    parser.add_argument("--top-k", type=int, default=None, help="Override (default: tuned per codec)")
    parser.add_argument("--palette-cache-dir", default="", help="Optional directory for cached encoded palette tensors")
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

    cuda_memory_fraction = _apply_cuda_memory_fraction()

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

    loaded_from_cache = False
    cache_file = None
    meta_file = None
    cache_key = ""
    if args.palette_cache_dir:
        cache_key = _palette_cache_key(args.codec, args.model, runtime_params, palette_files)
        cache_root = Path(args.palette_cache_dir)
        cache_file = cache_root / f"{cache_key}.npz"
        meta_file = cache_root / f"{cache_key}.json"
        loaded_from_cache = _load_palette_cache(synth, cache_file, meta_file, cache_key, palette_files)

    if not loaded_from_cache:
        result = synth.build_dataset(palette_files, aug_checkbox=False)
        msg = str(result.get("message", ""))
        if "Done!" not in msg and "Codebook grains" not in msg:
            raise RuntimeError(f"Palette build failed: {msg}")
        if cache_file is not None and meta_file is not None:
            _save_palette_cache(synth, cache_file, meta_file, cache_key, palette_files, runtime_params)

    sr, audio, debug = synth.morph_audio(args.source, return_debug=True)
    out_wav = Path(args.output_wav)
    out_tokens = Path(args.tokens_npy)
    out_match = Path(args.match_indices_npy)
    out_latency = Path(args.latency_json)
    out_diagnostics = Path(args.diagnostics_json) if args.diagnostics_json else None
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    out_tokens.parent.mkdir(parents=True, exist_ok=True)
    out_match.parent.mkdir(parents=True, exist_ok=True)
    out_latency.parent.mkdir(parents=True, exist_ok=True)
    if out_diagnostics is not None:
        out_diagnostics.parent.mkdir(parents=True, exist_ok=True)

    audio_f32 = np.asarray(audio, dtype=np.float32) / 32767.0
    sf.write(out_wav, audio_f32, sr)
    tokens_arr = np.asarray(debug["tokens"], dtype=np.int32)
    match_arr = np.asarray(debug["match_indices"], dtype=np.int32)
    np.save(out_tokens, tokens_arr)
    np.save(out_match, match_arr)

    timings = debug.get("timings", {})
    diagnostics = debug.get("diagnostics", {})
    if out_diagnostics is not None:
        out_diagnostics.write_text(json.dumps(diagnostics, indent=2))

    sequence = diagnostics.get("sequence", {}) if isinstance(diagnostics, dict) else {}
    token_change_rates = diagnostics.get("token_change_rates", {}) if isinstance(diagnostics, dict) else {}
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
        "cuda_memory_fraction": cuda_memory_fraction if cuda_memory_fraction is not None else "",
        "palette_cache_hit": bool(loaded_from_cache),
        "palette_cache_file": str(cache_file) if cache_file is not None else "",
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
        "diagnostics_json": str(out_diagnostics) if out_diagnostics is not None else "",
        "objective_j": sequence.get("objective_j", float("nan")),
        "emission_cost": sequence.get("emission_cost", float("nan")),
        "transition_cost": sequence.get("transition_cost", float("nan")),
        "weighted_transition_cost": sequence.get("weighted_transition_cost", float("nan")),
        "sequence_runtime_ms": sequence.get("runtime_ms", float("nan")),
        "file_switch_rate": sequence.get("file_switch_rate", float("nan")),
        "adjacent_step_rate": sequence.get("adjacent_step_rate", float("nan")),
        "coarse_transfer_fraction": diagnostics.get("coarse_transfer_fraction", float("nan")) if isinstance(diagnostics, dict) else float("nan"),
        "coarse_fallback_fraction": diagnostics.get("coarse_fallback_fraction", float("nan")) if isinstance(diagnostics, dict) else float("nan"),
        "token_change_rate_coarse": token_change_rates.get("coarse", float("nan")),
        "token_change_rate_middle": token_change_rates.get("middle", float("nan")),
        "token_change_rate_fine": token_change_rates.get("fine", float("nan")),
        "token_change_rate_overall": token_change_rates.get("overall", float("nan")),
    }
    out_latency.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
