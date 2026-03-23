#!/usr/bin/env python3
"""Evaluation harness for neural-morphing codec/matching experiments.

This tool supports:
1) Deterministic manifest preparation for internal held-out evaluation.
2) Running a fixed 4-ablation matrix (greedy/beam x full/RVQ-group swap) via
   an external runner command template.
3) Computing requested metrics:
   - FAD (if `frechet_audio_distance` is installed)
   - Index jitter
   - Token discontinuity (normalized Hamming between consecutive frames)
   - Waveform discontinuity (short-time energy jump at boundaries)
   - Chromagram difference
4) Aggregated reporting with bootstrap CIs, paired significance, and latency.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import librosa
import numpy as np
import soundfile as sf
from scipy.stats import wilcoxon


ABLATIONS = [
    {"id": "greedy_full_layer", "matcher": "greedy", "swap": "full_layer"},
    {"id": "greedy_rvq_group", "matcher": "greedy", "swap": "rvq_group"},
    {"id": "beam_full_layer", "matcher": "beam", "swap": "full_layer"},
    {"id": "beam_rvq_group", "matcher": "beam", "swap": "rvq_group"},
]


@dataclass
class ClipEntry:
    clip_id: str
    source: Path
    reference: Path


def _audio_files(root: Path) -> List[Path]:
    exts = {".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif"}
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in exts and p.is_file()])


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _bootstrap_ci(values: List[float], alpha: float = 0.05, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    arr = np.asarray([v for v in values if not math.isnan(v)], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        boots[i] = np.mean(sample)
    lo = np.percentile(boots, 100 * (alpha / 2))
    hi = np.percentile(boots, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def _parse_manifest(path: Path) -> tuple[List[Path], List[ClipEntry], dict]:
    data = json.loads(path.read_text())
    palette = [Path(p) for p in data.get("palette_train", [])]
    source_eval = []
    for row in data.get("source_eval", []):
        source_eval.append(
            ClipEntry(
                clip_id=row["id"],
                source=Path(row["source"]),
                reference=Path(row["reference"]),
            )
        )
    return palette, source_eval, data


def _ensure_tokens_codebook_major(tokens: np.ndarray) -> np.ndarray:
    arr = np.asarray(tokens)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D token array, got shape={arr.shape}")
    # Heuristic: codebook depth is typically <= 256; frame count is usually larger.
    if arr.shape[0] <= arr.shape[1]:
        return arr.astype(np.int32, copy=False)
    return arr.T.astype(np.int32, copy=False)


def metric_index_jitter(match_indices: np.ndarray) -> float:
    indices = np.asarray(match_indices, dtype=np.float64).reshape(-1)
    if indices.size < 2:
        return float("nan")
    return float(np.mean(np.abs(indices[1:] - indices[:-1])))


def metric_token_discontinuity(tokens_2d: np.ndarray) -> float:
    tokens = _ensure_tokens_codebook_major(tokens_2d)
    if tokens.shape[1] < 2:
        return float("nan")
    diff = tokens[:, 1:] != tokens[:, :-1]
    frame_hamming = np.mean(diff.astype(np.float64), axis=0)
    return float(np.mean(frame_hamming))


def metric_waveform_discontinuity(
    audio: np.ndarray,
    sample_rate: int,
    boundaries_samples: np.ndarray,
    window_ms: float = 10.0,
) -> float:
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    w = max(1, int(round(sample_rate * (window_ms / 1000.0))))
    if boundaries_samples.size == 0:
        return float("nan")

    jumps = []
    for b in boundaries_samples.astype(int):
        if b - w < 0 or b + w >= audio.shape[0]:
            continue
        left = audio[b - w : b]
        right = audio[b : b + w]
        e_left = float(np.mean(left * left) + 1e-12)
        e_right = float(np.mean(right * right) + 1e-12)
        db_jump = abs(10.0 * math.log10(e_right) - 10.0 * math.log10(e_left))
        jumps.append(db_jump)
    if not jumps:
        return float("nan")
    return float(np.mean(jumps))


def metric_chromagram_difference(output_audio: np.ndarray, ref_audio: np.ndarray, sample_rate: int) -> float:
    if output_audio.ndim > 1:
        output_audio = np.mean(output_audio, axis=1)
    if ref_audio.ndim > 1:
        ref_audio = np.mean(ref_audio, axis=1)
    min_len = min(output_audio.shape[0], ref_audio.shape[0])
    if min_len < 2048:
        return float("nan")
    output_audio = output_audio[:min_len]
    ref_audio = ref_audio[:min_len]
    chroma_out = librosa.feature.chroma_stft(y=output_audio, sr=sample_rate)
    chroma_ref = librosa.feature.chroma_stft(y=ref_audio, sr=sample_rate)
    frames = min(chroma_out.shape[1], chroma_ref.shape[1])
    if frames == 0:
        return float("nan")
    return float(np.mean(np.abs(chroma_out[:, :frames] - chroma_ref[:, :frames])))


def metric_fad(reference_wavs: List[Path], generated_wavs: List[Path], model_name: str = "vggish") -> float:
    try:
        from frechet_audio_distance import FrechetAudioDistance  # type: ignore
    except Exception:
        return float("nan")

    if not reference_wavs or not generated_wavs:
        return float("nan")

    with tempfile.TemporaryDirectory(prefix="nm_fad_ref_") as ref_dir, tempfile.TemporaryDirectory(
        prefix="nm_fad_gen_"
    ) as gen_dir:
        ref_root = Path(ref_dir)
        gen_root = Path(gen_dir)
        for i, wav in enumerate(reference_wavs):
            (ref_root / f"ref_{i:05d}.wav").symlink_to(wav.resolve())
        for i, wav in enumerate(generated_wavs):
            (gen_root / f"gen_{i:05d}.wav").symlink_to(wav.resolve())

        fad = FrechetAudioDistance(model_name=model_name, sample_rate=16000, use_pca=False, use_activation=False)
        score = fad.score(str(ref_root), str(gen_root))
        return float(score)


def _paired_significance(
    rows: List[dict],
    metric_key: str,
    baseline_id: str = "greedy_full_layer",
) -> Dict[str, dict]:
    by_ablation: Dict[str, Dict[str, float]] = {}
    for row in rows:
        if math.isnan(row.get(metric_key, float("nan"))):
            continue
        by_ablation.setdefault(row["ablation_id"], {})[row["clip_id"]] = row[metric_key]

    if baseline_id not in by_ablation:
        return {}

    result: Dict[str, dict] = {}
    baseline = by_ablation[baseline_id]
    for ablation_id, values in by_ablation.items():
        if ablation_id == baseline_id:
            continue
        common = sorted(set(baseline.keys()) & set(values.keys()))
        if len(common) < 3:
            result[ablation_id] = {"n": len(common), "p_value": float("nan")}
            continue
        a = np.asarray([baseline[c] for c in common], dtype=np.float64)
        b = np.asarray([values[c] for c in common], dtype=np.float64)
        try:
            stat = wilcoxon(a, b, zero_method="wilcox", correction=False, alternative="two-sided")
            p_value = float(stat.pvalue)
        except Exception:
            p_value = float("nan")
        result[ablation_id] = {"n": len(common), "p_value": p_value}
    return result


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    samples, sr = sf.read(str(path), always_2d=False)
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 1:
        return samples, int(sr)
    return samples, int(sr)


def _run_command_template(template: str, context: dict) -> None:
    command = template.format(**context)
    subprocess.run(command, shell=True, check=True)


def prepare_manifest(args: argparse.Namespace) -> None:
    seed = int(args.seed)
    rng = np.random.default_rng(seed)

    if args.all_audio_dir:
        all_files = _audio_files(Path(args.all_audio_dir))
        if len(all_files) < 3:
            raise ValueError("Need at least 3 audio files for deterministic split")
        shuffled = all_files.copy()
        rng.shuffle(shuffled)

        n_total = len(shuffled)
        n_palette = max(1, int(round(n_total * args.palette_ratio)))
        n_source = max(1, int(round(n_total * args.source_ratio)))
        n_refs = max(1, n_total - n_palette - n_source)

        palette = shuffled[:n_palette]
        source = shuffled[n_palette : n_palette + n_source]
        refs = shuffled[n_palette + n_source : n_palette + n_source + n_refs]
        if not refs:
            refs = source
    else:
        if not args.palette_dir or not args.source_dir:
            raise ValueError("Provide --palette-dir and --source-dir (or use --all-audio-dir)")
        palette = _audio_files(Path(args.palette_dir))
        source = _audio_files(Path(args.source_dir))
        refs = _audio_files(Path(args.reference_dir)) if args.reference_dir else source

    if not palette:
        raise ValueError("No palette files found")
    if not source:
        raise ValueError("No source_eval files found")
    if not refs:
        raise ValueError("No reference files found")

    refs_by_stem = {p.stem: p for p in refs}
    source_eval = []
    for i, src in enumerate(source):
        ref = refs_by_stem.get(src.stem, refs[i % len(refs)])
        source_eval.append({"id": src.stem, "source": str(src.resolve()), "reference": str(ref.resolve())})

    manifest = {
        "seed": seed,
        "palette_train": [str(p.resolve()) for p in palette],
        "source_eval": source_eval,
        "notes": "Deterministic split for internal evaluation.",
    }
    out = Path(args.output)
    _write_json(out, manifest)
    print(f"Wrote manifest: {out}")
    print(f"palette_train={len(palette)} source_eval={len(source_eval)}")


def evaluate(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    palette, clips, manifest = _parse_manifest(manifest_path)
    palette_manifest = out_root / "palette_train.txt"
    palette_manifest.write_text("\n".join(str(p) for p in palette))

    rows: List[dict] = []
    refs_for_fad: List[Path] = []
    generated_for_fad: Dict[str, List[Path]] = {ab["id"]: [] for ab in ABLATIONS}

    for ablation in ABLATIONS:
        ab_id = ablation["id"]
        for clip in clips:
            run_dir = out_root / "runs" / ab_id / clip.clip_id
            run_dir.mkdir(parents=True, exist_ok=True)

            output_wav = run_dir / "output.wav"
            tokens_npy = run_dir / "tokens.npy"
            match_indices_npy = run_dir / "match_indices.npy"
            latency_json = run_dir / "latency.json"
            config_json = run_dir / "config.json"
            _write_json(
                config_json,
                {
                    "ablation": ablation,
                    "clip_id": clip.clip_id,
                    "source": str(clip.source),
                    "reference": str(clip.reference),
                    "palette_manifest": str(palette_manifest),
                },
            )

            if args.runner_cmd:
                context = {
                    "source": shlex.quote(str(clip.source)),
                    "reference": shlex.quote(str(clip.reference)),
                    "palette_manifest": shlex.quote(str(palette_manifest)),
                    "output_wav": shlex.quote(str(output_wav)),
                    "tokens_npy": shlex.quote(str(tokens_npy)),
                    "match_indices_npy": shlex.quote(str(match_indices_npy)),
                    "latency_json": shlex.quote(str(latency_json)),
                    "config_json": shlex.quote(str(config_json)),
                    "matcher": ablation["matcher"],
                    "swap": ablation["swap"],
                    "ablation_id": ab_id,
                }
                if not args.dry_run:
                    _run_command_template(args.runner_cmd, context)
                else:
                    print("DRY RUN:", args.runner_cmd.format(**context))

            if not output_wav.exists():
                rows.append(
                    {
                        "ablation_id": ab_id,
                        "clip_id": clip.clip_id,
                        "status": "missing_output",
                        "fad": float("nan"),
                        "index_jitter": float("nan"),
                        "token_discontinuity": float("nan"),
                        "waveform_discontinuity_db": float("nan"),
                        "chroma_difference": float("nan"),
                    }
                )
                continue

            out_audio, out_sr = _load_audio(output_wav)
            ref_audio, ref_sr = _load_audio(clip.reference)
            if ref_sr != out_sr:
                if ref_audio.ndim == 1:
                    ref_audio = librosa.resample(ref_audio, orig_sr=ref_sr, target_sr=out_sr)
                else:
                    chans = [
                        librosa.resample(ref_audio[:, ch], orig_sr=ref_sr, target_sr=out_sr)
                        for ch in range(ref_audio.shape[1])
                    ]
                    min_len = min(len(ch) for ch in chans)
                    ref_audio = np.stack([ch[:min_len] for ch in chans], axis=1)
                ref_sr = out_sr

            token_discontinuity = float("nan")
            if tokens_npy.exists():
                tokens = np.load(tokens_npy)
                token_discontinuity = metric_token_discontinuity(tokens)

            index_jitter = float("nan")
            boundaries = np.array([], dtype=np.int64)
            if match_indices_npy.exists():
                match_indices = np.load(match_indices_npy)
                index_jitter = metric_index_jitter(match_indices)
                n = int(np.asarray(match_indices).reshape(-1).shape[0])
                if n > 1:
                    boundaries = np.linspace(0, len(out_audio), num=n + 1, dtype=np.int64)[1:-1]

            waveform_disc = metric_waveform_discontinuity(out_audio, out_sr, boundaries)
            chroma_diff = metric_chromagram_difference(out_audio, ref_audio, out_sr)

            row = {
                "ablation_id": ab_id,
                "clip_id": clip.clip_id,
                "status": "ok",
                "fad": float("nan"),  # filled later per-ablation
                "index_jitter": index_jitter,
                "token_discontinuity": token_discontinuity,
                "waveform_discontinuity_db": waveform_disc,
                "chroma_difference": chroma_diff,
                "output_wav": str(output_wav),
                "tokens_npy": str(tokens_npy) if tokens_npy.exists() else "",
                "match_indices_npy": str(match_indices_npy) if match_indices_npy.exists() else "",
                "latency_json": str(latency_json) if latency_json.exists() else "",
            }
            rows.append(row)
            refs_for_fad.append(clip.reference.resolve())
            generated_for_fad[ab_id].append(output_wav.resolve())

    # Compute per-ablation FAD once and broadcast to rows.
    fad_by_ablation: Dict[str, float] = {}
    for ab in ABLATIONS:
        ab_id = ab["id"]
        fad_by_ablation[ab_id] = metric_fad(reference_wavs=refs_for_fad, generated_wavs=generated_for_fad[ab_id])
        for row in rows:
            if row["ablation_id"] == ab_id:
                row["fad"] = fad_by_ablation[ab_id]

    # Aggregate summary.
    metrics = ["fad", "index_jitter", "token_discontinuity", "waveform_discontinuity_db", "chroma_difference"]
    summary = {"ablation_summary": {}, "significance": {}, "latency": {}}
    for ab in ABLATIONS:
        ab_id = ab["id"]
        ab_rows = [r for r in rows if r["ablation_id"] == ab_id and r["status"] == "ok"]
        metric_summary = {}
        for m in metrics:
            vals = [float(r[m]) for r in ab_rows if not math.isnan(float(r[m]))]
            if not vals:
                metric_summary[m] = {"mean": float("nan"), "std": float("nan"), "ci95": [float("nan"), float("nan")]}
                continue
            lo, hi = _bootstrap_ci(vals, alpha=0.05, n_boot=args.bootstrap, seed=args.seed)
            metric_summary[m] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "ci95": [lo, hi],
                "n": len(vals),
            }
        summary["ablation_summary"][ab_id] = metric_summary

    for m in ["index_jitter", "token_discontinuity", "waveform_discontinuity_db", "chroma_difference"]:
        summary["significance"][m] = _paired_significance(rows, m, baseline_id="greedy_full_layer")

    # Latency aggregation (if runner emits latency JSON files).
    for ab in ABLATIONS:
        ab_id = ab["id"]
        encode_ms = []
        decode_ms = []
        rtf = []
        for row in rows:
            if row["ablation_id"] != ab_id:
                continue
            latency_path = row.get("latency_json", "")
            if not latency_path:
                continue
            lp = Path(latency_path)
            if not lp.exists():
                continue
            payload = json.loads(lp.read_text())
            e = float(payload.get("encode_ms", float("nan")))
            d = float(payload.get("decode_ms", float("nan")))
            sec = float(payload.get("audio_seconds", float("nan")))
            if not math.isnan(e):
                encode_ms.append(e)
            if not math.isnan(d):
                decode_ms.append(d)
            if sec > 0 and not math.isnan(e) and not math.isnan(d):
                rtf.append(((e + d) / 1000.0) / sec)
        summary["latency"][ab_id] = {
            "encode_ms_p50": float(np.percentile(encode_ms, 50)) if encode_ms else float("nan"),
            "encode_ms_p95": float(np.percentile(encode_ms, 95)) if encode_ms else float("nan"),
            "decode_ms_p50": float(np.percentile(decode_ms, 50)) if decode_ms else float("nan"),
            "decode_ms_p95": float(np.percentile(decode_ms, 95)) if decode_ms else float("nan"),
            "rtf_mean": float(np.mean(rtf)) if rtf else float("nan"),
            "rtf_p95": float(np.percentile(rtf, 95)) if rtf else float("nan"),
            "n": len(rtf),
        }

    # Persist reports.
    report_dir = out_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    csv_path = report_dir / "per_clip_metrics.csv"
    keys = [
        "ablation_id",
        "clip_id",
        "status",
        "fad",
        "index_jitter",
        "token_discontinuity",
        "waveform_discontinuity_db",
        "chroma_difference",
        "output_wav",
        "tokens_npy",
        "match_indices_npy",
        "latency_json",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})

    _write_json(report_dir / "summary.json", summary)
    _write_json(
        report_dir / "reproducibility.json",
        {
            "manifest": str(manifest_path.resolve()),
            "palette_count": len(palette),
            "source_eval_count": len(clips),
            "ablations": ABLATIONS,
            "runner_cmd": args.runner_cmd or "",
            "seed": args.seed,
            "bootstrap": args.bootstrap,
        },
    )
    print(f"Wrote reports to: {report_dir}")
    print(f"Per-clip CSV: {csv_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate neural morphing codec/matching experiments.")
    sub = p.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="Prepare deterministic internal evaluation manifest.")
    prep.add_argument("--output", required=True, help="Output manifest JSON path")
    prep.add_argument("--seed", type=int, default=1234, help="Deterministic seed")
    prep.add_argument("--all-audio-dir", default="", help="Single corpus directory to split deterministically")
    prep.add_argument("--palette-dir", default="", help="Palette-train directory (if not using --all-audio-dir)")
    prep.add_argument("--source-dir", default="", help="Source-eval directory (if not using --all-audio-dir)")
    prep.add_argument("--reference-dir", default="", help="Reference-original directory (optional)")
    prep.add_argument("--palette-ratio", type=float, default=0.6, help="Split ratio for palette_train when using --all-audio-dir")
    prep.add_argument("--source-ratio", type=float, default=0.2, help="Split ratio for source_eval when using --all-audio-dir")
    prep.set_defaults(func=prepare_manifest)

    ev = sub.add_parser("evaluate", help="Run ablation evaluation + compute metrics/reports.")
    ev.add_argument("--manifest", required=True, help="Manifest JSON from `prepare`")
    ev.add_argument("--output-dir", required=True, help="Output directory for runs/reports")
    ev.add_argument(
        "--runner-cmd",
        default="",
        help=(
            "Optional shell command template to generate outputs for each clip+ablation. "
            "Placeholders: {source} {reference} {palette_manifest} {output_wav} {tokens_npy} "
            "{match_indices_npy} {latency_json} {config_json} {matcher} {swap} {ablation_id}"
        ),
    )
    ev.add_argument("--dry-run", action="store_true", help="Print expanded runner commands without executing.")
    ev.add_argument("--seed", type=int, default=1234, help="Seed for bootstrap CI.")
    ev.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap iterations for CI.")
    ev.set_defaults(func=evaluate)

    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
