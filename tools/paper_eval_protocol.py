#!/usr/bin/env python3
"""Paper-ready evaluation protocol runner for neural morphing."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import soundfile as sf
from scipy.stats import rankdata, wilcoxon


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = REPO_ROOT / "tools" / "evaluate_morphing.py"
PARITY_SCRIPT = REPO_ROOT / "tools" / "compare_realtime_parity.py"

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif"}
DEFAULT_SEEDS = [1234, 2234, 3234]
DEFAULT_CHUNK_SIZES = [8192, 16384, 32768]

DEFAULT_RUNNER_CMD = (
    "{python_bin} tools/run_morph_ablation.py --codec {codec} --palette-manifest {palette_manifest} "
    "--source {source} --output-wav {output_wav} --tokens-npy {tokens_npy} --match-indices-npy {match_indices_npy} "
    "--latency-json {latency_json} --diagnostics-json {diagnostics_json} --matcher {matcher} --swap {swap} --temperature {temperature} --threshold {threshold} "
    "--continuity {continuity} --rvq-focus {rvq_focus} --unit {unit} --stride {stride} --top-k {top_k} --seed {seed}"
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

QUALITY_METRICS = ["fad", "spectral_convergence", "log_spectral_distance"]
STRUCTURE_METRICS = [
    "index_jitter",
    "token_discontinuity",
    "waveform_discontinuity_db",
    "chroma_difference",
    "envelope_correlation",
    "boundary_phase_jump",
    "source_onset_f1",
    "source_onset_deviation_ms",
    "transient_strength_correlation",
    "bandwise_envelope_correlation",
    "boundary_click_energy",
    "output_to_source_distance",
    "output_to_nearest_palette_distance",
    "palette_embedding_shift",
    "file_switch_rate",
    "adjacent_step_rate",
    "token_change_rate_coarse",
    "token_change_rate_middle",
    "token_change_rate_fine",
    "token_change_rate_overall",
]

HEALTH_METRICS_HIGHER_BETTER = [
    "encode_success_rate",
    "decode_success_rate",
    "token_layout_valid_rate",
    "channel_consistency_rate",
    "determinism_pass_rate",
    "envelope_correlation_mean",
]
HEALTH_METRICS_LOWER_BETTER = [
    "clipping_fraction_max",
    "duration_drift_abs_ms_p95",
    "failure_count_sum",
    "retry_count_sum",
]

LOWER_BETTER = {
    "fad": True,
    "spectral_convergence": True,
    "log_spectral_distance": True,
    "index_jitter": True,
    "token_discontinuity": True,
    "waveform_discontinuity_db": True,
    "chroma_difference": True,
    "envelope_correlation": False,
    "boundary_phase_jump": True,
    "source_onset_f1": False,
    "source_onset_deviation_ms": True,
    "transient_strength_correlation": False,
    "bandwise_envelope_correlation": False,
    "boundary_click_energy": True,
    "output_to_source_distance": True,
    "output_to_nearest_palette_distance": True,
    "palette_embedding_shift": False,
    "file_switch_rate": True,
    "adjacent_step_rate": False,
    "token_change_rate_coarse": False,
    "token_change_rate_middle": False,
    "token_change_rate_fine": False,
    "token_change_rate_overall": False,
}


@dataclass
class AggregateOptions:
    bootstrap: int = 2000
    seed: int = 1234
    duration_drift_gate_ms: float = 120.0
    determinism_gate_rate: float = 1.0
    envelope_corr_gate: float = 0.90
    clipping_gate_fraction: float = 1e-4


def _split_arg(raw: str, cast=int) -> List:
    items = [p.strip() for p in str(raw).split(",") if p.strip()]
    return [cast(x) for x in items]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _to_float(value, default=float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _mean(values: Iterable[float]) -> float:
    arr = [v for v in values if not math.isnan(v)]
    if not arr:
        return float("nan")
    return float(np.mean(arr))


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


def _summarize_values(values: List[float], bootstrap: int, seed: int) -> dict:
    arr = [float(v) for v in values if not math.isnan(float(v))]
    if not arr:
        return {"mean": float("nan"), "std": float("nan"), "ci95": [float("nan"), float("nan")], "n": 0}
    lo, hi = _bootstrap_ci(arr, alpha=0.05, n_boot=bootstrap, seed=seed)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "ci95": [lo, hi],
        "n": len(arr),
    }


def benjamini_hochberg(p_values: List[float]) -> List[float]:
    indexed = [(i, float(p)) for i, p in enumerate(p_values) if not math.isnan(float(p))]
    q_values = [float("nan")] * len(p_values)
    if not indexed:
        return q_values

    indexed.sort(key=lambda x: x[1])
    m = len(indexed)
    raw_q = [0.0] * m
    for rank, (_, p) in enumerate(indexed, start=1):
        raw_q[rank - 1] = min(1.0, (p * m) / rank)

    monotonic_q = [0.0] * m
    running = 1.0
    for idx in range(m - 1, -1, -1):
        running = min(running, raw_q[idx])
        monotonic_q[idx] = running

    for (orig_idx, _), q in zip(indexed, monotonic_q):
        q_values[orig_idx] = float(q)
    return q_values


def _run_command(cmd: List[str], log_path: Path | None = None) -> None:
    rendered = " ".join(shlex.quote(c) for c in cmd)
    completed = subprocess.run(cmd, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    payload = f"$ {rendered}\n\n[exit={completed.returncode}]\n"
    if completed.stdout:
        payload += f"\nSTDOUT\n{completed.stdout}\n"
    if completed.stderr:
        payload += f"\nSTDERR\n{completed.stderr}\n"

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(payload)

    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {rendered}\n{payload}")


def _shell_quote(value: str) -> str:
    if sys.platform.startswith("win"):
        return subprocess.list2cmdline([str(value)])
    return shlex.quote(str(value))


def _audio_files(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS])


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _scan_split(root: Path, split: str) -> List[dict]:
    rows: List[dict] = []
    for wav in _audio_files(root):
        try:
            info = sf.info(str(wav))
        except Exception:
            continue
        rel = wav.relative_to(root)
        category = rel.parts[0] if len(rel.parts) > 1 else "uncategorized"
        rows.append(
            {
                "split": split,
                "id": wav.stem,
                "path": str(wav.resolve()),
                "duration_s": float(info.duration),
                "sr": int(info.samplerate),
                "channels": int(info.channels),
                "category": category,
            }
        )
    return rows


def _summarize_split(rows: List[dict]) -> dict:
    durations = [float(r["duration_s"]) for r in rows]
    if not rows:
        return {
            "count": 0,
            "duration_total_s": 0.0,
            "duration_median_s": float("nan"),
            "duration_range_5_to_15s_rate": float("nan"),
            "sr_histogram": {},
            "channels_histogram": {},
            "categories": {},
        }
    sr_hist: Dict[str, int] = {}
    ch_hist: Dict[str, int] = {}
    cat_hist: Dict[str, int] = {}
    for r in rows:
        sr_hist[str(int(r["sr"]))] = sr_hist.get(str(int(r["sr"])), 0) + 1
        ch_hist[str(int(r["channels"]))] = ch_hist.get(str(int(r["channels"])), 0) + 1
        cat = str(r["category"])
        cat_hist[cat] = cat_hist.get(cat, 0) + 1
    in_range = [d for d in durations if 5.0 <= d <= 15.0]
    return {
        "count": len(rows),
        "duration_total_s": float(np.sum(durations)),
        "duration_median_s": float(np.median(durations)),
        "duration_range_5_to_15s_rate": float(len(in_range) / max(1, len(durations))),
        "sr_histogram": dict(sorted(sr_hist.items())),
        "channels_histogram": dict(sorted(ch_hist.items())),
        "categories": dict(sorted(cat_hist.items())),
    }


def _overlap_report(palette_rows: List[dict], source_rows: List[dict], ref_rows: List[dict]) -> dict:
    def _paths(rows: List[dict]) -> set[str]:
        return {str(Path(r["path"]).resolve()) for r in rows}

    palette_paths = _paths(palette_rows)
    source_paths = _paths(source_rows)
    ref_paths = _paths(ref_rows)

    overlap_path_palette_source = sorted(palette_paths & source_paths)
    overlap_path_palette_ref = sorted(palette_paths & ref_paths)

    palette_hashes = {}
    source_hashes = {}
    ref_hashes = {}
    for r in palette_rows:
        try:
            palette_hashes[_sha256_file(Path(r["path"]))] = r["path"]
        except Exception:
            continue
    for r in source_rows:
        try:
            source_hashes[_sha256_file(Path(r["path"]))] = r["path"]
        except Exception:
            continue
    for r in ref_rows:
        try:
            ref_hashes[_sha256_file(Path(r["path"]))] = r["path"]
        except Exception:
            continue

    overlap_hash_palette_source = []
    overlap_hash_palette_ref = []
    for h, p in palette_hashes.items():
        if h in source_hashes:
            overlap_hash_palette_source.append({"sha256": h, "palette": p, "source": source_hashes[h]})
        if h in ref_hashes:
            overlap_hash_palette_ref.append({"sha256": h, "palette": p, "reference": ref_hashes[h]})

    return {
        "path_overlap_palette_source": overlap_path_palette_source,
        "path_overlap_palette_reference": overlap_path_palette_ref,
        "hash_overlap_palette_source": overlap_hash_palette_source,
        "hash_overlap_palette_reference": overlap_hash_palette_ref,
        "ok": not (
            overlap_path_palette_source
            or overlap_path_palette_ref
            or overlap_hash_palette_source
            or overlap_hash_palette_ref
        ),
    }


def audit_data(args: argparse.Namespace) -> None:
    palette_dir = Path(args.palette_dir)
    source_dir = Path(args.source_dir)
    reference_dir = Path(args.reference_dir)
    out_json = Path(args.report_json)
    metadata_csv = Path(args.metadata_csv)

    palette_rows = _scan_split(palette_dir, "palette_train")
    source_rows = _scan_split(source_dir, "source_eval")
    ref_rows = _scan_split(reference_dir, "original_refs")
    all_rows = palette_rows + source_rows + ref_rows

    metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    with metadata_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "id", "path", "duration_s", "sr", "channels", "category"],
        )
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    overlap = _overlap_report(palette_rows, source_rows, ref_rows)
    summary = {
        "palette_train": _summarize_split(palette_rows),
        "source_eval": _summarize_split(source_rows),
        "original_refs": _summarize_split(ref_rows),
    }

    warnings: List[str] = []
    if summary["palette_train"]["count"] < 150:
        warnings.append("palette_train has fewer than 150 clips (recommended 150-400).")
    if summary["source_eval"]["count"] < 50:
        warnings.append("source_eval has fewer than 50 clips (recommended 50-150).")
    if summary["original_refs"]["count"] < 50:
        warnings.append("original_refs has fewer than 50 clips (recommended 50-150).")
    if not overlap["ok"]:
        warnings.append("Split leakage detected by path and/or hash overlap.")

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "palette_dir": str(palette_dir.resolve()),
            "source_dir": str(source_dir.resolve()),
            "reference_dir": str(reference_dir.resolve()),
        },
        "metadata_csv": str(metadata_csv.resolve()),
        "summary": summary,
        "overlap": overlap,
        "warnings": warnings,
        "ok": len(warnings) == 0,
    }
    _write_json(out_json, payload)
    print(f"Wrote data audit report: {out_json}")
    print(f"Wrote metadata CSV: {metadata_csv}")
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"- {w}")
        if args.strict:
            raise SystemExit(2)


def _codec_defaults(codec_id: str) -> dict:
    return dict(TUNED_CODEC_PARAMS.get((codec_id or "dac").strip().lower(), TUNED_CODEC_PARAMS["dac"]))


def _resolve_codec_params(codec_id: str, args: argparse.Namespace) -> dict:
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


def _read_csv_rows(path: Path) -> List[dict]:
    with path.open("r", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _extract_seed(run_dir: Path) -> int:
    repro = run_dir / "reports" / "reproducibility.json"
    if repro.exists():
        try:
            return int(json.loads(repro.read_text()).get("seed", 0))
        except Exception:
            return 0
    return 0


def _paired_significance(rows: List[dict], codec_id: str, metric_key: str, baseline_id: str = "greedy_full_layer") -> Dict[str, dict]:
    by_ablation: Dict[str, Dict[str, float]] = {}
    for row in rows:
        if row.get("codec_id") != codec_id:
            continue
        if row.get("status") != "ok":
            continue
        value = _to_float(row.get(metric_key, float("nan")))
        if math.isnan(value):
            continue
        sample_id = f"{row.get('_seed', 0)}::{row.get('clip_id', '')}"
        by_ablation.setdefault(str(row.get("ablation_id", "")), {})[sample_id] = value

    if baseline_id not in by_ablation:
        return {}

    baseline = by_ablation[baseline_id]
    out: Dict[str, dict] = {}
    pvals = []
    keys = []
    for ablation_id, mapping in sorted(by_ablation.items()):
        if ablation_id == baseline_id:
            continue
        common = sorted(set(baseline.keys()) & set(mapping.keys()))
        if len(common) < 3:
            out[ablation_id] = {
                "n": len(common),
                "paired_median_difference": float("nan"),
                "effect_size_rank_biserial": float("nan"),
                "p_value": float("nan"),
                "p_adj_bh": float("nan"),
                "significant_0p05": False,
            }
            continue
        a = np.asarray([baseline[k] for k in common], dtype=np.float64)
        b = np.asarray([mapping[k] for k in common], dtype=np.float64)
        diff = b - a
        nonzero = diff[np.abs(diff) > 1e-12]
        if nonzero.size == 0:
            p_val = float("nan")
        else:
            try:
                p_val = float(wilcoxon(a, b, zero_method="wilcox", correction=False, alternative="two-sided").pvalue)
            except Exception:
                p_val = float("nan")
        if nonzero.size:
            ranks = rankdata(np.abs(nonzero))
            total_rank = float(np.sum(ranks))
            rank_biserial = (
                float((np.sum(ranks[nonzero > 0.0]) - np.sum(ranks[nonzero < 0.0])) / total_rank)
                if total_rank > 0.0
                else float("nan")
            )
        else:
            rank_biserial = 0.0
        out[ablation_id] = {
            "n": len(common),
            "paired_median_difference": float(np.median(diff)),
            "effect_size_rank_biserial": rank_biserial,
            "p_value": p_val,
            "p_adj_bh": float("nan"),
            "significant_0p05": False,
        }
        pvals.append(p_val)
        keys.append(ablation_id)

    adj = benjamini_hochberg(pvals)
    for ablation_id, q_val in zip(keys, adj):
        out[ablation_id]["p_adj_bh"] = q_val
        out[ablation_id]["significant_0p05"] = bool(not math.isnan(q_val) and q_val < 0.05)
    return out


def _normalize_values(metric_name: str, values_by_condition: Dict[str, float], lower_better: bool) -> Dict[str, float]:
    vals = [v for v in values_by_condition.values() if not math.isnan(v)]
    if not vals:
        return {k: float("nan") for k in values_by_condition}
    lo, hi = min(vals), max(vals)
    if abs(hi - lo) < 1e-12:
        return {k: (1.0 if not math.isnan(v) else float("nan")) for k, v in values_by_condition.items()}

    out = {}
    for k, raw in values_by_condition.items():
        if math.isnan(raw):
            out[k] = float("nan")
        elif lower_better:
            out[k] = float((hi - raw) / (hi - lo))
        else:
            out[k] = float((raw - lo) / (hi - lo))
    del metric_name
    return out


def _condition_key(codec_id: str, ablation_id: str) -> str:
    return f"{codec_id}::{ablation_id}"


def aggregate_reports(run_dirs: List[Path], output_dir: Path, options: AggregateOptions) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []
    run_summaries: List[dict] = []
    leakage_ok_all = True
    fad_status_by_condition: Dict[str, set[str]] = {}

    for run_dir in run_dirs:
        csv_path = run_dir / "reports" / "per_clip_metrics.csv"
        summary_path = run_dir / "reports" / "summary.json"
        repro_path = run_dir / "reports" / "reproducibility.json"
        if not csv_path.exists():
            continue
        seed = _extract_seed(run_dir)
        for row in _read_csv_rows(csv_path):
            row["_seed"] = seed
            row["_run_dir"] = str(run_dir.resolve())
            rows.append(row)

        summary_payload = {}
        if summary_path.exists():
            try:
                summary_payload = json.loads(summary_path.read_text())
            except Exception:
                summary_payload = {}
        repro_payload = {}
        if repro_path.exists():
            try:
                repro_payload = json.loads(repro_path.read_text())
            except Exception:
                repro_payload = {}

        run_summaries.append(
            {
                "run_dir": str(run_dir.resolve()),
                "seed": seed,
                "summary_path": str(summary_path.resolve()) if summary_path.exists() else "",
                "reproducibility_path": str(repro_path.resolve()) if repro_path.exists() else "",
                "manifest": repro_payload.get("manifest", ""),
            }
        )
        leakage_ok = bool(summary_payload.get("system_health", {}).get("leakage", {}).get("ok", True))
        leakage_ok_all = leakage_ok_all and leakage_ok
        fad_status = summary_payload.get("system_health", {}).get("fad_status", {})
        for cond_key, status in fad_status.items():
            fad_status_by_condition.setdefault(str(cond_key), set()).add(str(status))

    ok_rows = [r for r in rows if str(r.get("status", "")) == "ok"]
    conditions = sorted(set((_condition_key(str(r.get("codec_id", "")), str(r.get("ablation_id", ""))) for r in ok_rows)))
    codecs = sorted(set(str(r.get("codec_id", "")) for r in ok_rows))
    ablations = sorted(set(str(r.get("ablation_id", "")) for r in ok_rows))

    quality_summary: Dict[str, Dict[str, dict]] = {}
    structure_summary: Dict[str, Dict[str, dict]] = {}
    latency_summary: Dict[str, Dict[str, dict]] = {}
    health_summary: Dict[str, dict] = {}

    for codec_id in codecs:
        quality_summary.setdefault(codec_id, {})
        structure_summary.setdefault(codec_id, {})
        latency_summary.setdefault(codec_id, {})
        for ablation_id in ablations:
            cond_rows = [r for r in ok_rows if str(r.get("codec_id")) == codec_id and str(r.get("ablation_id")) == ablation_id]
            if not cond_rows:
                continue
            condition_key = _condition_key(codec_id, ablation_id)
            quality_summary[codec_id][ablation_id] = {
                m: _summarize_values([_to_float(r.get(m, float("nan"))) for r in cond_rows], options.bootstrap, options.seed)
                for m in QUALITY_METRICS
            }
            structure_summary[codec_id][ablation_id] = {
                m: _summarize_values([_to_float(r.get(m, float("nan"))) for r in cond_rows], options.bootstrap, options.seed)
                for m in STRUCTURE_METRICS
            }

            enc = [_to_float(r.get("encode_ms", float("nan"))) for r in cond_rows if not math.isnan(_to_float(r.get("encode_ms", float("nan"))))]
            dec = [_to_float(r.get("decode_ms", float("nan"))) for r in cond_rows if not math.isnan(_to_float(r.get("decode_ms", float("nan"))))]
            rtf = [_to_float(r.get("end_to_end_rtf", float("nan"))) for r in cond_rows if not math.isnan(_to_float(r.get("end_to_end_rtf", float("nan"))))]
            latency_summary[codec_id][ablation_id] = {
                "encode_ms_p50": float(np.percentile(enc, 50)) if enc else float("nan"),
                "encode_ms_p95": float(np.percentile(enc, 95)) if enc else float("nan"),
                "decode_ms_p50": float(np.percentile(dec, 50)) if dec else float("nan"),
                "decode_ms_p95": float(np.percentile(dec, 95)) if dec else float("nan"),
                "rtf_mean": float(np.mean(rtf)) if rtf else float("nan"),
                "rtf_p95": float(np.percentile(rtf, 95)) if rtf else float("nan"),
                "n": len(cond_rows),
            }

            drift_vals = [abs(_to_float(r.get("duration_drift_ms", float("nan")))) for r in cond_rows if not math.isnan(_to_float(r.get("duration_drift_ms", float("nan"))))]
            clip_vals = [_to_float(r.get("clipping_fraction", float("nan"))) for r in cond_rows if not math.isnan(_to_float(r.get("clipping_fraction", float("nan"))))]
            health_summary[condition_key] = {
                "n_ok": len(cond_rows),
                "encode_success_rate": _mean(_to_float(r.get("encode_ok", float("nan"))) for r in cond_rows),
                "decode_success_rate": _mean(_to_float(r.get("decode_ok", float("nan"))) for r in cond_rows),
                "token_layout_valid_rate": _mean(_to_float(r.get("token_layout_valid", float("nan"))) for r in cond_rows),
                "channel_consistency_rate": _mean(_to_float(r.get("channel_consistency_ok", float("nan"))) for r in cond_rows),
                "determinism_pass_rate": _mean(_to_float(r.get("determinism_pass", float("nan"))) for r in cond_rows),
                "envelope_correlation_mean": _mean(_to_float(r.get("envelope_correlation", float("nan"))) for r in cond_rows),
                "clipping_fraction_max": float(np.max(clip_vals)) if clip_vals else float("nan"),
                "duration_drift_abs_ms_p95": float(np.percentile(drift_vals, 95)) if drift_vals else float("nan"),
                "failure_count_sum": float(np.sum([_to_float(r.get("failure_count", 0.0), default=0.0) for r in cond_rows])),
                "retry_count_sum": float(np.sum([_to_float(r.get("retry_count", 0.0), default=0.0) for r in cond_rows])),
            }

    gates = {}
    for condition_key, h in health_summary.items():
        gate_pass = (
            bool(leakage_ok_all)
            and _to_float(h.get("encode_success_rate", float("nan"))) >= 1.0
            and _to_float(h.get("decode_success_rate", float("nan"))) >= 1.0
            and _to_float(h.get("token_layout_valid_rate", float("nan"))) >= 1.0
            and (
                math.isnan(_to_float(h.get("duration_drift_abs_ms_p95", float("nan"))))
                or _to_float(h.get("duration_drift_abs_ms_p95", float("nan"))) <= float(options.duration_drift_gate_ms)
            )
            and (
                math.isnan(_to_float(h.get("determinism_pass_rate", float("nan"))))
                or _to_float(h.get("determinism_pass_rate", float("nan"))) >= float(options.determinism_gate_rate)
            )
            and (
                math.isnan(_to_float(h.get("envelope_correlation_mean", float("nan"))))
                or _to_float(h.get("envelope_correlation_mean", float("nan"))) >= float(options.envelope_corr_gate)
            )
            and (
                math.isnan(_to_float(h.get("clipping_fraction_max", float("nan"))))
                or _to_float(h.get("clipping_fraction_max", float("nan"))) <= float(options.clipping_gate_fraction)
            )
        )
        gates[condition_key] = {"pass": bool(gate_pass)}

    significance: Dict[str, Dict[str, dict]] = {}
    for codec_id in codecs:
        significance[codec_id] = {}
        for metric in QUALITY_METRICS + STRUCTURE_METRICS:
            significance[codec_id][metric] = _paired_significance(ok_rows, codec_id=codec_id, metric_key=metric, baseline_id="greedy_full_layer")

    score_rows = []
    for condition_key in conditions:
        codec_id, ablation_id = condition_key.split("::", 1)
        quality = {m: _to_float(quality_summary.get(codec_id, {}).get(ablation_id, {}).get(m, {}).get("mean", float("nan"))) for m in QUALITY_METRICS}
        structure = {m: _to_float(structure_summary.get(codec_id, {}).get(ablation_id, {}).get(m, {}).get("mean", float("nan"))) for m in STRUCTURE_METRICS}
        score_rows.append(
            {
                "condition_key": condition_key,
                "codec_id": codec_id,
                "ablation_id": ablation_id,
                "quality": quality,
                "structure": structure,
                "health": dict(health_summary.get(condition_key, {})),
                "gate_pass": bool(gates.get(condition_key, {}).get("pass", False)),
            }
        )

    metric_values: Dict[str, Dict[str, float]] = {}
    for metric in QUALITY_METRICS + STRUCTURE_METRICS:
        metric_values[metric] = {row["condition_key"]: row["quality"].get(metric, float("nan")) if metric in QUALITY_METRICS else row["structure"].get(metric, float("nan")) for row in score_rows}

    for metric in HEALTH_METRICS_HIGHER_BETTER + HEALTH_METRICS_LOWER_BETTER:
        metric_values[metric] = {row["condition_key"]: _to_float(row["health"].get(metric, float("nan"))) for row in score_rows}

    norm_values: Dict[str, Dict[str, float]] = {}
    for metric in QUALITY_METRICS + STRUCTURE_METRICS:
        norm_values[metric] = _normalize_values(metric, metric_values[metric], lower_better=LOWER_BETTER[metric])
    for metric in HEALTH_METRICS_HIGHER_BETTER:
        norm_values[metric] = _normalize_values(metric, metric_values[metric], lower_better=False)
    for metric in HEALTH_METRICS_LOWER_BETTER:
        norm_values[metric] = _normalize_values(metric, metric_values[metric], lower_better=True)

    ranked = []
    for row in score_rows:
        key = row["condition_key"]
        quality_score = _mean(norm_values[m].get(key, float("nan")) for m in QUALITY_METRICS)
        structure_score = _mean(norm_values[m].get(key, float("nan")) for m in STRUCTURE_METRICS)
        health_score = _mean(
            [norm_values[m].get(key, float("nan")) for m in HEALTH_METRICS_HIGHER_BETTER + HEALTH_METRICS_LOWER_BETTER]
        )
        overall_score = _mean([quality_score, structure_score, health_score])
        ranked.append(
            {
                "condition_key": key,
                "codec_id": row["codec_id"],
                "ablation_id": row["ablation_id"],
                "gate_pass": row["gate_pass"],
                "quality_score": quality_score,
                "structure_score": structure_score,
                "health_score": health_score,
                "overall_score": overall_score,
            }
        )
    ranked.sort(key=lambda r: (0 if r["gate_pass"] else 1, -(r["overall_score"] if not math.isnan(r["overall_score"]) else -1e9)))

    combined_csv = output_dir / "combined_per_clip_metrics.csv"
    if rows:
        keys = sorted(set().union(*[set(r.keys()) for r in rows]))
        with combined_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    system_health_rows = []
    for cond_key, h in sorted(health_summary.items()):
        codec_id, ablation_id = cond_key.split("::", 1)
        system_health_rows.append(
            {
                "condition_key": cond_key,
                "codec_id": codec_id,
                "ablation_id": ablation_id,
                "gate_pass": bool(gates.get(cond_key, {}).get("pass", False)),
                **h,
            }
        )

    table_system_health_csv = output_dir / "table_system_health.csv"
    if system_health_rows:
        with table_system_health_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(system_health_rows[0].keys()))
            writer.writeheader()
            writer.writerows(system_health_rows)

    quality_structure_rows = []
    for codec_id, ab_map in quality_summary.items():
        for ablation_id, metrics in ab_map.items():
            for metric, stats in metrics.items():
                quality_structure_rows.append(
                    {
                        "condition_key": _condition_key(codec_id, ablation_id),
                        "codec_id": codec_id,
                        "ablation_id": ablation_id,
                        "panel": "quality",
                        "metric": metric,
                        "mean": stats["mean"],
                        "median": stats.get("median", float("nan")),
                        "std": stats["std"],
                        "ci95_lo": stats["ci95"][0],
                        "ci95_hi": stats["ci95"][1],
                        "n": stats["n"],
                    }
                )
    for codec_id, ab_map in structure_summary.items():
        for ablation_id, metrics in ab_map.items():
            for metric, stats in metrics.items():
                quality_structure_rows.append(
                    {
                        "condition_key": _condition_key(codec_id, ablation_id),
                        "codec_id": codec_id,
                        "ablation_id": ablation_id,
                        "panel": "structure",
                        "metric": metric,
                        "mean": stats["mean"],
                        "median": stats.get("median", float("nan")),
                        "std": stats["std"],
                        "ci95_lo": stats["ci95"][0],
                        "ci95_hi": stats["ci95"][1],
                        "n": stats["n"],
                    }
                )

    table_quality_csv = output_dir / "table_quality_structure.csv"
    if quality_structure_rows:
        with table_quality_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(quality_structure_rows[0].keys()))
            writer.writeheader()
            writer.writerows(quality_structure_rows)

    significance_rows = []
    for codec_id, metric_map in significance.items():
        for metric, ab_map in metric_map.items():
            for ablation_id, stats in ab_map.items():
                significance_rows.append(
                    {
                        "codec_id": codec_id,
                        "metric": metric,
                        "baseline_id": "greedy_full_layer",
                        "ablation_id": ablation_id,
                        "n": stats.get("n", 0),
                        "p_value": stats.get("p_value", float("nan")),
                        "p_adj_bh": stats.get("p_adj_bh", float("nan")),
                        "paired_median_difference": stats.get("paired_median_difference", float("nan")),
                        "effect_size_rank_biserial": stats.get("effect_size_rank_biserial", float("nan")),
                        "significant_0p05": stats.get("significant_0p05", False),
                    }
                )
    table_significance_csv = output_dir / "table_significance.csv"
    if significance_rows:
        with table_significance_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(significance_rows[0].keys()))
            writer.writeheader()
            writer.writerows(significance_rows)

    figure_radar_rows = ranked
    figure_radar_csv = output_dir / "figure_radar_scores.csv"
    if figure_radar_rows:
        with figure_radar_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(figure_radar_rows[0].keys()))
            writer.writeheader()
            writer.writerows(figure_radar_rows)

    figure_pareto_rows = []
    for row in ranked:
        codec_id = row["codec_id"]
        ablation_id = row["ablation_id"]
        lat = latency_summary.get(codec_id, {}).get(ablation_id, {})
        figure_pareto_rows.append({**row, **lat})
    figure_pareto_csv = output_dir / "figure_latency_quality_pareto.csv"
    if figure_pareto_rows:
        with figure_pareto_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(figure_pareto_rows[0].keys()))
            writer.writeheader()
            writer.writerows(figure_pareto_rows)

    fad_status_compact = {k: sorted(v) for k, v in fad_status_by_condition.items()}

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runs": run_summaries,
        "dataset": {
            "rows_total": len(rows),
            "rows_ok": len(ok_rows),
            "conditions": conditions,
            "codecs": codecs,
            "ablations": ablations,
            "unique_seed_clip_pairs": len(set(f"{r.get('_seed', 0)}::{r.get('clip_id', '')}" for r in ok_rows)),
        },
        "quality_metrics": quality_summary,
        "structure_metrics": structure_summary,
        "latency": latency_summary,
        "system_health": {
            "leakage_ok": bool(leakage_ok_all),
            "conditions": health_summary,
            "gates": gates,
            "gate_thresholds": {
                "duration_drift_gate_ms": options.duration_drift_gate_ms,
                "determinism_gate_rate": options.determinism_gate_rate,
                "envelope_corr_gate": options.envelope_corr_gate,
                "clipping_gate_fraction": options.clipping_gate_fraction,
            },
            "fad_status": fad_status_compact,
        },
        "significance": significance,
        "ranked_conditions": ranked,
        "artifacts": {
            "combined_per_clip_metrics_csv": str(combined_csv.resolve()) if combined_csv.exists() else "",
            "table_system_health_csv": str(table_system_health_csv.resolve()) if table_system_health_csv.exists() else "",
            "table_quality_structure_csv": str(table_quality_csv.resolve()) if table_quality_csv.exists() else "",
            "table_significance_csv": str(table_significance_csv.resolve()) if table_significance_csv.exists() else "",
            "figure_radar_scores_csv": str(figure_radar_csv.resolve()) if figure_radar_csv.exists() else "",
            "figure_latency_quality_pareto_csv": str(figure_pareto_csv.resolve()) if figure_pareto_csv.exists() else "",
        },
    }

    _write_json(output_dir / "aggregated_summary.json", summary)
    _write_json(
        output_dir / "appendix_package.json",
        {
            "generated_at_utc": summary["generated_at_utc"],
            "git_commit": _git_commit(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "run_dirs": [str(p.resolve()) for p in run_dirs],
            "summary_json": str((output_dir / "aggregated_summary.json").resolve()),
            "artifact_index": summary["artifacts"],
        },
    )
    (output_dir / "paper_report.md").write_text(_render_markdown_report(summary))
    return summary


def _git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True)
        return out.strip()
    except Exception:
        return ""


def _render_markdown_report(summary: dict) -> str:
    rows = summary.get("ranked_conditions", [])
    gates = summary.get("system_health", {}).get("gates", {})
    pass_count = sum(1 for g in gates.values() if bool(g.get("pass", False)))
    total_count = len(gates)
    fad_status = summary.get("system_health", {}).get("fad_status", {})
    fad_missing = [k for k, statuses in fad_status.items() if "missing_dependency" in statuses]
    lines = []
    lines.append("# Paper-Ready Evaluation Summary")
    lines.append("")
    lines.append(f"- Generated: `{summary.get('generated_at_utc', '')}`")
    lines.append(f"- Rows OK: `{summary.get('dataset', {}).get('rows_ok', 0)}`")
    lines.append(f"- Conditions: `{len(summary.get('dataset', {}).get('conditions', []))}`")
    lines.append(f"- Health gates pass: `{pass_count}/{total_count}`")
    lines.append("")
    if fad_missing:
        lines.append("## FAD Status")
        lines.append("")
        lines.append("- FAD is missing for one or more conditions (`missing_dependency`). Install `frechet_audio_distance` for publishable quality tables.")
        lines.append("")
    lines.append("## Top Ranked Conditions")
    lines.append("")
    lines.append("| rank | condition | gate_pass | quality | structure | health | overall |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: |")
    for idx, row in enumerate(rows[:10], start=1):
        lines.append(
            f"| {idx} | `{row.get('condition_key','')}` | `{row.get('gate_pass', False)}` | "
            f"{_fmt(row.get('quality_score'))} | {_fmt(row.get('structure_score'))} | "
            f"{_fmt(row.get('health_score'))} | {_fmt(row.get('overall_score'))} |"
        )
    lines.append("")
    lines.append("## Artifact Index")
    lines.append("")
    for k, v in sorted(summary.get("artifacts", {}).items()):
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Significance table uses paired Wilcoxon vs `greedy_full_layer` with Benjamini-Hochberg correction.")
    lines.append("- Use the generated CSVs directly for paper tables and figure plots.")
    return "\n".join(lines)


def _fmt(x) -> str:
    v = _to_float(x, default=float("nan"))
    if math.isnan(v):
        return "nan"
    return f"{v:.4f}"


def run_protocol(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    python_bin = args.python_bin
    runner_cmd = str(args.runner_cmd).replace("{python_bin}", _shell_quote(str(python_bin)))
    seeds = _split_arg(args.seeds, cast=int) if args.seeds else list(DEFAULT_SEEDS)

    if args.manifest:
        manifest_path = Path(args.manifest)
    else:
        manifest_path = Path(args.manifest_out) if args.manifest_out else (output_root / "paper_manifest.json")

    if not args.skip_data_audit and args.palette_dir and args.source_dir and args.reference_dir:
        audit_ns = argparse.Namespace(
            palette_dir=args.palette_dir,
            source_dir=args.source_dir,
            reference_dir=args.reference_dir,
            report_json=str(output_root / "data_audit.json"),
            metadata_csv=str(output_root / "clip_metadata.csv"),
            strict=False,
        )
        audit_data(audit_ns)

    if not args.skip_prepare:
        if not (args.palette_dir and args.source_dir and args.reference_dir):
            raise ValueError("Provide --palette-dir, --source-dir, --reference-dir unless --skip-prepare is set with --manifest.")
        prepare_cmd = [
            python_bin,
            str(EVAL_SCRIPT),
            "prepare",
            "--palette-dir",
            args.palette_dir,
            "--source-dir",
            args.source_dir,
            "--reference-dir",
            args.reference_dir,
            "--output",
            str(manifest_path),
            "--seed",
            str(int(seeds[0] if seeds else 1234)),
        ]
        _run_command(prepare_cmd, log_path=output_root / "prepare.log")

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    run_dirs = []
    for seed in seeds:
        run_dir = output_root / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_dirs.append(run_dir)

        evaluate_cmd = [
            python_bin,
            str(EVAL_SCRIPT),
            "evaluate",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(run_dir),
            "--runner-cmd",
            runner_cmd,
            "--codecs",
            args.codecs,
            "--ablations",
            args.ablations,
            "--determinism-runs",
            str(int(args.determinism_runs)),
            "--bootstrap",
            str(int(args.bootstrap)),
            "--seed",
            str(int(seed)),
            "--duration-drift-gate-ms",
            str(float(args.duration_drift_gate_ms)),
            "--determinism-gate-rate",
            str(float(args.determinism_gate_rate)),
            "--envelope-corr-gate",
            str(float(args.envelope_corr_gate)),
            "--clipping-gate-fraction",
            str(float(args.clipping_gate_fraction)),
            "--top-n-presets",
            str(int(args.top_n_presets)),
            "--palette-metric-limit",
            str(int(args.palette_metric_limit)),
        ]
        if int(args.clip_limit) > 0:
            evaluate_cmd.extend(["--clip-limit", str(int(args.clip_limit))])
        if args.resume_existing:
            evaluate_cmd.append("--resume-existing")
        for flag, value in (
            ("--temperature", args.temperature),
            ("--threshold", args.threshold),
            ("--continuity", args.continuity),
            ("--rvq-focus", args.rvq_focus),
            ("--unit", args.unit),
            ("--stride", args.stride),
            ("--top-k", args.top_k),
        ):
            if value is not None:
                evaluate_cmd.extend([flag, str(value)])
        _run_command(evaluate_cmd, log_path=run_dir / "evaluate.log")

        validate_cmd = [
            python_bin,
            str(EVAL_SCRIPT),
            "validate",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(run_dir),
            "--report-json",
            str(run_dir / "reports" / "validation.json"),
        ]
        _run_command(validate_cmd, log_path=run_dir / "validate.log")

        export_cmd = [
            python_bin,
            str(EVAL_SCRIPT),
            "export-presets",
            "--report-dir",
            str(run_dir / "reports"),
            "--output-json",
            str(run_dir / "reports" / "presets.json"),
            "--top-n",
            str(int(args.top_n_presets)),
        ]
        _run_command(export_cmd, log_path=run_dir / "export_presets.log")

    if not args.no_aggregate:
        summary = aggregate_reports(
            run_dirs=run_dirs,
            output_dir=output_root / "paper_reports",
            options=AggregateOptions(
                bootstrap=int(args.bootstrap),
                seed=int(seeds[0] if seeds else 1234),
                duration_drift_gate_ms=float(args.duration_drift_gate_ms),
                determinism_gate_rate=float(args.determinism_gate_rate),
                envelope_corr_gate=float(args.envelope_corr_gate),
                clipping_gate_fraction=float(args.clipping_gate_fraction),
            ),
        )
        print(f"Wrote aggregated report: {output_root / 'paper_reports' / 'aggregated_summary.json'}")
        print(f"Top condition: {summary.get('ranked_conditions', [{}])[0].get('condition_key', 'n/a') if summary.get('ranked_conditions') else 'n/a'}")


def aggregate_command(args: argparse.Namespace) -> None:
    run_dirs = [Path(p) for p in args.run_dirs]
    summary = aggregate_reports(
        run_dirs=run_dirs,
        output_dir=Path(args.output_dir),
        options=AggregateOptions(
            bootstrap=int(args.bootstrap),
            seed=int(args.seed),
            duration_drift_gate_ms=float(args.duration_drift_gate_ms),
            determinism_gate_rate=float(args.determinism_gate_rate),
            envelope_corr_gate=float(args.envelope_corr_gate),
            clipping_gate_fraction=float(args.clipping_gate_fraction),
        ),
    )
    print(f"Wrote aggregated report: {Path(args.output_dir) / 'aggregated_summary.json'}")
    print(f"Conditions: {len(summary.get('dataset', {}).get('conditions', []))}")


def parity_sweep(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    python_bin = args.python_bin
    chunk_sizes = _split_arg(args.chunk_sizes, cast=int) if args.chunk_sizes else list(DEFAULT_CHUNK_SIZES)

    sources: List[Path] = []
    if args.manifest:
        data = json.loads(Path(args.manifest).read_text())
        for row in data.get("source_eval", []):
            sources.append(Path(str(row.get("source", ""))))
    elif args.source_list:
        for line in Path(args.source_list).read_text().splitlines():
            line = line.strip()
            if line:
                sources.append(Path(line))
    elif args.source:
        sources = [Path(args.source)]
    else:
        raise ValueError("Provide one of --manifest, --source-list, or --source.")

    if int(args.max_clips) > 0:
        sources = sources[: int(args.max_clips)]
    if not sources:
        raise ValueError("No sources found for parity sweep.")

    defaults = _resolve_codec_params(args.codec, args)
    per_clip_records = []
    for chunk in chunk_sizes:
        for src in sources:
            clip_id = src.stem
            clip_out = output_dir / f"chunk_{int(chunk)}" / clip_id
            clip_out.mkdir(parents=True, exist_ok=True)

            cmd = [
                python_bin,
                str(PARITY_SCRIPT),
                "--palette-manifest",
                args.palette_manifest,
                "--source",
                str(src),
                "--output-dir",
                str(clip_out),
                "--codec",
                args.codec,
                "--matcher",
                args.matcher,
                "--swap",
                args.swap,
                "--temperature",
                str(defaults["temperature"]),
                "--threshold",
                str(defaults["threshold"]),
                "--continuity",
                str(defaults["continuity"]),
                "--rvq-focus",
                str(defaults["rvq_focus"]),
                "--unit",
                str(defaults["unit"]),
                "--stride",
                str(defaults["stride"]),
                "--top-k",
                str(defaults["top_k"]),
                "--chunk-samples",
                str(int(chunk)),
            ]
            if args.palette_cache_dir:
                cmd.extend(["--palette-cache-dir", args.palette_cache_dir])
            if args.plugin_render_dir:
                plugin_path = Path(args.plugin_render_dir) / f"{clip_id}.wav"
                if plugin_path.exists():
                    cmd.extend(["--plugin-render-wav", str(plugin_path)])
            _run_command(cmd, log_path=clip_out / "parity.log")

            summary_path = clip_out / "parity_summary.json"
            if not summary_path.exists():
                continue
            payload = json.loads(summary_path.read_text())
            metrics = payload.get("metrics", {}).get("proxy_vs_full", {})
            record = {"chunk_samples": int(chunk), "clip_id": clip_id}
            record.update({k: _to_float(v, default=float("nan")) for k, v in metrics.items() if k != "pair"})
            per_clip_records.append(record)

    if per_clip_records:
        clip_csv = output_dir / "parity_per_clip.csv"
        with clip_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_clip_records[0].keys()))
            writer.writeheader()
            writer.writerows(per_clip_records)

    metrics = [
        "spectral_convergence",
        "log_spectral_distance",
        "envelope_correlation",
        "chroma_difference",
        "waveform_discontinuity_db",
        "boundary_phase_jump",
    ]
    summary_rows = []
    for chunk in chunk_sizes:
        chunk_rows = [r for r in per_clip_records if int(r.get("chunk_samples", 0)) == int(chunk)]
        for metric in metrics:
            vals = [_to_float(r.get(metric, float("nan"))) for r in chunk_rows if not math.isnan(_to_float(r.get(metric, float("nan"))))]
            if not vals:
                mean = float("nan")
                std = float("nan")
                n = 0
            else:
                mean = float(np.mean(vals))
                std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                n = len(vals)
            summary_rows.append(
                {
                    "chunk_samples": int(chunk),
                    "metric": metric,
                    "mean": mean,
                    "std": std,
                    "n": n,
                }
            )

    curve_csv = output_dir / "parity_curve.csv"
    if summary_rows:
        with curve_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    _write_json(
        output_dir / "parity_sweep_summary.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "codec": args.codec,
            "matcher": args.matcher,
            "swap": args.swap,
            "chunk_sizes": [int(c) for c in chunk_sizes],
            "sources_count": len(sources),
            "defaults_used": defaults,
            "per_clip_csv": str((output_dir / "parity_per_clip.csv").resolve()) if (output_dir / "parity_per_clip.csv").exists() else "",
            "parity_curve_csv": str(curve_csv.resolve()) if curve_csv.exists() else "",
            "summary_rows": summary_rows,
        },
    )
    print(f"Wrote parity sweep report: {output_dir / 'parity_sweep_summary.json'}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Paper-ready evaluation protocol utilities.")
    sub = p.add_subparsers(dest="command", required=True)

    audit_p = sub.add_parser("audit-data", help="Audit dataset splits and emit metadata/overlap report.")
    audit_p.add_argument("--palette-dir", required=True)
    audit_p.add_argument("--source-dir", required=True)
    audit_p.add_argument("--reference-dir", required=True)
    audit_p.add_argument("--report-json", required=True)
    audit_p.add_argument("--metadata-csv", required=True)
    audit_p.add_argument("--strict", action="store_true", help="Exit non-zero when warnings are present.")
    audit_p.set_defaults(func=audit_data)

    run_p = sub.add_parser("run", help="Run the full paper-eval protocol (prepare/evaluate/validate/export/aggregate).")
    run_p.add_argument("--palette-dir", default="")
    run_p.add_argument("--source-dir", default="")
    run_p.add_argument("--reference-dir", default="")
    run_p.add_argument("--manifest", default="", help="Use existing manifest instead of preparing one.")
    run_p.add_argument("--manifest-out", default="", help="Output manifest path when preparing.")
    run_p.add_argument("--output-root", required=True)
    run_p.add_argument("--python-bin", default=sys.executable)
    run_p.add_argument("--runner-cmd", default=DEFAULT_RUNNER_CMD)
    run_p.add_argument("--codecs", default="dac,spectrostream")
    run_p.add_argument("--ablations", default="greedy_full_layer,greedy_rvq_group,beam_full_layer,beam_rvq_group")
    run_p.add_argument("--seeds", default="1234,2234,3234")
    run_p.add_argument("--determinism-runs", type=int, default=2)
    run_p.add_argument("--resume-existing", action="store_true", help="Reuse complete per-clip outputs when rerunning a partial evaluation.")
    run_p.add_argument("--bootstrap", type=int, default=2000)
    run_p.add_argument("--clip-limit", type=int, default=0)
    run_p.add_argument("--top-n-presets", type=int, default=8)
    run_p.add_argument("--palette-metric-limit", type=int, default=64)
    run_p.add_argument("--duration-drift-gate-ms", type=float, default=120.0)
    run_p.add_argument("--determinism-gate-rate", type=float, default=1.0)
    run_p.add_argument("--envelope-corr-gate", type=float, default=0.90)
    run_p.add_argument("--clipping-gate-fraction", type=float, default=1e-4)
    run_p.add_argument("--temperature", type=float, default=None)
    run_p.add_argument("--threshold", type=float, default=None)
    run_p.add_argument("--continuity", type=float, default=None)
    run_p.add_argument("--rvq-focus", dest="rvq_focus", type=float, default=None)
    run_p.add_argument("--unit", type=int, default=None)
    run_p.add_argument("--stride", type=int, default=None)
    run_p.add_argument("--top-k", dest="top_k", type=int, default=None)
    run_p.add_argument("--skip-prepare", action="store_true")
    run_p.add_argument("--skip-data-audit", action="store_true")
    run_p.add_argument("--no-aggregate", action="store_true", help="Skip final cross-seed aggregation.")
    run_p.set_defaults(func=run_protocol)

    agg_p = sub.add_parser("aggregate", help="Aggregate multiple evaluate runs into paper tables/figures.")
    agg_p.add_argument("--run-dirs", nargs="+", required=True, help="List of evaluate run directories.")
    agg_p.add_argument("--output-dir", required=True)
    agg_p.add_argument("--bootstrap", type=int, default=2000)
    agg_p.add_argument("--seed", type=int, default=1234)
    agg_p.add_argument("--duration-drift-gate-ms", type=float, default=120.0)
    agg_p.add_argument("--determinism-gate-rate", type=float, default=1.0)
    agg_p.add_argument("--envelope-corr-gate", type=float, default=0.90)
    agg_p.add_argument("--clipping-gate-fraction", type=float, default=1e-4)
    agg_p.set_defaults(func=aggregate_command)

    parity_p = sub.add_parser("parity-sweep", help="Run realtime parity sweep across chunk sizes.")
    parity_p.add_argument("--palette-manifest", required=True)
    parity_p.add_argument("--manifest", default="", help="Evaluation manifest JSON (uses source_eval paths).")
    parity_p.add_argument("--source-list", default="", help="Text file with one source path per line.")
    parity_p.add_argument("--source", default="", help="Single source path.")
    parity_p.add_argument("--output-dir", required=True)
    parity_p.add_argument("--python-bin", default=sys.executable)
    parity_p.add_argument("--codec", choices=["dac", "spectrostream"], default="dac")
    parity_p.add_argument("--matcher", choices=["beam", "greedy"], default="beam")
    parity_p.add_argument("--swap", choices=["full_layer", "rvq_group"], default="full_layer")
    parity_p.add_argument("--chunk-sizes", default="8192,16384,32768")
    parity_p.add_argument("--max-clips", type=int, default=0)
    parity_p.add_argument("--plugin-render-dir", default="", help="Optional directory with {clip_id}.wav plugin renders.")
    parity_p.add_argument("--palette-cache-dir", default="", help="Optional directory for encoded palette cache reuse.")
    parity_p.add_argument("--temperature", type=float, default=None)
    parity_p.add_argument("--threshold", type=float, default=None)
    parity_p.add_argument("--continuity", type=float, default=None)
    parity_p.add_argument("--rvq-focus", dest="rvq_focus", type=float, default=None)
    parity_p.add_argument("--unit", type=int, default=None)
    parity_p.add_argument("--stride", type=int, default=None)
    parity_p.add_argument("--top-k", dest="top_k", type=int, default=None)
    parity_p.set_defaults(func=parity_sweep)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
