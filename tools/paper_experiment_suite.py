#!/usr/bin/env python3
"""Paper artifact builders for the Neural Morphing evaluation package.

This module consumes completed `tools/evaluate_morphing.py evaluate` run
directories and writes the named CSV/TEX/PDF artifacts used in the manuscript.
It deliberately separates analysis from rendering: expensive DAC/SpectroStream
runs happen in `evaluate_morphing.py`; this script turns those results into
paper tables, diagnostic plots, sweeps, and listening manifests.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import rankdata, wilcoxon


DEFAULT_MAIN_METRICS = [
    "spectral_convergence",
    "log_spectral_distance",
    "envelope_correlation",
    "index_jitter",
    "source_onset_f1",
    "output_to_nearest_palette_distance",
    "palette_embedding_shift",
]

SEQUENCE_METRICS = [
    "objective_j",
    "emission_cost",
    "transition_cost",
    "weighted_transition_cost",
    "index_jitter",
    "file_switch_rate",
    "adjacent_step_rate",
    "sequence_runtime_ms",
]

SOURCE_STRUCTURE_METRICS = [
    "envelope_correlation",
    "source_onset_f1",
    "transient_strength_correlation",
    "bandwise_envelope_correlation",
]

PALETTE_TRANSFER_METRICS = [
    "palette_embedding_shift",
    "token_change_rate_middle",
    "token_change_rate_fine",
]


def _to_float(value, default=float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted(set().union(*(r.keys() for r in rows))) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_rows(run_dirs: Iterable[str]) -> list[dict]:
    rows: list[dict] = []
    for raw in run_dirs:
        run_dir = Path(raw)
        csv_path = run_dir / "reports" / "per_clip_metrics.csv"
        repro_path = run_dir / "reports" / "reproducibility.json"
        seed = 0
        palette_count = float("nan")
        if repro_path.exists():
            try:
                repro = json.loads(repro_path.read_text())
                seed = int(repro.get("seed", 0))
                palette_count = _to_float(repro.get("palette_count", float("nan")))
            except Exception:
                pass
        for row in _read_csv(csv_path):
            row["_run_dir"] = str(run_dir.resolve())
            row["_seed"] = seed
            row["palette_count"] = palette_count
            cfg_path = Path(row.get("config_json", ""))
            if cfg_path.exists():
                try:
                    cfg = json.loads(cfg_path.read_text())
                    params = cfg.get("params", {})
                    for key in ("temperature", "threshold", "continuity", "rvq_focus", "unit", "stride", "top_k"):
                        row[key] = params.get(key, row.get(key, ""))
                except Exception:
                    pass
            rows.append(row)
    return rows


def _bootstrap_ci(values: list[float], n_boot: int, seed: int) -> tuple[float, float]:
    arr = np.asarray([v for v in values if not math.isnan(v)], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        means[i] = float(np.mean(rng.choice(arr, size=arr.size, replace=True)))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _bh_adjust(pvals: list[float]) -> list[float]:
    indexed = [(i, p) for i, p in enumerate(pvals) if not math.isnan(p)]
    q = [float("nan")] * len(pvals)
    if not indexed:
        return q
    indexed.sort(key=lambda x: x[1])
    m = len(indexed)
    running = 1.0
    for rank_from_end, (idx, p) in enumerate(reversed(indexed), start=1):
        rank = m - rank_from_end + 1
        running = min(running, min(1.0, p * m / rank))
        q[idx] = float(running)
    return q


def _effect_rank_biserial(diff: np.ndarray) -> float:
    nonzero = diff[np.abs(diff) > 1e-12]
    if nonzero.size == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero))
    total = float(np.sum(ranks))
    if total <= 0.0:
        return float("nan")
    return float((np.sum(ranks[nonzero > 0.0]) - np.sum(ranks[nonzero < 0.0])) / total)


def _paired_stats(rows: list[dict], metric: str, condition_key: str, baseline_key: str) -> dict:
    by_condition: dict[str, dict[str, float]] = {}
    for row in rows:
        if str(row.get("status", "")) != "ok":
            continue
        key = f"{row.get('codec_id')}::{row.get('ablation_id')}"
        value = _to_float(row.get(metric, float("nan")))
        if math.isnan(value):
            continue
        sample_id = f"{row.get('_seed', 0)}::{row.get('clip_id', '')}"
        by_condition.setdefault(key, {})[sample_id] = value
    base = by_condition.get(baseline_key, {})
    comp = by_condition.get(condition_key, {})
    common = sorted(set(base) & set(comp))
    if len(common) < 3:
        return {
            "paired_n": len(common),
            "paired_median_difference": float("nan"),
            "wilcoxon_p": float("nan"),
            "effect_size_rank_biserial": float("nan"),
        }
    a = np.asarray([base[k] for k in common], dtype=np.float64)
    b = np.asarray([comp[k] for k in common], dtype=np.float64)
    diff = b - a
    try:
        p_value = float(wilcoxon(a, b, zero_method="wilcox", correction=False, alternative="two-sided").pvalue)
    except Exception:
        p_value = float("nan")
    return {
        "paired_n": len(common),
        "paired_median_difference": float(np.median(diff)),
        "wilcoxon_p": p_value,
        "effect_size_rank_biserial": _effect_rank_biserial(diff),
    }


def _summarize_conditions(
    rows: list[dict],
    metrics: list[str],
    codec: str,
    ablations: list[str],
    baseline: str,
    bootstrap: int,
    seed: int,
) -> list[dict]:
    out: list[dict] = []
    ok_rows = [r for r in rows if r.get("status") == "ok" and str(r.get("codec_id")) == codec]
    p_slots: list[tuple[int, float]] = []
    for ablation in ablations:
        condition_key = f"{codec}::{ablation}"
        condition_rows = [r for r in ok_rows if str(r.get("ablation_id")) == ablation]
        for metric in metrics:
            values = [_to_float(r.get(metric, float("nan"))) for r in condition_rows]
            values = [v for v in values if not math.isnan(v)]
            ci_lo, ci_hi = _bootstrap_ci(values, bootstrap, seed)
            paired = _paired_stats(ok_rows, metric, condition_key, f"{codec}::{baseline}")
            row = {
                "codec_id": codec,
                "ablation_id": ablation,
                "metric": metric,
                "n": len(values),
                "mean": float(np.mean(values)) if values else float("nan"),
                "median": float(np.median(values)) if values else float("nan"),
                "ci95_lo": ci_lo,
                "ci95_hi": ci_hi,
                **paired,
                "wilcoxon_p_bh": float("nan"),
            }
            p_slots.append((len(out), row["wilcoxon_p"]))
            out.append(row)
    qvals = _bh_adjust([p for _, p in p_slots])
    for (idx, _), q in zip(p_slots, qvals):
        out[idx]["wilcoxon_p_bh"] = q
    return out


def _fmt_num(value) -> str:
    v = _to_float(value)
    if math.isnan(v):
        return "nan"
    return f"{v:.4g}"


def _write_latex_table(path: Path, rows: list[dict], columns: list[str], caption: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{" + "l" * len(columns) + "}",
        "\\toprule",
        " & ".join(c.replace("_", "\\_") for c in columns) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(_fmt_num(row.get(c)) if isinstance(row.get(c), (float, int)) else str(row.get(c, "")).replace("_", "\\_") for c in columns) + " \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\end{table}",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _require_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    return plt, PdfPages


def _plot_metric_distributions(path: Path, rows: list[dict], metrics: list[str], codec: str, ablations: list[str]) -> None:
    plt, PdfPages = _require_matplotlib()
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        for metric in metrics:
            data = []
            labels = []
            for ablation in ablations:
                vals = [
                    _to_float(r.get(metric, float("nan")))
                    for r in rows
                    if r.get("status") == "ok" and r.get("codec_id") == codec and r.get("ablation_id") == ablation
                ]
                vals = [v for v in vals if not math.isnan(v)]
                if vals:
                    data.append(vals)
                    labels.append(ablation)
            if not data:
                continue
            fig, ax = plt.subplots(figsize=(9, 4.8))
            ax.boxplot(data, labels=labels, showfliers=False)
            ax.set_title(metric.replace("_", " "))
            ax.set_ylabel(metric)
            ax.tick_params(axis="x", rotation=25)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def _plot_sequence_pareto(path: Path, rows: list[dict], codec: str) -> None:
    plt, _ = _require_matplotlib()
    path.parent.mkdir(parents=True, exist_ok=True)
    by_ablation = {}
    for row in rows:
        if row.get("status") != "ok" or row.get("codec_id") != codec:
            continue
        ablation = str(row.get("ablation_id", ""))
        by_ablation.setdefault(ablation, {"runtime": [], "objective": [], "jitter": []})
        by_ablation[ablation]["runtime"].append(_to_float(row.get("sequence_runtime_ms", float("nan"))))
        by_ablation[ablation]["objective"].append(_to_float(row.get("objective_j", float("nan"))))
        by_ablation[ablation]["jitter"].append(_to_float(row.get("index_jitter", float("nan"))))
    fig, ax = plt.subplots(figsize=(7, 5))
    for ablation, vals in sorted(by_ablation.items()):
        runtime = [v for v in vals["runtime"] if not math.isnan(v)]
        objective = [v for v in vals["objective"] if not math.isnan(v)]
        if not runtime or not objective:
            continue
        ax.scatter(np.mean(runtime), np.mean(objective), label=ablation)
    ax.set_xlabel("Sequence runtime (ms)")
    ax.set_ylabel("Objective J")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _load_diagnostics(rows: list[dict]) -> list[dict]:
    diagnostics = []
    for row in rows:
        path = Path(str(row.get("diagnostics_json", "")))
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        payload["_row"] = row
        diagnostics.append(payload)
    return diagnostics


def _rvq_threshold_rows(diagnostics: list[dict], thresholds: list[float], quantiles: list[float]) -> list[dict]:
    emissions = []
    for diag in diagnostics:
        for grain in diag.get("selected_grains", []):
            value = _to_float(grain.get("emission", float("nan")))
            if not math.isnan(value):
                emissions.append(value)
    if not emissions:
        return []
    arr = np.asarray(emissions, dtype=np.float64)
    rows = []
    for theta in thresholds:
        rows.append({"threshold": float(theta), "threshold_label": f"{theta:.2f}", "coarse_transfer_fraction": float(np.mean(arr <= theta)), "n": int(arr.size)})
    for q in quantiles:
        theta = float(np.quantile(arr, q))
        rows.append({"threshold": theta, "threshold_label": f"q{int(round(q * 100)):02d}", "coarse_transfer_fraction": float(np.mean(arr <= theta)), "n": int(arr.size)})
    return rows


def _plot_rvq_diagnostics(
    diagnostics: list[dict],
    threshold_rows: list[dict],
    emission_pdf: Path,
    gate_pdf: Path,
) -> None:
    plt, _ = _require_matplotlib()
    emission_pdf.parent.mkdir(parents=True, exist_ok=True)
    gate_pdf.parent.mkdir(parents=True, exist_ok=True)
    bands = {"coarse": [], "middle": [], "fine": []}
    for diag in diagnostics:
        for grain in diag.get("selected_grains", []):
            dists = grain.get("group_dists", [])
            for idx, name in enumerate(("coarse", "middle", "fine")):
                if idx < len(dists):
                    value = _to_float(dists[idx])
                    if not math.isnan(value):
                        bands[name].append(value)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    data = [vals for vals in bands.values() if vals]
    labels = [name for name, vals in bands.items() if vals]
    if data:
        ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_ylabel("Selected emission distance")
    ax.set_title("Emission distributions by RVQ band")
    fig.tight_layout()
    fig.savefig(emission_pdf)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    numeric_rows = [r for r in threshold_rows if not str(r["threshold_label"]).startswith("q")]
    if numeric_rows:
        ax.plot([r["threshold"] for r in numeric_rows], [r["coarse_transfer_fraction"] for r in numeric_rows], marker="o")
    ax.set_xlabel("Threshold theta")
    ax.set_ylabel("Coarse transfer fraction")
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(gate_pdf)
    plt.close(fig)


def _group_summary(rows: list[dict], group_keys: list[str], metrics: list[str], bootstrap: int, seed: int) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = tuple(str(row.get(k, "")) for k in group_keys)
        groups.setdefault(key, []).append(row)
    out = []
    for key, group_rows in sorted(groups.items()):
        base = {k: v for k, v in zip(group_keys, key)}
        for metric in metrics:
            vals = [_to_float(r.get(metric, float("nan"))) for r in group_rows]
            vals = [v for v in vals if not math.isnan(v)]
            lo, hi = _bootstrap_ci(vals, bootstrap, seed)
            out.append(
                {
                    **base,
                    "metric": metric,
                    "n": len(vals),
                    "mean": float(np.mean(vals)) if vals else float("nan"),
                    "median": float(np.median(vals)) if vals else float("nan"),
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                }
            )
    return out


def dac_main(args: argparse.Namespace) -> None:
    rows = _load_rows(args.run_dirs)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    ablations = [a.strip() for a in args.ablations.split(",") if a.strip()]
    out_rows = _summarize_conditions(rows, metrics, args.codec, ablations, args.baseline, args.bootstrap, args.seed)
    fields = [
        "codec_id",
        "ablation_id",
        "metric",
        "n",
        "mean",
        "median",
        "ci95_lo",
        "ci95_hi",
        "paired_n",
        "paired_median_difference",
        "wilcoxon_p",
        "wilcoxon_p_bh",
        "effect_size_rank_biserial",
    ]
    _write_csv(Path(args.output_csv), out_rows, fields)
    _write_latex_table(Path(args.output_tex), out_rows, fields[:8], "Main 96-pair DAC objective evaluation.", "tab:dac-main-full96")
    _plot_metric_distributions(Path(args.output_fig), rows, metrics, args.codec, ablations)


def sequence(args: argparse.Namespace) -> None:
    rows = _load_rows(args.run_dirs)
    out_rows = _group_summary(rows, ["codec_id", "ablation_id"], SEQUENCE_METRICS, args.bootstrap, args.seed)
    _write_csv(Path(args.output_csv), out_rows)
    _write_latex_table(
        Path(args.output_tex),
        out_rows,
        ["codec_id", "ablation_id", "metric", "n", "mean", "median", "ci95_lo", "ci95_hi"],
        "Sequence optimizer comparison.",
        "tab:sequence-optimizer",
    )
    _plot_sequence_pareto(Path(args.output_fig), rows, args.codec)


def rvq_diagnostics(args: argparse.Namespace) -> None:
    rows = _load_rows(args.run_dirs)
    diagnostics = _load_diagnostics(rows)
    thresholds = [_to_float(x) for x in args.thresholds.split(",") if x.strip()]
    quantiles = [_to_float(x) for x in args.quantiles.split(",") if x.strip()]
    threshold_rows = _rvq_threshold_rows(diagnostics, thresholds, quantiles)
    _write_csv(Path(args.output_csv), threshold_rows, ["threshold_label", "threshold", "coarse_transfer_fraction", "n"])
    _plot_rvq_diagnostics(
        diagnostics,
        threshold_rows,
        Path(args.emission_fig),
        Path(args.gate_fig),
    )


def rho_sweep(args: argparse.Namespace) -> None:
    rows = _load_rows(args.run_dirs)
    out_rows = _group_summary(rows, ["rvq_focus", "codec_id", "ablation_id"], SOURCE_STRUCTURE_METRICS + PALETTE_TRANSFER_METRICS, args.bootstrap, args.seed)
    _write_csv(Path(args.output_csv), out_rows)
    _write_latex_table(
        Path(args.output_tex),
        out_rows,
        ["rvq_focus", "codec_id", "ablation_id", "metric", "n", "mean", "median"],
        "Rho control sweep.",
        "tab:rho-sweep",
    )
    _plot_control_curve(Path(args.output_fig), out_rows, "rvq_focus", "rho")


def _plot_control_curve(path: Path, rows: list[dict], x_key: str, x_label: str) -> None:
    plt, _ = _require_matplotlib()
    grouped: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        x = _to_float(row.get(x_key, float("nan")))
        y = _to_float(row.get("mean", float("nan")))
        metric = str(row.get("metric", ""))
        if math.isnan(x) or math.isnan(y):
            continue
        grouped.setdefault(metric, []).append((x, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for metric, pairs in sorted(grouped.items()):
        pairs.sort()
        ax.plot([p[0] for p in pairs], [p[1] for p in pairs], marker="o", label=metric)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Metric mean")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def grain_hop(args: argparse.Namespace) -> None:
    rows = _load_rows(args.run_dirs)
    metrics = SOURCE_STRUCTURE_METRICS + PALETTE_TRANSFER_METRICS + ["sequence_runtime_ms", "end_to_end_rtf"]
    out_rows = _group_summary(rows, ["unit", "stride", "codec_id", "ablation_id"], metrics, args.bootstrap, args.seed)
    _write_csv(Path(args.output_csv), out_rows)


def runtime_scaling(args: argparse.Namespace) -> None:
    rows = _load_rows(args.run_dirs)
    metrics = ["encode_ms", "decode_ms", "sequence_runtime_ms", "total_ms", "end_to_end_rtf"]
    out_rows = _group_summary(rows, ["palette_count", "codec_id", "ablation_id"], metrics, args.bootstrap, args.seed)
    _write_csv(Path(args.palette_scaling_csv), out_rows)


def listening_manifest(args: argparse.Namespace) -> None:
    rows = [r for r in _load_rows(args.run_dirs) if r.get("status") == "ok"]
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    by_clip: dict[str, dict[str, dict]] = {}
    for row in rows:
        key = str(row.get("clip_id", ""))
        cond = str(row.get("ablation_id", ""))
        if cond in conditions:
            by_clip.setdefault(key, {})[cond] = row
    eligible = [clip for clip, mapping in by_clip.items() if all(c in mapping for c in conditions)]
    rng = random.Random(args.seed)
    rng.shuffle(eligible)
    eligible = eligible[: int(args.sets)]
    manifest = []
    for set_idx, clip_id in enumerate(eligible, start=1):
        for condition in conditions:
            row = by_clip[clip_id][condition]
            cfg_path = Path(row.get("config_json", ""))
            source = ""
            if cfg_path.exists():
                try:
                    source = json.loads(cfg_path.read_text()).get("source", "")
                except Exception:
                    pass
            manifest.append(
                {
                    "set_id": f"stim_{set_idx:02d}",
                    "clip_id": clip_id,
                    "condition": condition,
                    "source_path": source,
                    "stimulus_path": row.get("output_wav", ""),
                }
            )
    _write_csv(Path(args.output_csv), manifest, ["set_id", "clip_id", "condition", "source_path", "stimulus_path"])


def claim_table(args: argparse.Namespace) -> None:
    rows = [
        {
            "Claim": "Beam/sequence optimization improves palette-path continuity.",
            "Experiment": "Sequence optimizer comparison",
            "Split": "test",
            "Metric": "J, transition cost, index jitter, file-switch rate",
            "Result": "Fill from sequence_optimizer_comparison.csv",
            "Limitation": "Bounded by retained Top-K candidates.",
        },
        {
            "Claim": "RVQ-group transfer exposes source-structure vs palette-detail control.",
            "Experiment": "RVQ threshold/band and rho sweeps",
            "Split": "dev/test",
            "Metric": "token change rates, source/onset metrics, palette distance",
            "Result": "Fill from rvq_threshold_sweep.csv and rho_sweep.csv",
            "Limitation": "Coarse gate claim depends on activation fraction.",
        },
        {
            "Claim": "The method is deployable under realistic palette/chunk/runtime constraints.",
            "Experiment": "Chunk/block parity and palette scaling",
            "Split": "test/runtime",
            "Metric": "RTF, latency, RSS, parity SC/LSD, underruns",
            "Result": "Fill from runtime_breakdown.tex and palette_scaling.csv",
            "Limitation": "Hardware and backend dependent.",
        },
    ]
    _write_latex_table(Path(args.output_tex), rows, ["Claim", "Experiment", "Split", "Metric", "Result", "Limitation"], "Claim-to-evidence map.", "tab:claim-evidence")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build paper result artifacts from completed evaluation runs.")
    sub = p.add_subparsers(dest="command", required=True)

    main_p = sub.add_parser("dac-main", help="Write results/dac_main_full96.csv, table, and metric distribution PDF.")
    main_p.add_argument("--run-dirs", nargs="+", required=True)
    main_p.add_argument("--codec", default="dac")
    main_p.add_argument("--ablations", default="greedy_full_layer,greedy_rvq_group,beam_full_layer,beam_rvq_group")
    main_p.add_argument("--baseline", default="greedy_full_layer")
    main_p.add_argument("--metrics", default=",".join(DEFAULT_MAIN_METRICS))
    main_p.add_argument("--bootstrap", type=int, default=2000)
    main_p.add_argument("--seed", type=int, default=1234)
    main_p.add_argument("--output-csv", default="results/dac_main_full96.csv")
    main_p.add_argument("--output-tex", default="tables/dac_main_full96.tex")
    main_p.add_argument("--output-fig", default="figures/dac_metric_distributions.pdf")
    main_p.set_defaults(func=dac_main)

    seq_p = sub.add_parser("sequence", help="Summarize Greedy/Smoothing/Beam/Viterbi sequence diagnostics.")
    seq_p.add_argument("--run-dirs", nargs="+", required=True)
    seq_p.add_argument("--codec", default="dac")
    seq_p.add_argument("--bootstrap", type=int, default=2000)
    seq_p.add_argument("--seed", type=int, default=1234)
    seq_p.add_argument("--output-csv", default="results/sequence_optimizer_comparison.csv")
    seq_p.add_argument("--output-tex", default="tables/sequence_optimizer_comparison.tex")
    seq_p.add_argument("--output-fig", default="figures/beam_viterbi_pareto.pdf")
    seq_p.set_defaults(func=sequence)

    rvq_p = sub.add_parser("rvq-diagnostics", help="Build RVQ emission and coarse-gate diagnostics.")
    rvq_p.add_argument("--run-dirs", nargs="+", required=True)
    rvq_p.add_argument("--thresholds", default="0.40,0.55,0.70,0.85,1.00")
    rvq_p.add_argument("--quantiles", default="0.10,0.25,0.50")
    rvq_p.add_argument("--output-csv", default="results/rvq_threshold_sweep.csv")
    rvq_p.add_argument("--emission-fig", default="figures/emission_distribution_by_band.pdf")
    rvq_p.add_argument("--gate-fig", default="figures/coarse_gate_activation_curve.pdf")
    rvq_p.set_defaults(func=rvq_diagnostics)

    rho_p = sub.add_parser("rho-sweep", help="Summarize rho/rvq_focus sweep runs.")
    rho_p.add_argument("--run-dirs", nargs="+", required=True)
    rho_p.add_argument("--bootstrap", type=int, default=2000)
    rho_p.add_argument("--seed", type=int, default=1234)
    rho_p.add_argument("--output-csv", default="results/rho_sweep.csv")
    rho_p.add_argument("--output-tex", default="tables/rho_sweep.tex")
    rho_p.add_argument("--output-fig", default="figures/rho_control_curve.pdf")
    rho_p.set_defaults(func=rho_sweep)

    gh_p = sub.add_parser("grain-hop", help="Summarize grain/hop sweep runs.")
    gh_p.add_argument("--run-dirs", nargs="+", required=True)
    gh_p.add_argument("--bootstrap", type=int, default=2000)
    gh_p.add_argument("--seed", type=int, default=1234)
    gh_p.add_argument("--output-csv", default="results/grain_hop_sweep.csv")
    gh_p.set_defaults(func=grain_hop)

    rt_p = sub.add_parser("runtime-scaling", help="Summarize palette scaling/runtime runs.")
    rt_p.add_argument("--run-dirs", nargs="+", required=True)
    rt_p.add_argument("--bootstrap", type=int, default=2000)
    rt_p.add_argument("--seed", type=int, default=1234)
    rt_p.add_argument("--palette-scaling-csv", default="results/palette_scaling.csv")
    rt_p.set_defaults(func=runtime_scaling)

    listen_p = sub.add_parser("listening-manifest", help="Create a balanced listening stimuli manifest.")
    listen_p.add_argument("--run-dirs", nargs="+", required=True)
    listen_p.add_argument("--conditions", default="greedy_rvq_group,beam_rvq_group,viterbi_rvq_group")
    listen_p.add_argument("--sets", type=int, default=12)
    listen_p.add_argument("--seed", type=int, default=1234)
    listen_p.add_argument("--output-csv", default="listening/stimuli_manifest.csv")
    listen_p.set_defaults(func=listening_manifest)

    claim_p = sub.add_parser("claim-table", help="Write manuscript claim-to-evidence table scaffold.")
    claim_p.add_argument("--output-tex", default="tables/claim_to_evidence.tex")
    claim_p.set_defaults(func=claim_table)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
