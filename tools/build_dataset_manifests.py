#!/usr/bin/env python3
"""Build immutable paper dataset manifests for Neural Morphing.

The tool scans palette/source/reference audio folders, joins optional metadata,
computes file hashes and durations, and writes the five CSVs used by the paper
protocol:

  data/manifests/palette_freesound.csv
  data/manifests/source_lofi_drums.csv
  data/manifests/reference_lofi_drums.csv
  data/manifests/eval_pairs_dev.csv
  data/manifests/eval_pairs_test.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import soundfile as sf


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif", ".m4a", ".aac"}

DEFAULT_ALLOWED_LICENSES = (
    "CC0-1.0",
    "CC-BY-3.0",
    "CC-BY-4.0",
    "CC-BY-SA-3.0",
    "CC-BY-SA-4.0",
)

CLIP_FIELDS = [
    "clip_id",
    "path",
    "sha256",
    "license",
    "source_url",
    "source_ref",
    "duration",
    "sample_rate",
    "channels",
    "split",
    "tempo_bpm",
    "tempo_bucket",
    "crop_start",
    "crop_end",
    "is_palette",
    "is_source",
    "is_reference",
    "license_filter",
    "license_ok",
]

PAIR_FIELDS = [
    "pair_id",
    "split",
    "source_clip_id",
    "source_path",
    "source_sha256",
    "reference_clip_id",
    "reference_path",
    "reference_sha256",
    "tempo_bucket",
    "legacy_32",
    "pair_index",
]


def _audio_files(root: Path) -> list[Path]:
    if not root:
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS)


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _stable_clip_id(path: Path, prefix: str, existing: set[str]) -> str:
    stem = re.sub(r"[^A-Za-z0-9_]+", "_", path.stem).strip("_").lower() or "clip"
    candidate = f"{prefix}_{stem}"
    if candidate not in existing:
        existing.add(candidate)
        return candidate
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:8]
    candidate = f"{prefix}_{stem}_{digest}"
    existing.add(candidate)
    return candidate


def _read_metadata(path: str) -> dict[str, dict]:
    if not path:
        return {}
    meta_path = Path(path)
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {meta_path}")
    out: dict[str, dict] = {}
    with meta_path.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            normalized = {str(k).strip(): ("" if v is None else str(v).strip()) for k, v in row.items()}
            keys = [
                normalized.get("path", ""),
                normalized.get("clip_id", ""),
                normalized.get("source_ref", ""),
                normalized.get("source_url", ""),
            ]
            for key in keys:
                if key:
                    out[key] = normalized
                    try:
                        out[str(Path(key).resolve())] = normalized
                    except Exception:
                        pass
    return out


def _metadata_for(path: Path, clip_id: str, metadata: dict[str, dict]) -> dict:
    candidates = [str(path), str(path.resolve()), path.name, path.stem, clip_id]
    for key in candidates:
        if key in metadata:
            return metadata[key]
    return {}


def _safe_float(value: str | float | int | None, default: float) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _estimate_tempo(path: Path, sample_rate: int, skip_tempo: bool) -> tuple[float, str]:
    if skip_tempo:
        return float("nan"), "unknown"
    try:
        y, sr = librosa.load(str(path), sr=sample_rate, mono=True, duration=30.0)
        if y.size < max(512, sr // 4):
            return float("nan"), "unknown"
        tempo_arr = librosa.beat.tempo(y=y, sr=sr, aggregate=None)
        tempo = float(np.nanmedian(tempo_arr)) if np.asarray(tempo_arr).size else float("nan")
    except Exception:
        return float("nan"), "unknown"
    if not math.isfinite(tempo) or tempo <= 0.0:
        return float("nan"), "unknown"
    if tempo < 80.0:
        bucket = "slow_lt80"
    elif tempo < 110.0:
        bucket = "mid_80_110"
    elif tempo < 140.0:
        bucket = "up_110_140"
    else:
        bucket = "fast_ge140"
    return tempo, bucket


def _license_ok(license_id: str, allowed: set[str]) -> bool:
    value = license_id.strip().upper()
    if not value:
        return False
    if "NC" in value or "NONCOMMERCIAL" in value:
        return False
    if "ND" in value or "NO-DERIV" in value or "NODERIV" in value:
        return False
    return license_id.strip() in allowed


def _write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _scan_split(
    root: Path,
    split: str,
    prefix: str,
    metadata: dict[str, dict],
    allowed_licenses: set[str],
    license_filter: str,
    skip_tempo: bool,
    existing_ids: set[str],
) -> list[dict]:
    rows: list[dict] = []
    for audio_path in _audio_files(root):
        try:
            info = sf.info(str(audio_path))
        except Exception:
            continue
        clip_id = _stable_clip_id(audio_path, prefix, existing_ids)
        meta = _metadata_for(audio_path, clip_id, metadata)
        duration = float(info.duration)
        crop_start = _safe_float(meta.get("crop_start"), 0.0)
        crop_end = _safe_float(meta.get("crop_end"), duration)
        tempo_meta = _safe_float(meta.get("tempo_bpm"), float("nan"))
        if math.isfinite(tempo_meta):
            tempo_bpm = tempo_meta
            tempo_bucket = str(meta.get("tempo_bucket", "")).strip() or _tempo_bucket(tempo_bpm)
        else:
            tempo_bpm, tempo_bucket = _estimate_tempo(audio_path, int(info.samplerate), skip_tempo)
        license_id = str(meta.get("license", "")).strip()
        source_url = str(meta.get("source_url", "")).strip()
        source_ref = str(meta.get("source_ref", "")).strip() or source_url
        row = {
            "clip_id": clip_id,
            "path": str(audio_path.resolve()),
            "sha256": _sha256_file(audio_path),
            "license": license_id,
            "source_url": source_url,
            "source_ref": source_ref,
            "duration": f"{duration:.6f}",
            "sample_rate": int(info.samplerate),
            "channels": int(info.channels),
            "split": str(meta.get("split", "")).strip() or split,
            "tempo_bpm": f"{tempo_bpm:.3f}" if math.isfinite(tempo_bpm) else "",
            "tempo_bucket": tempo_bucket,
            "crop_start": f"{max(0.0, crop_start):.6f}",
            "crop_end": f"{min(duration, max(crop_start, crop_end)):.6f}",
            "is_palette": int(split == "palette"),
            "is_source": int(split == "source"),
            "is_reference": int(split == "reference"),
            "license_filter": license_filter,
            "license_ok": int(_license_ok(license_id, allowed_licenses)),
        }
        rows.append(row)
    return rows


def _tempo_bucket(tempo: float) -> str:
    if not math.isfinite(tempo):
        return "unknown"
    if tempo < 80.0:
        return "slow_lt80"
    if tempo < 110.0:
        return "mid_80_110"
    if tempo < 140.0:
        return "up_110_140"
    return "fast_ge140"


def _pair_rows(
    source_rows: list[dict],
    ref_rows: list[dict],
    seed: int,
    dev_pairs: int,
    test_pairs: int,
    legacy_count: int,
) -> tuple[list[dict], list[dict]]:
    refs_by_stem = {Path(r["path"]).stem.lower(): r for r in ref_rows}
    refs = list(ref_rows)
    if not source_rows or not refs:
        return [], []

    pairs = []
    for idx, src in enumerate(source_rows):
        ref = refs_by_stem.get(Path(src["path"]).stem.lower(), refs[idx % len(refs)])
        pairs.append((src, ref))

    rng = random.Random(seed)
    rng.shuffle(pairs)
    dev_count = max(0, min(int(dev_pairs), len(pairs)))
    remaining = pairs[dev_count:]
    test_count = len(remaining) if int(test_pairs) <= 0 else min(int(test_pairs), len(remaining))
    dev = _format_pairs(pairs[:dev_count], "dev", 0, 0)
    test = _format_pairs(remaining[:test_count], "test", legacy_count, 0)
    return dev, test


def _format_pairs(pairs: list[tuple[dict, dict]], split: str, legacy_count: int, offset: int) -> list[dict]:
    rows = []
    for i, (src, ref) in enumerate(pairs):
        index = offset + i
        rows.append(
            {
                "pair_id": f"{split}_{index:03d}_{src['clip_id']}__{ref['clip_id']}",
                "split": split,
                "source_clip_id": src["clip_id"],
                "source_path": src["path"],
                "source_sha256": src["sha256"],
                "reference_clip_id": ref["clip_id"],
                "reference_path": ref["path"],
                "reference_sha256": ref["sha256"],
                "tempo_bucket": src.get("tempo_bucket") or ref.get("tempo_bucket") or "unknown",
                "legacy_32": int(split == "test" and i < int(legacy_count)),
                "pair_index": index,
            }
        )
    return rows


def _write_eval_json(path: Path, palette_rows: list[dict], pair_rows: list[dict], seed: int, notes: str | None = None) -> None:
    payload = {
        "seed": int(seed),
        "palette_train": [r["path"] for r in palette_rows],
        "source_eval": [
            {
                "id": r["pair_id"],
                "source": r["source_path"],
                "reference": r["reference_path"],
                "source_clip_id": r["source_clip_id"],
                "reference_clip_id": r["reference_clip_id"],
                "legacy_32": bool(int(r.get("legacy_32", 0))),
            }
            for r in pair_rows
        ],
        "notes": notes or "Generated from immutable CSV manifests by tools/build_dataset_manifests.py.",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _summary(rows_by_name: dict[str, list[dict]], allowed: set[str]) -> dict:
    missing = {}
    license_bad = {}
    for name, rows in rows_by_name.items():
        if name.startswith("eval_pairs_"):
            missing[name] = {}
            license_bad[name] = 0
            continue
        missing[name] = {
            "license": sum(1 for r in rows if not str(r.get("license", "")).strip()),
            "source_ref": sum(1 for r in rows if not str(r.get("source_ref", "")).strip()),
        }
        license_bad[name] = sum(1 for r in rows if not bool(int(r.get("license_ok", 0))))
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "allowed_license_filter": sorted(allowed),
        "counts": {name: len(rows) for name, rows in rows_by_name.items()},
        "missing_required_metadata": missing,
        "license_filter_failures": license_bad,
    }


def build_manifests(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    metadata = _read_metadata(args.metadata_csv)
    allowed = {x.strip() for x in str(args.allowed_licenses).split(",") if x.strip()}
    license_filter = "exact-allowed-licenses: " + ",".join(sorted(allowed))
    existing_ids: set[str] = set()

    palette_rows = _scan_split(
        Path(args.palette_dir),
        "palette",
        "pal",
        metadata,
        allowed,
        license_filter,
        bool(args.skip_tempo),
        existing_ids,
    )
    source_rows = _scan_split(
        Path(args.source_dir),
        "source",
        "src",
        metadata,
        allowed,
        license_filter,
        bool(args.skip_tempo),
        existing_ids,
    )
    ref_rows = _scan_split(
        Path(args.reference_dir),
        "reference",
        "ref",
        metadata,
        allowed,
        license_filter,
        bool(args.skip_tempo),
        existing_ids,
    )
    dev_pairs, test_pairs = _pair_rows(
        source_rows,
        ref_rows,
        seed=int(args.seed),
        dev_pairs=int(args.dev_pairs),
        test_pairs=int(args.test_pairs),
        legacy_count=int(args.legacy_count),
    )

    _write_csv(out_dir / "palette_freesound.csv", palette_rows, CLIP_FIELDS)
    _write_csv(out_dir / "source_lofi_drums.csv", source_rows, CLIP_FIELDS)
    _write_csv(out_dir / "reference_lofi_drums.csv", ref_rows, CLIP_FIELDS)
    _write_csv(out_dir / "eval_pairs_dev.csv", dev_pairs, PAIR_FIELDS)
    _write_csv(out_dir / "eval_pairs_test.csv", test_pairs, PAIR_FIELDS)
    _write_eval_json(out_dir / "eval_manifest_dev.json", palette_rows, dev_pairs, int(args.seed))
    _write_eval_json(out_dir / "eval_manifest_test.json", palette_rows, test_pairs, int(args.seed))
    diagnostic_pairs = []
    if int(args.diagnostic_pairs) > 0:
        diagnostic_count = min(int(args.diagnostic_pairs), len(test_pairs))
        diagnostic_split = f"diagnostic{diagnostic_count}_from_test"
        diagnostic_pairs = [{**row, "split": diagnostic_split} for row in test_pairs[:diagnostic_count]]
        _write_csv(out_dir / f"eval_pairs_diagnostic{diagnostic_count}.csv", diagnostic_pairs, PAIR_FIELDS)
        _write_eval_json(
            out_dir / f"eval_manifest_diagnostic{diagnostic_count}.json",
            palette_rows,
            diagnostic_pairs,
            int(args.seed),
            notes=(
                f"Bounded diagnostic subset derived from the first {diagnostic_count} pairs of "
                "eval_manifest_test.json because no disjoint dev pairs remain when all available "
                "pairs are assigned to the main test manifest."
            ),
        )

    summary = _summary(
        {
            "palette_freesound": palette_rows,
            "source_lofi_drums": source_rows,
            "reference_lofi_drums": ref_rows,
            "eval_pairs_dev": dev_pairs,
            "eval_pairs_test": test_pairs,
            **({f"eval_pairs_diagnostic{len(diagnostic_pairs)}": diagnostic_pairs} if diagnostic_pairs else {}),
        },
        allowed,
    )
    summary_path = out_dir / "dataset_manifest_audit.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    problems = sum(v for group in summary["missing_required_metadata"].values() for v in group.values())
    problems += sum(summary["license_filter_failures"].values())
    print(f"Wrote manifests under: {out_dir}")
    print(f"Wrote audit: {summary_path}")
    if problems and args.strict:
        raise SystemExit(f"Strict manifest build failed with {problems} metadata/license issue(s).")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build immutable Neural Morphing dataset manifests.")
    p.add_argument("--palette-dir", required=True)
    p.add_argument("--source-dir", required=True)
    p.add_argument("--reference-dir", required=True)
    p.add_argument("--metadata-csv", default="", help="Optional CSV keyed by path or clip_id with license/source/crop fields.")
    p.add_argument("--out-dir", default="data/manifests")
    p.add_argument("--allowed-licenses", default=",".join(DEFAULT_ALLOWED_LICENSES))
    p.add_argument("--dev-pairs", type=int, default=24)
    p.add_argument("--test-pairs", type=int, default=96, help="Use 0 to keep all non-dev pairs.")
    p.add_argument("--diagnostic-pairs", type=int, default=0, help="Optional diagnostic subset size copied from the fixed test order.")
    p.add_argument("--legacy-count", type=int, default=32)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--skip-tempo", action="store_true", help="Skip librosa tempo estimation.")
    p.add_argument("--strict", action="store_true", help="Exit non-zero if license/source metadata is incomplete.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    build_manifests(args)


if __name__ == "__main__":
    main()
