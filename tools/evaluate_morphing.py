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
import hashlib
import json
import math
import os
import random
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import stft
from scipy.stats import wilcoxon


ABLATIONS = [
    {"id": "greedy_full_layer", "matcher": "greedy", "swap": "full_layer"},
    {"id": "greedy_rvq_group", "matcher": "greedy", "swap": "rvq_group"},
    {"id": "greedy_smooth_full_layer", "matcher": "greedy_smooth", "swap": "full_layer"},
    {"id": "greedy_smooth_rvq_group", "matcher": "greedy_smooth", "swap": "rvq_group"},
    {"id": "beam_full_layer", "matcher": "beam", "swap": "full_layer"},
    {"id": "beam_rvq_group", "matcher": "beam", "swap": "rvq_group"},
    {"id": "viterbi_full_layer", "matcher": "viterbi", "swap": "full_layer"},
    {"id": "viterbi_rvq_group", "matcher": "viterbi", "swap": "rvq_group"},
    {"id": "beam_identity", "matcher": "beam", "swap": "identity"},
    {"id": "beam_coarse_gated", "matcher": "beam", "swap": "coarse_gated"},
    {"id": "beam_coarse_forced", "matcher": "beam", "swap": "coarse_forced"},
    {"id": "beam_middle_only", "matcher": "beam", "swap": "middle_only"},
    {"id": "beam_fine_only", "matcher": "beam", "swap": "fine_only"},
    {"id": "beam_middle_fine", "matcher": "beam", "swap": "middle_fine"},
    {"id": "beam_full_layer_forced", "matcher": "beam", "swap": "full_layer_forced"},
]

DEFAULT_CODECS = ["dac", "spectrostream"]

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


def _unique_paths(paths: Iterable[Path]) -> List[Path]:
    seen = set()
    out: List[Path] = []
    for path in paths:
        resolved = Path(path).resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        out.append(resolved)
    return out


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


def _safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    c = np.corrcoef(a, b)
    if c.shape != (2, 2):
        return float("nan")
    return float(c[0, 1])


def _to_mono_audio(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return np.mean(arr, axis=1, dtype=np.float32)
    return np.asarray(arr.reshape(-1), dtype=np.float32)


def _align_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    aa = _to_mono_audio(a)
    bb = _to_mono_audio(b)
    n = min(aa.shape[0], bb.shape[0])
    if n <= 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    return aa[:n], bb[:n]


def _rms_envelope(audio: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    x = _to_mono_audio(audio)
    if x.size < frame_length:
        return np.asarray([], dtype=np.float32)
    n_frames = 1 + (x.size - frame_length) // hop_length
    env = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_length
        frame = x[start : start + frame_length]
        env[i] = np.sqrt(np.mean(frame * frame) + 1e-12)
    return env


def _stft_mag_phase(audio: np.ndarray, sample_rate: int, n_fft: int = 1024, hop_length: int = 256):
    del sample_rate
    x = _to_mono_audio(audio)
    if x.size < n_fft:
        return np.zeros((n_fft // 2 + 1, 0), dtype=np.float32), np.zeros((n_fft // 2 + 1, 0), dtype=np.float32)
    _, _, z = stft(x, nperseg=n_fft, noverlap=n_fft - hop_length, boundary=None, padded=False)
    mag = np.abs(z).astype(np.float32, copy=False)
    phase = np.angle(z).astype(np.float32, copy=False)
    return mag, phase


def metric_spectral_convergence(output_audio: np.ndarray, ref_audio: np.ndarray, sample_rate: int) -> float:
    out, ref = _align_pair(output_audio, ref_audio)
    mag_out, _ = _stft_mag_phase(out, sample_rate)
    mag_ref, _ = _stft_mag_phase(ref, sample_rate)
    frames = min(mag_out.shape[1], mag_ref.shape[1])
    if frames == 0:
        return float("nan")
    diff = mag_ref[:, :frames] - mag_out[:, :frames]
    num = np.linalg.norm(diff, ord="fro")
    den = np.linalg.norm(mag_ref[:, :frames], ord="fro") + 1e-12
    return float(num / den)


def metric_log_spectral_distance(output_audio: np.ndarray, ref_audio: np.ndarray, sample_rate: int) -> float:
    out, ref = _align_pair(output_audio, ref_audio)
    mag_out, _ = _stft_mag_phase(out, sample_rate)
    mag_ref, _ = _stft_mag_phase(ref, sample_rate)
    frames = min(mag_out.shape[1], mag_ref.shape[1])
    if frames == 0:
        return float("nan")
    log_out = 20.0 * np.log10(np.maximum(mag_out[:, :frames], 1e-8))
    log_ref = 20.0 * np.log10(np.maximum(mag_ref[:, :frames], 1e-8))
    return float(np.mean(np.sqrt(np.mean((log_ref - log_out) ** 2, axis=0))))


def metric_envelope_correlation(output_audio: np.ndarray, ref_audio: np.ndarray) -> float:
    out, ref = _align_pair(output_audio, ref_audio)
    env_out = _rms_envelope(out)
    env_ref = _rms_envelope(ref)
    n = min(env_out.size, env_ref.size)
    if n == 0:
        return float("nan")
    return _safe_corrcoef(env_out[:n], env_ref[:n])


def metric_boundary_phase_jump(audio: np.ndarray, sample_rate: int, boundaries_samples: np.ndarray) -> float:
    mono = _to_mono_audio(audio)
    if boundaries_samples.size == 0 or mono.size < 2048:
        return float("nan")

    n_fft = 1024
    hop = 256
    _, phase = _stft_mag_phase(mono, sample_rate, n_fft=n_fft, hop_length=hop)
    if phase.shape[1] < 2:
        return float("nan")

    jumps = []
    for b in boundaries_samples.astype(int):
        frame = int(round(b / hop))
        if frame <= 0 or frame >= phase.shape[1]:
            continue
        left = phase[:, frame - 1]
        right = phase[:, frame]
        delta = np.angle(np.exp(1j * (right - left)))
        jumps.append(float(np.mean(np.abs(delta))))
    if not jumps:
        return float("nan")
    return float(np.mean(jumps))


def metric_clipping_fraction(audio: np.ndarray, threshold: float = 0.999) -> float:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(np.abs(arr.reshape(-1)) >= float(threshold)))


def metric_onset_preservation(
    output_audio: np.ndarray,
    source_audio: np.ndarray,
    sample_rate: int,
    tolerance_ms: float = 50.0,
) -> tuple[float, float]:
    out, src = _align_pair(output_audio, source_audio)
    if out.size < 2048 or src.size < 2048:
        return float("nan"), float("nan")
    try:
        out_frames = librosa.onset.onset_detect(y=out, sr=sample_rate, units="frames", backtrack=False)
        src_frames = librosa.onset.onset_detect(y=src, sr=sample_rate, units="frames", backtrack=False)
    except Exception:
        return float("nan"), float("nan")
    out_times = librosa.frames_to_time(out_frames, sr=sample_rate)
    src_times = librosa.frames_to_time(src_frames, sr=sample_rate)
    if src_times.size == 0:
        return float("nan"), float("nan")
    if out_times.size == 0:
        return 0.0, float("nan")

    tol = float(tolerance_ms) / 1000.0
    used: set[int] = set()
    deviations = []
    true_pos = 0
    for src_t in src_times:
        distances = np.abs(out_times - src_t)
        order = np.argsort(distances)
        match_idx = None
        for idx in order:
            if int(idx) in used:
                continue
            if float(distances[idx]) <= tol:
                match_idx = int(idx)
            break
        if match_idx is not None:
            used.add(match_idx)
            true_pos += 1
            deviations.append(abs(float(out_times[match_idx] - src_t)) * 1000.0)
    precision = true_pos / max(1, out_times.size)
    recall = true_pos / max(1, src_times.size)
    f1 = 0.0 if precision + recall <= 0.0 else 2.0 * precision * recall / (precision + recall)
    dev = float(np.mean(deviations)) if deviations else float("nan")
    return float(f1), dev


def metric_transient_strength_correlation(output_audio: np.ndarray, source_audio: np.ndarray, sample_rate: int) -> float:
    out, src = _align_pair(output_audio, source_audio)
    if out.size < 2048 or src.size < 2048:
        return float("nan")
    try:
        out_env = librosa.onset.onset_strength(y=out, sr=sample_rate)
        src_env = librosa.onset.onset_strength(y=src, sr=sample_rate)
    except Exception:
        return float("nan")
    n = min(out_env.size, src_env.size)
    if n == 0:
        return float("nan")
    return _safe_corrcoef(np.asarray(out_env[:n]), np.asarray(src_env[:n]))


def metric_bandwise_envelope_correlation(output_audio: np.ndarray, source_audio: np.ndarray, sample_rate: int) -> float:
    out, src = _align_pair(output_audio, source_audio)
    n_fft = 2048
    hop = 512
    mag_out, _ = _stft_mag_phase(out, sample_rate, n_fft=n_fft, hop_length=hop)
    mag_src, _ = _stft_mag_phase(src, sample_rate, n_fft=n_fft, hop_length=hop)
    frames = min(mag_out.shape[1], mag_src.shape[1])
    if frames == 0:
        return float("nan")
    freqs = np.linspace(0.0, sample_rate / 2.0, mag_out.shape[0])
    bands = [(0.0, 250.0), (250.0, 2000.0), (2000.0, sample_rate / 2.0 + 1.0)]
    corrs = []
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            continue
        env_out = np.sqrt(np.mean(mag_out[mask, :frames] ** 2, axis=0) + 1e-12)
        env_src = np.sqrt(np.mean(mag_src[mask, :frames] ** 2, axis=0) + 1e-12)
        corr = _safe_corrcoef(env_out, env_src)
        if not math.isnan(corr):
            corrs.append(corr)
    return float(np.mean(corrs)) if corrs else float("nan")


def metric_boundary_click_energy(audio: np.ndarray, sample_rate: int, boundaries_samples: np.ndarray, window_ms: float = 4.0) -> float:
    mono = _to_mono_audio(audio)
    if mono.size < 4 or boundaries_samples.size == 0:
        return float("nan")
    w = max(2, int(round(sample_rate * window_ms / 1000.0)))
    vals = []
    diff = np.diff(mono, prepend=mono[0])
    for b in boundaries_samples.astype(int):
        lo = max(0, b - w)
        hi = min(diff.size, b + w)
        if hi <= lo:
            continue
        click = float(np.mean(diff[lo:hi] ** 2) + 1e-12)
        local = float(np.mean(mono[lo:hi] ** 2) + 1e-12)
        vals.append(10.0 * math.log10(click / local))
    return float(np.mean(vals)) if vals else float("nan")


def metric_output_to_source_distance(output_audio: np.ndarray, source_audio: np.ndarray, sample_rate: int) -> float:
    return metric_log_spectral_distance(output_audio, source_audio, sample_rate)


def _resample_like(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if int(orig_sr) == int(target_sr):
        return audio
    if audio.ndim == 1:
        return librosa.resample(audio, orig_sr=int(orig_sr), target_sr=int(target_sr))
    chans = [librosa.resample(audio[:, ch], orig_sr=int(orig_sr), target_sr=int(target_sr)) for ch in range(audio.shape[1])]
    min_len = min(len(ch) for ch in chans)
    return np.stack([ch[:min_len] for ch in chans], axis=1)


def metric_nearest_palette_distance(
    output_audio: np.ndarray,
    sample_rate: int,
    palette_cache: list[tuple[Path, np.ndarray, int]],
) -> tuple[float, str]:
    if not palette_cache:
        return float("nan"), ""
    best = float("inf")
    best_path = ""
    for path, pal_audio, pal_sr in palette_cache:
        try:
            pal = _resample_like(pal_audio, pal_sr, sample_rate)
            distance = metric_log_spectral_distance(output_audio, pal, sample_rate)
        except Exception:
            distance = float("nan")
        if not math.isnan(distance) and distance < best:
            best = float(distance)
            best_path = str(path)
    if best == float("inf"):
        return float("nan"), ""
    return best, best_path


def metric_fad(reference_wavs: List[Path], generated_wavs: List[Path], model_name: str = "vggish") -> dict:
    try:
        from frechet_audio_distance import FrechetAudioDistance  # type: ignore
    except Exception:
        return {"score": float("nan"), "status": "missing_dependency"}

    if not reference_wavs or not generated_wavs:
        return {"score": float("nan"), "status": "insufficient_data"}

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
        try:
            score = fad.score(str(ref_root), str(gen_root))
            return {"score": float(score), "status": "ok"}
        except Exception:
            return {"score": float("nan"), "status": "error"}


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


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _check_manifest_leakage(palette: List[Path], clips: List[ClipEntry]) -> dict:
    palette_set = {p.resolve() for p in palette}
    source_set = {c.source.resolve() for c in clips}
    ref_set = {c.reference.resolve() for c in clips}

    path_overlap_palette_source = sorted(str(p) for p in (palette_set & source_set))
    path_overlap_palette_reference = sorted(str(p) for p in (palette_set & ref_set))

    palette_hashes: Dict[str, Path] = {}
    source_hashes: Dict[str, Path] = {}
    ref_hashes: Dict[str, Path] = {}
    hash_overlap_palette_source: List[dict] = []
    hash_overlap_palette_reference: List[dict] = []

    for p in palette_set:
        try:
            palette_hashes[_sha256_file(p)] = p
        except Exception:
            continue
    for p in source_set:
        try:
            source_hashes[_sha256_file(p)] = p
        except Exception:
            continue
    for p in ref_set:
        try:
            ref_hashes[_sha256_file(p)] = p
        except Exception:
            continue

    for h, pp in palette_hashes.items():
        if h in source_hashes:
            hash_overlap_palette_source.append({"palette": str(pp), "source": str(source_hashes[h]), "sha256": h})
        if h in ref_hashes:
            hash_overlap_palette_reference.append({"palette": str(pp), "reference": str(ref_hashes[h]), "sha256": h})

    return {
        "path_overlap_palette_source": path_overlap_palette_source,
        "path_overlap_palette_reference": path_overlap_palette_reference,
        "hash_overlap_palette_source": hash_overlap_palette_source,
        "hash_overlap_palette_reference": hash_overlap_palette_reference,
        "ok": not (
            path_overlap_palette_source
            or path_overlap_palette_reference
            or hash_overlap_palette_source
            or hash_overlap_palette_reference
        ),
    }


def _run_command_template(template: str, context: dict) -> None:
    command = template.format(**context)
    subprocess.run(command, shell=True, check=True)


def _shell_quote(value: str) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline([str(value)])
    return shlex.quote(str(value))


def _parse_csv_rows(csv_path: Path) -> List[dict]:
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append(dict(row))
    return rows


def _mean(values: Iterable[float]) -> float:
    arr = [v for v in values if not math.isnan(v)]
    if not arr:
        return float("nan")
    return float(np.mean(arr))


def _to_float(value, default=float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


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


def _parse_list_arg(raw: str, default_values: List[str]) -> List[str]:
    if not raw:
        return list(default_values)
    items = [p.strip() for p in raw.split(",") if p.strip()]
    return items or list(default_values)


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


def evaluate(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    palette, clips, manifest = _parse_manifest(manifest_path)
    if int(getattr(args, "clip_limit", 0) or 0) > 0:
        clips = clips[: int(args.clip_limit)]
    leakage = _check_manifest_leakage(palette, clips)

    selected_ablations = _parse_list_arg(args.ablations, [ab["id"] for ab in ABLATIONS])
    ablation_map = {ab["id"]: ab for ab in ABLATIONS}
    unknown_ablations = [ab for ab in selected_ablations if ab not in ablation_map]
    if unknown_ablations:
        raise ValueError(f"Unknown ablation id(s): {unknown_ablations}")
    selected_ablation_defs = [ablation_map[ab] for ab in selected_ablations]

    selected_codecs = _parse_list_arg(args.codecs, DEFAULT_CODECS)
    if len(selected_codecs) > 1 and args.runner_cmd and "{codec}" not in args.runner_cmd:
        raise ValueError("Runner command must include {codec} when evaluating multiple codecs.")

    palette_manifest = out_root / "palette_train.txt"
    palette_manifest.write_text("\n".join(str(p) for p in palette))
    palette_metric_limit = max(0, int(getattr(args, "palette_metric_limit", 64)))
    palette_cache: list[tuple[Path, np.ndarray, int]] = []
    for pal_path in palette[:palette_metric_limit]:
        try:
            pal_audio, pal_sr = _load_audio(pal_path)
            palette_cache.append((pal_path, pal_audio, pal_sr))
        except Exception:
            continue

    rows: List[dict] = []
    refs_for_fad: Dict[str, List[Path]] = {}
    generated_for_fad: Dict[str, List[Path]] = {}

    for codec_id in selected_codecs:
        codec_params = _resolve_runtime_params(codec_id, args)
        refs_for_fad.setdefault(codec_id, [])
        for ablation in selected_ablation_defs:
            ab_id = ablation["id"]
            condition_key = f"{codec_id}::{ab_id}"
            generated_for_fad.setdefault(condition_key, [])
            for clip in clips:
                run_dir = out_root / "runs" / codec_id / ab_id / clip.clip_id
                run_dir.mkdir(parents=True, exist_ok=True)

                output_wav = run_dir / "output.wav"
                tokens_npy = run_dir / "tokens.npy"
                match_indices_npy = run_dir / "match_indices.npy"
                latency_json = run_dir / "latency.json"
                diagnostics_json = run_dir / "diagnostics.json"
                config_json = run_dir / "config.json"
                config_payload = {
                    "codec": codec_id,
                    "ablation": ablation,
                    "clip_id": clip.clip_id,
                    "source": str(clip.source),
                    "reference": str(clip.reference),
                    "palette_manifest": str(palette_manifest),
                    "params": {
                        "temperature": codec_params["temperature"],
                        "threshold": codec_params["threshold"],
                        "continuity": codec_params["continuity"],
                        "rvq_focus": codec_params["rvq_focus"],
                        "unit": codec_params["unit"],
                        "stride": codec_params["stride"],
                        "top_k": codec_params["top_k"],
                        "seed": args.seed,
                    },
                }
                _write_json(config_json, config_payload)

                outputs_ready = (
                    output_wav.exists()
                    and tokens_npy.exists()
                    and match_indices_npy.exists()
                    and latency_json.exists()
                )
                if args.runner_cmd and not (args.resume_existing and outputs_ready):
                    context = {
                        "codec": codec_id,
                        "source": _shell_quote(str(clip.source)),
                        "reference": _shell_quote(str(clip.reference)),
                        "palette_manifest": _shell_quote(str(palette_manifest)),
                        "output_wav": _shell_quote(str(output_wav)),
                        "tokens_npy": _shell_quote(str(tokens_npy)),
                        "match_indices_npy": _shell_quote(str(match_indices_npy)),
                        "latency_json": _shell_quote(str(latency_json)),
                        "diagnostics_json": _shell_quote(str(diagnostics_json)),
                        "config_json": _shell_quote(str(config_json)),
                        "matcher": ablation["matcher"],
                        "swap": ablation["swap"],
                        "ablation_id": ab_id,
                        "temperature": codec_params["temperature"],
                        "threshold": codec_params["threshold"],
                        "continuity": codec_params["continuity"],
                        "rvq_focus": codec_params["rvq_focus"],
                        "unit": codec_params["unit"],
                        "stride": codec_params["stride"],
                        "top_k": codec_params["top_k"],
                        "seed": args.seed,
                    }
                    if not args.dry_run:
                        _run_command_template(args.runner_cmd, context)
                    else:
                        print("DRY RUN:", args.runner_cmd.format(**context))
                elif args.runner_cmd and args.resume_existing and outputs_ready:
                    print(f"resume-existing: reusing {output_wav}")

                if not output_wav.exists():
                    rows.append(
                        {
                            "codec_id": codec_id,
                            "ablation_id": ab_id,
                            "clip_id": clip.clip_id,
                            "status": "missing_output",
                            "output_wav": str(output_wav),
                            "tokens_npy": str(tokens_npy),
                            "match_indices_npy": str(match_indices_npy),
                            "latency_json": str(latency_json),
                            "diagnostics_json": str(diagnostics_json),
                            "config_json": str(config_json),
                        }
                    )
                    continue

                out_audio, out_sr = _load_audio(output_wav)
                ref_audio, ref_sr = _load_audio(clip.reference)
                src_audio, src_sr = _load_audio(clip.source)
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
                if src_sr != out_sr:
                    if src_audio.ndim == 1:
                        src_audio = librosa.resample(src_audio, orig_sr=src_sr, target_sr=out_sr)
                    else:
                        chans = [
                            librosa.resample(src_audio[:, ch], orig_sr=src_sr, target_sr=out_sr)
                            for ch in range(src_audio.shape[1])
                        ]
                        min_len = min(len(ch) for ch in chans)
                        src_audio = np.stack([ch[:min_len] for ch in chans], axis=1)
                    src_sr = out_sr

                tokens = None
                token_discontinuity = float("nan")
                token_layout_valid = False
                token_shape = []
                if tokens_npy.exists():
                    tokens = np.load(tokens_npy)
                    token_discontinuity = metric_token_discontinuity(tokens)
                    token_layout_valid = bool(tokens.ndim == 2 and tokens.shape[0] > 0 and tokens.shape[1] > 0)
                    token_shape = list(tokens.shape)

                index_jitter = float("nan")
                boundaries = np.array([], dtype=np.int64)
                match_indices = None
                if match_indices_npy.exists():
                    match_indices = np.load(match_indices_npy)
                    index_jitter = metric_index_jitter(match_indices)
                    n = int(np.asarray(match_indices).reshape(-1).shape[0])
                    if n > 1:
                        boundaries = np.linspace(0, len(out_audio), num=n + 1, dtype=np.int64)[1:-1]

                waveform_disc = metric_waveform_discontinuity(out_audio, out_sr, boundaries)
                chroma_diff = metric_chromagram_difference(out_audio, ref_audio, out_sr)
                spectral_conv = metric_spectral_convergence(out_audio, ref_audio, out_sr)
                log_spectral_dist = metric_log_spectral_distance(out_audio, ref_audio, out_sr)
                envelope_corr = metric_envelope_correlation(out_audio, src_audio)
                phase_jump = metric_boundary_phase_jump(out_audio, out_sr, boundaries)
                clipping_fraction = metric_clipping_fraction(out_audio)
                source_onset_f1, source_onset_deviation_ms = metric_onset_preservation(out_audio, src_audio, out_sr)
                transient_strength_correlation = metric_transient_strength_correlation(out_audio, src_audio, out_sr)
                bandwise_envelope_correlation = metric_bandwise_envelope_correlation(out_audio, src_audio, out_sr)
                boundary_click_energy = metric_boundary_click_energy(out_audio, out_sr, boundaries)
                output_to_source_distance = metric_output_to_source_distance(out_audio, src_audio, out_sr)
                output_to_nearest_palette_distance, nearest_palette_path = metric_nearest_palette_distance(
                    out_audio,
                    out_sr,
                    palette_cache,
                )
                palette_embedding_shift = (
                    output_to_source_distance - output_to_nearest_palette_distance
                    if not math.isnan(output_to_source_distance) and not math.isnan(output_to_nearest_palette_distance)
                    else float("nan")
                )

                latency_payload = {}
                if latency_json.exists():
                    try:
                        latency_payload = json.loads(latency_json.read_text())
                    except Exception:
                        latency_payload = {}
                diagnostics_payload = {}
                if diagnostics_json.exists():
                    try:
                        diagnostics_payload = json.loads(diagnostics_json.read_text())
                    except Exception:
                        diagnostics_payload = {}

                encode_ok = bool(latency_payload.get("encode_ok", True))
                decode_ok = bool(latency_payload.get("decode_ok", True))
                layout_ok_payload = latency_payload.get("token_layout_valid", token_layout_valid)
                token_layout_valid = token_layout_valid and bool(layout_ok_payload)
                duration_drift_samples = _to_float(latency_payload.get("duration_drift_samples", float("nan")))
                duration_drift_ms = _to_float(latency_payload.get("duration_drift_ms", float("nan")))
                source_channels = _to_float(latency_payload.get("source_channels", src_audio.shape[1] if src_audio.ndim > 1 else 1))
                output_channels = _to_float(latency_payload.get("output_channels", out_audio.shape[1] if out_audio.ndim > 1 else 1))
                channel_consistency_ok = bool(source_channels == output_channels)
                encode_ms = _to_float(latency_payload.get("encode_ms", float("nan")))
                decode_ms = _to_float(latency_payload.get("decode_ms", float("nan")))
                total_ms = _to_float(latency_payload.get("total_ms", float("nan")))
                audio_seconds = _to_float(latency_payload.get("audio_seconds", float("nan")))
                end_to_end_rtf = _to_float(latency_payload.get("end_to_end_rtf", float("nan")))
                failure_count = int(_to_float(latency_payload.get("failure_count", 0), default=0.0))
                retry_count = int(_to_float(latency_payload.get("retry_count", 0), default=0.0))
                palette_cache_hit = bool(latency_payload.get("palette_cache_hit", False))
                palette_cache_file = str(latency_payload.get("palette_cache_file", ""))
                sequence_runtime_ms = _to_float(latency_payload.get("sequence_runtime_ms", float("nan")))
                objective_j = _to_float(latency_payload.get("objective_j", float("nan")))
                emission_cost = _to_float(latency_payload.get("emission_cost", float("nan")))
                transition_cost = _to_float(latency_payload.get("transition_cost", float("nan")))
                weighted_transition_cost = _to_float(latency_payload.get("weighted_transition_cost", float("nan")))
                file_switch_rate = _to_float(latency_payload.get("file_switch_rate", float("nan")))
                adjacent_step_rate = _to_float(latency_payload.get("adjacent_step_rate", float("nan")))
                coarse_transfer_fraction = _to_float(latency_payload.get("coarse_transfer_fraction", float("nan")))
                coarse_fallback_fraction = _to_float(latency_payload.get("coarse_fallback_fraction", float("nan")))
                token_change_rate_coarse = _to_float(latency_payload.get("token_change_rate_coarse", float("nan")))
                token_change_rate_middle = _to_float(latency_payload.get("token_change_rate_middle", float("nan")))
                token_change_rate_fine = _to_float(latency_payload.get("token_change_rate_fine", float("nan")))
                token_change_rate_overall = _to_float(latency_payload.get("token_change_rate_overall", float("nan")))

                determinism_hashes = []
                base_hash = latency_payload.get("determinism_hash", "")
                if isinstance(base_hash, str) and base_hash:
                    determinism_hashes.append(base_hash)
                elif tokens is not None and match_indices is not None:
                    h = hashlib.sha256()
                    h.update(tokens.astype(np.int32, copy=False).tobytes(order="C"))
                    h.update(np.asarray(match_indices, dtype=np.int32).tobytes(order="C"))
                    determinism_hashes.append(h.hexdigest())

                if args.runner_cmd and not args.dry_run and args.determinism_runs > 1:
                    for rep in range(1, int(args.determinism_runs)):
                        det_dir = run_dir / f"determinism_{rep:02d}"
                        det_dir.mkdir(parents=True, exist_ok=True)
                        det_output_wav = det_dir / "output.wav"
                        det_tokens_npy = det_dir / "tokens.npy"
                        det_match_indices_npy = det_dir / "match_indices.npy"
                        det_latency_json = det_dir / "latency.json"
                        det_diagnostics_json = det_dir / "diagnostics.json"
                        det_config_json = det_dir / "config.json"
                        _write_json(det_config_json, config_payload)
                        det_context = {
                            "codec": codec_id,
                            "source": _shell_quote(str(clip.source)),
                            "reference": _shell_quote(str(clip.reference)),
                            "palette_manifest": _shell_quote(str(palette_manifest)),
                            "output_wav": _shell_quote(str(det_output_wav)),
                            "tokens_npy": _shell_quote(str(det_tokens_npy)),
                            "match_indices_npy": _shell_quote(str(det_match_indices_npy)),
                            "latency_json": _shell_quote(str(det_latency_json)),
                            "diagnostics_json": _shell_quote(str(det_diagnostics_json)),
                            "config_json": _shell_quote(str(det_config_json)),
                            "matcher": ablation["matcher"],
                            "swap": ablation["swap"],
                            "ablation_id": ab_id,
                            "temperature": codec_params["temperature"],
                            "threshold": codec_params["threshold"],
                            "continuity": codec_params["continuity"],
                            "rvq_focus": codec_params["rvq_focus"],
                            "unit": codec_params["unit"],
                            "stride": codec_params["stride"],
                            "top_k": codec_params["top_k"],
                            "seed": args.seed,
                        }
                        _run_command_template(args.runner_cmd, det_context)
                        det_hash = ""
                        if det_latency_json.exists():
                            try:
                                det_payload = json.loads(det_latency_json.read_text())
                                det_hash = str(det_payload.get("determinism_hash", ""))
                            except Exception:
                                det_hash = ""
                        if not det_hash and det_tokens_npy.exists() and det_match_indices_npy.exists():
                            h = hashlib.sha256()
                            h.update(np.load(det_tokens_npy).astype(np.int32, copy=False).tobytes(order="C"))
                            h.update(np.load(det_match_indices_npy).astype(np.int32, copy=False).tobytes(order="C"))
                            det_hash = h.hexdigest()
                        if det_hash:
                            determinism_hashes.append(det_hash)

                determinism_pass = float("nan")
                if determinism_hashes:
                    determinism_pass = float(len(set(determinism_hashes)) == 1)

                row = {
                    "codec_id": codec_id,
                    "ablation_id": ab_id,
                    "clip_id": clip.clip_id,
                    "status": "ok",
                    "fad": float("nan"),  # filled later per condition
                    "index_jitter": index_jitter,
                    "token_discontinuity": token_discontinuity,
                    "waveform_discontinuity_db": waveform_disc,
                    "chroma_difference": chroma_diff,
                    "spectral_convergence": spectral_conv,
                    "log_spectral_distance": log_spectral_dist,
                    "envelope_correlation": envelope_corr,
                    "boundary_phase_jump": phase_jump,
                    "source_onset_f1": source_onset_f1,
                    "source_onset_deviation_ms": source_onset_deviation_ms,
                    "transient_strength_correlation": transient_strength_correlation,
                    "bandwise_envelope_correlation": bandwise_envelope_correlation,
                    "boundary_click_energy": boundary_click_energy,
                    "output_to_source_distance": output_to_source_distance,
                    "output_to_nearest_palette_distance": output_to_nearest_palette_distance,
                    "palette_embedding_shift": palette_embedding_shift,
                    "nearest_palette_path": nearest_palette_path,
                    "clipping_fraction": clipping_fraction,
                    "encode_ok": float(encode_ok),
                    "decode_ok": float(decode_ok),
                    "token_layout_valid": float(token_layout_valid),
                    "duration_drift_samples": duration_drift_samples,
                    "duration_drift_ms": duration_drift_ms,
                    "channel_consistency_ok": float(channel_consistency_ok),
                    "source_channels": source_channels,
                    "output_channels": output_channels,
                    "determinism_pass": determinism_pass,
                    "failure_count": float(failure_count),
                    "retry_count": float(retry_count),
                    "palette_cache_hit": float(palette_cache_hit),
                    "palette_cache_file": palette_cache_file,
                    "encode_ms": encode_ms,
                    "decode_ms": decode_ms,
                    "total_ms": total_ms,
                    "audio_seconds": audio_seconds,
                    "end_to_end_rtf": end_to_end_rtf,
                    "objective_j": objective_j,
                    "emission_cost": emission_cost,
                    "transition_cost": transition_cost,
                    "weighted_transition_cost": weighted_transition_cost,
                    "sequence_runtime_ms": sequence_runtime_ms,
                    "file_switch_rate": file_switch_rate,
                    "adjacent_step_rate": adjacent_step_rate,
                    "coarse_transfer_fraction": coarse_transfer_fraction,
                    "coarse_fallback_fraction": coarse_fallback_fraction,
                    "token_change_rate_coarse": token_change_rate_coarse,
                    "token_change_rate_middle": token_change_rate_middle,
                    "token_change_rate_fine": token_change_rate_fine,
                    "token_change_rate_overall": token_change_rate_overall,
                    "selected_emission_mean": _to_float(diagnostics_payload.get("selected_emission_mean", float("nan"))),
                    "selected_emission_median": _to_float(diagnostics_payload.get("selected_emission_median", float("nan"))),
                    "tokens_shape": json.dumps(token_shape),
                    "output_wav": str(output_wav),
                    "tokens_npy": str(tokens_npy) if tokens_npy.exists() else "",
                    "match_indices_npy": str(match_indices_npy) if match_indices_npy.exists() else "",
                    "latency_json": str(latency_json) if latency_json.exists() else "",
                    "diagnostics_json": str(diagnostics_json) if diagnostics_json.exists() else "",
                    "config_json": str(config_json),
                }
                rows.append(row)
                refs_for_fad[codec_id].append(clip.reference.resolve())
                generated_for_fad[condition_key].append(output_wav.resolve())

    fad_status: Dict[str, str] = {}
    for codec_id in selected_codecs:
        for ablation in selected_ablation_defs:
            ab_id = ablation["id"]
            condition_key = f"{codec_id}::{ab_id}"
            reference_wavs = _unique_paths(refs_for_fad.get(codec_id, []))
            generated_wavs = _unique_paths(generated_for_fad.get(condition_key, []))
            fad_payload = metric_fad(
                reference_wavs=reference_wavs,
                generated_wavs=generated_wavs,
            )
            fad_score = float(fad_payload.get("score", float("nan")))
            fad_status[condition_key] = str(fad_payload.get("status", "unknown"))
            for row in rows:
                if row.get("codec_id") == codec_id and row.get("ablation_id") == ab_id:
                    row["fad"] = fad_score

    quality_metric_names = ["fad", "spectral_convergence", "log_spectral_distance"]
    structure_metric_names = [
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
    all_metrics = quality_metric_names + structure_metric_names
    summary = {
        "quality_metrics": {},
        "structure_metrics": {},
        "system_health": {
            "leakage": leakage,
            "fad_status": fad_status,
            "conditions": {},
            "gates": {},
        },
        "significance": {},
        "latency": {},
    }

    for codec_id in selected_codecs:
        summary["quality_metrics"].setdefault(codec_id, {})
        summary["structure_metrics"].setdefault(codec_id, {})
        summary["latency"].setdefault(codec_id, {})
        rows_codec = [r for r in rows if r.get("codec_id") == codec_id]
        for ablation in selected_ablation_defs:
            ab_id = ablation["id"]
            cond_rows = [r for r in rows_codec if r.get("ablation_id") == ab_id and r.get("status") == "ok"]

            quality_summary = {}
            for m in quality_metric_names:
                vals = [float(r[m]) for r in cond_rows if not math.isnan(float(r[m]))]
                if not vals:
                    quality_summary[m] = {"mean": float("nan"), "std": float("nan"), "ci95": [float("nan"), float("nan")], "n": 0}
                else:
                    lo, hi = _bootstrap_ci(vals, alpha=0.05, n_boot=args.bootstrap, seed=args.seed)
                    quality_summary[m] = {
                        "mean": float(np.mean(vals)),
                        "median": float(np.median(vals)),
                        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                        "ci95": [lo, hi],
                        "n": len(vals),
                    }
            summary["quality_metrics"][codec_id][ab_id] = quality_summary

            structure_summary = {}
            for m in structure_metric_names:
                vals = [float(r[m]) for r in cond_rows if not math.isnan(float(r[m]))]
                if not vals:
                    structure_summary[m] = {"mean": float("nan"), "std": float("nan"), "ci95": [float("nan"), float("nan")], "n": 0}
                else:
                    lo, hi = _bootstrap_ci(vals, alpha=0.05, n_boot=args.bootstrap, seed=args.seed)
                    structure_summary[m] = {
                        "mean": float(np.mean(vals)),
                        "median": float(np.median(vals)),
                        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                        "ci95": [lo, hi],
                        "n": len(vals),
                    }
            summary["structure_metrics"][codec_id][ab_id] = structure_summary

            encode_ms = [float(r["encode_ms"]) for r in cond_rows if not math.isnan(float(r["encode_ms"]))]
            decode_ms = [float(r["decode_ms"]) for r in cond_rows if not math.isnan(float(r["decode_ms"]))]
            rtf = [float(r["end_to_end_rtf"]) for r in cond_rows if not math.isnan(float(r["end_to_end_rtf"]))]
            summary["latency"][codec_id][ab_id] = {
                "encode_ms_p50": float(np.percentile(encode_ms, 50)) if encode_ms else float("nan"),
                "encode_ms_p95": float(np.percentile(encode_ms, 95)) if encode_ms else float("nan"),
                "decode_ms_p50": float(np.percentile(decode_ms, 50)) if decode_ms else float("nan"),
                "decode_ms_p95": float(np.percentile(decode_ms, 95)) if decode_ms else float("nan"),
                "rtf_mean": float(np.mean(rtf)) if rtf else float("nan"),
                "rtf_p95": float(np.percentile(rtf, 95)) if rtf else float("nan"),
                "n": len(rtf),
            }

            condition_health = {
                "n_ok": len(cond_rows),
                "encode_success_rate": _mean(float(r["encode_ok"]) for r in cond_rows),
                "decode_success_rate": _mean(float(r["decode_ok"]) for r in cond_rows),
                "token_layout_valid_rate": _mean(float(r["token_layout_valid"]) for r in cond_rows),
                "channel_consistency_rate": _mean(float(r["channel_consistency_ok"]) for r in cond_rows),
                "determinism_pass_rate": _mean(float(r["determinism_pass"]) for r in cond_rows),
                "envelope_correlation_mean": _mean(float(r["envelope_correlation"]) for r in cond_rows),
                "clipping_fraction_max": (
                    float(np.max([_to_float(r.get("clipping_fraction", float("nan"))) for r in cond_rows if not math.isnan(_to_float(r.get("clipping_fraction", float("nan"))))]))
                    if any(not math.isnan(_to_float(r.get("clipping_fraction", float("nan")))) for r in cond_rows)
                    else float("nan")
                ),
                "duration_drift_abs_ms_p95": (
                    float(np.percentile([abs(float(r["duration_drift_ms"])) for r in cond_rows if not math.isnan(float(r["duration_drift_ms"]))], 95))
                    if any(not math.isnan(float(r["duration_drift_ms"])) for r in cond_rows)
                    else float("nan")
                ),
                "failure_count_sum": float(np.sum([float(r["failure_count"]) for r in cond_rows])) if cond_rows else 0.0,
                "retry_count_sum": float(np.sum([float(r["retry_count"]) for r in cond_rows])) if cond_rows else 0.0,
            }
            summary["system_health"]["conditions"][f"{codec_id}::{ab_id}"] = condition_health

        rows_codec_for_sig = [r for r in rows if r.get("codec_id") == codec_id]
        summary["significance"][codec_id] = {}
        for m in structure_metric_names:
            summary["significance"][codec_id][m] = _paired_significance(rows_codec_for_sig, m, baseline_id="greedy_full_layer")

    condition_keys = sorted(summary["system_health"]["conditions"].keys())
    for key in condition_keys:
        h = summary["system_health"]["conditions"][key]
        gate_pass = (
            bool(leakage["ok"])
            and _to_float(h.get("encode_success_rate", float("nan"))) >= 1.0
            and _to_float(h.get("decode_success_rate", float("nan"))) >= 1.0
            and _to_float(h.get("token_layout_valid_rate", float("nan"))) >= 1.0
            and (
                math.isnan(_to_float(h.get("duration_drift_abs_ms_p95", float("nan"))))
                or _to_float(h.get("duration_drift_abs_ms_p95", float("nan"))) <= float(args.duration_drift_gate_ms)
            )
            and (
                math.isnan(_to_float(h.get("determinism_pass_rate", float("nan"))))
                or _to_float(h.get("determinism_pass_rate", float("nan"))) >= float(args.determinism_gate_rate)
            )
            and (
                math.isnan(_to_float(h.get("envelope_correlation_mean", float("nan"))))
                or _to_float(h.get("envelope_correlation_mean", float("nan"))) >= float(args.envelope_corr_gate)
            )
            and (
                math.isnan(_to_float(h.get("clipping_fraction_max", float("nan"))))
                or _to_float(h.get("clipping_fraction_max", float("nan"))) <= float(args.clipping_gate_fraction)
            )
        )
        summary["system_health"]["gates"][key] = {"pass": bool(gate_pass)}

    conditions_for_ranking = []
    for key in condition_keys:
        codec_id, ab_id = key.split("::", 1)
        quality_means = {m: _to_float(summary["quality_metrics"][codec_id][ab_id][m]["mean"]) for m in quality_metric_names}
        structure_means = {m: _to_float(summary["structure_metrics"][codec_id][ab_id][m]["mean"]) for m in structure_metric_names}
        conditions_for_ranking.append(
            {
                "condition_key": key,
                "codec_id": codec_id,
                "ablation_id": ab_id,
                "quality": quality_means,
                "structure": structure_means,
                "gate_pass": bool(summary["system_health"]["gates"][key]["pass"]),
            }
        )

    lower_better = {
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
    metric_values_by_name: Dict[str, List[float]] = {m: [] for m in lower_better}
    for cond in conditions_for_ranking:
        for m in quality_metric_names:
            v = cond["quality"][m]
            if not math.isnan(v):
                metric_values_by_name[m].append(v)
        for m in structure_metric_names:
            v = cond["structure"][m]
            if not math.isnan(v):
                metric_values_by_name[m].append(v)

    metric_minmax = {}
    for m, vals in metric_values_by_name.items():
        if vals:
            metric_minmax[m] = (min(vals), max(vals))
        else:
            metric_minmax[m] = (float("nan"), float("nan"))

    ranked_presets = []
    for cond in conditions_for_ranking:
        def score_metric(metric_name: str, raw: float) -> float:
            lo, hi = metric_minmax[metric_name]
            if math.isnan(raw) or math.isnan(lo) or math.isnan(hi):
                return float("nan")
            if abs(hi - lo) < 1e-12:
                return 1.0
            if lower_better[metric_name]:
                return float((hi - raw) / (hi - lo))
            return float((raw - lo) / (hi - lo))

        quality_scores = [score_metric(m, cond["quality"][m]) for m in quality_metric_names]
        structure_scores = [score_metric(m, cond["structure"][m]) for m in structure_metric_names]
        q = _mean(quality_scores)
        s = _mean(structure_scores)
        composite = _mean([q, s])
        ranked_presets.append(
            {
                "condition_key": cond["condition_key"],
                "codec_id": cond["codec_id"],
                "ablation_id": cond["ablation_id"],
                "gate_pass": cond["gate_pass"],
                "quality_score": q,
                "structure_score": s,
                "composite_score": composite,
            }
        )

    ranked_presets.sort(
        key=lambda r: (
            0 if r["gate_pass"] else 1,
            -(r["composite_score"] if not math.isnan(r["composite_score"]) else -1e9),
        )
    )
    top_n = max(1, int(args.top_n_presets))
    ranked_presets = ranked_presets[:top_n]

    for preset in ranked_presets:
        codec_id = preset["codec_id"]
        ab_id = preset["ablation_id"]
        matching_rows = [r for r in rows if r.get("codec_id") == codec_id and r.get("ablation_id") == ab_id and r.get("status") == "ok"]
        params = {}
        if matching_rows:
            cfg_path = Path(matching_rows[0].get("config_json", ""))
            if cfg_path.exists():
                try:
                    cfg = json.loads(cfg_path.read_text())
                    params = cfg.get("params", {})
                except Exception:
                    params = {}
        preset["params"] = params

    report_dir = out_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    csv_path = report_dir / "per_clip_metrics.csv"
    keys = [
        "codec_id",
        "ablation_id",
        "clip_id",
        "status",
        "fad",
        "index_jitter",
        "token_discontinuity",
        "waveform_discontinuity_db",
        "chroma_difference",
        "spectral_convergence",
        "log_spectral_distance",
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
        "nearest_palette_path",
        "clipping_fraction",
        "encode_ok",
        "decode_ok",
        "token_layout_valid",
        "duration_drift_samples",
        "duration_drift_ms",
        "channel_consistency_ok",
        "source_channels",
        "output_channels",
        "determinism_pass",
        "failure_count",
        "retry_count",
        "palette_cache_hit",
        "palette_cache_file",
        "encode_ms",
        "decode_ms",
        "total_ms",
        "audio_seconds",
        "end_to_end_rtf",
        "objective_j",
        "emission_cost",
        "transition_cost",
        "weighted_transition_cost",
        "sequence_runtime_ms",
        "file_switch_rate",
        "adjacent_step_rate",
        "coarse_transfer_fraction",
        "coarse_fallback_fraction",
        "token_change_rate_coarse",
        "token_change_rate_middle",
        "token_change_rate_fine",
        "token_change_rate_overall",
        "selected_emission_mean",
        "selected_emission_median",
        "tokens_shape",
        "output_wav",
        "tokens_npy",
        "match_indices_npy",
        "latency_json",
        "diagnostics_json",
        "config_json",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})

    _write_json(report_dir / "summary.json", summary)
    _write_json(report_dir / "ranked_presets.json", {"presets": ranked_presets})
    _write_json(
        report_dir / "reproducibility.json",
        {
            "manifest": str(manifest_path.resolve()),
            "palette_count": len(palette),
            "source_eval_count": len(clips),
            "ablations": selected_ablations,
            "codecs": selected_codecs,
            "runner_cmd": args.runner_cmd or "",
            "seed": args.seed,
            "bootstrap": args.bootstrap,
            "determinism_runs": args.determinism_runs,
            "palette_metric_limit": args.palette_metric_limit,
            "params_cli_overrides": {
                "temperature": args.temperature,
                "threshold": args.threshold,
                "continuity": args.continuity,
                "rvq_focus": args.rvq_focus,
                "unit": args.unit,
                "stride": args.stride,
                "top_k": args.top_k,
            },
            "params_by_codec": {codec_id: _resolve_runtime_params(codec_id, args) for codec_id in selected_codecs},
        },
    )
    print(f"Wrote reports to: {report_dir}")
    print(f"Per-clip CSV: {csv_path}")


def _make_evaluate_namespace(base_args: argparse.Namespace, output_dir: Path, params: dict) -> argparse.Namespace:
    return argparse.Namespace(
        manifest=base_args.manifest,
        output_dir=str(output_dir),
        runner_cmd=base_args.runner_cmd,
        dry_run=base_args.dry_run,
        seed=base_args.seed,
        bootstrap=base_args.bootstrap,
        codecs=base_args.codecs,
        ablations=base_args.ablations,
        determinism_runs=base_args.determinism_runs,
        duration_drift_gate_ms=base_args.duration_drift_gate_ms,
        determinism_gate_rate=base_args.determinism_gate_rate,
        envelope_corr_gate=base_args.envelope_corr_gate,
        clipping_gate_fraction=base_args.clipping_gate_fraction,
        top_n_presets=base_args.top_n_presets,
        clip_limit=base_args.clip_limit,
        palette_metric_limit=base_args.palette_metric_limit,
        temperature=params["temperature"],
        threshold=params["threshold"],
        continuity=params["continuity"],
        rvq_focus=params["rvq_focus"],
        unit=params["unit"],
        stride=params["stride"],
        top_k=params["top_k"],
    )


def _sample_trial_params(rng: random.Random) -> dict:
    unit = rng.randint(1, 8)
    stride = rng.randint(1, unit)
    return {
        "temperature": rng.uniform(0.1, 1.5),
        "threshold": rng.uniform(0.2, 1.6),
        "continuity": rng.uniform(0.0, 1.0),
        "rvq_focus": rng.uniform(0.0, 1.0),
        "unit": unit,
        "stride": stride,
        "top_k": rng.randint(1, 8),
    }


def search(args: argparse.Namespace) -> None:
    if not args.runner_cmd:
        raise ValueError("--runner-cmd is required for `search`.")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    trials = max(1, int(args.trials))
    rows = []

    for trial_idx in range(trials):
        params = _sample_trial_params(rng)
        trial_dir = out_root / "trials" / f"trial_{trial_idx:04d}"
        eval_args = _make_evaluate_namespace(args, trial_dir, params)
        evaluate(eval_args)

        ranked_path = trial_dir / "reports" / "ranked_presets.json"
        best_score = float("nan")
        best_gate = False
        if ranked_path.exists():
            payload = json.loads(ranked_path.read_text())
            presets = payload.get("presets", [])
            if presets:
                best_score = _to_float(presets[0].get("composite_score", float("nan")))
                best_gate = bool(presets[0].get("gate_pass", False))
        rows.append(
            {
                "trial_id": trial_idx,
                "score": best_score,
                "gate_pass": best_gate,
                "params": params,
                "trial_dir": str(trial_dir),
            }
        )
        print(f"[search] trial={trial_idx} score={best_score:.6f} gate_pass={best_gate} params={params}")

    rows.sort(
        key=lambda r: (
            0 if r["gate_pass"] else 1,
            -(r["score"] if not math.isnan(float(r["score"])) else -1e9),
        )
    )
    top_n = max(1, int(args.top_n_presets))
    out_payload = {
        "trials": rows,
        "top_presets": rows[:top_n],
        "seed": args.seed,
        "trials_count": trials,
    }
    _write_json(out_root / "search_results.json", out_payload)
    print(f"Wrote search results: {out_root / 'search_results.json'}")


def validate(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    palette, clips, _ = _parse_manifest(manifest_path)
    leakage = _check_manifest_leakage(palette, clips)

    result = {
        "manifest": str(manifest_path.resolve()),
        "leakage": leakage,
        "runs_health": {},
    }

    if args.output_dir:
        out_root = Path(args.output_dir)
        csv_path = out_root / "reports" / "per_clip_metrics.csv"
        if csv_path.exists():
            rows = _parse_csv_rows(csv_path)
            total = len(rows)
            ok = [r for r in rows if r.get("status") == "ok"]
            encode_rate = _mean(_to_float(r.get("encode_ok", float("nan"))) for r in ok)
            decode_rate = _mean(_to_float(r.get("decode_ok", float("nan"))) for r in ok)
            layout_rate = _mean(_to_float(r.get("token_layout_valid", float("nan"))) for r in ok)
            det_rate = _mean(_to_float(r.get("determinism_pass", float("nan"))) for r in ok)
            drift_vals = [abs(_to_float(r.get("duration_drift_ms", float("nan")))) for r in ok if not math.isnan(_to_float(r.get("duration_drift_ms", float("nan"))))]
            env_corr = _mean(_to_float(r.get("envelope_correlation", float("nan"))) for r in ok)
            clip_vals = [_to_float(r.get("clipping_fraction", float("nan"))) for r in ok if not math.isnan(_to_float(r.get("clipping_fraction", float("nan"))))]
            result["runs_health"] = {
                "rows_total": total,
                "rows_ok": len(ok),
                "encode_success_rate": encode_rate,
                "decode_success_rate": decode_rate,
                "token_layout_valid_rate": layout_rate,
                "determinism_pass_rate": det_rate,
                "duration_drift_abs_ms_p95": float(np.percentile(drift_vals, 95)) if drift_vals else float("nan"),
                "envelope_correlation_mean": env_corr,
                "clipping_fraction_max": float(np.max(clip_vals)) if clip_vals else float("nan"),
            }
        else:
            result["runs_health"] = {"status": "missing_per_clip_metrics_csv", "path": str(csv_path)}

    _write_json(Path(args.report_json), result)
    print(f"Wrote validation report: {args.report_json}")


def export_presets(args: argparse.Namespace) -> None:
    report_dir = Path(args.report_dir)
    summary_path = report_dir / "summary.json"
    csv_path = report_dir / "per_clip_metrics.csv"
    ranked_path = report_dir / "ranked_presets.json"
    if not summary_path.exists() or not csv_path.exists():
        raise FileNotFoundError("Missing summary/per_clip reports. Run evaluate first.")

    summary = json.loads(summary_path.read_text())
    rows = _parse_csv_rows(csv_path)
    conditions = summary.get("system_health", {}).get("conditions", {})
    gates = summary.get("system_health", {}).get("gates", {})
    ranked = []
    if ranked_path.exists():
        try:
            ranked = json.loads(ranked_path.read_text()).get("presets", [])
        except Exception:
            ranked = []

    presets = []
    if ranked:
        for item in ranked:
            key = str(item.get("condition_key", ""))
            if "::" not in key:
                continue
            codec_id, ablation_id = key.split("::", 1)
            cond_rows = [r for r in rows if r.get("codec_id") == codec_id and r.get("ablation_id") == ablation_id and r.get("status") == "ok"]
            if not cond_rows:
                continue
            health = conditions.get(key, {})
            cfg_path = Path(cond_rows[0].get("config_json", ""))
            params = dict(item.get("params", {}))
            if not params and cfg_path.exists():
                try:
                    params = json.loads(cfg_path.read_text()).get("params", {})
                except Exception:
                    params = {}
            presets.append(
                {
                    "condition_key": key,
                    "codec_id": codec_id,
                    "ablation_id": ablation_id,
                    "gate_pass": bool(item.get("gate_pass", gates.get(key, {}).get("pass", False))),
                    "quality_score": _to_float(item.get("quality_score", float("nan"))),
                    "structure_score": _to_float(item.get("structure_score", float("nan"))),
                    "composite_score": _to_float(item.get("composite_score", float("nan"))),
                    "quality_metrics": summary.get("quality_metrics", {}).get(codec_id, {}).get(ablation_id, {}),
                    "structure_metrics": summary.get("structure_metrics", {}).get(codec_id, {}).get(ablation_id, {}),
                    "system_health": health,
                    "params": params,
                }
            )
    else:
        for key, health in conditions.items():
            codec_id, ablation_id = key.split("::", 1)
            cond_rows = [r for r in rows if r.get("codec_id") == codec_id and r.get("ablation_id") == ablation_id and r.get("status") == "ok"]
            if not cond_rows:
                continue
            cfg_path = Path(cond_rows[0].get("config_json", ""))
            params = {}
            if cfg_path.exists():
                try:
                    params = json.loads(cfg_path.read_text()).get("params", {})
                except Exception:
                    params = {}
            presets.append(
                {
                    "condition_key": key,
                    "codec_id": codec_id,
                    "ablation_id": ablation_id,
                    "gate_pass": bool(gates.get(key, {}).get("pass", False)),
                    "quality_score": float("nan"),
                    "structure_score": float("nan"),
                    "composite_score": float("nan"),
                    "quality_metrics": summary.get("quality_metrics", {}).get(codec_id, {}).get(ablation_id, {}),
                    "structure_metrics": summary.get("structure_metrics", {}).get(codec_id, {}).get(ablation_id, {}),
                    "system_health": health,
                    "params": params,
                }
            )

    presets.sort(
        key=lambda p: (
            0 if p["gate_pass"] else 1,
            -(p["composite_score"] if not math.isnan(_to_float(p["composite_score"])) else -1e9),
        )
    )
    top_n = max(1, int(args.top_n))
    _write_json(Path(args.output_json), {"presets": presets[:top_n], "source_report_dir": str(report_dir)})
    print(f"Wrote presets: {args.output_json}")


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
            "{match_indices_npy} {latency_json} {diagnostics_json} {config_json} {matcher} {swap} {ablation_id}"
        ),
    )
    ev.add_argument("--dry-run", action="store_true", help="Print expanded runner commands without executing.")
    ev.add_argument("--seed", type=int, default=1234, help="Seed for bootstrap CI.")
    ev.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap iterations for CI.")
    ev.add_argument("--codecs", default="dac,spectrostream", help="Comma-separated codecs to evaluate.")
    ev.add_argument("--ablations", default="greedy_full_layer,greedy_rvq_group,beam_full_layer,beam_rvq_group", help="Comma-separated ablation ids to evaluate.")
    ev.add_argument("--determinism-runs", type=int, default=2, help="Repeated runs per condition/clip for determinism check.")
    ev.add_argument("--resume-existing", action="store_true", help="Reuse complete per-clip runner outputs instead of regenerating them.")
    ev.add_argument("--duration-drift-gate-ms", type=float, default=120.0, help="Gate threshold for duration drift abs p95.")
    ev.add_argument("--determinism-gate-rate", type=float, default=1.0, help="Gate threshold for determinism pass rate.")
    ev.add_argument("--envelope-corr-gate", type=float, default=0.90, help="Minimum envelope correlation mean for health gate.")
    ev.add_argument("--clipping-gate-fraction", type=float, default=1e-4, help="Maximum clipping fraction for health gate.")
    ev.add_argument("--top-n-presets", type=int, default=8, help="Top-N ranked presets exported from evaluation.")
    ev.add_argument("--clip-limit", type=int, default=0, help="Optional cap on number of source_eval clips (0=all).")
    ev.add_argument("--palette-metric-limit", type=int, default=64, help="Palette clips to cache for nearest-palette transfer metrics.")
    ev.add_argument("--temperature", type=float, default=None, help="Global override (default: tuned per codec)")
    ev.add_argument("--threshold", type=float, default=None, help="Global override (default: tuned per codec)")
    ev.add_argument("--continuity", type=float, default=None, help="Global override (default: tuned per codec)")
    ev.add_argument("--rvq-focus", dest="rvq_focus", type=float, default=None, help="Global override (default: tuned per codec)")
    ev.add_argument("--unit", type=int, default=None, help="Global override (default: tuned per codec)")
    ev.add_argument("--stride", type=int, default=None, help="Global override (default: tuned per codec)")
    ev.add_argument("--top-k", dest="top_k", type=int, default=None, help="Global override (default: tuned per codec)")
    ev.set_defaults(func=evaluate)

    search_p = sub.add_parser("search", help="Random-search hyperparameters and rank presets with system-health gates.")
    search_p.add_argument("--manifest", required=True, help="Manifest JSON from `prepare`")
    search_p.add_argument("--output-dir", required=True, help="Output directory for search artifacts")
    search_p.add_argument("--runner-cmd", required=True, help="Runner command template (same placeholders as evaluate)")
    search_p.add_argument("--trials", type=int, default=12, help="Number of random trials")
    search_p.add_argument("--seed", type=int, default=1234, help="Seed for sampling/search")
    search_p.add_argument("--bootstrap", type=int, default=500, help="Bootstrap iterations per trial")
    search_p.add_argument("--codecs", default="dac,spectrostream")
    search_p.add_argument("--ablations", default="greedy_full_layer,greedy_rvq_group,beam_full_layer,beam_rvq_group")
    search_p.add_argument("--determinism-runs", type=int, default=2)
    search_p.add_argument("--duration-drift-gate-ms", type=float, default=120.0)
    search_p.add_argument("--determinism-gate-rate", type=float, default=1.0)
    search_p.add_argument("--envelope-corr-gate", type=float, default=0.90)
    search_p.add_argument("--clipping-gate-fraction", type=float, default=1e-4)
    search_p.add_argument("--top-n-presets", type=int, default=8)
    search_p.add_argument("--clip-limit", type=int, default=0)
    search_p.add_argument("--palette-metric-limit", type=int, default=64)
    search_p.add_argument("--dry-run", action="store_true")
    search_p.set_defaults(func=search)

    validate_p = sub.add_parser("validate", help="Framework/system-health validation (leakage + run outputs).")
    validate_p.add_argument("--manifest", required=True, help="Manifest JSON from `prepare`")
    validate_p.add_argument("--output-dir", default="", help="Optional evaluate output-dir to validate run artifacts.")
    validate_p.add_argument("--report-json", required=True, help="Validation report path")
    validate_p.set_defaults(func=validate)

    export_p = sub.add_parser("export-presets", help="Export ranked presets from existing evaluate reports.")
    export_p.add_argument("--report-dir", required=True, help="Directory containing summary.json and per_clip_metrics.csv")
    export_p.add_argument("--output-json", required=True, help="Output presets JSON")
    export_p.add_argument("--top-n", type=int, default=8, help="Top-N presets to export")
    export_p.set_defaults(func=export_presets)

    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
