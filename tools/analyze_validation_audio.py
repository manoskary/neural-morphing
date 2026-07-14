#!/usr/bin/env python3
import argparse
import csv
import hashlib
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def load_mono(path: Path, sample_rate: int | None = None):
    audio, sr = sf.read(path, always_2d=True, dtype="float32")
    audio = audio.mean(axis=1)
    if sample_rate and sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate
    return np.asarray(audio, dtype=np.float32), sr


def features(audio: np.ndarray, sr: int):
    rms = float(np.sqrt(np.mean(audio * audio) + 1.0e-12))
    peak = float(np.max(np.abs(audio)) + 1.0e-12)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64, fmax=sr / 2)
    mel_db = librosa.power_to_db(mel + 1.0e-12, ref=1.0)
    frame_rms = librosa.feature.rms(y=audio, frame_length=1024, hop_length=256)[0]
    return {
        "rms_db": 20.0 * np.log10(rms),
        "peak_db": 20.0 * np.log10(peak),
        "crest_db": 20.0 * np.log10(peak / rms),
        "centroid_hz": float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))),
        "flatness": float(np.mean(librosa.feature.spectral_flatness(y=audio))),
        "onset_strength": float(np.mean(librosa.onset.onset_strength(y=audio, sr=sr))),
        "silent_frames_pct": 100.0 * float(np.mean(frame_rms < 10.0 ** (-60.0 / 20.0))),
        "max_sample_delta": float(np.max(np.abs(np.diff(audio)))) if len(audio) > 1 else 0.0,
        "clip_samples": int(np.sum(np.abs(audio) >= 0.999)),
        "mel_db": mel_db,
        "envelope": frame_rms,
    }


def distance(first: np.ndarray, second: np.ndarray):
    frames = min(first.shape[-1], second.shape[-1])
    return float(np.mean(np.abs(first[..., :frames] - second[..., :frames])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--palette", type=Path, required=True)
    parser.add_argument("--renders", type=Path, required=True)
    parser.add_argument("--pattern", default="*.wav")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    render_paths = sorted(args.renders.glob(args.pattern))
    if not render_paths:
        raise SystemExit("No render files found")

    _, target_sr = load_mono(render_paths[0])
    source = features(*load_mono(args.source, target_sr))
    palette = features(*load_mono(args.palette, target_sr))
    baseline = None
    rows = []
    for path in render_paths:
        audio, sr = load_mono(path, target_sr)
        current = features(audio, sr)
        if "baseline" in path.stem:
            baseline = current
        rows.append((path, audio, current))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = ["name", "sha256", "rms_db", "peak_db", "crest_db", "centroid_hz", "flatness",
                      "onset_strength", "silent_frames_pct", "max_sample_delta", "clip_samples",
                      "mel_distance_source_db", "mel_distance_palette_db", "mel_distance_baseline_db"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for path, audio, current in rows:
            row = {key: current[key] for key in fieldnames if key in current}
            row.update({
                "name": path.stem,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
                "mel_distance_source_db": distance(current["mel_db"], source["mel_db"]),
                "mel_distance_palette_db": distance(current["mel_db"], palette["mel_db"]),
                "mel_distance_baseline_db": distance(current["mel_db"], baseline["mel_db"]) if baseline else 0.0,
            })
            writer.writerow(row)


if __name__ == "__main__":
    main()
