#!/usr/bin/env python3
"""Unit tests for multi-metric evaluation helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
import soundfile as sf

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from evaluate_morphing import (
    ClipEntry,
    _check_manifest_leakage,
    metric_envelope_correlation,
    metric_log_spectral_distance,
    metric_spectral_convergence,
)


class MultiMetricTests(unittest.TestCase):
    def setUp(self):
        self.sr = 16000
        t = np.arange(self.sr, dtype=np.float32) / self.sr
        self.ref = 0.2 * np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)
        self.out_same = self.ref.copy()
        self.out_diff = 0.2 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)

    def test_spectral_convergence_identity(self):
        value = metric_spectral_convergence(self.out_same, self.ref, self.sr)
        self.assertTrue(np.isfinite(value))
        self.assertLess(value, 1e-6)

    def test_log_spectral_distance_identity(self):
        value = metric_log_spectral_distance(self.out_same, self.ref, self.sr)
        self.assertTrue(np.isfinite(value))
        self.assertLess(value, 1e-6)

    def test_envelope_corr_identity_vs_diff(self):
        same = metric_envelope_correlation(self.out_same, self.ref)
        diff = metric_envelope_correlation(self.out_diff, self.ref)
        self.assertTrue(np.isfinite(same))
        self.assertGreater(same, 0.99)
        self.assertTrue(np.isnan(diff) or diff <= same)

    def test_manifest_leakage_hash_overlap(self):
        with tempfile.TemporaryDirectory(prefix="nm_eval_test_") as td:
            root = Path(td)
            palette_wav = root / "palette.wav"
            source_wav = root / "source.wav"
            ref_wav = root / "ref.wav"

            sf.write(palette_wav, self.ref, self.sr)
            sf.write(source_wav, self.ref, self.sr)  # same content on purpose
            sf.write(ref_wav, self.out_diff, self.sr)

            clips = [ClipEntry(clip_id="c1", source=source_wav, reference=ref_wav)]
            report = _check_manifest_leakage([palette_wav], clips)
            self.assertFalse(report["ok"])
            self.assertGreaterEqual(len(report["hash_overlap_palette_source"]), 1)


if __name__ == "__main__":
    unittest.main()
