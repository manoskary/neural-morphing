#!/usr/bin/env python3
"""Unit tests for multi-metric evaluation helpers."""

from __future__ import annotations

import tempfile
import unittest
import argparse
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
    _resolve_runtime_params,
    metric_clipping_fraction,
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

    def test_clipping_fraction(self):
        unclipped = np.linspace(-0.8, 0.8, 1000, dtype=np.float32)
        clipped = np.clip(np.linspace(-1.2, 1.2, 1000, dtype=np.float32), -1.0, 1.0)
        frac_unclipped = metric_clipping_fraction(unclipped)
        frac_clipped = metric_clipping_fraction(clipped)
        self.assertTrue(np.isfinite(frac_unclipped))
        self.assertTrue(np.isfinite(frac_clipped))
        self.assertLess(frac_unclipped, frac_clipped)

    def test_codec_runtime_defaults_resolution(self):
        args = argparse.Namespace(
            temperature=None,
            threshold=None,
            continuity=None,
            rvq_focus=None,
            unit=None,
            stride=None,
            top_k=None,
        )
        dac = _resolve_runtime_params("dac", args)
        spectro = _resolve_runtime_params("spectrostream", args)
        self.assertEqual(dac["unit"], 7)
        self.assertEqual(dac["top_k"], 7)
        self.assertEqual(spectro["unit"], 2)
        self.assertEqual(spectro["top_k"], 8)
        self.assertNotEqual(dac["threshold"], spectro["threshold"])

    def test_codec_runtime_override_resolution(self):
        args = argparse.Namespace(
            temperature=0.8,
            threshold=0.9,
            continuity=0.2,
            rvq_focus=0.7,
            unit=4,
            stride=3,
            top_k=5,
        )
        resolved = _resolve_runtime_params("spectrostream", args)
        self.assertEqual(resolved["temperature"], 0.8)
        self.assertEqual(resolved["threshold"], 0.9)
        self.assertEqual(resolved["continuity"], 0.2)
        self.assertEqual(resolved["rvq_focus"], 0.7)
        self.assertEqual(resolved["unit"], 4)
        self.assertEqual(resolved["stride"], 3)
        self.assertEqual(resolved["top_k"], 5)


if __name__ == "__main__":
    unittest.main()
