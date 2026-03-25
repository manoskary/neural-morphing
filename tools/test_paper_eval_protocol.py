#!/usr/bin/env python3
"""Unit tests for paper evaluation protocol utilities."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from paper_eval_protocol import AggregateOptions, aggregate_reports, benjamini_hochberg


class PaperProtocolTests(unittest.TestCase):
    def test_bh_correction_monotonic(self):
        pvals = [0.04, 0.01, 0.20, float("nan")]
        qvals = benjamini_hochberg(pvals)
        self.assertEqual(len(qvals), 4)
        self.assertTrue(qvals[1] <= qvals[0])
        self.assertTrue(qvals[2] >= qvals[0])

    def test_aggregate_reports_smoke(self):
        with tempfile.TemporaryDirectory(prefix="nm_paper_agg_") as td:
            root = Path(td)
            run_dir = root / "seed_1234"
            reports = run_dir / "reports"
            reports.mkdir(parents=True, exist_ok=True)

            rows = []
            for clip_id, base_jitter, beam_jitter in (("c1", 10.0, 8.0), ("c2", 12.0, 9.0), ("c3", 14.0, 10.0)):
                rows.append(
                    {
                        "codec_id": "dac",
                        "ablation_id": "greedy_full_layer",
                        "clip_id": clip_id,
                        "status": "ok",
                        "fad": "nan",
                        "spectral_convergence": "0.3",
                        "log_spectral_distance": "8.0",
                        "index_jitter": str(base_jitter),
                        "token_discontinuity": "0.9",
                        "waveform_discontinuity_db": "4.0",
                        "chroma_difference": "0.05",
                        "envelope_correlation": "0.95",
                        "boundary_phase_jump": "1.0",
                        "encode_ok": "1",
                        "decode_ok": "1",
                        "token_layout_valid": "1",
                        "channel_consistency_ok": "1",
                        "determinism_pass": "1",
                        "duration_drift_ms": "0.0",
                        "clipping_fraction": "0.0",
                        "failure_count": "0",
                        "retry_count": "0",
                        "encode_ms": "10.0",
                        "decode_ms": "11.0",
                        "end_to_end_rtf": "0.8",
                    }
                )
                rows.append(
                    {
                        "codec_id": "dac",
                        "ablation_id": "beam_full_layer",
                        "clip_id": clip_id,
                        "status": "ok",
                        "fad": "nan",
                        "spectral_convergence": "0.25",
                        "log_spectral_distance": "7.5",
                        "index_jitter": str(beam_jitter),
                        "token_discontinuity": "0.85",
                        "waveform_discontinuity_db": "3.8",
                        "chroma_difference": "0.04",
                        "envelope_correlation": "0.96",
                        "boundary_phase_jump": "0.9",
                        "encode_ok": "1",
                        "decode_ok": "1",
                        "token_layout_valid": "1",
                        "channel_consistency_ok": "1",
                        "determinism_pass": "1",
                        "duration_drift_ms": "0.0",
                        "clipping_fraction": "0.0",
                        "failure_count": "0",
                        "retry_count": "0",
                        "encode_ms": "9.0",
                        "decode_ms": "10.0",
                        "end_to_end_rtf": "0.7",
                    }
                )

            csv_path = reports / "per_clip_metrics.csv"
            with csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            (reports / "summary.json").write_text(
                json.dumps(
                    {
                        "system_health": {
                            "leakage": {"ok": True},
                            "fad_status": {
                                "dac::greedy_full_layer": "missing_dependency",
                                "dac::beam_full_layer": "missing_dependency",
                            },
                        }
                    }
                )
            )
            (reports / "reproducibility.json").write_text(json.dumps({"seed": 1234, "manifest": "dummy.json"}))

            out_dir = root / "paper_reports"
            summary = aggregate_reports(
                run_dirs=[run_dir],
                output_dir=out_dir,
                options=AggregateOptions(bootstrap=100, seed=1234),
            )
            self.assertTrue((out_dir / "aggregated_summary.json").exists())
            self.assertTrue((out_dir / "table_significance.csv").exists())
            self.assertEqual(summary["dataset"]["rows_ok"], 6)
            self.assertEqual(len(summary["ranked_conditions"]), 2)


if __name__ == "__main__":
    unittest.main()

