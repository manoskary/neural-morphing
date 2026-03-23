#!/usr/bin/env python3
"""Unit tests for bridge codec contract and token layout conversions."""

import base64
import unittest

import numpy as np
from fastapi.testclient import TestClient

import server


class DummyAdapter(server.CodecAdapter):
    codec_id = "dac"

    def ensure_loaded(self) -> None:
        return

    def metadata(self) -> server.CodecMetadata:
        return server.CodecMetadata(
            sample_rate=44100,
            codebook_count=2,
            embedding_dim=4,
            required_input_channels=1,
            frame_rate_hz=100.0,
        )

    def encode_samples(self, samples: np.ndarray, sample_rate: int) -> server.TokenBlock:
        del sample_rate
        frames = max(samples.shape[0] // 16, 1)
        codebooks = self.metadata().codebook_count
        tokens = np.arange(frames * codebooks, dtype=np.int32).reshape(codebooks, frames).reshape(-1).tolist()
        return server.TokenBlock(B=1, T=frames, codebooks=codebooks, tokens=tokens)

    def tokens_to_vector_row(self, block: server.TokenBlock, frame_index: int) -> np.ndarray:
        del block
        return np.asarray([float(frame_index), 1.0, 2.0, 3.0], dtype=np.float32)

    def decode_tokens(self, block: server.TokenBlock) -> np.ndarray:
        samples = max(block.T * 16, 1)
        return np.zeros((samples, 1), dtype=np.float32)


class DummyRuntime:
    def __init__(self):
        self.active_codec = "dac"
        self.supported_codecs = ["dac", "spectrostream"]
        self._adapter = DummyAdapter()

    def get_adapter(self, codec_id=None):
        del codec_id
        return self._adapter

    def set_codec(self, codec_id: str):
        self.active_codec = codec_id
        return self._adapter

    def capabilities(self):
        md = self._adapter.metadata()
        return server.CapabilitiesResponse(
            ok=True,
            supported_codecs=self.supported_codecs,
            active_codec=self.active_codec,
            sample_rate=md.sample_rate,
            codebook_count=md.codebook_count,
            embedding_dim=md.embedding_dim,
            required_input_channels=md.required_input_channels,
            frame_rate_hz=md.frame_rate_hz,
            token_layout=server.TOKEN_LAYOUT_CODEBOOK_MAJOR,
            supports_pcm_endpoints=True,
        )


class CodecContractTests(unittest.TestCase):
    def setUp(self):
        self._runtime_backup = server.runtime
        server.runtime = DummyRuntime()
        self.client = TestClient(server.app)

    def tearDown(self):
        server.runtime = self._runtime_backup

    def test_token_layout_roundtrip(self):
        frame_major = np.asarray([[0, 1], [2, 3], [4, 5]], dtype=np.int32)
        codebook_major = server._frame_major_to_codebook_major(frame_major)
        reconstructed = server._codebook_major_to_frame_major(codebook_major.tolist(), codebooks=2, frames=3)
        np.testing.assert_array_equal(frame_major, reconstructed)

    def test_capabilities_schema(self):
        response = self.client.get("/capabilities")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["active_codec"], "dac")
        self.assertEqual(data["token_layout"], server.TOKEN_LAYOUT_CODEBOOK_MAJOR)
        self.assertIn("supported_codecs", data)
        self.assertTrue(data["supports_pcm_endpoints"])

    def test_encode_pcm_schema(self):
        payload = np.zeros((128, 1), dtype=np.float32).reshape(-1)
        response = self.client.post(
            "/encode_pcm",
            json={
                "sample_rate": 44100,
                "channels": 1,
                "num_samples": 128,
                "dtype": server.PCM_DTYPE_FLOAT32,
                "pcm_layout": server.PCM_LAYOUT_INTERLEAVED,
                "pcm_b64": base64.b64encode(payload.tobytes(order="C")).decode("utf-8"),
            },
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["token_layout"], server.TOKEN_LAYOUT_CODEBOOK_MAJOR)
        self.assertGreater(data["T"], 0)
        self.assertEqual(data["codebooks"], 2)

    def test_tokens_to_vectors_batch_schema(self):
        response = self.client.post(
            "/tokens_to_vectors_batch",
            json={
                "B": 1,
                "T": 4,
                "codebooks": 2,
                "tokens": [0, 1, 2, 3, 4, 5, 6, 7],
                "start_frame": 1,
                "frame_count": 2,
                "token_layout": server.TOKEN_LAYOUT_CODEBOOK_MAJOR,
            },
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["count"], 2)
        self.assertEqual(data["D"], 4)
        self.assertEqual(len(data["vectors"]), 2)
        self.assertEqual(len(data["vectors"][0]), 4)


if __name__ == "__main__":
    unittest.main()
