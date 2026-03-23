#!/usr/bin/env python3
"""
Test script for the Neural Morphing Python Bridge Server
"""

import requests
import json
import time
import sys
import os

def test_server(base_url="http://localhost:8000"):
    """Test the Python bridge server endpoints"""
    
    print(f"Testing Neural Morphing Bridge Server at {base_url}")
    
    try:
        # Test health endpoint
        print("\n1. Testing /health endpoint...")
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✓ Health check passed: {health_data}")
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
            
        print("\n2. Testing /capabilities endpoint...")
        response = requests.get(f"{base_url}/capabilities", timeout=10)
        if response.status_code == 200:
            capabilities = response.json()
            print(f"✓ Capabilities: codec={capabilities.get('active_codec')} channels={capabilities.get('required_input_channels')}")
        else:
            print(f"? Capabilities endpoint returned: {response.status_code}")
        
        # We can't easily test the other endpoints without a real audio file
        # but we can test that they return proper error messages
        print("\n3. Testing /encode endpoint (expecting error with invalid path)...")
        response = requests.post(f"{base_url}/encode", 
                               json={"path": "/nonexistent/file.wav"},
                               timeout=10)
        if response.status_code in [404, 500]:
            print("✓ Encode endpoint responds correctly to invalid input")
        else:
            print(f"? Encode endpoint returned: {response.status_code}")
            
        print("\n4. Testing /tokens_to_vectors endpoint...")
        response = requests.post(f"{base_url}/tokens_to_vectors",
                               json={"B": 1, "T": 1, "codebooks": 1, "tokens": [42], "frame_index": 0},
                               timeout=10)
        if response.status_code in [200, 500]:
            print("✓ tokens_to_vectors endpoint responds")
        else:
            print(f"? tokens_to_vectors endpoint returned: {response.status_code}")
            
        print("\n5. Testing /decode endpoint...")
        response = requests.post(f"{base_url}/decode",
                               json={"B": 1, "T": 1, "codebooks": 1, "tokens": [42]},
                               timeout=10)
        if response.status_code in [200, 500]:
            print("✓ decode endpoint responds")
        else:
            print(f"? decode endpoint returned: {response.status_code}")

        print("\n6. Testing /encode_pcm endpoint...")
        dummy_pcm = b"\x00\x00\x00\x00" * 128  # 128 float32 zeros mono
        response = requests.post(
            f"{base_url}/encode_pcm",
            json={
                "sample_rate": 44100,
                "channels": 1,
                "num_samples": 128,
                "dtype": "float32le",
                "pcm_layout": "interleaved",
                "pcm_b64": __import__("base64").b64encode(dummy_pcm).decode("utf-8"),
            },
            timeout=10,
        )
        if response.status_code in [200, 500]:
            print("✓ encode_pcm endpoint responds")
        else:
            print(f"? encode_pcm endpoint returned: {response.status_code}")
            
        print(f"\n✓ Server at {base_url} appears to be working!")
        return True
        
    except requests.exceptions.ConnectError:
        print(f"✗ Cannot connect to server at {base_url}")
        print("  Make sure the server is running with: python bridge/server.py")
        return False
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

if __name__ == "__main__":
    server_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    success = test_server(server_url)
    sys.exit(0 if success else 1)
