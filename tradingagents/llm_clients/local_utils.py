"""Utilities for local LLM inference: health checks, validation, recommendations."""

import os
import subprocess
from typing import Dict, List, Optional, Tuple

import requests


RECOMMENDED_MODELS_32GB: Dict[str, List[Dict[str, str]]] = {
    "quick_think": [
        {
            "name": "Qwen3-8B-Q4_K_M",
            "ollama": "qwen3:8b",
            "gguf": "Qwen3-8B-Q4_K_M.gguf",
            "size": "~5 GB",
            "description": "Fast 8B model, strong tool-calling, ideal for analyst agents",
        },
        {
            "name": "Qwen3-4B-Q4_K_M",
            "ollama": "qwen3:4b",
            "gguf": "Qwen3-4B-Q4_K_M.gguf",
            "size": "~3 GB",
            "description": "Ultra-light 4B model for budget setups",
        },
    ],
    "deep_think": [
        {
            "name": "Qwen3-30B-A3B-Q4_K_M",
            "ollama": "qwen3:30b-a3b",
            "gguf": "Qwen3-30B-A3B-Q4_K_M.gguf",
            "size": "~18 GB",
            "description": "MoE 30B model (only 3B active), excellent reasoning at low memory cost",
        },
        {
            "name": "Qwen3-14B-Q4_K_M",
            "ollama": "qwen3:14b",
            "gguf": "Qwen3-14B-Q4_K_M.gguf",
            "size": "~9 GB",
            "description": "Dense 14B model, strong reasoning for mid-range setups",
        },
    ],
}


def check_ollama_health(base_url: str = "http://localhost:11434") -> Tuple[bool, str]:
    """Check if Ollama server is running and responsive.

    Returns:
        (is_healthy, message)
    """
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            model_names = [m["name"] for m in models]
            return True, f"Ollama running with {len(models)} model(s): {', '.join(model_names) or 'none'}"
        return False, f"Ollama returned status {resp.status_code}"
    except requests.ConnectionError:
        return False, "Ollama server not reachable. Start with: ollama serve"
    except requests.Timeout:
        return False, "Ollama server timed out"
    except Exception as e:
        return False, f"Ollama health check failed: {e}"


def check_ollama_model(model_name: str, base_url: str = "http://localhost:11434") -> bool:
    """Check if a specific model is available in Ollama."""
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            available = [m["name"] for m in models]
            return any(model_name in name for name in available)
    except Exception:
        pass
    return False


def pull_ollama_model(model_name: str) -> Tuple[bool, str]:
    """Pull a model using the ollama CLI.

    Returns:
        (success, message)
    """
    try:
        result = subprocess.run(
            ["ollama", "pull", model_name],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode == 0:
            return True, f"Successfully pulled {model_name}"
        return False, f"Failed to pull {model_name}: {result.stderr}"
    except FileNotFoundError:
        return False, "ollama CLI not found. Install from: https://ollama.com"
    except subprocess.TimeoutExpired:
        return False, f"Timed out pulling {model_name} (>10 min)"
    except Exception as e:
        return False, f"Error pulling model: {e}"


def validate_gguf_path(path: str) -> Tuple[bool, str]:
    """Validate that a GGUF model file exists and is readable.

    Returns:
        (is_valid, message)
    """
    if not path:
        return False, "No model path provided"
    if not os.path.isfile(path):
        return False, f"File not found: {path}"
    if not path.lower().endswith(".gguf"):
        return False, f"File does not have .gguf extension: {path}"
    if not os.access(path, os.R_OK):
        return False, f"File not readable: {path}"

    size_bytes = os.path.getsize(path)
    size_gb = size_bytes / (1024 ** 3)
    return True, f"Valid GGUF file ({size_gb:.1f} GB)"


def estimate_memory_usage(model_path: str) -> Optional[float]:
    """Estimate RAM usage in GB based on GGUF file size.

    Rule of thumb: runtime memory ~ 1.1-1.2x file size for inference.
    """
    if not os.path.isfile(model_path):
        return None
    size_bytes = os.path.getsize(model_path)
    return round(size_bytes * 1.15 / (1024 ** 3), 1)


def get_model_recommendations(available_ram_gb: float = 32.0) -> Dict[str, List[Dict]]:
    """Get model recommendations that fit within available RAM.

    Returns filtered recommendations from RECOMMENDED_MODELS_32GB.
    """
    SIZE_MAP = {
        "~3 GB": 3.0,
        "~5 GB": 5.0,
        "~9 GB": 9.0,
        "~18 GB": 18.0,
    }

    result = {"quick_think": [], "deep_think": []}

    for tier in ("quick_think", "deep_think"):
        for model in RECOMMENDED_MODELS_32GB[tier]:
            size = SIZE_MAP.get(model["size"], 99.0)
            if size <= available_ram_gb * 0.8:
                result[tier].append(model)

    return result
