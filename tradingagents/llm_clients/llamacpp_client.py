"""LlamaCpp client for direct local inference from GGUF model files."""

import os
from typing import Any, Optional

from .base_client import BaseLLMClient
from .validators import validate_model


class LlamaCppClient(BaseLLMClient):
    """Client for local inference via llama-cpp-python.

    Loads GGUF model files directly in-process without requiring
    an external server. Supports GPU offloading and configurable
    context window.

    Requires: pip install tradingagents[local]
    (installs llama-cpp-python and langchain-community)
    """

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, base_url, **kwargs)
        self._validate_dependencies()

    @staticmethod
    def _validate_dependencies():
        try:
            import llama_cpp  # noqa: F401
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required for local inference. "
                "Install with: pip install tradingagents[local]"
            )

    def get_llm(self) -> Any:
        """Return a configured ChatLlamaCpp instance."""
        from langchain_community.chat_models import ChatLlamaCpp

        model_path = self.kwargs.get("model_path")
        if not model_path:
            raise ValueError(
                "model_path is required for llamacpp provider. "
                "Set local_model_path_deep / local_model_path_quick in config."
            )
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"GGUF model file not found: {model_path}"
            )

        llm_kwargs = {
            "model_path": model_path,
            "n_gpu_layers": self.kwargs.get("n_gpu_layers", -1),
            "n_ctx": self.kwargs.get("n_ctx", 4096),
            "n_batch": self.kwargs.get("n_batch", 512),
            "verbose": False,
        }

        if "temperature" in self.kwargs:
            llm_kwargs["temperature"] = self.kwargs["temperature"]
        if "callbacks" in self.kwargs:
            llm_kwargs["callbacks"] = self.kwargs["callbacks"]

        return ChatLlamaCpp(**llm_kwargs)

    def validate_model(self) -> bool:
        return validate_model("llamacpp", self.model)
