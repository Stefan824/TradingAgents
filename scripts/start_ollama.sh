#!/usr/bin/env bash
set -euo pipefail

# ── Configure these ──────────────────────────────────────────────────
OLLAMA_INSTALL_DIR="${OLLAMA_INSTALL_DIR:-$HOME/ollama-install}"
OLLAMA_CUDA_VERSION="${OLLAMA_CUDA_VERSION:-cuda_v12}"
# ─────────────────────────────────────────────────────────────────────

OLLAMA_BIN="$OLLAMA_INSTALL_DIR/bin/ollama"
OLLAMA_LIB="$OLLAMA_INSTALL_DIR/lib/ollama"
CUDA_LIB="$OLLAMA_LIB/$OLLAMA_CUDA_VERSION"

if [ ! -x "$OLLAMA_BIN" ]; then
    echo "ERROR: Ollama binary not found at $OLLAMA_BIN"
    echo ""
    echo "Either set OLLAMA_INSTALL_DIR or install Ollama:"
    echo "  mkdir -p \$HOME/ollama-install && cd \$HOME/ollama-install"
    echo "  curl -L -o /tmp/ollama.tar.zst \\"
    echo "    https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst"
    echo "  tar --zstd -xf /tmp/ollama.tar.zst"
    exit 1
fi

if [ ! -d "$CUDA_LIB" ]; then
    echo "WARNING: CUDA libs not found at $CUDA_LIB — running CPU-only"
    echo "         Set OLLAMA_CUDA_VERSION to cuda_v12 or cuda_v13"
fi

export LD_LIBRARY_PATH="${OLLAMA_LIB}:${CUDA_LIB}:${LD_LIBRARY_PATH:-}"

echo "Ollama binary:  $OLLAMA_BIN"
echo "Runner libs:    $OLLAMA_LIB"
echo "CUDA libs:      $CUDA_LIB"
echo ""

# Run from the install dir so Ollama finds ../lib/ollama/ relative to the binary.
# Do NOT set OLLAMA_LLM_LIBRARY -- it tells Ollama to skip all other runners.
exec "$OLLAMA_BIN" serve "$@"
