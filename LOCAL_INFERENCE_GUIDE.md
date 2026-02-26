# Local LLM Inference -- Deployment & Debugging Guide

## What Was Built

Three commits (`1dcb238`, `812232a`, `d0f7428`) added local LLM inference as an
alternative to API-based inference. Every LLM call in the pipeline now supports
a local inference option. Users choose their provider via config or CLI.

### New Files

| File | Purpose |
|------|---------|
| `tradingagents/llm_clients/llamacpp_client.py` | Direct GGUF model loading via llama-cpp-python (no server needed) |
| `tradingagents/llm_clients/mock_client.py` | Mock LLM for testing (returns canned trading responses) |
| `tradingagents/llm_clients/local_utils.py` | Ollama health checks, GGUF validation, Qwen 3 model recommendations |
| `test_local_integration.py` | Full pipeline integration test using mock provider |

### Modified Files

| File | What Changed |
|------|-------------|
| `tradingagents/llm_clients/factory.py` | Routes `llamacpp`, `mock` providers |
| `tradingagents/llm_clients/validators.py` | Accepts any model name for `llamacpp`/`mock` |
| `tradingagents/llm_clients/__init__.py` | Exports `MockLLMClient` |
| `tradingagents/llm_clients/openai_client.py` | Ollama server/model health check on `get_llm()` |
| `tradingagents/default_config.py` | Added `local_model_path_deep`, `local_model_path_quick`, `local_n_gpu_layers`, `local_n_ctx`, `local_n_batch` |
| `tradingagents/graph/trading_graph.py` | Passes per-model paths and local kwargs for llamacpp provider |
| `cli/utils.py` | LlamaCpp provider option, Qwen 3 model menus, local config prompts |
| `cli/main.py` | Wires local config, normalizes provider name, Ollama readiness check |
| `pyproject.toml` | Optional `[local]` dependency group |
| `requirements.txt` | Commented optional deps |

---

## Architecture

All LLM calls go through a factory pattern:

```
config["llm_provider"] --> create_llm_client() --> BaseLLMClient.get_llm() --> LangChain BaseChatModel
```

Two LLM instances are created: `deep_thinking_llm` (research/risk managers)
and `quick_thinking_llm` (analysts, researchers, trader, debators).

Providers available: `openai`, `anthropic`, `google`, `xai`, `ollama`,
`openrouter`, `llamacpp` (new), `mock` (new).

The agents use two invocation patterns:
1. **Direct invoke**: `llm.invoke(prompt)` -- researchers, trader, debators, managers
2. **Tool-bound chain**: `(prompt | llm.bind_tools(tools)).invoke(messages)` -- all 4 analysts

Both patterns work with any provider including local ones.

---

## Running on a 4090 Server (24 GB VRAM, ~32 GB RAM)

### Option A: Ollama (Recommended for First Run)

Ollama handles model management and GPU offloading automatically.

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull Qwen 3 models
ollama pull qwen3:8b          # ~5 GB, quick-think
ollama pull qwen3:30b-a3b     # ~18 GB, deep-think (MoE, fits in 24GB VRAM)

# 3. Verify
ollama list

# 4. Run the pipeline
python -c "
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
config['llm_provider'] = 'ollama'
config['deep_think_llm'] = 'qwen3:30b-a3b'
config['quick_think_llm'] = 'qwen3:8b'
config['max_debate_rounds'] = 1

ta = TradingAgentsGraph(debug=True, config=config)
_, decision = ta.propagate('NVDA', '2024-05-10')
print(f'Decision: {decision}')
"
```

### Option B: LlamaCpp (Direct, No Server)

Loads GGUF files directly in-process. More control, no Ollama dependency.

```bash
# 1. Install local inference deps
pip install tradingagents[local]
# If llama-cpp-python fails, install with CUDA:
# CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

# 2. Download Qwen 3 GGUF models (from HuggingFace)
# Quick-think: https://huggingface.co/Qwen/Qwen3-8B-GGUF
# Deep-think:  https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF
# Get the Q4_K_M quantization files

# 3. Run the pipeline
python -c "
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
config['llm_provider'] = 'llamacpp'
config['deep_think_llm'] = 'qwen3-30b-a3b'
config['quick_think_llm'] = 'qwen3-8b'
config['local_model_path_deep'] = '/path/to/Qwen3-30B-A3B-Q4_K_M.gguf'
config['local_model_path_quick'] = '/path/to/Qwen3-8B-Q4_K_M.gguf'
config['local_n_gpu_layers'] = -1   # all layers to GPU
config['local_n_ctx'] = 4096        # start conservative
config['max_debate_rounds'] = 1

ta = TradingAgentsGraph(debug=True, config=config)
_, decision = ta.propagate('NVDA', '2024-05-10')
print(f'Decision: {decision}')
"
```

### Option C: Verify Setup with Mock First

Before downloading any models, verify the pipeline works:

```bash
python test_local_integration.py
```

This runs all 15 agents end-to-end with mock responses (~5 seconds, no GPU).
Expected output: `PASSED: All checks passed`.

---

## 4090 Memory Budget

The RTX 4090 has 24 GB VRAM. Recommended model pairing:

| Role | Model | VRAM | Notes |
|------|-------|------|-------|
| Quick-think | Qwen3-8B Q4_K_M | ~5 GB | Fast, good tool-calling |
| Deep-think | Qwen3-30B-A3B Q4_K_M | ~18 GB | MoE: only 3B params active per token |
| **Total** | | **~23 GB** | Fits in 24 GB VRAM with margin |

If both models need to be loaded simultaneously (they do -- the pipeline
alternates between them), Ollama handles this with model hot-swapping.
For llamacpp, both are loaded in-process so you need ~23 GB VRAM available.

**If VRAM is tight**, reduce to:
- Quick: Qwen3-4B Q4_K_M (~3 GB)
- Deep: Qwen3-14B Q4_K_M (~9 GB)
- Total: ~12 GB, leaves room for other processes

**CPU fallback**: Set `local_n_gpu_layers = 0` to run entirely on CPU/RAM.
Slower but works if GPU is busy.

---

## Likely Issues & Debugging

### 1. Tool Calling Failures

The analyst agents (market, social, news, fundamentals) use `llm.bind_tools(tools)`.
If the local model doesn't support tool/function calling, the agent will fail or
produce empty reports.

**Symptoms**: Empty `market_report`, `sentiment_report`, etc. Or errors about
tool call parsing.

**Fix**: Qwen 3 models have native tool-calling support. If using a different
model, ensure it supports the ChatML tool-calling format. In Ollama, verify with:
```bash
ollama run qwen3:8b "Call the function get_stock_data with symbol=NVDA, start_date=2024-05-01, end_date=2024-05-10"
```

### 2. Context Window Overflow

Some agents build very long prompts (especially researchers and managers that
concatenate all analyst reports). If context is too small, responses get truncated.

**Symptoms**: Incomplete or cut-off responses, or errors about token limits.

**Fix**: Increase `local_n_ctx`. Start with 4096, bump to 8192 or 16384 if needed.
For 4090 with Q4 models, 8192 is safe. Watch VRAM usage with `nvidia-smi`.

### 3. Ollama Not Running

**Symptoms**: `ConnectionError` or warning "Ollama server not reachable".

**Fix**: `ollama serve` in a separate terminal. The health check in
`openai_client.py` logs a warning but doesn't block -- the actual error comes
later when the LLM tries to call the API.

### 4. GGUF File Not Found (llamacpp)

**Symptoms**: `FileNotFoundError: GGUF model file not found`.

**Fix**: Check the path in `config['local_model_path_deep']` and
`config['local_model_path_quick']`. Must be absolute paths to `.gguf` files.

### 5. llama-cpp-python CUDA Build

**Symptoms**: Inference runs on CPU even with `-1` GPU layers, or import errors.

**Fix**: Reinstall with CUDA support:
```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install --force-reinstall llama-cpp-python
```
Verify: `python -c "import llama_cpp; print(llama_cpp.llama_supports_gpu_offload())"`

### 6. Mock Routing Mismatch

If you modify agent prompts, the mock may return wrong canned responses.

**How routing works**: `mock_client.py` function `_route_response()` matches
keywords from the prompt text to select a response. Key matchers:
- Market analyst: `"analyzing financial markets"`
- Social analyst: `"social media"` + `"company specific news"`
- News analyst: `"news researcher"` + `"recent news and trends"`
- Fundamentals: `"fundamental information"`
- Bull/Bear: `"bull analyst"` / `"bear analyst"`
- Research Manager: `"portfolio manager and debate facilitator"`
- Trader: `"trading agent"` + `"investment decision"`
- Risk analysts: `"aggressive/conservative/neutral risk analyst"`
- Risk Judge: `"risk management judge"`
- Signal extraction: `"extract the investment decision"`

### 7. Bear Researcher Returns Bull Response

Known mock limitation: the bear researcher prompt contains "Bull Analyst" in the
debate history, so the mock's keyword routing may match the bull response instead.
This only affects mock testing, not real inference. The mock test still passes
because the pipeline completes regardless of response content.

---

## Key Code Paths

**Factory**: `tradingagents/llm_clients/factory.py:create_llm_client()`
- Entry point for all LLM client creation

**Config flow**: `default_config.py` -> `trading_graph.py:_get_provider_kwargs()`
-> `create_llm_client()` -> `LlamaCppClient.get_llm()` or `OpenAIClient.get_llm()`

**Agent invocation**: All agents are in `tradingagents/agents/`. They receive
the LLM as a constructor argument and call `llm.invoke()` or
`(prompt | llm.bind_tools(tools)).invoke()`.

**Graph wiring**: `tradingagents/graph/setup.py:GraphSetup.setup_graph()` builds
the LangGraph `StateGraph` with all agent nodes and conditional edges.

**Conditional routing**: `tradingagents/graph/conditional_logic.py` determines
agent flow based on tool_calls presence (analysts) and debate round counts.

---

## Quick Validation Checklist

```bash
# 1. Mock test (no model needed)
python test_local_integration.py
# Expected: PASSED, ~5 seconds

# 2. Ollama health check
python -c "from tradingagents.llm_clients.local_utils import check_ollama_health; print(check_ollama_health())"
# Expected: (True, 'Ollama running with N model(s): ...')

# 3. Single model inference test (Ollama)
python -c "
from tradingagents.llm_clients import create_llm_client
client = create_llm_client('ollama', 'qwen3:8b')
llm = client.get_llm()
print(llm.invoke('What is 2+2?').content[:200])
"

# 4. Full pipeline with real local model
python -c "
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
config = DEFAULT_CONFIG.copy()
config['llm_provider'] = 'ollama'
config['deep_think_llm'] = 'qwen3:8b'  # use same model for both to start
config['quick_think_llm'] = 'qwen3:8b'
config['max_debate_rounds'] = 1
ta = TradingAgentsGraph(debug=True, config=config)
_, decision = ta.propagate('NVDA', '2024-05-10')
print(f'Decision: {decision}')
"
```
