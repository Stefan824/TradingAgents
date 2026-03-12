# Agent Experiment — Pilot Runner

Batch-runs the TradingAgents framework over pre-declared pilot windows to produce
per-day trading decisions (BUY / SELL / HOLD → 1 / -1 / 0) for comparison against
the `deep-trading` forecasting baselines.

**This module does not modify any TradingAgents or deep-trading code.**

## Pilot Windows (2024 BTC-USD, data-driven)

Selected for regime diversity using 2025 rolling 20-day benchmark metrics:

| Window | Dates | Regime | Selection Criterion |
|--------|-------|--------|---------------------|
| 1 | Feb 24 – Mar 15 | High-Volatility Shock | Max rolling 20-day daily volatility in 2025 |
| 2 | Apr 9 – Apr 28 | Bull Breakout | Best rolling 20-day benchmark return in 2025 |
| 3 | Nov 3 – Nov 22 | Bear Drawdown | Worst rolling 20-day benchmark return in 2025 |

## Prerequisites

1. **TradingAgents installed** — `pip install -e .` from `TradingAgents/TradingAgents/`
2. **Ollama running** with models pulled:
   ```bash
   ollama pull qwen3:8b
   ollama pull qwen3:30b-a3b
   ollama serve
   ```
   Or use LlamaCpp — see `LOCAL_INFERENCE_GUIDE.md`.

## Usage

From the `TradingAgents/TradingAgents/` directory:

```bash
# Dry run (mock LLM, no GPU, validates pipeline ~30s)
python -m agent_experiment.scripts.run_pilot --dry-run

# Full run with Ollama
python -m agent_experiment.scripts.run_pilot

# Custom config
python -m agent_experiment.scripts.run_pilot --config agent_experiment/configs/pilot.yaml

# Verbose logging
python -m agent_experiment.scripts.run_pilot --dry-run -v
```

## Output

Artifacts are saved to `outputs/<run_id>/`:

```
outputs/20260311T120000Z/
├── signals.csv       # date, window_idx, decision_raw, position, latency_s, error
├── metadata.json     # experiment config, model ids, window specs
└── summary.txt       # human-readable run summary
```

### signals.csv columns

| Column | Description |
|--------|-------------|
| `date` | Calendar date (YYYY-MM-DD) |
| `window_idx` | Pilot window index (0, 1, 2) |
| `decision_raw` | Raw agent output (BUY / SELL / HOLD) |
| `position` | Mapped position: 1.0, -1.0, or 0.0 |
| `latency_s` | Inference time in seconds |
| `error` | Error message if the run failed for this date |

## Configuration

Edit `configs/pilot.yaml`:

```yaml
symbol_agent: "BTC-USD"           # yfinance ticker for the agent
llm_provider: "ollama"            # ollama, llamacpp, or mock
quick_think_llm: "qwen3:8b"
deep_think_llm: "qwen3:30b-a3b"
max_debate_rounds: 1
selected_analysts: [market, news, fundamentals]
```

## Design Decisions

- **One decision per calendar day** — Agent calls `propagate(symbol, date)` once per date.
  Crypto markets are 24/7 so every calendar day has data.
- **Tri-state positions** — BUY=1, SELL=-1, HOLD=0 (aligned with deep-trading pilot requirements).
- **Mock dry-run** — Uses the existing mock LLM provider for pipeline validation without GPU.
- **No TradingAgents modifications** — All code is additive; imports use the public API only.
