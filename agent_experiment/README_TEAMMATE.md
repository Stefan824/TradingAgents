# Teammate Guide — Running the Agent + Model Walk-Forward Experiment

This guide tells you exactly how to run the experiment defined in
`agent_experiment/configs/walkforward.yaml`. Read this first before touching
anything.

## What This Experiment Does

- Three-way comparison: **Pure Model** (deep-trading) vs **Pure Agent** vs **Agent + Model**.
- This config runs the **Agent + Model** track: a TradingAgents multi-agent
  pipeline where a new **Model Analyst** feeds four deep-trading model metrics
  (LSTM / XGBoost / ARIMA-GARCH / XGB-LSTM Ensemble) into the debate.
- Output: one **BUY / SELL / HOLD** decision per 45-day walk-forward window,
  for BTC-USD and ETH-USD, covering 2023-01-01 → 2025-12-31.

## Current Config Snapshot (`configs/walkforward.yaml`)

| Setting | Value |
|---|---|
| Symbols | `BTC-USD` (BTC/USDT), `ETH-USD` (ETH/USDT) |
| Test period | 2023-01-01 → 2025-12-31 |
| Window size | 45 days (~24 windows per symbol) |
| LLM provider | Ollama |
| Quick-think LLM | `qwen3:8b` |
| Deep-think LLM | `qwen3:30b-a3b` |
| Analysts | `market`, `news`, `fundamentals`, `model` |
| Debate rounds | 1 (Bull/Bear), 1 (Risk) |
| Retries | 3 |
| Model artifacts | `agent_experiment/model_artifacts/` (bundled) |

## 1. Environment Setup (once)

```bash
cd TradingAgents
pip install -e .
pip install pyyaml
```

## 2. Start Ollama and Pull Models

```bash
ollama serve &
ollama pull qwen3:8b         # ~5.2 GB
ollama pull qwen3:30b-a3b    # ~18 GB
ollama list                  # verify both present
```

On the PSC cluster, store models on ocean (home quota is too small):

```bash
export OLLAMA_MODELS=/ocean/projects/cis260081p/<your_id>/.ollama/models
mkdir -p $OLLAMA_MODELS
```

See `README_PSC.md` for the full PSC setup (binary download, SLURM template).

## 3. Dry Run First (always)

Validates the full pipeline end-to-end with a mock LLM. Takes ~30 seconds and
requires no GPU.

```bash
python -m agent_experiment.scripts.run_walkforward --dry-run -v
```

If this fails, do **not** submit a full run — fix it first.

## 4. Full Run

```bash
# Both symbols
python -m agent_experiment.scripts.run_walkforward -v

# Single symbol
python -m agent_experiment.scripts.run_walkforward --symbol BTC-USD -v

# Custom output directory (recommended — avoid overwriting others' results)
python -m agent_experiment.scripts.run_walkforward \
  --output-dir outputs/walkforward_<yourname>_<date> -v
```

On PSC, submit via SLURM — template is in `README_PSC.md` §3.

```bash
sbatch train_walkforward.sh
squeue -u $USER
tail -f /ocean/projects/cis260081p/shared/logs/agent-model-wf-<JOBID>.out
```

## 5. Output

```
outputs/walkforward_<run_id>/
├── BTC-USD/
│   ├── signals.csv        # one row per 45-day window
│   └── metadata.json      # config snapshot (for reproducibility)
├── ETH-USD/
│   ├── signals.csv
│   └── metadata.json
└── all_signals.csv        # only created when >1 symbol is run
```

### `signals.csv` columns

| Column | Meaning |
|---|---|
| `symbol` | `BTC-USD` / `ETH-USD` |
| `window_idx` | Walk-forward window index (0, 1, 2, …) |
| `window_start` / `window_end` | Window dates (YYYY-MM-DD) |
| `decision_raw` | Raw agent output (BUY / SELL / HOLD) |
| `position` | Mapped position: 1.0 / -1.0 / 0.0 |
| `latency_s` | Inference time per window |
| `attempts` | >1 means retries happened |
| `error` | Error message if the window failed |

## 6. Rules — Please Do Not Break These

1. **Do not change `test_start`, `test_end`, or `walk_forward_retrain_days`.**
   They must match deep-trading's `default.yaml`, otherwise results cannot be
   compared across tracks.
2. **Do not modify `model_artifacts/`.** These are the frozen deep-trading
   outputs. Changing them causes data leakage.
3. **Always use a unique `--output-dir`.** Do not overwrite teammates' runs.
4. **For ablations, copy the YAML to a new file** (e.g.
   `walkforward_no_model.yaml`) and point `--config` at it. Do not edit the
   shared `walkforward.yaml` in place.
5. **Never run training/inference on a PSC login node.** Use `sbatch` or
   `interact`.
6. **Run `--dry-run` before every new configuration** to catch config errors
   before burning GPU hours.

## 7. Ablations — Suggested Splits

Copy `walkforward.yaml` → new file, edit only the relevant field, run with
`--config <new_file> --output-dir outputs/ablation_<name>`.

| Ablation | Field to change | Purpose |
|---|---|---|
| No Model Analyst | remove `model` from `selected_analysts` | Isolate the contribution of model metrics |
| Market-only | `selected_analysts: [market, model]` | Test minimal context |
| Fewer debate rounds | `max_debate_rounds: 1` → keep, or try higher values (e.g. `2`, `3`) | Test whether more debate helps. Setting to `0` short-circuits the debate but is untested — inspect the graph before relying on it. |
| Smaller LLM | `deep_think_llm: qwen3:8b` | Cost/quality trade-off |
| Shorter window | `walk_forward_retrain_days: 30` | Sensitivity to window length |

Coordinate who runs which ablation in the team channel and tag output
directories with your name.

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: tradingagents` | Ran `pip install -e .`? |
| `ollama: command not found` | `export PATH="$HOME/bin:$PATH"` (PSC) |
| Disk full when pulling models | Set `OLLAMA_MODELS` to ocean path |
| LLM errors / retries in logs | Built-in retry (3×) will handle transient failures; check logs |
| SLURM job stuck in `PD` | Try a different GPU (`v100-32` queues faster) |
| `sbatch: This does not look like a batch script` | First line of `.sh` must be `#!/bin/bash`, no leading whitespace |

## 9. Where to Look

- Config spec: `agent_experiment/experiment/walkforward_config.py`
- Runner logic: `agent_experiment/experiment/walkforward_runner.py`
- Entry point: `agent_experiment/scripts/run_walkforward.py`
- Model Analyst: `tradingagents/agents/analysts/model_analyst.py`
- Model metrics tool: `tradingagents/agents/utils/model_metrics_tool.py`
- Full architecture: `agent_experiment/README_WALKFORWARD.md`
- PSC setup: `agent_experiment/README_PSC.md`

Ping the channel before making structural changes (new analyst, new metric,
new symbol). Happy experimenting.
