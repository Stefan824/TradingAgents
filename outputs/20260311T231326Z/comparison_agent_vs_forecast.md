# Agent vs Forecasting Models — Pilot Comparison

**Evaluation period:** 60 days across 3 pilot windows (2025)
- Window 0: 2025-02-24 → 2025-03-15
- Window 1: 2025-04-09 → 2025-04-28
- Window 2: 2025-11-03 → 2025-11-22

All strategies evaluated on the **exact same bars** using deep-trading metrics.

---

## 1. Aggregate Comparison (All Pilot Windows)

| strategy | cumulative_return | annualized_return | sharpe | sortino | max_drawdown | calmar | excess_cumulative_return | information_ratio | hit_rate | profit_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **trading_agent** | 0.2477 | 2.8473 | 3.2176 | 5.6591 | -0.1018 | 27.98 | 0.4164 | 2.5210 | 0.4914 | 1.1510 |
| arima_garch | -0.1042 | -0.4882 | -0.7496 | -1.1616 | -0.3075 | -1.5876 | 0.0645 | 0.5279 | 0.4965 | 0.9759 |
| buy_and_hold | -0.1687 | -0.6753 | -1.4733 | -2.2391 | -0.2626 | -2.5711 | 0.0000 | 0.0000 | 0.5118 | 0.9533 |
| lstm | 0.1427 | 1.2531 | 1.6061 | 2.6205 | -0.1562 | 8.0201 | 0.3114 | 2.8100 | 0.5139 | 1.0535 |
| macd | 0.2269 | 2.4722 | 2.2929 | 4.1212 | -0.1188 | 20.82 | 0.3956 | 2.5014 | 0.4861 | 1.0773 |
| sma_cross | 0.0075 | 0.0466 | 0.3886 | 0.5737 | -0.4142 | 0.1126 | 0.1762 | 1.1694 | 0.5160 | 1.0127 |
| xgb_lstm_ensemble | 0.3027 | 4.0010 | 2.8753 | 4.6125 | -0.2065 | 19.38 | 0.4714 | 3.6417 | 0.5312 | 1.0979 |
| xgboost | 0.1264 | 1.0643 | 1.4673 | 2.3414 | -0.2236 | 4.7605 | 0.2951 | 2.5607 | 0.5188 | 1.0488 |

---

## 2. Benchmark Reference

- **Buy-and-hold (pilot windows):** cumulative return = -0.1687
