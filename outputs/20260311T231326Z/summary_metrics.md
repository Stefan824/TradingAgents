# Agent Pilot — Summary Metrics

Same metric pipeline as deep-trading forecasting baselines (`summary_metrics`).

---

## 1. Aggregate (All Pilot Windows)


| strategy      | cumulative_return | annualized_return | annualized_volatility | downside_volatility_annualized | sharpe | sortino | max_drawdown | calmar | var_95  | cvar_95 | ulcer_index | time_under_water_ratio | benchmark_cumulative_return | benchmark_annualized_return | excess_cumulative_return | tracking_error_annualized | information_ratio | turnover_mean | turnover_annualized | hit_rate | avg_holding_bars | profit_factor |
| ------------- | ----------------- | ----------------- | --------------------- | ------------------------------ | ------ | ------- | ------------ | ------ | ------- | ------- | ----------- | ---------------------- | --------------------------- | --------------------------- | ------------------------ | ------------------------- | ----------------- | ------------- | ------------------- | -------- | ---------------- | ------------- |
| trading_agent | 0.2477            | 2.8473            | 0.4501                | 0.2559                         | 3.2176 | 5.6591  | -0.1018      | 27.98  | -0.0065 | -0.0106 | 0.0366      | 0.9632                 | -0.1687                     | -0.6753                     | 0.4164                   | 0.9420                    | 2.5210            | 0.0250        | 219.2               | 0.4914   | 45.33            | 1.1510        |


**Diagnostics:** 

- Bars: 1440 | Long: 5.0% | Short: 51.7% | Flat: 43.3% | Flips: 31

### 1.1 Regime Metrics (Aggregate)


| regime   | cumulative_return | annualized_return | annualized_volatility | sharpe | sortino | max_drawdown | calmar | cvar_95 |
| -------- | ----------------- | ----------------- | --------------------- | ------ | ------- | ------------ | ------ | ------- |
| high_vol | 0.1580            | 18.62             | 0.6565                | 4.8601 | 9.0613  | -0.1008      | 184.7  | -0.0147 |
| low_vol  | 0.0775            | 0.9140            | 0.3232                | 2.1701 | 3.5338  | -0.1125      | 8.1273 | -0.0082 |


---

## 2. Per-Window Metrics

### 2.1 Window 0: 2025-02-24 → 2025-03-15


| strategy           | cumulative_return | annualized_return | annualized_volatility | downside_volatility_annualized | sharpe | sortino | max_drawdown | calmar | var_95  | cvar_95 | ulcer_index | time_under_water_ratio | benchmark_cumulative_return | benchmark_annualized_return | excess_cumulative_return | tracking_error_annualized | information_ratio | turnover_mean | turnover_annualized | hit_rate | avg_holding_bars | profit_factor |
| ------------------ | ----------------- | ----------------- | --------------------- | ------------------------------ | ------ | ------- | ------------ | ------ | ------- | ------- | ----------- | ---------------------- | --------------------------- | --------------------------- | ------------------------ | ------------------------- | ----------------- | ------------- | ------------------- | -------- | ---------------- | ------------- |
| trading_agent (w0) | 0.0046            | 0.0875            | 0.5617                | 0.3399                         | 0.4291 | 0.7090  | -0.1018      | 0.8597 | -0.0080 | -0.0144 | 0.0493      | 0.9812                 | -0.1238                     | -0.9105                     | 0.1284                   | 1.2380                    | 1.8885            | 0.0229        | 200.9               | 0.4564   | 47.83            | 1.0180        |



| regime   | cumulative_return | annualized_return | sharpe  | sortino | max_drawdown | calmar  | cvar_95 |
| -------- | ----------------- | ----------------- | ------- | ------- | ------------ | ------- | ------- |
| high_vol | 0.0213            | 2.5995            | 1.9407  | 3.4112  | -0.0839      | 30.98   | -0.0199 |
| low_vol  | -0.0163           | -0.3489           | -0.9265 | -1.4140 | -0.0716      | -4.8715 | -0.0103 |


*Bars: 480 | Long: 5.0% | Short: 55.0% | Flat: 40.0% | Flips: 10*

### 2.2 Window 1: 2025-04-09 → 2025-04-28


| strategy           | cumulative_return | annualized_return | annualized_volatility | downside_volatility_annualized | sharpe | sortino | max_drawdown | calmar | var_95  | cvar_95 | ulcer_index | time_under_water_ratio | benchmark_cumulative_return | benchmark_annualized_return | excess_cumulative_return | tracking_error_annualized | information_ratio | turnover_mean | turnover_annualized | hit_rate | avg_holding_bars | profit_factor |
| ------------------ | ----------------- | ----------------- | --------------------- | ------------------------------ | ------ | ------- | ------------ | ------ | ------- | ------- | ----------- | ---------------------- | --------------------------- | --------------------------- | ------------------------ | ------------------------- | ----------------- | ------------- | ------------------- | -------- | ---------------- | ------------- |
| trading_agent (w1) | 0.0405            | 1.0660            | 0.3858                | 0.1831                         | 2.0713 | 4.3640  | -0.0769      | 13.87  | -0.0047 | -0.0077 | 0.0320      | 0.9750                 | 0.2401                      | 49.92                       | -0.1996                  | 0.6157                    | -5.2788           | 0.0354        | 310.5               | 0.4599   | 31.89            | 1.1030        |



| regime   | cumulative_return | annualized_return | sharpe  | sortino | max_drawdown | calmar    | cvar_95 |
| -------- | ----------------- | ----------------- | ------- | ------- | ------------ | --------- | ------- |
| high_vol | 0.1079            | 511.2             | 10.9    | 30.39   | -0.0244      | 2.099e+04 | -0.0084 |
| low_vol  | -0.0608           | -0.8055           | -6.4447 | -9.4619 | -0.0592      | -13.61    | -0.0071 |


*Bars: 480 | Long: 10.0% | Short: 50.0% | Flat: 40.0% | Flips: 13*

### 2.3 Window 2: 2025-11-03 → 2025-11-22


| strategy           | cumulative_return | annualized_return | annualized_volatility | downside_volatility_annualized | sharpe | sortino | max_drawdown | calmar | var_95  | cvar_95 | ulcer_index | time_under_water_ratio | benchmark_cumulative_return | benchmark_annualized_return | excess_cumulative_return | tracking_error_annualized | information_ratio | turnover_mean | turnover_annualized | hit_rate | avg_holding_bars | profit_factor |
| ------------------ | ----------------- | ----------------- | --------------------- | ------------------------------ | ------ | ------- | ------------ | ------ | ------- | ------- | ----------- | ---------------------- | --------------------------- | --------------------------- | ------------------------ | ------------------------- | ----------------- | ------------- | ------------------- | -------- | ---------------- | ------------- |
| trading_agent (w2) | 0.1843            | 20.95             | 0.3771                | 0.2120                         | 8.3815 | 14.91   | -0.0413      | 507.7  | -0.0065 | -0.0091 | 0.0160      | 0.8854                 | -0.2350                     | -0.9925                     | 0.4193                   | 0.8608                    | 9.1680            | 0.0167        | 146.1               | 0.5667   | 60               | 1.4147        |



| regime   | cumulative_return | annualized_return | sharpe | sortino | max_drawdown | calmar | cvar_95 |
| -------- | ----------------- | ----------------- | ------ | ------- | ------------ | ------ | ------- |
| high_vol | 0.0316            | 5.6443            | 4.9308 | 8.6627  | -0.0335      | 168.3  | -0.0096 |
| low_vol  | 0.1480            | 35.64             | 10.01  | 17.92   | -0.0232      | 1535   | -0.0087 |


*Bars: 480 | Long: 0.0% | Short: 50.0% | Flat: 50.0% | Flips: 7*

---

## 3. Metric Definitions (from deep-trading)


| Metric                   | Description                             |
| ------------------------ | --------------------------------------- |
| cumulative_return        | Total strategy return                   |
| annualized_return        | Annualized return                       |
| sharpe                   | Sharpe ratio                            |
| max_drawdown             | Maximum drawdown                        |
| sortino                  | Sortino ratio                           |
| calmar                   | Calmar ratio (return / max drawdown)    |
| excess_cumulative_return | Strategy − benchmark                    |
| information_ratio        | Excess return / tracking error          |
| hit_rate                 | Fraction of bars with correct direction |
| turnover_annualized      | Annualized turnover                     |
| profit_factor            | Gains / losses                          |


