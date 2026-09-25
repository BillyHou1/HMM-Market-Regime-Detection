# HMM Market Regime Detection

Boyu Hou, University of Bristol

A Gaussian hidden Markov model (HMM) is used to label market regimes for SPY, based on daily prices from January 2010 to November 2025. Four features are computed for each day: the 5-day return, 20-day volatility, the deviation from the 60-day moving average, and 20-day downside volatility. Each feature is z-scored over the 252 days ending that day. No later data therefore enters the scaling. The model is fitted on the 2,962 days before 2023, and the number of states is chosen by BIC between 2 and 5. However, BIC fell at every step, so 5 is the upper limit of the search rather than a clear optimum. The states are named by mean volatility. For every day, the state is obtained by forward filtering, which uses only the observations available up to that day and never looks at the rest of the series.

## Results

The states are evaluated through a one-day 95% VaR. For each state, the VaR is set to the 5th percentile of training-period returns on the days that follow it, and the VaR for day t is taken from the filtered state of day t-1. Ten assets are tested from January 2023 to December 2025, which gives 752 trading days, or 1,096 days for bitcoin. The SPY model is compared with seven asset-class models, each fitted on a basket of two to five tickers before 2023. The asset-class models come closer to the nominal 5% on 7 of the 10 assets. However, the gain is small. The mean absolute gap is reduced only from 1.90 to 1.84 percentage points, and both the Kupiec and Christoffersen tests are passed on just 3 assets by each approach. For EFA, EEM, AAPL and bitcoin, violation rates stay between 1.3% and 3.1% under both models, so the VaR is too conservative for these assets.

| Asset | Days | SPY model rate | Kupiec p | Christoffersen p | Asset-class model rate | Kupiec p | Christoffersen p |
|---|---:|---:|---:|---:|---:|---:|---:|
| TLT | 752 | 6.12% | 0.174 | 0.477 | 7.71% | 0.002 | 0.456 |
| USO | 752 | 3.32% | 0.025 | 0.007 | 4.12% | 0.255 | 0.006 |
| GLD | 752 | 4.26% | 0.337 | 0.200 | 4.65% | 0.660 | 0.309 |
| VNQ | 752 | 5.45% | 0.575 | 0.087 | 4.92% | 0.920 | 0.137 |
| UUP | 752 | 6.65% | 0.048 | 0.152 | 8.24% | 0.000 | 0.033 |
| BTC-USD | 1096 | 1.28% | 0.000 | 0.547 | 1.64% | 0.000 | 0.438 |
| EFA | 752 | 2.39% | 0.000 | 0.347 | 2.26% | 0.000 | 0.375 |
| EEM | 752 | 2.39% | 0.000 | 0.446 | 3.06% | 0.009 | 0.732 |
| SPY | 752 | 3.46% | 0.040 | 0.296 | 4.12% | 0.255 | 0.537 |
| AAPL | 752 | 2.13% | 0.000 | 0.037 | 2.79% | 0.003 | 0.119 |

## Usage

```bash
pip install -r requirements.txt
python src/main.py
python src/train_multi.py
python src/backtest_walkforward.py
```

## References

- Rabiner, L. R. (1989). A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition. *Proceedings of the IEEE*, 77(2), 257-286.
- Hamilton, J. D. (1989). A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle. *Econometrica*, 57(2), 357-384.
- RiskMetrics Group (1996). *RiskMetrics Technical Document*. J.P. Morgan.
- Sortino, F. A., and Price, L. N. (1994). Performance Measurement in a Downside Risk Framework. *Journal of Investing*, 3(3), 59-64.