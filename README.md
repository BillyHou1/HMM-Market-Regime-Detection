# HMM Market Regime Detection

## 1. Project Abstract

Financial markets do not behave consistently over time. A strategy that performs well during a stable uptrend may fail during a crisis. The primary challenge is identifying the current regime of the market.

To address this, I utilize a Hidden Markov Model to classify market days into distinct states based on four key features: return, volatility, momentum, and downside risk. The last feature is particularly important because traditional measures of volatility treat a 2% gain and a 2% loss as equivalent, which they are not. To accurately capture actual risk, I calculate volatility using only negative returns.

## 2. Methodology

**Dataset:** SPY daily prices, January 2011 to November 2025—approximately 3,500 observations.

### A. Feature Set

| Feature | Calculation | Window |
|---------|-------------|--------|
| Return | 5-day cumulative return | 5 days |
| Volatility | Annualized std of daily returns | 20 days |
| Momentum | Price deviation from 60-day MA | 60 days |
| Downside Risk | Annualized std of negative returns only | 20 days |

### B. Hidden Markov Model

I fit a Gaussian HMM and tested 3 to 7 states. BIC selected 5 as optimal.

The model found:
- **Steady Bull** accounts for about 55% of trading days. Low volatility, positive momentum.
- **Quiet Decline** is roughly 20%. The market drifts lower but without panic.
- **Correction** at 15%. Volatility picks up, momentum turns negative.
- **High Volatility** around 7%. Risk is elevated across all measures.
- **Crisis** is rare, about 3%. March 2020 falls here.

### C. Validation

Each feature is z-scored using a 252-day rolling window. Only past data is used. The model cannot see future prices during training or prediction.

## 3. Usage

**Requirements:** Python 3.8+, hmmlearn, pandas, numpy, yfinance

```bash
pip install -r requirements.txt
python src/hmm_model.py
```

## 4. References

- Rabiner, L.R. (1989). "A Tutorial on Hidden Markov Models". Proceedings of the IEEE.
- Hamilton, J.D. (1989). "A New Approach to Economic Analysis of Nonstationary Time Series". Econometrica.
- RiskMetrics Group (1996). RiskMetrics Technical Document. J.P. Morgan.
- Sortino, F. & Price, L. (1994). "Performance Measurement in a Downside Risk Framework". Journal of Investing.

---

*Billy Hou | University of Bristol | 2025*
