# 📊 XAU/USD Advanced Quantitative Analytics Dashboard

An institutional-grade, real-time quantitative trading dashboard built with **Python** and **Streamlit**. It integrates statistical models, automated machine learning (SARIMA), multi-asset macro correlations, and Smart Money Concepts (SMC) to deliver real-time diagnostic market verdicts.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Status](https://img.shields.io/badge/Status-Active-success)

---

## 🚀 Key Features
- **4H Closed-Bar Diagnostic Engine:** Filters out market noise by evaluating structural market shifts, Fair Value Gaps (FVGs), and Monte Carlo simulations exclusively on completed 4-hour candles.
- **Monte Carlo Terminal Probabilities:** Simulates 100 geometric Brownian motion paths to calculate exact bullish vs. bearish forward distributions.
- **Automated SARIMA Optimization:** Automatically calculates data stationarity ($ADF$) and fits optimal parameters with AIC scoring.
- **Cross-Asset Macro Correlation Matrix:** Tracks live rolling correlations between Gold, Silver, the U.S. Dollar Index (DXY), and Real Yields.
- **Institutional Order Flow (SMC):** Detects active Market Structure Shifts (MSS) and unmitigated Fair Value Gaps.

---

## 🛠️ Tech Stack & Libraries
* **Frontend/UI:** Streamlit, Plotly, TradingView Widgets
* **Quantitative/Stats:** Statsmodels (SARIMAX, ADF), Scikit-Learn (PCA, Linear Regression), NumPy, Pandas
* **Data Feeds:** Twelve Data API & Yahoo Finance SDK (`yfinance`)

---

## ⚙️ Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/airshadolomhodeen-alt/xauusd-dashboard.git](https://github.com/airshadolomhodeen-alt/xauusd-dashboard.git)
   cd xauusd-dashboard
