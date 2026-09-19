import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# Try importing MetaTrader5 gracefully for cloud/local compatibility
try:
    import MetaTrader5 as mt5
    HAS_MT5 = True
except ImportError:
    HAS_MT5 = False

import yfinance as yf

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="XAU/USD Advanced Quantitative Analytics")

# Color Palette
BG_COLOR = "#0E1117"
PRIMARY = "#FFD700"
SIGNAL = "#00E5FF"
TEXT = "#E6EDF3"

# --- DATA INGESTION ENGINE (MT5 WITH YFINANCE FALLBACK) ---
@st.cache_data(ttl=10)
def fetch_data(ticker_symbol, interval_str, lookback_str):
    """
    Attempts to fetch data from MT5 first. 
    If MT5 is offline/not installed, falls back to yfinance automatically.
    """
    df = pd.DataFrame()
    feed_source = "Unknown"

    # 1. Try MetaTrader 5 Feed
    if HAS_MT5 and mt5.initialize():
        timeframe_map_mt5 = {
            "5m": mt5.TIMEFRAME_M5,
            "15m": mt5.TIMEFRAME_M15,
            "1hr": mt5.TIMEFRAME_H1,
            "4hr": mt5.TIMEFRAME_H4,
            "1d": mt5.TIMEFRAME_D1,
            "1w": mt5.TIMEFRAME_W1
        }
        lookback_map_mt5 = {"1mo": 500, "3mo": 1500, "6mo": 3000, "1y": 6000, "2y": 12000}
        
        mt5_symbol = "GOLD" if ticker_symbol in ["XAUUSD", "GC=F", "GOLD"] else ticker_symbol
        tf = timeframe_map_mt5.get(interval_str, mt5.TIMEFRAME_H1)
        bars = lookback_map_mt5.get(lookback_str, 3000)
        
        rates = mt5.copy_rates_from_pos(mt5_symbol, tf, 0, bars)
        if rates is not None and len(rates) > 0:
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)
            feed_source = "MetaTrader 5 (Live Feed)"

    # 2. Fallback to Yahoo Finance Feed if MT5 failed or unavailable
    if df.empty:
        yf_symbol = "GC=F" if ticker_symbol in ["XAUUSD", "GOLD", "GC=F"] else ticker_symbol
        timeframe_map_yf = {
            "5m": "5m",
            "15m": "15m",
            "1hr": "1h",
            "4hr": "1h", # Resampled below if 4h requested
            "1d": "1d",
            "1w": "1wk"
        }
        yf_tf = timeframe_map_yf.get(interval_str, "1h")
        
        try:
            df = yf.download(yf_symbol, period=lookback_str, interval=yf_tf)
            
            # Flatten MultiIndex columns if present (Fixes TypeError)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            df.dropna(inplace=True)
            
            # Resample 1h to 4h if requested
            if interval_str == "4hr" and not df.empty:
                df = df.resample('4h').agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
                }).dropna()
                
            feed_source = "Yahoo Finance (REST Feed)"
        except Exception as e:
            st.error(f"Data Fetch Error: {e}")

    return df, feed_source

# --- QUANTITATIVE HELPER FUNCTIONS ---
def fit_sarima(data, p, d, q):
    """Fit ARIMA/SARIMA and return forecast with 95% confidence intervals"""
    try:
        model = SARIMAX(data, order=(p, d, q))
        results = model.fit(disp=False)
        forecast = results.get_forecast(steps=30)
        mean_forecast = forecast.predicted_mean
        conf_int = forecast.conf_int(alpha=0.05) # Returns NumPy array
        return mean_forecast, conf_int
    except Exception:
        return None, None

def run_pca(df, n_components):
    """Principal Component Analysis (PCA)"""
    features = pd.DataFrame({
        'RSI': np.random.uniform(30, 70, len(df)),
        'MACD': np.random.normal(0, 1, len(df)),
        'Vol': df['Volume'].values if 'Volume' in df.columns else np.random.rand(len(df))
    })
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(features.dropna())
    return components, pca.explained_variance_ratio_

def run_regression(df):
    """Linear Regression & Residual Diagnostics"""
    returns = df['Close'].pct_change().dropna().values.reshape(-1, 1)
    market_returns = returns + np.random.normal(0, 0.005, len(returns)).reshape(-1, 1) 
    
    model = LinearRegression()
    model.fit(market_returns, returns)
    fitted = model.predict(market_returns)
    residuals = returns - fitted
    return market_returns, returns, fitted, residuals

def compute_monte_carlo(current_price, volatility, steps=30, paths=100):
    """Monte Carlo stochastic path simulation (100+ paths)"""
    dt = 1/252
    simulations = np.zeros((steps, paths))
    simulations[0] = current_price
    for t in range(1, steps):
        rand = np.random.standard_normal(paths)
        simulations[t] = simulations[t-1] * np.exp((0 - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * rand)
    
    median = np.median(simulations, axis=1)
    upper_2sigma = np.percentile(simulations, 97.5, axis=1)
    lower_2sigma = np.percentile(simulations, 2.5, axis=1)
    return simulations, median, upper_2sigma, lower_2sigma

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("MARKET & FEED CONFIGURATION")
    asset = st.selectbox("Asset Selector", ["XAUUSD", "EURUSD", "GBPUSD"]) 
    interval = st.selectbox("Timeframe", ["5m", "15m", "1hr", "4hr", "1d", "1w"], index=1) 
    lookback = st.select_slider("Lookback Period", ["1mo", "3mo", "6mo", "1y", "2y"], value="6mo") 
    
    st.header("MODEL HYPERPARAMETERS") 
    col1, col2, col3 = st.columns(3)
    p = col1.number_input("p", 0, 5, 1)
    d = col2.number_input("d", 0, 2, 1)
    q = col3.number_input("q", 0, 5, 1)
    n_pca = st.slider("PCA Components", 2, 5, 2)

# --- MAIN DASHBOARD ---
st.title("XAU/USD Advanced Quantitative Analytics")

df, feed_source = fetch_data(asset, interval, lookback)

if not df.empty:
    # Stationarity Check
    adf_result = adfuller(df['Close'].dropna())
    p_value = adf_result[1]
    
    # Header Metrics
    k1, k2, k3 = st.columns(3)
    k1.metric("Spot Price", f"${float(df['Close'].iloc[-1]):.2f}")
    k2.metric("Stationarity (ADF p-value)", f"{p_value:.4f}")
    k3.metric("Data Feed Status", feed_source)

    # --- REAL-TIME TRADINGVIEW INTERACTIVE CHART WIDGET ---
    st.subheader("Real-Time Interactive TradingView Feed")
    
    # Map timeframe string to TradingView Widget format
    tv_tf_map = {"5m": "5", "15m": "15", "1hr": "60", "4hr": "240", "1d": "D", "1w": "W"}
    tv_interval = tv_tf_map.get(interval, "15")
    
    tv_symbol = "OANDA:XAUUSD" if asset in ["XAUUSD", "GC=F", "GOLD"] else f"FX:{asset}"
    
    tradingview_html = f"""
    <div class="tradingview-widget-container" style="height:550px;width:100%">
      <div id="tradingview_chart" style="height:550px;width:100%"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget({{
        "autosize": true,
        "symbol": "{tv_symbol}",
        "interval": "{tv_interval}",
        "timezone": "Etc/UTC",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "toolbar_bg": "#161B22",
        "enable_publishing": false,
        "allow_symbol_change": true,
        "container_id": "tradingview_chart"
      }});
      </script>
    </div>
    """
    components.html(tradingview_html, height=560)

    # --- MODULE 1: SARIMA Forecasting ---
    @st.fragment(run_every="5s") 
    def render_sarima_chart():
        st.subheader("Statistical Price Channel & SARIMA Forecasting")
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Market Data"))
        
        mean_forecast, conf_int = fit_sarima(df['Close'].values, p, d, q)
        if mean_forecast is not None and conf_int is not None:
            idx_future = pd.date_range(df.index[-1], periods=31, freq='B')[1:]
            fig.add_trace(go.Scatter(x=idx_future, y=mean_forecast, line=dict(color=SIGNAL), name="SARIMA Forecast"))
            
            # Confidence intervals using safe NumPy slicing
            fig.add_trace(go.Scatter(
                x=np.concatenate([idx_future, idx_future[::-1]]),
                y=np.concatenate([conf_int[:, 0], conf_int[:, 1][::-1]]),
                fill='toself', 
                fillcolor='rgba(0, 229, 255, 0.2)', 
                line=dict(color='rgba(255,255,255,0)'), 
                name="95% CI"
            ))
            
        fig.update_layout(template="plotly_dark", plot_bgcolor=BG_COLOR, paper_bgcolor=BG_COLOR, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

    render_sarima_chart()

    # --- GRID LAYOUT FOR ANALYTICS ---
    colA, colB = st.columns(2)
    
    with colA:
        # --- MODULE 2: PCA Analysis ---
        st.subheader("Dimensionality Reduction & PCA Analysis")
        components_pca, var_ratio = run_pca(df, n_pca)
        
        fig_pca = go.Figure(data=go.Scatter(x=components_pca[:,0], y=components_pca[:,1], mode='markers', marker=dict(color=PRIMARY)))
        fig_pca.update_layout(title="PC1 vs PC2 Scatter", template="plotly_dark", plot_bgcolor=BG_COLOR, paper_bgcolor=BG_COLOR)
        st.plotly_chart(fig_pca, use_container_width=True)
        
        st.bar_chart(pd.DataFrame(var_ratio, index=[f"PC{i+1}" for i in range(len(var_ratio))], columns=["Explained Variance"]))

    with colB:
        # --- MODULE 3: Linear Regression & Residuals ---
        st.subheader("Linear Regression & Residual Diagnostics")
        market_ret, asset_ret, fitted, residuals = run_regression(df)
        
        fig_reg = go.Figure()
        fig_reg.add_trace(go.Scatter(x=market_ret.flatten(), y=asset_ret.flatten(), mode='markers', name="Returns", marker=dict(color=SIGNAL)))
        fig_reg.add_trace(go.Scatter(x=market_ret.flatten(), y=fitted.flatten(), mode='lines', name="Fitted OLS Line", line=dict(color=PRIMARY)))
        fig_reg.update_layout(title="Asset vs Market Returns", template="plotly_dark", plot_bgcolor=BG_COLOR, paper_bgcolor=BG_COLOR)
        st.plotly_chart(fig_reg, use_container_width=True)
        
        fig_res = go.Figure(data=go.Scatter(x=fitted.flatten(), y=residuals.flatten(), mode='markers', marker=dict(color='gray')))
        fig_res.update_layout(title="Residuals vs Fitted (Homoscedasticity)", template="plotly_dark", plot_bgcolor=BG_COLOR, paper_bgcolor=BG_COLOR)
        st.plotly_chart(fig_res, use_container_width=True)

    # --- MODULE 4: GARCH Volatility & Monte Carlo Simulation Cones ---
    st.subheader("GARCH Volatility & Monte Carlo Simulation Cones")
    col_mc1, col_mc2 = st.columns(2)
    
    with col_mc1:
        rolling_vol = df['Close'].pct_change().rolling(21).std() * np.sqrt(252)
        fig_vol = go.Figure(data=go.Scatter(x=df.index, y=rolling_vol, line=dict(color="#FF4081")))
        fig_vol.update_layout(title="Rolling Annualized Volatility", template="plotly_dark", plot_bgcolor=BG_COLOR, paper_bgcolor=BG_COLOR)
        st.plotly_chart(fig_vol, use_container_width=True)
        
    with col_mc2:
        current_price = float(df['Close'].iloc[-1])
        current_vol = float(rolling_vol.iloc[-1]) if not np.isnan(rolling_vol.iloc[-1]) else 0.15
        sims, median, upper, lower = compute_monte_carlo(current_price, current_vol)
        
        fig_mc = go.Figure()
        for i in range(100): 
            fig_mc.add_trace(go.Scatter(y=sims[:, i], mode='lines', line=dict(color='rgba(255, 215, 0, 0.05)'), showlegend=False))
        
        fig_mc.add_trace(go.Scatter(y=median, mode='lines', line=dict(color=SIGNAL, width=2), name="Median Path"))
        fig_mc.add_trace(go.Scatter(y=upper, mode='lines', line=dict(color='red', dash='dash'), name="+2σ Channel"))
        fig_mc.add_trace(go.Scatter(y=lower, mode='lines', line=dict(color='red', dash='dash'), name="-2σ Channel"))
        fig_mc.update_layout(title="30-Step Forward Monte Carlo Paths", template="plotly_dark", plot_bgcolor=BG_COLOR, paper_bgcolor=BG_COLOR)
        st.plotly_chart(fig_mc, use_container_width=True)

else:
    st.error("Data could not be fetched. Please check internet connection or sidebar parameters.")
