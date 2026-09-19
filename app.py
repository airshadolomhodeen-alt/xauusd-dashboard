import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import time

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="XAU/USD Advanced Quantitative Analytics")

# Color Palette
BG_COLOR = "#0E1117"
PRIMARY = "#FFD700"
SIGNAL = "#00E5FF"
TEXT = "#E6EDF3"

# --- MODULAR FUNCTIONS ---
@st.cache_data(ttl=300)
def fetch_historical_data(ticker, interval, lookback):
    """Data Ingestion: yfinance for real-time market polling and historical OHLC data"""
    try:
        df = yf.download(ticker, period=lookback, interval=interval)
        
        # FIX: Flatten MultiIndex columns if present to extract standard 1D Series
        # This resolves the float formatting TypeError on df['Close'].iloc[-1]
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
            
        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error(f"Data Fetch Error: {e}")
        return pd.DataFrame()

def fit_sarima(data, p, d, q):
    """Fit ARIMA/SARIMA and return forecast with 95% confidence intervals"""
    try:
        model = SARIMAX(data, order=(p, d, q))
        results = model.fit(disp=False)
        forecast = results.get_forecast(steps=30)
        mean_forecast = forecast.predicted_mean
        conf_int = forecast.conf_int(alpha=0.05) # 95% CI
        return mean_forecast, conf_int
    except:
        return None, None

def run_pca(df, n_components):
    """Principal Component Analysis (PCA) scatter plot"""
    # Mocking yield curve, DXY, and tech indicators for PCA
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
    # Mock market index returns
    market_returns = returns + np.random.normal(0, 0.005, len(returns)).reshape(-1, 1) 
    
    model = LinearRegression()
    model.fit(market_returns, returns)
    fitted = model.predict(market_returns)
    residuals = returns - fitted
    return market_returns, returns, fitted, residuals

def compute_monte_carlo(current_price, volatility, steps=30, paths=100):
    """30-step forward Monte Carlo stochastic path simulation (100+ paths)"""
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
    asset = st.selectbox("Asset Selector", ["GC=F", "EURUSD=X", "^GSPC"]) 
    interval = st.selectbox("Timeframe", ["1m", "5m", "1h", "1d"], index=3) 
    lookback = st.select_slider("Lookback Period", ["1mo", "3mo", "6mo", "1y", "2y"], value="6mo") 
    
    st.header("MODEL HYPERPARAMETERS") 
    col1, col2, col3 = st.columns(3)
    p = col1.number_input("p", 0, 5, 1)
    d = col2.number_input("d", 0, 2, 1)
    q = col3.number_input("q", 0, 5, 1)
    n_pca = st.slider("PCA Components", 2, 5, 2)

# --- MAIN DASHBOARD ---
st.title("XAU/USD Advanced Quantitative Analytics")

df = fetch_historical_data(asset, interval, lookback)

if not df.empty:
    # Stationarity Check
    adf_result = adfuller(df['Close'].dropna())
    p_value = adf_result[1]
    
    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    # Using float conversion to ensure the metric displays properly even if typing is strict
    k1.metric("Spot Price", f"${float(df['Close'].iloc[-1]):.2f}")
    k2.metric("Stationarity (ADF p-value)", f"{p_value:.4f}")
    
    # --- MODULE 1: Live Chart & SARIMA Forecast ---
    @st.fragment(run_every="5s") 
    def render_live_chart():
        st.subheader("Real-Time Price Channel & SARIMA Forecasting")
        
        fig = go.Figure()
        # Main candlestick / line chart
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Market Data"))
        
        # SARIMA Overlay
        mean_forecast, conf_int = fit_sarima(df['Close'].values, p, d, q)
        if mean_forecast is not None:
            idx_future = pd.date_range(df.index[-1], periods=31, freq='B')[1:]
            fig.add_trace(go.Scatter(x=idx_future, y=mean_forecast, line=dict(color=SIGNAL), name="SARIMA Forecast"))
            
            # 95% confidence intervals (shaded fill)
            fig.add_trace(go.Scatter(x=np.concatenate([idx_future, idx_future[::-1]]),
                                     y=np.concatenate([conf_int.iloc[:, 0], conf_int.iloc[:, 1][::-1]]),
                                     fill='toself', fillcolor='rgba(0, 229, 255, 0.2)', line=dict(color='rgba(255,255,255,0)'), name="95% CI"))
            
        fig.update_layout(template="plotly_dark", plot_bgcolor=BG_COLOR, paper_bgcolor=BG_COLOR, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

    render_live_chart()

    # --- GRID LAYOUT ---
    colA, colB = st.columns(2)
    
    with colA:
        # --- MODULE 2: Dimensionality Reduction & PCA Analysis ---
        st.subheader("Dimensionality Reduction & PCA Analysis")
        components, var_ratio = run_pca(df, n_pca)
        
        # PC1 vs PC2 Scatter
        fig_pca = go.Figure(data=go.Scatter(x=components[:,0], y=components[:,1], mode='markers', marker=dict(color=PRIMARY)))
        fig_pca.update_layout(title="PC1 vs PC2", template="plotly_dark", plot_bgcolor=BG_COLOR, paper_bgcolor=BG_COLOR)
        st.plotly_chart(fig_pca, use_container_width=True)
        
        # Variance explained bar chart
        st.bar_chart(pd.DataFrame(var_ratio, index=[f"PC{i+1}" for i in range(len(var_ratio))], columns=["Explained Variance"]))

    with colB:
        # --- MODULE 3: Linear Regression & Residual Diagnostics ---
        st.subheader("Linear Regression & Residual Diagnostics")
        market_ret, asset_ret, fitted, residuals = run_regression(df)
        
        fig_reg = go.Figure()
        # Scatter plot with Fitted OLS Regression Line
        fig_reg.add_trace(go.Scatter(x=market_ret.flatten(), y=asset_ret.flatten(), mode='markers', name="Returns", marker=dict(color=SIGNAL)))
        fig_reg.add_trace(go.Scatter(x=market_ret.flatten(), y=fitted.flatten(), mode='lines', name="Fitted OLS Line", line=dict(color=PRIMARY)))
        fig_reg.update_layout(title="Asset vs Market Returns", template="plotly_dark", plot_bgcolor=BG_COLOR, paper_bgcolor=BG_COLOR)
        st.plotly_chart(fig_reg, use_container_width=True)
        
        # Residuals vs Fitted plot
        fig_res = go.Figure(data=go.Scatter(x=fitted.flatten(), y=residuals.flatten(), mode='markers', marker=dict(color='gray')))
        fig_res.update_layout(title="Residuals vs Fitted (Homoscedasticity Check)", template="plotly_dark", plot_bgcolor=BG_COLOR, paper_bgcolor=BG_COLOR)
        st.plotly_chart(fig_res, use_container_width=True)

    # --- MODULE 4: GARCH Volatility & Monte Carlo Simulation Cones ---
    st.subheader("GARCH Volatility & Monte Carlo Simulation Cones")
    col_mc1, col_mc2 = st.columns(2)
    
    with col_mc1:
        # Real-time rolling annualized volatility plot
        rolling_vol = df['Close'].pct_change().rolling(21).std() * np.sqrt(252)
        fig_vol = go.Figure(data=go.Scatter(x=df.index, y=rolling_vol, line=dict(color="#FF4081")))
        fig_vol.update_layout(title="Rolling Annualized Volatility (Proxy for GARCH)", template="plotly_dark", plot_bgcolor=BG_COLOR, paper_bgcolor=BG_COLOR)
        st.plotly_chart(fig_vol, use_container_width=True)
        
    with col_mc2:
        # Monte Carlo stochastic path simulation displaying median and ±2σ probability density channels
        current_price = float(df['Close'].iloc[-1])
        current_vol = float(rolling_vol.iloc[-1]) if not np.isnan(rolling_vol.iloc[-1]) else 0.15
        sims, median, upper, lower = compute_monte_carlo(current_price, current_vol)
        
        fig_mc = go.Figure()
        for i in range(100): # 100+ paths
            fig_mc.add_trace(go.Scatter(y=sims[:, i], mode='lines', line=dict(color='rgba(255, 215, 0, 0.05)'), showlegend=False))
        
        fig_mc.add_trace(go.Scatter(y=median, mode='lines', line=dict(color=SIGNAL, width=2), name="Median Path"))
        fig_mc.add_trace(go.Scatter(y=upper, mode='lines', line=dict(color='red', dash='dash'), name="+2σ Channel"))
        fig_mc.add_trace(go.Scatter(y=lower, mode='lines', line=dict(color='red', dash='dash'), name="-2σ Channel"))
        fig_mc.update_layout(title="30-Step Forward Monte Carlo", template="plotly_dark", plot_bgcolor=BG_COLOR, paper_bgcolor=BG_COLOR)
        st.plotly_chart(fig_mc, use_container_width=True)

else:
    st.warning("Failed to load historical data. Please check data connection.")
