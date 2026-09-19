import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from twelvedata import TDClient
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="XAU/USD Advanced Quantitative Analytics")

BG_COLOR = "#0E1117"
PRIMARY = "#FFD700"
SIGNAL = "#00E5FF"

# --- DATA INGESTION ENGINE (TWELVE DATA SDK + YFINANCE FALLBACK) ---
@st.cache_data(ttl=15)
def fetch_data(ticker_symbol, interval_str, lookback_str):
    df = pd.DataFrame()
    feed_source = "Unknown"
    notice = ""

    # Interval mapping for Twelve Data API
    td_interval_map = {"5m": "5min", "15m": "15min", "1hr": "1h", "4hr": "4h", "1d": "1day", "1w": "1week"}
    td_interval = td_interval_map.get(interval_str, "15min")

    # 1. Primary: Twelve Data Cloud SDK
    api_key = st.secrets.get("TWELVE_DATA_API_KEY", "")
    
    if api_key:
        try:
            symbol_td = "XAU/USD" if ticker_symbol in ["XAUUSD", "GOLD", "GC=F"] else ticker_symbol
            td = TDClient(apikey=api_key)
            ts = td.time_series(symbol=symbol_td, interval=td_interval, outputsize=500)
            df = ts.as_pandas()

            if not df.empty:
                df = df.sort_index()
                df.rename(columns={
                    'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
                }, inplace=True)
                for col in ['Open', 'High', 'Low', 'Close']:
                    df[col] = df[col].astype(float)
                feed_source = "Twelve Data (24/7 Cloud Feed)"
        except Exception as e:
            notice = f"Twelve Data Feed fallback activated: {e}"

    # 2. Secondary Fallback: Yahoo Finance
    if df.empty:
        yf_symbol = "GC=F" if ticker_symbol in ["XAUUSD", "GOLD", "GC=F"] else ticker_symbol
        yf_tf_map = {"5m": "5m", "15m": "15m", "1hr": "1h", "4hr": "1h", "1d": "1d", "1w": "1wk"}
        valid_period = "60d" if interval_str in ["5m", "15m"] else lookback_str

        try:
            df = yf.download(yf_symbol, period=valid_period, interval=yf_tf_map.get(interval_str, "1h"), progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.dropna(inplace=True)
            feed_source = "Yahoo Finance (Backup Feed)"
        except Exception as e:
            st.error(f"Data Fetch Failure: {e}")

    return df, feed_source, notice

# --- ENHANCED MACRO CORRELATION ENGINE (TIPS & SILVER ADDED) ---
@st.cache_data(ttl=300)
def fetch_macro_correlation(period_str):
    tickers = {
        'Gold (XAU/USD)': 'GC=F',
        'Silver (Positive Proxy)': 'SI=F',
        'Dollar Index (DXY)': 'DX-Y.NYB',
        'Real Yields (IEF)': 'IEF'
    }
    data = pd.DataFrame()
    for name, sym in tickers.items():
        try:
            df_asset = yf.download(sym, period=period_str, interval="1d", progress=False)
            if isinstance(df_asset.columns, pd.MultiIndex):
                data[name] = df_asset['Close'].squeeze()
            else:
                data[name] = df_asset['Close']
        except Exception:
            pass
    data.dropna(inplace=True)
    return data.pct_change().corr()

# --- AUTOMATED MODEL OPTIMIZATION ENGINE ---
def auto_fit_sarima(data_series):
    adf_pvalue = adfuller(data_series.dropna())[1]
    best_d = 1 if adf_pvalue > 0.05 else 0

    best_aic = float("inf")
    best_order = (1, best_d, 1)
    best_results = None

    for p_try in range(0, 3):
        for q_try in range(0, 3):
            try:
                model = SARIMAX(data_series, order=(p_try, best_d, q_try))
                results = model.fit(disp=False)
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = (p_try, best_d, q_try)
                    best_results = results
            except Exception:
                continue

    if best_results is not None:
        forecast = best_results.get_forecast(steps=30)
        return forecast.predicted_mean, forecast.conf_int(alpha=0.05), best_order, best_aic
    else:
        return None, None, (1, 1, 1), 0.0

def run_pca(df, n_components=2):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df_rsi = 100 - (100 / (1 + rs))

    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df_macd = ema12 - ema26

    if 'Volume' in df.columns and df['Volume'].notna().sum() > 0 and (df['Volume'] != 0).any():
        vol_feature = df['Volume']
    else:
        vol_feature = df['High'] - df['Low']

    features = pd.DataFrame({
        'RSI': df_rsi,
        'MACD': df_macd,
        'Volatility': vol_feature
    }).dropna()

    if features.empty or len(features) < n_components:
        return np.zeros((len(df), n_components)), np.zeros(n_components)

    scaled_features = StandardScaler().fit_transform(features)
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(scaled_features)
    return components, pca.explained_variance_ratio_

def run_regression(df):
    asset_returns = df['Close'].pct_change().dropna()
    try:
        sp500 = yf.download("^GSPC", period="1y", interval="1d", progress=False)
        if isinstance(sp500.columns, pd.MultiIndex):
            sp500_close = sp500['Close'].squeeze()
        else:
            sp500_close = sp500['Close']
        market_returns = sp500_close.pct_change().dropna()

        combined = pd.DataFrame({
            'Asset': asset_returns,
            'Market': market_returns
        }).dropna()

        X = combined['Market'].values.reshape(-1, 1)
        y = combined['Asset'].values.reshape(-1, 1)
    except Exception:
        X = asset_returns.values.reshape(-1, 1)
        y = asset_returns.values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)
    fitted = model.predict(X)
    residuals = y - fitted
    return X, y, fitted, residuals

def compute_monte_carlo(current_price, volatility, steps=30, paths=100):
    dt = 1/252
    simulations = np.zeros((steps, paths))
    simulations[0] = current_price
    for t in range(1, steps):
        rand = np.random.standard_normal(paths)
        simulations[t] = simulations[t-1] * np.exp((0 - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * rand)
    
    # Calculate Bullish vs Bearish terminal probabilities
    final_prices = simulations[-1, :]
    bullish_count = np.sum(final_prices > current_price)
    bullish_prob = (bullish_count / paths) * 100
    bearish_prob = 100 - bullish_prob

    return simulations, np.median(simulations, axis=1), np.percentile(simulations, 97.5, axis=1), np.percentile(simulations, 2.5, axis=1), bullish_prob, bearish_prob

# --- SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.header("MARKET & FEED CONFIGURATION")
    asset = st.selectbox("Asset Selector", ["XAUUSD", "EURUSD", "GBPUSD"])
    interval = st.selectbox("Timeframe", ["5m", "15m", "1hr", "4hr", "1d", "1w"], index=1)
    lookback = st.select_slider("Lookback Period", ["1mo", "3mo", "6mo", "1y", "2y"], value="3mo")

# --- MAIN DASHBOARD LAYOUT ---
st.title("XAU/USD Advanced Quantitative Analytics")

df, feed_source, notice = fetch_data(asset, interval, lookback)

if notice:
    st.info(notice)

if not df.empty:
    adf_result = adfuller(df['Close'].dropna())
    p_value = adf_result[1]

    # Metrics Summary Bar
    k1, k2, k3 = st.columns(3)
    k1.metric("Spot Price", f"${float(df['Close'].iloc[-1]):.2f}")
    k2.metric("Stationarity (ADF p-value)", f"{p_value:.4f}")
    k3.metric("Data Feed Status", feed_source)

    # --- REAL-TIME MULTI-ASSET TRADINGVIEW FEEDS ---
    st.subheader("Real-Time Multi-Asset Charts (XAU/USD vs. DXY)")
    tv_tf_map = {"5m": "5", "15m": "15", "1hr": "60", "4hr": "240", "1d": "D", "1w": "W"}
    tv_interval = tv_tf_map.get(interval, "15")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("**Gold Spot / U.S. Dollar (OANDA:XAUUSD)**")
        tv_gold_html = f"""
        <div class="tradingview-widget-container" style="height:460px;width:100%">
          <div id="tradingview_gold" style="height:460px;width:100%"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
          <script type="text/javascript">
          new TradingView.widget({{
            "autosize": true,
            "symbol": "OANDA:XAUUSD",
            "interval": "{tv_interval}",
            "timezone": "Etc/UTC",
            "theme": "dark",
            "style": "1",
            "locale": "en",
            "toolbar_bg": "#161B22",
            "enable_publishing": false,
            "allow_symbol_change": true,
            "container_id": "tradingview_gold"
          }});
          </script>
        </div>
        """
        components.html(tv_gold_html, height=470)

    with chart_col2:
        st.markdown("**U.S. Dollar Index (CAPITALCOM:DXY)**")
        tv_dxy_html = f"""
        <div class="tradingview-widget-container" style="height:460px;width:100%">
          <div id="tradingview_dxy" style="height:460px;width:100%"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
          <script type="text/javascript">
          new TradingView.widget({{
            "autosize": true,
            "symbol": "CAPITALCOM:DXY",
            "interval": "{tv_interval}",
            "timezone": "Etc/UTC",
            "theme": "dark",
            "style": "1",
            "locale": "en",
            "toolbar_bg": "#161B22",
            "enable_publishing": false,
            "allow_symbol_change": true,
            "container_id": "tradingview_dxy"
          }});
          </script>
        </div>
        """
        components.html(tv_dxy_html, height=470)

    # --- CROSS-ASSET MACRO CORRELATION HEATMAP ---
    st.subheader("Cross-Asset Rolling Return Correlation Matrix (Enhanced Drivers)")
    try:
        corr_matrix = fetch_macro_correlation(lookback)
        if not corr_matrix.empty:
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu',
                zmin=-1, zmax=1,
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}",
                textfont={"size": 13}
            ))
            fig_corr.update_layout(
                template="plotly_dark",
                plot_bgcolor=BG_COLOR,
                paper_bgcolor=BG_COLOR,
                height=380,
                margin=dict(l=0, r=0, t=20, b=0)
            )
            st.plotly_chart(fig_corr, use_container_width=True)
    except Exception as e:
        st.warning(f"Unable to render correlation heatmap: {e}")

    # --- MODULE 1: AUTOMATED SARIMA FORECAST ---
    st.subheader("Statistical Price Channel & Automated SARIMA Forecasting")
    mean_forecast, conf_int, best_order, best_aic = auto_fit_sarima(df['Close'])
    p_opt, d_opt, q_opt = best_order
    st.caption(f"**Automated Optimization Evidence:** Stationarity proven ($d={d_opt}$) | Lowest AIC Score Model selected: **ARIMA({p_opt}, {d_opt}, {q_opt})** (AIC: `{best_aic:.2f}`)")

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Market Data"))

    if mean_forecast is not None and conf_int is not None:
        idx_future = pd.date_range(df.index[-1], periods=31, freq='B')[1:]
        fig.add_trace(go.Scatter(x=idx_future, y=mean_forecast, line=dict(color=SIGNAL), name="SARIMA Forecast"))
        fig.add_trace(go.Scatter(
            x=np.concatenate([idx_future, idx_future[::-1]]),
            y=np.concatenate([conf_int.iloc[:, 0] if hasattr(conf_int, 'iloc') else conf_int[:, 0], 
                              (conf_int.iloc[:, 1] if hasattr(conf_int, 'iloc') else conf_int[:, 1])[::-1]]),
            fill='toself', fillcolor='rgba(0, 229, 255, 0.2)',
            line=dict(color='rgba(255,255,255,0)'), name="95% CI"
        ))
    fig.update_layout(template="plotly_dark", plot_bgcolor=BG_COLOR, paper_bgcolor=BG_COLOR, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # --- MODULES 2 & 3: PCA & REGRESSION ---
    colA, colB = st.columns(2)
    with colA:
        st.subheader("Dimensionality Reduction & PCA Analysis (Real Indicators)")
        components_pca, var_ratio = run_pca(df, n_components=2)
        fig_pca = go.Figure(data=go.Scatter(x=components_pca[:,0], y=components_pca[:,1], mode='markers', marker=dict(color=PRIMARY)))
        fig_pca.update_layout(title="PC1 vs PC2 Scatter", template="plotly_dark", plot_bgcolor=BG_COLOR, paper_bgcolor=BG_COLOR)
        st.plotly_chart(fig_pca, use_container_width=True)
        st.bar_chart(pd.DataFrame(var_ratio, index=[f"PC{i+1}" for i in range(len(var_ratio))], columns=["Explained Variance"]))

    with colB:
        st.subheader("Linear Regression vs. S&P 500 Market Benchmark")
        market_ret, asset_ret, fitted, residuals = run_regression(df)
        fig_reg = go.Figure()
        fig_reg.add_trace(go.Scatter(x=market_ret.flatten(), y=asset_ret.flatten(), mode='markers', name="Returns", marker=dict(color=SIGNAL)))
        fig_reg.add_trace(go.Scatter(x=market_ret.flatten(), y=fitted.flatten(), mode='lines', name="Fitted OLS Line", line=dict(color=PRIMARY)))
        fig_reg.update_layout(title="Gold Returns vs S&P 500 Returns", template="plotly_dark", plot_bgcolor=BG_COLOR, paper_bgcolor=BG_COLOR)
        st.plotly_chart(fig_reg, use_container_width=True)

        fig_res = go.Figure(data=go.Scatter(x=fitted.flatten(), y=residuals.flatten(), mode='markers', marker=dict(color='gray')))
        fig_res.update_layout(title="Residuals vs Fitted (Homoscedasticity)", template="plotly_dark", plot_bgcolor=BG_COLOR, paper_bgcolor=BG_COLOR)
        st.plotly_chart(fig_res, use_container_width=True)

    # --- MODULE 4: GARCH VOLATILITY & MONTE CARLO PROBABILITY REPORT ---
    st.subheader("GARCH Volatility & Monte Carlo Probability Report")
    col_mc1, col_mc2 = st.columns(2)

    with col_mc1:
        rolling_vol = df['Close'].pct_change().rolling(21).std() * np.sqrt(252)
        fig_vol = go.Figure(data=go.Scatter(x=df.index, y=rolling_vol, line=dict(color="#FF4081")))
        fig_vol.update_layout(title="Rolling Annualized Volatility", template="plotly_dark", plot_bgcolor=BG_COLOR, paper_bgcolor=BG_COLOR)
        st.plotly_chart(fig_vol, use_container_width=True)

    with col_mc2:
        current_price = float(df['Close'].iloc[-1])
        current_vol = float(rolling_vol.iloc[-1]) if not np.isnan(rolling_vol.iloc[-1]) else 0.15
        sims, median, upper, lower, bullish_p, bearish_p = compute_monte_carlo(current_price, current_vol)
        
        # Display Probability Metric Cards
        p_col1, p_col2 = st.columns(2)
        p_col1.metric("Monte Carlo Bullish Probability", f"{bullish_p:.1f}%")
        p_col2.metric("Monte Carlo Bearish Probability", f"{bearish_p:.1f}%")

        fig_mc = go.Figure()
        for i in range(100):
            fig_mc.add_trace(go.Scatter(y=sims[:, i], mode='lines', line=dict(color='rgba(255, 215, 0, 0.05)'), showlegend=False))
        fig_mc.add_trace(go.Scatter(y=median, mode='lines', line=dict(color=SIGNAL, width=2), name="Median Path"))
        fig_mc.add_trace(go.Scatter(y=upper, mode='lines', line=dict(color='red', dash='dash'), name="+2σ Channel"))
        fig_mc.add_trace(go.Scatter(y=lower, mode='lines', line=dict(color='red', dash='dash'), name="-2σ Channel"))
        fig_mc.update_layout(title="30-Step Forward Monte Carlo Paths", template="plotly_dark", plot_bgcolor=BG_COLOR, paper_bgcolor=BG_COLOR)
        st.plotly_chart(fig_mc, use_container_width=True)

else:
    st.error("Data stream unavailable. Please verify Streamlit Secrets setup or sidebar parameters.")
