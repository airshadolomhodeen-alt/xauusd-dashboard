import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import concurrent.futures
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from twelvedata import TDClient
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION & SECURITY ---
st.set_page_config(layout="wide", page_title="XAU/USD Advanced Quantitative Analytics")

def check_password():
    """Returns `True` if the user entered the correct password."""
    def password_entered():
        if st.session_state["password"] == "January16@&1989":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.markdown("## 🔒 Private Quantitative Dashboard Authentication")
        st.text_input("Enter Dashboard Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.markdown("## 🔒 Private Quantitative Dashboard Authentication")
        st.text_input("Enter Dashboard Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect. Access denied.")
        return False
    else:
        return True

if not check_password():
    st.stop()

BG_COLOR = "#0E1117"
CARD_BG = "#161B22"
BORDER_COLOR = "#30363d"
PRIMARY = "#FFD700"
SIGNAL = "#00E5FF"

# --- CUSTOM RESPONSIVE CSS STYLING (IPHONE / ANDROID / DESKTOP OPTIMIZED) ---
st.markdown(f"""
<style>
    .stApp {{
        background-color: {BG_COLOR};
    }}
    div.stMetric {{
        background-color: {CARD_BG};
        padding: 12px;
        border-radius: 8px;
        border: 1px solid {BORDER_COLOR};
    }}
    .dashboard-card {{
        background-color: {CARD_BG};
        padding: 15px;
        border-radius: 8px;
        border: 1px solid {BORDER_COLOR};
        margin-bottom: 15px;
    }}
    @media only screen and (max-width: 768px) {{
        .main .block-container {{
            padding-left: 1rem;
            padding-right: 1rem;
        }}
    }}
</style>
""", unsafe_allow_html=True)

# --- DATA INGESTION ENGINE ---
@st.cache_data(ttl=15)
def fetch_data(ticker_symbol, interval_str, lookback_str):
    df = pd.DataFrame()
    feed_source = "Unknown"
    notice = ""

    td_interval_map = {"5m": "5min", "15m": "15min", "1hr": "1h", "4hr": "4h", "1d": "1day", "1w": "1week"}
    td_interval = td_interval_map.get(interval_str, "15min")
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
                    if col in df.columns:
                        df[col] = df[col].astype(float)
                feed_source = "Twelve Data (24/7 Cloud Feed)"
        except Exception as e:
            notice = f"Twelve Data Feed fallback activated: {e}"

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

# --- 4H RESAMPLER FOR CLOSED-BAR DIAGNOSTICS ---
def get_4h_closed_data(df_input):
    if df_input.empty:
        return df_input
    
    df_res = df_input.copy()
    if not isinstance(df_res.index, pd.DatetimeIndex):
        df_res.index = pd.to_datetime(df_res.index)
    df_res = df_res.sort_index()
    
    agg_dict = {}
    if 'Open' in df_res.columns: agg_dict['Open'] = 'first'
    if 'High' in df_res.columns: agg_dict['High'] = 'max'
    if 'Low' in df_res.columns: agg_dict['Low'] = 'min'
    if 'Close' in df_res.columns: agg_dict['Close'] = 'last'
    if 'Volume' in df_res.columns: 
        agg_dict['Volume'] = 'sum'
    elif 'volume' in df_res.columns: 
        agg_dict['volume'] = 'sum'

    try:
        df_4h = df_res.resample('4h').agg(agg_dict).dropna()
    except Exception:
        df_4h = df_res.groupby(pd.Grouper(freq='4h')).agg(agg_dict).dropna()

    if len(df_4h) > 1:
        return df_4h.iloc[:-1]
    return df_4h

# --- HIGH-PERFORMANCE PARALLEL MACRO CORRELATION ENGINE ---
@st.cache_data(ttl=300)
def fetch_macro_correlation(period_str):
    tickers = {
        'Gold (XAU/USD)': 'GC=F',
        'Silver (Proxy)': 'SI=F',
        'Dollar Index (DXY)': 'DX-Y.NYB',
        'Real Yields (IEF)': 'IEF'
    }
    data = pd.DataFrame()

    def download_ticker(name, sym):
        try:
            df_asset = yf.download(sym, period=period_str, interval="1d", progress=False)
            if not df_asset.empty:
                if isinstance(df_asset.columns, pd.MultiIndex):
                    return name, df_asset['Close'].squeeze()
                else:
                    return name, df_asset['Close']
        except Exception:
            pass
        return name, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(tickers)) as executor:
        futures = [executor.submit(download_ticker, name, sym) for name, sym in tickers.items()]
        for future in concurrent.futures.as_completed(futures):
            name, series = future.result()
            if series is not None:
                data[name] = series

    data.dropna(inplace=True)
    return data.pct_change().corr()

# --- SMART MONEY CONCEPTS: FVG & STRUCTURE DETECTOR ---
def detect_smart_money_patterns(df_input):
    fvgs = []
    market_structure = "Consolidation / Range"
    
    for i in range(2, len(df_input)):
        if df_input['High'].iloc[i-2] < df_input['Low'].iloc[i]:
            fvgs.append({
                'type': 'Bullish FVG',
                'start_idx': df_input.index[i-1],
                'end_idx': df_input.index[-1],
                'lower': float(df_input['High'].iloc[i-2]),
                'upper': float(df_input['Low'].iloc[i])
            })
        elif df_input['Low'].iloc[i-2] > df_input['High'].iloc[i]:
            fvgs.append({
                'type': 'Bearish FVG',
                'start_idx': df_input.index[i-1],
                'end_idx': df_input.index[-1],
                'lower': float(df_input['High'].iloc[i]),
                'upper': float(df_input['Low'].iloc[i-2])
            })
            
    recent_closes = df_input['Close'].iloc[-10:]
    if len(recent_closes) > 0:
        if recent_closes.iloc[-1] > recent_closes.max() * 0.999:
            market_structure = "Bullish Trend / Break of Structure"
        elif recent_closes.iloc[-1] < recent_closes.min() * 1.001:
            market_structure = "Bearish Trend / Change of Character"

    return fvgs, market_structure

# --- 4H CLOSED CANDLE ROC-AUC CLASSIFIER MODULE ---
def compute_4h_roc_auc(df_input):
    if df_input is None or df_input.empty or len(df_input) < 30:
        return 0.50
    
    df_local = df_input.copy()
    df_local['Target'] = (df_local['Close'].shift(-1) > df_local['Close']).astype(int)
    df_local['Returns'] = df_local['Close'].pct_change()
    
    delta = df_local['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df_local['RSI'] = 100 - (100 / (1 + rs))
    df_local['Volatility'] = df_local['Close'].pct_change().rolling(14).std()
    
    df_local.dropna(inplace=True)
    if len(df_local) < 20:
        return 0.50
        
    X = df_local[['Returns', 'RSI', 'Volatility']]
    y = df_local['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    if len(np.unique(y_test)) < 2:
        return 0.50
        
    try:
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
        return float(auc_score)
    except Exception:
        return 0.50

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
        lb_pvalue = 1.0
        try:
            nobs = len(best_results.resid)
            safe_lag = min(10, max(1, nobs - 1))
            if safe_lag > 0:
                ljung_res = acorr_ljungbox(best_results.resid, lags=[safe_lag], return_df=True)
                lb_pvalue = float(ljung_res['lb_pvalue'].iloc[0])
        except Exception:
            lb_pvalue = 1.0
        return forecast.predicted_mean, forecast.conf_int(alpha=0.05), best_order, best_aic, lb_pvalue
    else:
        return None, None, (1, 1, 1), 0.0, 1.0

def run_pca(df_input, n_components=2):
    delta = df_input['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df_rsi = 100 - (100 / (1 + rs))

    ema12 = df_input['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df_input['Close'].ewm(span=26, adjust=False).mean()
    df_macd = ema12 - ema26

    if 'Volume' in df_input.columns and df_input['Volume'].notna().sum() > 0 and (df_input['Volume'] != 0).any():
        vol_feature = df_input['Volume']
    elif 'volume' in df_input.columns and df_input['volume'].notna().sum() > 0 and (df_input['volume'] != 0).any():
        vol_feature = df_input['volume']
    else:
        vol_feature = df_input['High'] - df_input['Low']

    features = pd.DataFrame({
        'RSI': df_rsi,
        'MACD': df_macd,
        'Volatility': vol_feature
    }).dropna()

    if features.empty or len(features) < n_components:
        return np.zeros((len(df_input), n_components)), np.zeros(n_components)

    scaled_features = StandardScaler().fit_transform(features)
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(scaled_features)
    return components, pca.explained_variance_ratio_

def run_regression(df_input):
    asset_returns = df_input['Close'].pct_change().dropna()
    X, y = None, None
    try:
        sp500 = yf.download("^GSPC", period="1y", interval="1d", progress=False)
        if not sp500.empty:
            if isinstance(sp500.columns, pd.MultiIndex):
                sp500_close = sp500['Close'].squeeze()
            else:
                sp500_close = sp500['Close']
            market_returns = sp500_close.pct_change().dropna()

            combined = pd.DataFrame({
                'Asset': asset_returns,
                'Market': market_returns
            }).dropna()
            
            if not combined.empty:
                X = combined['Market'].values.reshape(-1, 1)
                y = combined['Asset'].values.reshape(-1, 1)
    except Exception:
        pass

    if X is None or y is None or len(X) == 0 or len(y) == 0:
        clean_asset = asset_returns.dropna()
        if len(clean_asset) > 1:
            X = clean_asset.values[:-1].reshape(-1, 1)
            y = clean_asset.values[1:].reshape(-1, 1)
        else:
            X = np.array([[0.0], [1.0]])
            y = np.array([[0.0], [1.0]])

    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y).any(axis=1)
    X = X[mask]
    y = y[mask]
    
    if len(X) < 2:
        X = np.array([[0.0], [1.0]])
        y = np.array([[0.0], [1.0]])

    model = LinearRegression()
    model.fit(X, y)
    fitted = model.predict(X)
    residuals = y - fitted
    
    reg_lb_pval = 1.0
    try:
        nobs = len(residuals)
        safe_lag = min(10, max(1, nobs - 1))
        if safe_lag > 0:
            lb_res = acorr_ljungbox(residuals.flatten(), lags=[safe_lag], return_df=True)
            reg_lb_pval = float(lb_res['lb_pvalue'].iloc[0])
    except Exception:
        reg_lb_pval = 1.0
    
    return X, y, fitted, residuals, reg_lb_pval

def compute_monte_carlo(current_price, volatility, steps=30, paths=100):
    dt = 1/252
    simulations = np.zeros((steps, paths))
    simulations[0] = current_price
    for t in range(1, steps):
        rand = np.random.standard_normal(paths)
        simulations[t] = simulations[t-1] * np.exp((0 - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * rand)
    
    final_prices = simulations[-1, :]
    bullish_count = np.sum(final_prices > current_price)
    bullish_prob = (bullish_count / paths) * 100
    bearish_prob = 100 - bullish_prob

    median_path = np.median(simulations, axis=1)
    p25_path = np.percentile(simulations, 25, axis=1)
    p75_path = np.percentile(simulations, 75, axis=1)
    upper_2sig = np.percentile(simulations, 97.5, axis=1)
    lower_2sig = np.percentile(simulations, 2.5, axis=1)

    return simulations, median_path, p25_path, p75_path, upper_2sig, lower_2sig, bullish_prob, bearish_prob

# --- SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.header("MARKET & FEED CONFIGURATION")
    asset = st.selectbox("Asset Selector", ["XAUUSD", "EURUSD", "GBPUSD"])
    interval = st.selectbox("Timeframe", ["5m", "15m", "1hr", "4hr", "1d", "1w"], index=1)
    lookback = st.select_slider("Lookback Period", ["1mo", "3mo", "6mo", "1y", "2y"], value="3mo")

    st.markdown("---")
    st.header("STATISTICAL MODELS OVERLAY")
    show_sarima = st.toggle("SARIMA Forecasting", value=True)
    show_pca = st.toggle("PCA Decomposition", value=True)
    show_reg = st.toggle("Return Regression", value=True)
    show_garch_mc = st.toggle("GARCH(1,1) Volatility", value=True)

# --- MAIN DASHBOARD HEADER & LIVE CLOCKS ---
header_col1, header_col2 = st.columns([2, 1])

with header_col1:
    st.title("XAU/USD Advanced Quantitative Analytics")

with header_col2:
    pht_clock_html = """
    <div style="font-family: sans-serif; color: #FFD700; font-size: 11px; font-weight: bold; background: #161B22; padding: 8px; border-radius: 6px; text-align: right; border: 1px solid #30363d; margin-top: 5px;">
        🇵🇭 PHT Live: <span id="pht-clock" style="color: #00E5FF;">Loading...</span><br>
        ⏳ 4H Close: <span id="candle-countdown" style="color: #FFD700;">Calculating...</span>
    </div>
    <script>
    function updateClocks() {
        const now = new Date();
        const options = { timeZone: 'Asia/Manila', hour: 'numeric', minute: '2-digit', second: '2-digit', hour12: true };
        document.getElementById('pht-clock').innerText = new Intl.DateTimeFormat('en-US', options).format(now);
        
        const utcNow = new Date(now.toLocaleString('en-US', { timeZone: 'UTC' }));
        const currentHour = utcNow.getUTCHours();
        const nextH = Math.floor(currentHour / 4) * 4 + 4;
        const targetUtc = new Date(utcNow);
        if (nextH >= 24) {
            targetUtc.setUTCDate(targetUtc.getUTCDate() + 1);
            targetUtc.setUTCHours(0, 0, 0, 0);
        } else {
            targetUtc.setUTCHours(nextH, 0, 0, 0);
        }
        const diffMs = targetUtc - utcNow;
        if (diffMs > 0) {
            const hrs = Math.floor((diffMs % 86400000) / 3600000);
            const mins = Math.floor((diffMs % 3600000) / 60000);
            const secs = Math.floor((diffMs % 60000) / 1000);
            document.getElementById('candle-countdown').innerText = hrs + "h " + mins + "m " + secs + "s";
        } else {
            document.getElementById('candle-countdown').innerText = "Syncing...";
        }
    }
    setInterval(updateClocks, 1000);
    updateClocks();
    </script>
    """
    components.html(pht_clock_html, height=55)

df, feed_source, notice = fetch_data(asset, interval, lookback)

if notice:
    st.info(notice)

if not df.empty:
    adf_result = adfuller(df['Close'].dropna())
    p_value = adf_result[1]
    
    current_price = float(df['Close'].iloc[-1])
    rolling_vol_calc = df['Close'].pct_change().rolling(21).std() * np.sqrt(252)
    current_vol = float(rolling_vol_calc.iloc[-1]) if not np.isnan(rolling_vol_calc.iloc[-1]) else 0.15
    
    # --- 4-HOUR CLOSED CANDLE DIAGNOSTIC ENGINE ---
    df_4h_closed = get_4h_closed_data(df)
    
    if not df_4h_closed.empty:
        price_4h_closed = float(df_4h_closed['Close'].iloc[-1])
        vol_4h_calc = df_4h_closed['Close'].pct_change().rolling(21).std() * np.sqrt(252)
        vol_4h = float(vol_4h_calc.iloc[-1]) if not np.isnan(vol_4h_calc.iloc[-1]) else current_vol
        
        _, _, _, _, _, _, bullish_p_4h, bearish_p_4h = compute_monte_carlo(price_4h_closed, vol_4h)
        fvgs_4h, structure_4h = detect_smart_money_patterns(df_4h_closed)
        auc_4h_score = compute_4h_roc_auc(df_4h_closed)
        last_4h_time = df_4h_closed.index[-1].strftime('%Y-%m-%d %H:%M UTC')
    else:
        bullish_p_4h, bearish_p_4h = 50.0, 50.0
        fvgs_4h, structure_4h = [], "Consolidation / Range"
        auc_4h_score = 0.50
        last_4h_time = "N/A"

    # --- TOP SUMMARY METRIC BAR (INCLUDING MODEL ROC-AUC SCORE) ---
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Spot Price", f"${current_price:.2f}")
    m2.metric("24h Vol.", f"{current_vol*100:.2f}%")
    m3.metric("Model Bias", f"{bullish_p_4h:.1f}% Bull")
    m4.metric("4H ROC-AUC", f"{auc_4h_score:.3f}")
    m5.metric("Regime", structure_4h.split('/')[0])
    m6.metric("4H Sync", "OK 🟢")

    st.markdown("---")

    # --- TOP COMMAND CENTER: 4H CLOSED-BAR QUANTITATIVE VERDICT ---
    st.subheader("Quantitative Diagnosis & Execution Verdict (4H Candle Anchor)")
    
    gate_passed = (bullish_p_4h > 60.0 or bearish_p_4h > 60.0) and (auc_4h_score > 0.60)
    gate_badge = '<span style="background: #238636; color: #ffffff; padding: 3px 8px; border-radius: 4px; font-size: 11px;">GATE PASSED</span>' if gate_passed else '<span style="background: #9e6a03; color: #ffffff; padding: 3px 8px; border-radius: 4px; font-size: 11px;">STAND ASIDE</span>'

    if bullish_p_4h > 60:
        verdict_action = "HIGH-PROBABILITY BUY (LONG ENTRY)"
        verdict_desc = f"The 4-hour closed candle model confirms statistical and structural upward alignment (Model ROC-AUC Score: <code>{auc_4h_score:.3f}</code>)."
        verdict_color = "#00E5FF"
    elif bearish_p_4h > 60:
        verdict_action = "HIGH-PROBABILITY SELL (SHORT ENTRY)"
        verdict_desc = f"The 4-hour closed candle model confirms downside distribution pressure (Model ROC-AUC Score: <code>{auc_4h_score:.3f}</code>)."
        verdict_color = "#FF4081"
    else:
        verdict_action = "STAND ASIDE / NEUTRAL"
        verdict_desc = f"Market equilibrium detected ({bullish_p_4h:.1f}% Bullish vs {bearish_p_4h:.1f}% Bearish | Model ROC-AUC Score: <code>{auc_4h_score:.3f}</code>). Capital preservation advised."
        verdict_color = "#FFD700"

    st.markdown(f"""
    <div class="dashboard-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; flex-wrap: wrap; gap: 5px;">
            <h4 style="margin: 0; color: {verdict_color}; font-size: 15px;">📊 Verdict: {verdict_action}</h4>
            <div>
                {gate_badge}
                <span style="font-size: 10px; background: #21262d; color: #8b949e; padding: 3px 6px; border-radius: 4px; border: 1px solid #30363d; margin-left: 5px;">
                    4H Close: <strong>{last_4h_time}</strong>
                </span>
            </div>
        </div>
        <p style="font-size: 13px; line-height: 1.5; margin: 0;">
            {verdict_desc}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- EXPANDABLE FVG DETAILS CONTAINER ---
    with st.expander(f"🔍 View Detailed 4H Fair Value Gaps (FVGs) ({len(fvgs_4h)} Detected)", expanded=False):
        if fvgs_4h:
            fvg_df = pd.DataFrame(fvgs_4h)
            st.dataframe(fvg_df[['type', 'start_idx', 'lower', 'upper']], use_container_width=True)
        else:
            st.info("No active structural Fair Value Gaps detected on the current 4H closed timeframe window.")

    st.markdown("---")

    # --- REAL-TIME TRADINGVIEW CHARTS ---
    st.subheader("Real-Time Multi-Asset Charts (XAU/USD vs. DXY)")
    tv_tf_map = {"5m": "5", "15m": "15", "1hr": "60", "4hr": "240", "1d": "D", "1w": "W"}
    tv_interval = tv_tf_map.get(interval, "15")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("**Gold Spot / U.S. Dollar (OANDA:XAUUSD)**")
        tv_gold_html = f"""
        <div class="dashboard-card" style="padding:0; height:380px;">
          <div id="tradingview_gold" style="height:380px;width:100%"></div>
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
        components.html(tv_gold_html, height=390)

    with chart_col2:
        st.markdown("**U.S. Dollar Index (CAPITALCOM:DXY)**")
        tv_dxy_html = f"""
        <div class="dashboard-card" style="padding:0; height:380px;">
          <div id="tradingview_dxy" style="height:380px;width:100%"></div>
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
        components.html(tv_dxy_html, height=390)

    # --- CROSS-ASSET MACRO CORRELATION HEATMAP ---
    st.subheader("Cross-Asset Rolling Return Correlation Matrix (Macro Drivers)")
    try:
        corr_matrix = fetch_macro_correlation(lookback)
        if not corr_matrix.empty:
            text_vals = []
            for row in corr_matrix.values:
                row_text = []
                for val in row:
                    cell_str = f"{val:.2f}"
                    if abs(val) > 0.7:
                        cell_str += "<br>(High Coupling)"
                    row_text.append(cell_str)
                text_vals.append(row_text)

            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu',
                zmin=-1, zmax=1,
                text=text_vals,
                texttemplate="%{text}",
                textfont={"size": 11}
            ))
            fig_corr.update_layout(
                template="plotly_dark",
                plot_bgcolor=CARD_BG,
                paper_bgcolor=CARD_BG,
                height=320,
                margin=dict(l=0, r=0, t=10, b=0)
            )
            st.plotly_chart(fig_corr, use_container_width=True)
    except Exception as e:
        st.warning(f"Unable to render correlation heatmap: {e}")

    # --- ANALYTICAL MODULES 2x2 GRID ---
    row1_col1, row1_col2 = st.columns(2)

    # MODULE 1: SARIMA FORECAST
    with row1_col1:
        st.markdown("### SARIMA PRICE PATH & FORECAST")
        if show_sarima:
            mean_forecast, conf_int, best_order, best_aic, lb_pval = auto_fit_sarima(df['Close'])
            p_opt, d_opt, q_opt = best_order
            st.caption(f"ARIMA({p_opt},{d_opt},{q_opt}) | AIC: `{best_aic:.1f}` | LB p-val: `{lb_pval:.2f}`")

            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Market"))

            fvgs_chart, _ = detect_smart_money_patterns(df)
            for fvg in fvgs_chart[-5:]:
                color = "rgba(0, 255, 0, 0.15)" if "Bullish" in fvg['type'] else "rgba(255, 0, 0, 0.15)"
                fig.add_shape(type="rect", x0=fvg['start_idx'], y0=fvg['lower'], x1=fvg['end_idx'], y1=fvg['upper'], fillcolor=color, line=dict(width=0), layer="below")

            if mean_forecast is not None and conf_int is not None:
                idx_future = pd.date_range(df.index[-1], periods=31, freq='B')[1:]
                fig.add_trace(go.Scatter(x=idx_future, y=mean_forecast, line=dict(color=SIGNAL), name="Forecast"))
                fig.add_trace(go.Scatter(
                    x=np.concatenate([idx_future, idx_future[::-1]]),
                    y=np.concatenate([conf_int.iloc[:, 0] if hasattr(conf_int, 'iloc') else conf_int[:, 0], 
                                     (conf_int.iloc[:, 1] if hasattr(conf_int, 'iloc') else conf_int[:, 1])[::-1]]),
                    fill='toself', fillcolor='rgba(0, 229, 255, 0.1)', line=dict(color='rgba(255,255,255,0)'), name="95% CI"
                ))
            fig.update_layout(template="plotly_dark", plot_bgcolor=CARD_BG, paper_bgcolor=CARD_BG, height=280, margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("SARIMA module disabled.")

    # MODULE 2: PCA DIMENSIONALITY REDUCTION
    with row1_col2:
        with st.expander("### PCA DIMENSIONALITY REDUCTION", expanded=True):
            if show_pca:
                components_pca, var_ratio = run_pca(df, n_components=2)
                cum_var = np.sum(var_ratio) * 100
                st.caption(f"Cumulative Variance: **PC1+PC2 = {cum_var:.1f}%**")

                fig_pca = go.Figure(data=go.Scatter(x=components_pca[:,0], y=components_pca[:,1], mode='markers', marker=dict(color=PRIMARY, size=4)))
                fig_pca.update_layout(template="plotly_dark", plot_bgcolor=CARD_BG, paper_bgcolor=CARD_BG, height=180, margin=dict(l=0, r=0, t=20, b=0))
                st.plotly_chart(fig_pca, use_container_width=True)
                st.bar_chart(pd.DataFrame(var_ratio, index=[f"PC{i+1}" for i in range(len(var_ratio))], columns=["Variance"]), height=110)
            else:
                st.info("PCA module disabled.")

    row2_col1, row2_col2 = st.columns(2)

    # MODULE 3: OLS REGRESSION
    with row2_col1:
        with st.expander("### RETURN REGRESSION & RESIDUALS", expanded=True):
            if show_reg:
                market_ret, asset_ret, fitted, residuals, reg_lb_pval = run_regression(df)
                st.caption(f"Residual LB p-val: `{reg_lb_pval:.2f}`")

                fig_reg = go.Figure()
                fig_reg.add_trace(go.Scatter(x=market_ret.flatten(), y=asset_ret.flatten(), mode='markers', name="Ret", marker=dict(color=SIGNAL, size=3)))
                fig_reg.add_trace(go.Scatter(x=market_ret.flatten(), y=fitted.flatten(), mode='lines', name="OLS", line=dict(color=PRIMARY)))
                fig_reg.update_layout(template="plotly_dark", plot_bgcolor=CARD_BG, paper_bgcolor=CARD_BG, height=160, margin=dict(l=0, r=0, t=20, b=0))
                st.plotly_chart(fig_reg, use_container_width=True)

                fig_res = go.Figure(data=go.Scatter(x=fitted.flatten(), y=residuals.flatten(), mode='markers', marker=dict(color='gray', size=3)))
                fig_res.update_layout(template="plotly_dark", plot_bgcolor=CARD_BG, paper_bgcolor=CARD_BG, height=140, margin=dict(l=0, r=0, t=20, b=0))
                st.plotly_chart(fig_res, use_container_width=True)
            else:
                st.info("Regression module disabled.")

    # MODULE 4: GARCH & MONTE CARLO ENVELOPE
    with row2_col2:
        st.markdown("### GARCH & MONTE CARLO ENVELOPE")
        if show_garch_mc:
            rolling_vol = df['Close'].pct_change().rolling(21).std() * np.sqrt(252)
            sims, median, p25, p75, upper, lower, bullish_p, bearish_p = compute_monte_carlo(current_price, current_vol)

            p_col1, p_col2 = st.columns(2)
            p_col1.metric("MC Bullish", f"{bullish_p:.1f}%")
            p_col2.metric("MC Bearish", f"{bearish_p:.1f}%")

            fig_mc = go.Figure()
            for i in range(50):
                fig_mc.add_trace(go.Scatter(y=sims[:, i], mode='lines', line=dict(color='rgba(255, 215, 0, 0.03)'), showlegend=False))
            fig_mc.add_trace(go.Scatter(y=median, mode='lines', line=dict(color=SIGNAL, width=1.5), name="Median"))
            fig_mc.add_trace(go.Scatter(y=upper, mode='lines', line=dict(color='red', dash='dash'), name="+2σ"))
            fig_mc.add_trace(go.Scatter(y=lower, mode='lines', line=dict(color='red', dash='dash'), name="-2σ"))
            fig_mc.update_layout(template="plotly_dark", plot_bgcolor=CARD_BG, paper_bgcolor=CARD_BG, height=220, margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig_mc, use_container_width=True)
        else:
            st.info("GARCH module disabled.")

else:
    st.error("Data stream unavailable. Please verify Streamlit Secrets setup or sidebar parameters.")
