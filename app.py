import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.optimize as sco
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Portfolio Optimizer & Risk Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E293B; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.0rem; color: #64748B; margin-bottom: 2rem; }
    .metric-card { background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Interactive Markowitz Portfolio Optimizer & Risk Engine</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Modern Portfolio Theory (MPT) Quantitative Analytics for JSE & Global Equities</div>', 
    unsafe_allow_html=True
)

# ==========================================
# 2. SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("🕹️ Model Parameters")

default_tickers = ["CPI.JO", "NPN.JO", "FSR.JO", "AGL.JO", "VOO", "QQQ"]
ticker_input = st.sidebar.text_input(
    "Asset Tickers (comma-separated):",
    value=", ".join(default_tickers),
    help="Use .JO for Johannesburg Stock Exchange tickers (e.g., CPI.JO, NPN.JO, FSR.JO, AGL.JO)."
)

tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

col_start, col_end = st.sidebar.columns(2)
start_date = col_start.date_input("Start Date", datetime.now() - timedelta(days=5*365))
end_date = col_end.date_input("End Date", datetime.now())

risk_free_rate_pct = st.sidebar.number_input(
    "Risk-Free Rate (%) [SA 10Y Yield]",
    min_value=0.0,
    max_value=20.0,
    value=8.95,
    step=0.05,
    help="Benchmark yield. South African 10-Year Government Bond Yield trades around 8.95%-9.10%."
)
rf_rate = risk_free_rate_pct / 100.0

num_simulations = st.sidebar.slider(
    "Monte Carlo Portfolios",
    min_value=1000,
    max_value=10000,
    value=5000,
    step=1000,
    help="Number of random portfolio allocations simulated to map the risk-return scatter."
)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Note on Multi-Currency Assets**:\n"
    "Daily percentage returns are scale-invariant ratios (r_t = ΔP / P_{t-1}). "
    "Therefore, JSE prices (quoted in ZAC) and US prices (quoted in USD) can be combined directly "
    "to compute accurate covariance matrices without needing daily exchange rate conversions."
)

# ==========================================
# 3. DATA INGESTION PIPELINE
# ==========================================
@st.cache_data(ttl=3600)
def fetch_market_data(ticker_list, start, end):
    try:
        data = yf.download(
            ticker_list,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
            group_by='ticker'
        )

        if data.empty:
            return pd.DataFrame(), pd.DataFrame()

        prices = data.copy()

        if isinstance(prices.columns, pd.MultiIndex):
            price_levels = set(prices.columns.get_level_values(1))
            if 'Adj Close' in price_levels:
                prices = prices.xs('Adj Close', level=1, axis=1)
            elif 'Close' in price_levels:
                prices = prices.xs('Close', level=1, axis=1)

        if isinstance(prices.columns, pd.MultiIndex):
            prices.columns = [str(col).strip().upper() for col in prices.columns.get_level_values(-1)]
        else:
            prices.columns = [str(col).strip().upper() for col in prices.columns]

        prices = prices.loc[:, ~prices.columns.duplicated()]
        prices = prices.dropna(how='all')
        returns = prices.pct_change().dropna()
        return prices, returns
    except Exception as e:
        st.error(f"Error fetching data from Yahoo Finance: {e}")
        return pd.DataFrame(), pd.DataFrame()

if len(tickers) < 2:
    st.warning("⚠️ Please specify at least two valid tickers to perform portfolio optimization.")
    st.stop()

with st.spinner("Fetching market data and computing statistical matrices..."):
    prices_df, returns_df = fetch_market_data(tickers, start_date, end_date)

if prices_df.empty or returns_df.empty:
    st.error("❌ Unable to retrieve market data. Please verify ticker symbols and date ranges.")
    st.stop()

available_tickers = [t for t in tickers if t in returns_df.columns]
if len(available_tickers) < 2:
    st.warning("⚠️ Not enough valid ticker data was returned to run the optimization. Please check the symbols and date range.")
    st.stop()

prices_df = prices_df[available_tickers]
returns_df = returns_df[available_tickers]
num_assets = len(available_tickers)

# ==========================================
# 4. MATHEMATICAL ENGINE & OPTIMIZATION
# ==========================================
mean_returns_daily = returns_df.mean()
cov_matrix_daily = returns_df.cov()

annualized_returns = mean_returns_daily * 252
annualized_cov = cov_matrix_daily * 252

def portfolio_performance(weights, mean_returns, cov_matrix):
    port_return = np.sum(mean_returns * weights)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return port_return, port_volatility

def min_neg_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
    p_ret, p_vol = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_vol

def min_variance(weights, mean_returns, cov_matrix):
    return portfolio_performance(weights, mean_returns, cov_matrix)[1] ** 2

# Optimization Constraints and Bounds (Long-Only, 100% Capital Allocation)
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
bounds = tuple((0.0, 1.0) for _ in range(num_assets))
initial_weights = np.array(num_assets * [1.0 / num_assets])

# Tangency Portfolio (Maximum Sharpe Ratio)
opt_sharpe = sco.minimize(
    min_neg_sharpe,
    initial_weights,
    args=(annualized_returns, annualized_cov, rf_rate),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

# Global Minimum Variance Portfolio (GMVP)
opt_var = sco.minimize(
    min_variance,
    initial_weights,
    args=(annualized_returns, annualized_cov),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

# Extract Optimal Results
max_sharpe_weights = opt_sharpe.x
ret_max_sp, vol_max_sp = portfolio_performance(max_sharpe_weights, annualized_returns, annualized_cov)
sharpe_max_sp = (ret_max_sp - rf_rate) / vol_max_sp

min_var_weights = opt_var.x
ret_min_var, vol_min_var = portfolio_performance(min_var_weights, annualized_returns, annualized_cov)
sharpe_min_var = (ret_min_var - rf_rate) / vol_min_var

# Monte Carlo Simulation
np.random.seed(42)
mc_weights = np.random.dirichlet(np.ones(num_assets), size=num_simulations)
mc_returns = np.dot(mc_weights, annualized_returns)
mc_volatilities = np.sqrt(np.einsum('ij,jk,ik->i', mc_weights, annualized_cov, mc_weights))
mc_sharpe = (mc_returns - rf_rate) / mc_volatilities

# ==========================================
# 5. VISUALIZATION & INTERACTIVE DASHBOARD
# ==========================================

# SECTION 1: Efficient Frontier Plot
st.subheader("📊 Markowitz Efficient Frontier")

fig_frontier = go.Figure()

# Monte Carlo Scatter
fig_frontier.add_trace(go.Scatter(
    x=mc_volatilities,
    y=mc_returns,
    mode='markers',
    marker=dict(
        color=mc_sharpe,
        colorscale='Viridis',
        colorbar=dict(title="Sharpe Ratio"),
        size=4,
        opacity=0.6
    ),
    name="Simulated Portfolios",
    hovertemplate="Volatility: %{x:.2%}<br>Expected Return: %{y:.2%}<extra></extra>"
))

# Individual Assets Markers
for i, ticker in enumerate(available_tickers):
    asset_ret = annualized_returns[ticker]
    asset_vol = np.sqrt(annualized_cov.iloc[i, i])
    fig_frontier.add_trace(go.Scatter(
        x=[asset_vol],
        y=[asset_ret],
        mode='markers+text',
        marker=dict(color='black', size=10, symbol='diamond'),
        text=[f"  {ticker}"],
        textposition="top right",
        name=f"Asset: {ticker}"
    ))

# Max Sharpe Portfolio Marker
fig_frontier.add_trace(go.Scatter(
    x=[vol_max_sp],
    y=[ret_max_sp],
    mode='markers',
    marker=dict(color='#EF4444', size=16, symbol='star'),
    name="Max Sharpe Ratio (Tangency)",
    hovertemplate=f"Max Sharpe Portfolio<br>Volatility: {vol_max_sp:.2%}<br>Return: {ret_max_sp:.2%}<extra></extra>"
))

# Minimum Variance Portfolio Marker
fig_frontier.add_trace(go.Scatter(
    x=[vol_min_var],
    y=[ret_min_var],
    mode='markers',
    marker=dict(color='#2563EB', size=14, symbol='square'),
    name="Minimum Variance (GMVP)",
    hovertemplate=f"Minimum Variance Portfolio<br>Volatility: {vol_min_var:.2%}<br>Return: {ret_min_var:.2%}<extra></extra>"
))

fig_frontier.update_layout(
    xaxis_title="Annualized Volatility (Risk)",
    yaxis_title="Annualized Expected Return",
    xaxis=dict(tickformat=".1%"),
    yaxis=dict(tickformat=".1%"),
    template="plotly_white",
    height=550,
    margin=dict(l=20, r=20, t=30, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig_frontier, width="stretch")

# SECTION 2: Optimal Portfolio Breakdown
st.subheader("⚖️ Strategy Performance & Capital Allocations")

col_ms, col_mv = st.columns(2)

with col_ms:
    st.markdown("### 🌟 Tangency Portfolio (Max Sharpe)")
    m1, m2, m3 = st.columns(3)
    m1.metric("Expected Return", f"{ret_max_sp:.2%}")
    m2.metric("Volatility", f"{vol_max_sp:.2%}")
    m3.metric("Sharpe Ratio", f"{sharpe_max_sp:.3f}")
    
    df_ms = pd.DataFrame({
        "Asset": available_tickers,
        "Weight": max_sharpe_weights
    }).sort_values(by="Weight", ascending=False)
    
    df_ms["Allocation (%)"] = (df_ms["Weight"] * 100).map("{:.2f}%".format)
    st.dataframe(df_ms[["Asset", "Allocation (%)"]], width="stretch", hide_index=True)

with col_mv:
    st.markdown("### 🛡️ Global Minimum Variance Portfolio")
    m1, m2, m3 = st.columns(3)
    m1.metric("Expected Return", f"{ret_min_var:.2%}")
    m2.metric("Volatility", f"{vol_min_var:.2%}")
    m3.metric("Sharpe Ratio", f"{sharpe_min_var:.3f}")
    
    df_mv = pd.DataFrame({
        "Asset": available_tickers,
        "Weight": min_var_weights
    }).sort_values(by="Weight", ascending=False)
    
    df_mv["Allocation (%)"] = (df_mv["Weight"] * 100).map("{:.2f}%".format)
    st.dataframe(df_mv[["Asset", "Allocation (%)"]], width="stretch", hide_index=True)

# SECTION 3: Visual Comparison & Correlation Matrix
col_bar, col_corr = st.columns(2)

with col_bar:
    st.subheader("📊 Asset Weight Allocations")
    alloc_df = pd.DataFrame({
        "Asset": available_tickers,
        "Max Sharpe": max_sharpe_weights,
        "Min Variance": min_var_weights
    }).melt(id_vars="Asset", var_name="Strategy", value_name="Weight")

    fig_bar = px.bar(
        alloc_df,
        x="Asset",
        y="Weight",
        color="Strategy",
        barmode="group",
        labels={"Weight": "Capital Allocation"},
        color_discrete_map={"Max Sharpe": "#EF4444", "Min Variance": "#2563EB"}
    )
    fig_bar.update_layout(yaxis=dict(tickformat=".0%"), template="plotly_white", height=400)
    st.plotly_chart(fig_bar, width="stretch")

with col_corr:
    st.subheader("🔗 Cross-Asset Correlation Heatmap")
    corr_matrix = returns_df.corr()
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        aspect="auto",
        zmin=-1, zmax=1
    )
    fig_corr.update_layout(template="plotly_white", height=400)
    st.plotly_chart(fig_corr, width="stretch")