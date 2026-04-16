from __future__ import annotations

import io
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import MarketDataFetcher, UniverseLoader
from src.factor_engine import FactorEngine
from src.ranker import FactorWeights, PortfolioRanker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Smart Beta Screener",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


@st.cache_data(show_spinner=False, ttl=3600)
def load_universe() -> List[str]:
    loader = UniverseLoader(cache_dir=DATA_DIR)
    return loader.load()


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_market_data(tickers: Tuple[str, ...], period: str) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    fetcher = MarketDataFetcher(period=period)
    return fetcher.fetch(list(tickers))


@st.cache_data(show_spinner=False)
def compute_factors(prices_json: str, fundamentals: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    prices = pd.read_json(io.StringIO(prices_json), orient="split")
    engine = FactorEngine(prices=prices, fundamentals=fundamentals)
    return engine.compute_all_factors()


def _bar_chart(series: pd.Series, title: str, x_label: str, top_n: int = 20) -> go.Figure:
    top = series.head(top_n).sort_values(ascending=True)

    if (top.values < 0).any():
        colors = ['#d62728' if val < 0 else '#2ca02c' for val in top.values]
    else:
        colors = ['#1f77b4'] * len(top)

    fig = px.bar(
        x=top.values, y=top.index, orientation="h", title=title,
        labels={"x": x_label, "y": "Ticker"}
    )
    fig.update_traces(marker_color=colors)
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def _scatter_factors(factor_df: pd.DataFrame, col_x: str, col_y: str) -> go.Figure:
    df = factor_df[[col_x, col_y]].dropna().reset_index()
    df.columns = pd.Index(["Ticker", col_x, col_y])
    fig = px.scatter(
        df, x=col_x, y=col_y, text="Ticker",
        title=f"{col_x.capitalize()} vs {col_y.capitalize()}"
    )
    fig.update_traces(textposition="top center", textfont_size=9, marker=dict(size=7, opacity=0.7))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def _radar_chart(row: pd.Series, ticker: str) -> go.Figure:
    z_cols = ["z_momentum", "z_volatility", "z_value", "z_quality"]
    labels = ["Momentum", "Low Vol", "Value", "Quality"]
    values = [row.get(c, 0.0) or 0.0 for c in z_cols]
    values_closed = values + [values[0]]
    labels_closed = labels + [labels[0]]

    fig = go.Figure(
        go.Scatterpolar(
            r=values_closed, theta=labels_closed, fill="toself",
            line_width=2
        )
    )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=False, title=f"{ticker} Factor Profile", height=340,
        margin=dict(l=40, r=40, t=50, b=20)
    )
    return fig


def main() -> None:
    st.markdown("<h1>Smart Beta Screener</h1>", unsafe_allow_html=True)
    st.markdown("<p>Factor Investing Dashboard | S&P 100 Universe</p>", unsafe_allow_html=True)
    st.divider()

    with st.sidebar:
        st.markdown("### Configuration")
        st.markdown("---")
        data_period = st.selectbox("Historical Data Period", options=["1y", "2y", "3y"], index=1)
        st.markdown("### Factor Weights")
        w_momentum = st.slider("Momentum", 0.0, 3.0, 1.0, 0.1)
        w_volatility = st.slider("Low Volatility", 0.0, 3.0, 1.0, 0.1)
        w_value = st.slider("Value", 0.0, 3.0, 1.0, 0.1)
        w_quality = st.slider("Quality", 0.0, 3.0, 1.0, 0.1)
        st.markdown("---")
        top_n = st.slider("Top N Display", 5, 50, 20, 5)
        st.markdown("---")
        run_btn = st.button("Run Screener")

    if "ranked_df" not in st.session_state:
        st.session_state["ranked_df"] = None
    if "factor_df" not in st.session_state:
        st.session_state["factor_df"] = None

    if run_btn:
        weights = FactorWeights(momentum=w_momentum, volatility=w_volatility, value=w_value, quality=w_quality)
        with st.spinner("Loading universe..."):
            tickers = load_universe()

        progress_bar = st.progress(0, text="Fetching market data...")
        try:
            prices, fundamentals = fetch_market_data(tickers=tuple(tickers), period=data_period)
        except Exception as exc:
            st.error(f"Market data fetch failed: {exc}")
            return

        progress_bar.progress(50, text="Computing factors...")
        try:
            factor_df = compute_factors(prices_json=prices.to_json(orient="split"), fundamentals=fundamentals)
        except Exception as exc:
            st.error(f"Factor computation failed: {exc}")
            return

        progress_bar.progress(80, text="Ranking...")
        try:
            ranker = PortfolioRanker(factor_df=factor_df, weights=weights)
            ranked_df = ranker.rank()
        except Exception as exc:
            st.error(f"Ranking failed: {exc}")
            return

        progress_bar.progress(100, text="Done")
        progress_bar.empty()
        st.session_state["ranked_df"] = ranked_df
        st.session_state["factor_df"] = factor_df

    ranked_df: Optional[pd.DataFrame] = st.session_state.get("ranked_df")
    factor_df: Optional[pd.DataFrame] = st.session_state.get("factor_df")

    if ranked_df is None:
        st.info("Adjust the factor weights in the sidebar and execute the screener.")
        return

    top1 = ranked_df.index[0]
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    col_k1.metric("Universe Size", f"{len(ranked_df)}")
    col_k2.metric("Top Ranked Stock", top1, f"Score: {ranked_df.loc[top1, 'composite_score']:.3f}")
    col_k3.metric("Avg Composite Score", f"{ranked_df['composite_score'].mean():.3f}")
    col_k4.metric("Score Std Dev", f"{ranked_df['composite_score'].std():.3f}")
    st.divider()

    tab_rank, tab_factors, tab_explore, tab_detail = st.tabs(
        ["Rankings", "Factor Analysis", "Factor Explorer", "Stock Detail"]
    )

    with tab_rank:
        st.markdown("### Top Ranked Stocks")
        col_chart, col_table = st.columns([1.2, 1])
        with col_chart:
            top_series = ranked_df["composite_score"].head(top_n)
            fig_rank = _bar_chart(top_series, title=f"Top {top_n} Composite Score", x_label="Composite Score",
                                  top_n=top_n)
            st.plotly_chart(fig_rank, theme="streamlit", width="stretch")
        with col_table:
            display_cols = ["rank", "composite_score", "z_momentum", "z_volatility", "z_value", "z_quality"]
            display_df = ranked_df[display_cols].head(top_n).copy()
            display_df.columns = pd.Index(["Rank", "Score", "Mom Z", "Vol Z", "Val Z", "Qual Z"])

            styled_df = display_df.style.format({c: "{:.3f}" for c in display_df.columns if c != "Rank"}) \
                .map(lambda x: 'color: #2ca02c;' if x > 0 else ('color: #d62728;' if x < 0 else ''),
                     subset=["Mom Z", "Vol Z", "Val Z", "Qual Z"])

            st.dataframe(styled_df, height=420, width="stretch")

    with tab_factors:
        st.markdown("### Factor Distributions")
        factor_cols = ["momentum", "volatility", "value", "quality"]
        factor_labels = {
            "momentum": ("Momentum", "12M-1M Return"),
            "volatility": ("Low Volatility", "Negative Ann. Vol."),
            "value": ("Value", "Negative P/E Ratio"),
            "quality": ("Quality", "ROE")
        }
        col_a, col_b = st.columns(2)
        pairs = [(factor_cols[0], col_a), (factor_cols[1], col_b), (factor_cols[2], col_a), (factor_cols[3], col_b)]
        for fc, col in pairs:
            label, x_lab = factor_labels[fc]
            with col:
                if fc in factor_df.columns:
                    series = factor_df[fc].dropna().sort_values(ascending=False)
                    fig_f = _bar_chart(series, title=f"{label}", x_label=x_lab, top_n=20)
                    st.plotly_chart(fig_f, theme="streamlit", width="stretch")

    with tab_explore:
        st.markdown("### Cross-Factor Scatter Plot")
        available_factors = [c for c in factor_df.columns if factor_df[c].notna().sum() > 2]
        col_sx, col_sy = st.columns(2)
        with col_sx:
            x_factor = st.selectbox("X-axis factor", available_factors, index=0)
        with col_sy:
            y_factor = st.selectbox("Y-axis factor", [f for f in available_factors if f != x_factor], index=0)

        fig_scatter = _scatter_factors(factor_df, x_factor, y_factor)
        st.plotly_chart(fig_scatter, theme="streamlit", width="stretch")

        st.markdown("### Factor Correlation Matrix")
        corr_df = factor_df[available_factors].corr()
        fig_corr = px.imshow(corr_df, text_auto=".2f", color_continuous_scale="RdYlGn",
                             title="Pairwise Factor Correlation", aspect="auto", zmin=-1, zmax=1)
        fig_corr.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_corr, theme="streamlit", width="stretch")

    with tab_detail:
        st.markdown("### Individual Stock Factor Profile")
        ticker_list = ranked_df.index.tolist()
        selected_ticker = st.selectbox("Select a ticker", options=ticker_list,
                                       format_func=lambda t: f"#{int(ranked_df.loc[t, 'rank'])}  {t}")

        if selected_ticker:
            row = ranked_df.loc[selected_ticker]
            col_d1, col_d2 = st.columns([1, 1.4])
            with col_d1:
                st.markdown(f"#### {selected_ticker}")
                metrics = {
                    "Composite Score": f"{row['composite_score']:.4f}",
                    "Rank": f"#{int(row['rank'])} / {len(ranked_df)}",
                    "Z Momentum": f"{row.get('z_momentum', float('nan')):.3f}",
                    "Z Low Vol": f"{row.get('z_volatility', float('nan')):.3f}",
                    "Z Value": f"{row.get('z_value', float('nan')):.3f}",
                    "Z Quality": f"{row.get('z_quality', float('nan')):.3f}",
                }
                for label, val in metrics.items():
                    c1, c2 = st.columns([1.6, 1])
                    c1.markdown(f"<span>{label}</span>", unsafe_allow_html=True)
                    c2.markdown(f"<b>{val}</b>", unsafe_allow_html=True)
            with col_d2:
                fig_radar = _radar_chart(row, selected_ticker)
                st.plotly_chart(fig_radar, theme="streamlit", width="stretch")

    st.divider()
    csv_data = ranked_df.to_csv()
    st.download_button(label="Export Rankings to CSV", data=csv_data, file_name="smart_beta_rankings.csv",
                       mime="text/csv")


if __name__ == "__main__":
    main()