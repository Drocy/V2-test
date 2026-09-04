
import io
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image as RLImage
)

st.set_page_config(
    page_title="Indigo Market Intelligence Engine V2",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

ENGINE_FILE = "indigo_v2_engine_streamlit_complete.pkl"

# No default/flagship portfolio. Indigo starts with the full verified data-layer universe;
# the client configuration determines which instruments are actually analysed.

@st.cache_resource
def load_engine(path):
    with open(path, "rb") as f:
        return pickle.load(f)


if not os.path.exists(ENGINE_FILE):
    st.error(f"Engine artifact not found: {ENGINE_FILE}")
    st.stop()

engine = load_engine(ENGINE_FILE)
clean_data = engine.get("data_integrity", {}).get("clean_market_data", {})
metadata = engine.get("metadata", {})


def asset_prices(data, ticker):
    df = data[ticker].copy()
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    vol_col = "Volume" if "Volume" in df.columns else None
    out = pd.DataFrame({
        "Date": pd.to_datetime(df[date_col]),
        "Price": pd.to_numeric(df[price_col], errors="coerce"),
    })
    if vol_col:
        out["Volume"] = pd.to_numeric(df[vol_col], errors="coerce")
    return out.dropna(subset=["Date", "Price"]).sort_values("Date").drop_duplicates("Date")


@st.cache_data
def build_price_table(data, tickers):
    frames = []
    for ticker in tickers:
        if ticker not in data:
            continue
        p = asset_prices(data, ticker)[["Date", "Price"]].rename(columns={"Price": ticker})
        frames.append(p.set_index("Date"))
    if not frames:
        return pd.DataFrame()
    prices = pd.concat(frames, axis=1).sort_index()
    # Preserve crypto weekend observations while treating unavailable market prices
    # as unchanged until the next observation.
    return prices.ffill().dropna(how="all")


def portfolio_returns(prices, weights):
    selected = [x for x, w in weights.items() if w > 0 and x in prices.columns]
    if not selected:
        return pd.Series(dtype=float)
    p = prices[selected].ffill()
    r = p.pct_change().fillna(0.0)
    w = pd.Series({x: weights[x] for x in selected}, dtype=float)
    w = w / w.sum()
    return r.mul(w, axis=1).sum(axis=1).rename("Portfolio_Return")


def var_es(returns, confidence=0.95):
    r = pd.Series(returns).dropna()
    if len(r) == 0:
        return np.nan, np.nan
    q = r.quantile(1 - confidence)
    tail = r[r <= q]
    return float(-q), float(-tail.mean() if len(tail) else -q)


def max_drawdown(returns):
    r = pd.Series(returns).dropna()
    if len(r) == 0:
        return np.nan
    wealth = (1 + r).cumprod()
    dd = wealth / wealth.cummax() - 1
    return float(dd.min())


def annualized_vol(returns):
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        return np.nan
    return float(r.std(ddof=1) * np.sqrt(365))


def sharpe(returns):
    r = pd.Series(returns).dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return np.nan
    return float(r.mean() / r.std(ddof=1) * np.sqrt(365))


def total_return(returns):
    r = pd.Series(returns).dropna()
    return float((1 + r).prod() - 1) if len(r) else np.nan


def portfolio_metrics(pr, confidence):
    v, es = var_es(pr, confidence)
    return {
        "Portfolio Return": total_return(pr),
        "Annualized Volatility": annualized_vol(pr),
        f"VaR {int(confidence*100)}% 1D": v,
        f"ES {int(confidence*100)}% 1D": es,
        "Maximum Drawdown": max_drawdown(pr),
        "Sharpe (Rf=0)": sharpe(pr),
        "Observations": int(pr.dropna().shape[0]),
        "Latest Date": pr.index.max() if len(pr) else pd.NaT,
    }


def risk_decomposition(asset_returns, weights):
    cols = [x for x in weights if x in asset_returns.columns and weights[x] > 0]
    if not cols:
        return pd.DataFrame()
    r = asset_returns[cols].dropna(how="all").fillna(0)
    w = pd.Series({x: weights[x] for x in cols}, dtype=float)
    w = w / w.sum()
    cov = r.cov() * 365
    pv = float(w.values @ cov.values @ w.values)
    port_vol = np.sqrt(max(pv, 0))
    if port_vol == 0:
        mrc = pd.Series(0.0, index=cols)
    else:
        mrc = cov.dot(w) / port_vol
    rc = w * mrc
    out = pd.DataFrame({
        "Asset": cols,
        "Weight": [w[x] for x in cols],
        "Marginal_Risk": [mrc[x] for x in cols],
        "Risk_Contribution": [rc[x] for x in cols],
    })
    out["Risk_Contribution_Pct"] = (
        out["Risk_Contribution"] / out["Risk_Contribution"].sum()
        if out["Risk_Contribution"].sum() != 0 else 0
    )
    return out.sort_values("Risk_Contribution_Pct", ascending=False).reset_index(drop=True)


def stress_table(asset_returns, weights):
    # Transparent deterministic shocks for an executive stress view.
    scenarios = {
        "Broad Risk-Off": {x: -0.10 for x in weights},
        "Equity Shock": {x: -0.15 if not x.endswith("-USD") else -0.05 for x in weights},
        "Technology Shock": {x: -0.20 if x in ["AAPL","MSFT","NVDA","AMZN","GOOGL","TSLA"] else -0.05 for x in weights},
        "Crypto Shock": {x: -0.35 if x.endswith("-USD") else -0.03 for x in weights},
        "Severe Multi-Asset": {x: -0.25 if not x.endswith("-USD") else -0.45 for x in weights},
    }
    rows = []
    for name, shocks in scenarios.items():
        ret = sum(weights[x] * shocks.get(x, 0) for x in weights)
        rows.append({
            "Scenario": name,
            "Estimated Portfolio Return": ret,
            "Estimated Portfolio Loss": -ret
        })
    return pd.DataFrame(rows)


def liquidity_table(data, weights, window=20):
    rows = []
    for ticker, weight in weights.items():
        if weight <= 0 or ticker not in data:
            continue
        df = asset_prices(data, ticker)
        dollar = df["Price"] * df.get("Volume", pd.Series(index=df.index, dtype=float))
        avg_dv = dollar.tail(window).mean()
        ret = df["Price"].pct_change().tail(window).dropna()
        vol = ret.std(ddof=1) * np.sqrt(365) if len(ret) > 1 else np.nan
        # Transparent relative score; higher dollar volume and lower volatility improve score.
        dv_score = np.clip(np.log10(max(avg_dv, 1)) / 10 * 100, 0, 100)
        vol_penalty = np.clip((vol if pd.notna(vol) else 0) * 100, 0, 80)
        score = float(np.clip(dv_score - vol_penalty * 0.35, 0, 100))
        rows.append({
            "Asset": ticker,
            "Weight": weight,
            "Avg Daily Dollar Volume": avg_dv,
            "Annualized Volatility": vol,
            "Liquidity Score": score
        })
    return pd.DataFrame(rows).sort_values("Liquidity Score")


def latest_alerts(asset_returns, prices):
    rows = []
    for ticker in prices.columns:
        r = asset_returns[ticker].dropna() if ticker in asset_returns else pd.Series(dtype=float)
        if len(r) < 30:
            continue
        z = (r.iloc[-1] - r.tail(60).mean()) / r.tail(60).std(ddof=1) if r.tail(60).std(ddof=1) else 0
        dd = (prices[ticker] / prices[ticker].cummax() - 1).iloc[-1]
        triggers = []
        if abs(z) >= 3:
            triggers.append("Extreme return move")
        if dd <= -0.20:
            triggers.append("Drawdown > 20%")
        severity = "Critical" if len(triggers) >= 2 else ("High" if triggers else "Normal")
        rows.append({"Asset": ticker, "Severity": severity, "Triggers": ", ".join(triggers) or "None",
                     "Latest Return Z": z, "Current Drawdown": dd})
    return pd.DataFrame(rows).sort_values(["Severity", "Asset"], ascending=[True, True])


def fmt_pct(x):
    return "n/a" if pd.isna(x) else f"{x:.2%}"


def fmt_num(x):
    return "n/a" if pd.isna(x) else f"{x:,.2f}"


def intelligence_text(metrics, decomp, stress, liq, alerts):
    ret = metrics["Portfolio Return"]
    vol = metrics["Annualized Volatility"]
    var = next((v for k, v in metrics.items() if k.startswith("VaR")), np.nan)
    es = next((v for k, v in metrics.items() if k.startswith("ES")), np.nan)
    dd = metrics["Maximum Drawdown"]

    top = decomp.iloc[0] if not decomp.empty else None
    top_asset = top["Asset"] if top is not None else "n/a"
    top_rc = top["Risk_Contribution_Pct"] if top is not None else np.nan
    worst = stress.sort_values("Estimated Portfolio Return").iloc[0] if not stress.empty else None
    weak_liq = liq.sort_values("Liquidity Score").iloc[0] if not liq.empty else None
    alert_count = int((alerts["Severity"] != "Normal").sum()) if not alerts.empty else 0

    deductive = []
    inductive = []
    reasoning = []
    recommendations = []

    if pd.notna(top_rc):
        deductive.append(
            f"Because {top_asset} contributes approximately {top_rc:.1%} of measured portfolio volatility, "
            f"a change in that position's risk has a disproportionate effect on total portfolio risk."
        )
    if pd.notna(es) and pd.notna(var):
        deductive.append(
            f"Because Expected Shortfall ({es:.2%}) is larger than one-day VaR ({var:.2%}), "
            f"losses in the adverse tail are materially worse than the ordinary VaR threshold alone suggests."
        )
    if pd.notna(dd):
        deductive.append(
            f"Because the observed maximum drawdown is {dd:.2%}, capital impairment has historically been materially larger "
            f"than a single-day loss metric can describe."
        )
    if worst is not None:
        deductive.append(
            f"Under the defined {worst['Scenario']} shock, the estimated portfolio loss is {worst['Estimated Portfolio Loss']:.2%}; "
            f"therefore diversification does not eliminate joint downside exposure."
        )

    if not decomp.empty:
        concentration = decomp["Risk_Contribution_Pct"].head(3).sum()
        inductive.append(
            f"The three largest risk contributors account for approximately {concentration:.1%} of measured portfolio risk. "
            f"Across the observed data, this indicates that risk is being driven by a relatively small subset of exposures."
        )
    if not stress.empty:
        inductive.append(
            f"The stress set produces losses ranging from {stress['Estimated Portfolio Return'].min():.2%} to "
            f"{stress['Estimated Portfolio Return'].max():.2%}. This pattern indicates that the portfolio's resilience is scenario-dependent."
        )
    if not liq.empty and weak_liq is not None:
        inductive.append(
            f"{weak_liq['Asset']} has the lowest relative liquidity score in the configured portfolio. "
            f"That identifies it as the first exposure to examine when execution capacity or market depth becomes constrained."
        )
    if alert_count:
        inductive.append(
            f"{alert_count} portfolio/universe assets currently meet at least one configured exception condition. "
            f"The concentration of these exceptions should be reviewed before treating the portfolio as fully stable."
        )

    reasoning.append(
        f"Return of {fmt_pct(ret)} must be read alongside annualized volatility of {fmt_pct(vol)} and maximum drawdown of {fmt_pct(dd)}. "
        "These metrics answer different questions: return describes outcome, volatility describes typical variability, "
        "and drawdown describes the depth of capital decline from a previous peak."
    )
    reasoning.append(
        "VaR answers a threshold question—how large a loss is expected not to be exceeded on most ordinary days at the selected confidence level. "
        "Expected Shortfall goes further by measuring the average loss when that threshold is breached, making it more informative about tail severity."
    )
    reasoning.append(
        "Risk contribution explains where portfolio risk comes from rather than merely where capital is invested. "
        "An asset can have a modest weight but a large risk contribution when its volatility and covariance with the rest of the portfolio are high."
    )
    reasoning.append(
        "Stress testing is deliberately different from forecasting. It asks what the portfolio would lose under specified adverse shocks, "
        "so it exposes vulnerabilities that a historical average may not reveal."
    )

    recommendations.append("Review the largest risk contributors before changing allocations solely on the basis of portfolio weights.")
    recommendations.append("Use Expected Shortfall and drawdown alongside VaR rather than relying on a single headline risk number.")
    recommendations.append("Test the portfolio against the most adverse defined scenario before increasing gross exposure.")
    if weak_liq is not None:
        recommendations.append(f"Monitor {weak_liq['Asset']} for execution capacity if market conditions deteriorate.")
    if alert_count:
        recommendations.append("Investigate active exceptions and determine whether each is transient market noise or a persistent risk condition.")

    return {
        "deductive": deductive,
        "inductive": inductive,
        "reasoning": reasoning,
        "recommendations": recommendations
    }


def make_chart_bytes(kind, data, x, y, title, ylabel=None):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    if kind == "line":
        ax.plot(data[x], data[y])
    elif kind == "bar":
        ax.bar(data[x].astype(str), data[y])
        ax.tick_params(axis="x", rotation=45)
    ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def pdf_report(metrics, decomp, stress, liq, alerts, intelligence, portfolio_series, risk_series, config):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=14*mm, leftMargin=14*mm,
                            topMargin=14*mm, bottomMargin=14*mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8.5, leading=11))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading1"], spaceAfter=8))
    styles.add(ParagraphStyle(name="Sub", parent=styles["Heading2"], spaceAfter=5))
    story = []

    story += [
        Paragraph("INDIGO", styles["Title"]),
        Paragraph("Market Intelligence Engine V2 — Institutional Risk Report", styles["Heading2"]),
        Paragraph(f"Generated: {datetime.utcnow():%Y-%m-%d %H:%M UTC}", styles["Small"]),
        Spacer(1, 8),
        Paragraph("Executive Summary", styles["Section"]),
        Paragraph(
            f"The configured portfolio returned {fmt_pct(metrics['Portfolio Return'])} over the analysis window, "
            f"with annualized volatility of {fmt_pct(metrics['Annualized Volatility'])}, "
            f"maximum drawdown of {fmt_pct(metrics['Maximum Drawdown'])}, and selected-confidence tail measures shown below.",
            styles["BodyText"]),
        Spacer(1, 6)
    ]

    # KPI table
    kpi = [["Metric", "Value"]]
    for k, v in metrics.items():
        if isinstance(v, (float, np.floating)):
            val = fmt_pct(v) if any(s in k for s in ["Return", "Volatility", "VaR", "ES", "Drawdown"]) else fmt_num(v)
        else:
            val = str(v)
        kpi.append([k, val])
    t = Table(kpi, colWidths=[80*mm, 85*mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#20242c")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
    ]))
    story += [t, Spacer(1, 10)]

    story += [Paragraph("Portfolio Performance", styles["Section"])]
    chart1 = make_chart_bytes("line", portfolio_series.reset_index(), "Date", "Cumulative", "Portfolio Growth")
    story += [RLImage(chart1, width=175*mm, height=92*mm), PageBreak()]

    story += [Paragraph("Risk Analysis", styles["Section"])]
    chart2 = make_chart_bytes("bar", decomp, "Asset", "Risk_Contribution_Pct", "Risk Contribution by Asset", "Risk contribution")
    story += [RLImage(chart2, width=175*mm, height=92*mm), Spacer(1, 8)]

    def table_from_df(df, max_rows=20):
        d = df.head(max_rows).copy()
        data = [list(d.columns)] + d.astype(str).values.tolist()
        tab = Table(data, repeatRows=1)
        tab.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#20242c")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
            ("FONTSIZE", (0,0), (-1,-1), 6.5),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
        ]))
        return tab

    story += [
        Paragraph("Risk Decomposition", styles["Sub"]),
        table_from_df(decomp),
        Spacer(1, 10),
        Paragraph("Stress & Scenario", styles["Sub"]),
        table_from_df(stress),
        PageBreak(),
        Paragraph("Liquidity", styles["Section"]),
        table_from_df(liq),
        Spacer(1, 10),
        Paragraph("Alerts & Exceptions", styles["Sub"]),
        table_from_df(alerts),
        PageBreak(),
    ]

    story += [Paragraph("Written Institutional Intelligence", styles["Section"])]
    for title, items in [
        ("What the metrics mean", intelligence["reasoning"]),
        ("Deductive findings", intelligence["deductive"]),
        ("Inductive findings", intelligence["inductive"]),
        ("Recommendations", intelligence["recommendations"]),
    ]:
        story.append(Paragraph(title, styles["Sub"]))
        for item in items:
            story.append(Paragraph("• " + item, styles["BodyText"]))
            story.append(Spacer(1, 4))

    story += [
        Spacer(1, 10),
        Paragraph(
            "Configuration: "
            f"Confidence={config['confidence']:.0%}; "
            f"Window={config['window_label']}; "
            f"Portfolio weights sum={sum(config['weights'].values()):.2%}.",
            styles["Small"]
        )
    ]

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


# ---------------- UI ----------------

st.sidebar.markdown("## INDIGO")
st.sidebar.caption("Market Intelligence Engine V2")
st.sidebar.divider()

pages = [
    "Executive Overview", "Portfolio Configuration", "Market Intelligence",
    "Portfolio Risk", "Cross-Asset Risk", "Stress & Scenario",
    "Liquidity", "Alerts & Exceptions", "Institutional Intelligence",
    "Institutional Report"
]
page = st.sidebar.radio("Intelligence", pages)

# ---------------------------------------------------------------------------
# CLIENT PORTFOLIO CONFIGURATION — FULL UNIVERSE, ZERO DEFAULT HOLDINGS
# ---------------------------------------------------------------------------
# The portfolio is intentionally EMPTY at startup. Every instrument present in
# the verified data layer is displayed here. The client fills Configuration.
registry = engine.get("universe", {}).get("instrument_registry", {}) or {}

catalog_rows = []
for ticker, df in clean_data.items():
    meta = registry.get(ticker, {}) if isinstance(registry, dict) else {}
    catalog_rows.append({
        "Asset": str(ticker),
        "Name": meta.get("name", ticker),
        "Asset Class": meta.get("asset_class", "Unclassified"),
        "Sub-Class": meta.get("sub_class", ""),
        "Currency": meta.get("currency", ""),
        "Exchange": meta.get("exchange", ""),
        "Data Rows": int(len(df)),
    })

catalog = (
    pd.DataFrame(catalog_rows)
    .drop_duplicates("Asset")
    .sort_values(["Asset Class", "Asset"], kind="stable")
    .reset_index(drop=True)
)
catalog["Configuration"] = 0.0

if "portfolio_config" not in st.session_state:
    st.session_state.portfolio_config = catalog.copy()
else:
    # Keep the full current universe while preserving any weights already entered.
    old = st.session_state.portfolio_config.set_index("Asset")
    fresh = catalog.set_index("Asset")
    common = old.index.intersection(fresh.index)
    fresh.loc[common, "Configuration"] = pd.to_numeric(
        old.loc[common, "Configuration"], errors="coerce"
    ).fillna(0.0)
    st.session_state.portfolio_config = fresh.reset_index()[catalog.columns]

if page == "Portfolio Configuration":
    st.subheader("Client Portfolio Configuration")
    st.success(
        f"FULL VERIFIED UNIVERSE: {len(catalog):,} instruments loaded. "
        "No default holdings. Configuration starts at 0% for every instrument."
    )

    a, b, c = st.columns([1, 1, 2])
    with a:
        class_filter = st.selectbox(
            "Asset class", ["All"] + sorted(catalog["Asset Class"].fillna("Unclassified").unique())
        )
    with b:
        search = st.text_input("Search ticker / instrument", "")
    with c:
        st.caption(
            "Enter portfolio weights directly in the Configuration column. "
            "Only instruments with a value greater than 0% enter the risk engine."
        )

    visible = st.session_state.portfolio_config.copy()
    if class_filter != "All":
        visible = visible[visible["Asset Class"] == class_filter]
    if search.strip():
        q = search.strip().lower()
        visible = visible[
            visible["Asset"].str.lower().str.contains(q, na=False)
            | visible["Name"].astype(str).str.lower().str.contains(q, na=False)
        ]

    edited = st.data_editor(
        visible,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        key="full_universe_portfolio_editor",
        disabled=[
            "Asset", "Name", "Asset Class", "Sub-Class",
            "Currency", "Exchange", "Data Rows"
        ],
        column_order=[
            "Asset", "Name", "Asset Class", "Sub-Class",
            "Currency", "Exchange", "Data Rows", "Configuration"
        ],
        column_config={
            "Configuration": st.column_config.NumberColumn(
                "Configuration",
                help="Client portfolio weight. Leave at 0% if not held.",
                min_value=0.0,
                max_value=100.0,
                step=0.1,
                format="%.1f%%",
            )
        },
    )

    # Merge only the rows visible in the filtered editor back into the full universe.
    full = st.session_state.portfolio_config.set_index("Asset")
    changes = edited.set_index("Asset")
    full.loc[changes.index, "Configuration"] = pd.to_numeric(
        changes["Configuration"], errors="coerce"
    ).fillna(0.0)
    st.session_state.portfolio_config = full.reset_index()[catalog.columns]

    configured = st.session_state.portfolio_config[
        st.session_state.portfolio_config["Configuration"] > 0
    ].copy()

    if configured.empty:
        st.warning("Portfolio is empty — this is intentional. Configure the client's holdings above.")
        weights = {}
    else:
        raw = dict(zip(configured["Asset"], configured["Configuration"]))
        total = sum(raw.values())
        weights = {k: v / total for k, v in raw.items()} if total > 0 else {}
        st.info(
            f"{len(configured):,} holdings configured | "
            f"Entered weights: {total:.2f}% | "
            "Analysis weights are normalised to 100%."
        )
        st.dataframe(
            configured[["Asset", "Name", "Asset Class", "Configuration"]],
            use_container_width=True,
            hide_index=True,
        )
else:
    configured = st.session_state.portfolio_config[
        st.session_state.portfolio_config["Configuration"] > 0
    ].copy()
    raw = dict(zip(configured["Asset"], configured["Configuration"]))
    total = sum(raw.values())
    weights = {k: v / total for k, v in raw.items()} if total > 0 else {}

with st.sidebar.expander("Analysis controls", expanded=False):
    window_label = st.selectbox("Analysis window", ["1Y", "2Y", "5Y", "MAX"], index=1)
    confidence = st.selectbox("Tail-risk confidence", [0.95, 0.99], index=0, format_func=lambda x: f"{x:.0%}")
    st.caption(f"Configured instruments: {len(weights)}")

prices_all = build_price_table(clean_data, list(weights.keys())) if weights else pd.DataFrame()
if not prices_all.empty:
    if window_label == "1Y":
        cutoff = prices_all.index.max() - pd.Timedelta(days=365)
    elif window_label == "2Y":
        cutoff = prices_all.index.max() - pd.Timedelta(days=730)
    elif window_label == "5Y":
        cutoff = prices_all.index.max() - pd.Timedelta(days=1825)
    else:
        cutoff = prices_all.index.min()
    prices = prices_all.loc[prices_all.index >= cutoff].copy()
    asset_rets = prices.pct_change()
    pr = portfolio_returns(prices, weights)
    metrics = portfolio_metrics(pr, confidence)
    decomp = risk_decomposition(asset_rets, weights)
    stress = stress_table(asset_rets, weights)
    liq = liquidity_table(clean_data, weights)
    alerts = latest_alerts(asset_rets, prices)
    intel = intelligence_text(metrics, decomp, stress, liq, alerts)
    cum = (1 + pr).cumprod()
    portfolio_series = pd.DataFrame({"Date": cum.index, "Cumulative": cum.values})
    rolling_vol = pr.rolling(30).std() * np.sqrt(365)
    risk_series = pd.DataFrame({"Date": rolling_vol.index, "RollingVol": rolling_vol.values})
else:
    prices = pd.DataFrame()
    asset_rets = pd.DataFrame()
    pr = pd.Series(dtype=float)
    metrics = {}
    decomp = pd.DataFrame()
    stress = pd.DataFrame()
    liq = pd.DataFrame()
    alerts = pd.DataFrame()
    intel = {"deductive": [], "inductive": [], "recommendations": []}
    portfolio_series = pd.DataFrame(columns=["Date", "Cumulative"])
    risk_series = pd.DataFrame(columns=["Date", "RollingVol"])

if pr.empty and page != "Portfolio Configuration":
    st.error("No portfolio is configured. Open Portfolio Configuration and enter the client holdings/weights first.")
    st.stop()

config = {"confidence": confidence, "window_label": window_label, "weights": weights}

st.sidebar.divider()
st.sidebar.caption(f"Engine {metadata.get('version', 'V2')}")
st.sidebar.caption(f"Data through {prices_all.index.max():%Y-%m-%d}" if not prices_all.empty else "No portfolio data selected")

st.title(page)

if page == "Executive Overview":
    st.caption("Decision-level view of portfolio outcome, risk, concentration and current exceptions.")
    cols = st.columns(5)
    vals = [
        ("Return", fmt_pct(metrics["Portfolio Return"])),
        ("Volatility", fmt_pct(metrics["Annualized Volatility"])),
        ("VaR", fmt_pct(next(v for k,v in metrics.items() if k.startswith("VaR")))),
        ("ES", fmt_pct(next(v for k,v in metrics.items() if k.startswith("ES")))),
        ("Max drawdown", fmt_pct(metrics["Maximum Drawdown"])),
    ]
    for c, (lab, val) in zip(cols, vals):
        c.metric(lab, val)
    st.divider()
    st.subheader("Executive intelligence")
    for item in intel["deductive"][:3]:
        st.info(item)
    if not decomp.empty:
        st.subheader("Where risk is coming from")
        st.dataframe(decomp, use_container_width=True, hide_index=True)
    st.subheader("Portfolio trajectory")
    st.line_chart(portfolio_series.set_index("Date"))

elif page == "Portfolio Configuration":
    st.subheader("Portfolio parameters")
    st.write("Configure the portfolio and analysis assumptions before interpreting risk.")
    st.write(f"Normalised portfolio weights: **{sum(weights.values()):.0%}**")
    st.dataframe(pd.DataFrame({"Asset": list(weights), "Weight": [f"{v:.2%}" for v in weights.values()]}),
                 use_container_width=True, hide_index=True)
    st.info("Weights are normalised to 100% for analysis. Changing these controls recalculates the downstream risk views.")

elif page == "Market Intelligence":
    mi = []
    for ticker in weights:
        if ticker not in prices:
            continue
        r = asset_rets[ticker].dropna()
        mi.append({
            "Asset": ticker,
            "Last Price": prices[ticker].iloc[-1],
            "Return 20D": (prices[ticker].iloc[-1] / prices[ticker].iloc[-21] - 1) if len(prices[ticker]) > 21 else np.nan,
            "Return 63D": (prices[ticker].iloc[-1] / prices[ticker].iloc[-64] - 1) if len(prices[ticker]) > 64 else np.nan,
            "Volatility 20D": r.tail(20).std() * np.sqrt(365),
            "Max Drawdown": max_drawdown(r),
            "Latest Return Z": (r.iloc[-1] - r.tail(60).mean()) / r.tail(60).std() if len(r) >= 60 and r.tail(60).std() else np.nan
        })
    mi = pd.DataFrame(mi)
    st.dataframe(mi.style.format({
        "Last Price": "{:,.2f}", "Return 20D": "{:.2%}", "Return 63D": "{:.2%}",
        "Volatility 20D": "{:.2%}", "Max Drawdown": "{:.2%}", "Latest Return Z": "{:.2f}"
    }), use_container_width=True)
    st.subheader("Market return comparison")
    chart = mi.set_index("Asset")["Return 20D"].sort_values()
    st.bar_chart(chart)

elif page == "Portfolio Risk":
    st.caption("Portfolio exposure, concentration, tail risk and risk contribution.")
    st.dataframe(pd.DataFrame([metrics]).T.rename(columns={0: "Value"}), use_container_width=True)
    st.subheader("Risk contribution")
    st.bar_chart(decomp.set_index("Asset")["Risk_Contribution_Pct"])
    st.dataframe(decomp.style.format({"Weight":"{:.2%}", "Marginal_Risk":"{:.4f}",
                                      "Risk_Contribution":"{:.4f}", "Risk_Contribution_Pct":"{:.2%}"}),
                 use_container_width=True, hide_index=True)
    st.subheader("30-day rolling annualized volatility")
    st.line_chart(risk_series.set_index("Date"))

elif page == "Cross-Asset Risk":
    st.subheader("Correlation structure")
    corr = asset_rets.corr()
    st.dataframe(corr.style.format("{:.2f}"), use_container_width=True)
    st.subheader("Average absolute correlation")
    avg = corr.abs().replace(1, np.nan).mean().sort_values(ascending=False)
    st.bar_chart(avg)

elif page == "Stress & Scenario":
    st.caption("Deterministic scenario analysis. These are shocks, not forecasts.")
    st.dataframe(stress.style.format({
        "Estimated Portfolio Return":"{:.2%}",
        "Estimated Portfolio Loss":"{:.2%}"
    }), use_container_width=True, hide_index=True)
    st.bar_chart(stress.set_index("Scenario")["Estimated Portfolio Loss"])

elif page == "Liquidity":
    st.caption("Relative liquidity screening based on recent traded dollar volume and volatility. It is not a full order-book market-impact model.")
    st.dataframe(liq.style.format({
        "Weight":"{:.2%}", "Avg Daily Dollar Volume":"{:,.0f}",
        "Annualized Volatility":"{:.2%}", "Liquidity Score":"{:.1f}"
    }), use_container_width=True, hide_index=True)
    st.bar_chart(liq.set_index("Asset")["Liquidity Score"])

elif page == "Alerts & Exceptions":
    st.subheader("Current exceptions")
    active = alerts[alerts["Severity"] != "Normal"]
    if active.empty:
        st.success("No active exception conditions under the configured rules.")
    else:
        st.warning(f"{len(active)} assets currently meet at least one exception condition.")
        st.dataframe(active, use_container_width=True, hide_index=True)

elif page == "Institutional Intelligence":
    st.caption("Written decision intelligence: what the measures mean, what the evidence suggests, why it matters, and what should be reviewed.")
    st.subheader("What the risk metrics mean")
    for x in intel["reasoning"]:
        st.write("• " + x)
    st.subheader("Deductive analysis")
    for x in intel["deductive"]:
        st.info(x)
    st.subheader("Inductive analysis")
    for x in intel["inductive"]:
        st.write("• " + x)
    st.subheader("Recommendations")
    for x in intel["recommendations"]:
        st.success(x)

elif page == "Institutional Report":
    st.caption("Complete institutional report: executive summary, graphical analysis, tables, interpretation, reasoning and recommendations.")
    st.subheader("Report at a glance")
    cols = st.columns(4)
    cols[0].metric("Return", fmt_pct(metrics["Portfolio Return"]))
    cols[1].metric("Volatility", fmt_pct(metrics["Annualized Volatility"]))
    cols[2].metric("Max drawdown", fmt_pct(metrics["Maximum Drawdown"]))
    cols[3].metric("Active exceptions", str((alerts["Severity"] != "Normal").sum()))
    st.subheader("Graphical analysis")
    st.line_chart(portfolio_series.set_index("Date"))
    st.bar_chart(decomp.set_index("Asset")["Risk_Contribution_Pct"])
    st.subheader("Written analysis")
    for title, items in [
        ("Reasoning", intel["reasoning"]),
        ("Deductive findings", intel["deductive"]),
        ("Inductive findings", intel["inductive"]),
        ("Recommendations", intel["recommendations"]),
    ]:
        st.markdown(f"### {title}")
        for x in items:
            st.write("• " + x)
    st.subheader("Tables")
    st.markdown("**Portfolio risk**")
    st.dataframe(pd.DataFrame([metrics]), use_container_width=True)
    st.markdown("**Risk decomposition**")
    st.dataframe(decomp, use_container_width=True, hide_index=True)
    st.markdown("**Stress scenarios**")
    st.dataframe(stress, use_container_width=True, hide_index=True)
    st.markdown("**Liquidity**")
    st.dataframe(liq, use_container_width=True, hide_index=True)
    st.markdown("**Alerts**")
    st.dataframe(alerts, use_container_width=True, hide_index=True)

    pdf_bytes = pdf_report(metrics, decomp, stress, liq, alerts, intel, portfolio_series, risk_series, config)
    st.download_button(
        "Download Institutional Report (PDF)",
        data=pdf_bytes,
        file_name=f"indigo_v2_institutional_report_{datetime.utcnow():%Y%m%d_%H%M}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
