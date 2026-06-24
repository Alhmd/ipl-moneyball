"""
IPL Moneyball — Interactive Dashboard
======================================
Streamlit app over real_merged_ranked.csv (v3: Cricsheet performance +
Kaggle auction prices, merged on surname+year). Run with:
    streamlit run dashboard.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA_PATH = "/Users/syedalhmdhusainkazmi/Desktop/ipl-moneyball/all_players_value.csv"

st.set_page_config(page_title="IPL Moneyball", layout="wide", page_icon="🏏")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

df = load_data()

st.title("🏏 IPL Moneyball — Auction Price vs. Real Performance")
st.caption(
    "Does IPL auction price track on-field output, or reputation? "
    "Regression-residual model on real Cricsheet ball-by-ball data merged with real Kaggle auction prices."
)

# ── Sidebar filters ──────────────────────────────────────────────────────
st.sidebar.header("Filters")
price_types = st.sidebar.multiselect("Price type", sorted(df["price_type"].unique()), default=list(df["price_type"].unique()))
roles = st.sidebar.multiselect("Role", sorted(df["Role"].dropna().unique()), default=list(df["Role"].dropna().unique()))
origins = st.sidebar.multiselect("Player Origin", sorted(df["Player Origin"].dropna().unique()), default=list(df["Player Origin"].dropna().unique()))
year_min, year_max = int(df["year"].min()), int(df["year"].max())
year_range = st.sidebar.slider("Year range", year_min, year_max, (year_min, year_max))

base = df[df["price_type"].isin(price_types)]
is_auction = base["price_type"] == "Auction"
# Role/Origin filters only meaningfully apply to auction rows (retained icon players don't carry those fields)
passes_role_origin = (~is_auction) | (base["Role"].isin(roles) & base["Player Origin"].isin(origins))
fdf = base[passes_role_origin & base["year"].between(year_range[0], year_range[1])].copy()

# ── KPI row ───────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Player-seasons (filtered)", len(fdf))
c2.metric("Most undervalued (z)", f"{fdf['z'].max():.1f}σ" if len(fdf) else "—")
c3.metric("Most overpriced (z)", f"{fdf['z'].min():.1f}σ" if len(fdf) else "—")
c4.metric("Avg residual", f"{fdf['value_residual'].mean():.2f}" if len(fdf) else "—")

tab_dash, tab_lookup, tab_stars, tab_method = st.tabs(
    ["📊 Dashboard", "🔍 Player Lookup", "⭐ Big Names", "📝 Methodology & What Was Done"]
)

# ── Dashboard tab ─────────────────────────────────────────────────────────
with tab_dash:
    left, right = st.columns([2, 1])

    with left:
        auc = fdf[fdf["price_type"] == "Auction"]
        slope, intercept = np.polyfit(np.log(auc["Amount"].clip(lower=1)), auc["performance"], 1) if len(auc) > 1 else (0, 0)
        fig = px.scatter(
            fdf, x="Amount", y="performance", color="z", symbol="price_type",
            color_continuous_scale="RdYlGn", hover_data=["Player", "year", "Role", "Team", "Player Origin", "price_type"],
            log_x=True, labels={"Amount": "Price (₹, log scale)", "performance": "Composite performance"},
            title="Price vs. Performance — color = value residual (z-score), shape = auction vs. retention",
        )
        if len(auc) > 1:
            xs = np.linspace(fdf["Amount"].min(), fdf["Amount"].max(), 100)
            ys = slope * np.log(xs.clip(min=1)) + intercept
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Auction-fitted expectation", line=dict(color="black", dash="dash")))
        fig.update_layout(height=480)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig2 = px.histogram(fdf, x="value_residual", nbins=25, title="Residual distribution")
        fig2.add_vline(x=fdf["value_residual"].mean() if len(fdf) else 0, line_dash="dash")
        fig2.update_layout(height=480, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Rankings")
    n = st.slider("Show top N", 5, 30, 10)
    col_a, col_b = st.columns(2)
    show_cols = ["Player", "year", "price_type", "Team", "Amount", "performance", "z"]
    with col_a:
        st.markdown("**🟢 Most undervalued**")
        st.dataframe(fdf.sort_values("z", ascending=False).head(n)[show_cols], hide_index=True, use_container_width=True)
    with col_b:
        st.markdown("**🔴 Most overpriced**")
        st.dataframe(fdf.sort_values("z").head(n)[show_cols], hide_index=True, use_container_width=True)

    st.subheader("💰 MVP / LVP by Price Bracket")
    st.caption("Brackets in ₹2 Cr steps; everything ₹22 Cr and above is grouped as one open-ended bracket.")
    bdf = fdf.copy()
    bdf["Amount_Cr"] = bdf["Amount"] / 1e7
    bin_edges = list(range(0, 24, 2)) + [np.inf]
    bin_labels = [f"₹{b}-{b+2} Cr" for b in range(0, 22, 2)] + ["₹22+ Cr"]
    bdf["price_bracket"] = pd.cut(bdf["Amount_Cr"], bins=bin_edges, labels=bin_labels, right=False)

    bracket_rows = []
    for label in bin_labels:
        sub = bdf[bdf["price_bracket"] == label]
        if len(sub) == 0:
            continue
        mvp = sub.loc[sub["z"].idxmax()]
        lvp = sub.loc[sub["z"].idxmin()]
        bracket_rows.append({
            "Price Bracket": label,
            "Players": len(sub),
            "MVP (best value)": f"{mvp['Player']} ({int(mvp['year'])}, {mvp['price_type']})",
            "MVP z": round(mvp["z"], 2),
            "LVP (worst value)": f"{lvp['Player']} ({int(lvp['year'])}, {lvp['price_type']})",
            "LVP z": round(lvp["z"], 2),
        })
    if bracket_rows:
        st.dataframe(pd.DataFrame(bracket_rows), hide_index=True, use_container_width=True)
    else:
        st.info("No player-seasons in the current filter to bracket.")

# ── Player lookup tab ────────────────────────────────────────────────────
with tab_lookup:
    player = st.selectbox("Pick a player", sorted(df["Player"].unique()))
    pdf = df[df["Player"] == player].sort_values("year")
    if len(pdf):
        fig3 = px.bar(pdf, x="year", y="z", color="z", color_continuous_scale="RdYlGn",
                      title=f"{player} — value residual (z) by season")
        fig3.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig3, use_container_width=True)
        st.dataframe(pdf[["year", "price_type", "Team", "Amount", "runs", "wickets", "economy", "performance", "z"]], hide_index=True, use_container_width=True)
    else:
        st.info("No seasons found for this player after merging with performance data.")

# ── Big Names tab ─────────────────────────────────────────────────────────
STAR_PLAYERS = [
    "Virat Kohli", "Rohit Sharma", "MS Dhoni", "Hardik Pandya", "KL Rahul",
    "Suryakumar Yadav", "Shubman Gill", "Rishabh Pant", "Jasprit Bumrah",
    "Ravindra Jadeja", "Andre Russell", "Rashid Khan", "Sunil Narine",
    "Jos Buttler", "David Warner", "AB de Villiers", "Chris Gayle",
    "Kane Williamson", "Yuzvendra Chahal", "Shikhar Dhawan", "Sanju Samson",
]

with tab_stars:
    st.subheader("⭐ Are the Biggest Names Actually Worth Their Price?")
    available_stars = [p for p in STAR_PLAYERS if p in df["Player"].unique()]
    missing_stars = [p for p in STAR_PLAYERS if p not in df["Player"].unique()]
    selected = st.multiselect("Star players", available_stars, default=available_stars)
    if missing_stars:
        st.caption(f"Not in this dataset (no auction or 2025-retention record matched): {', '.join(missing_stars)}")

    sdf = df[df["Player"].isin(selected)].sort_values(["Player", "year"])
    if len(sdf):
        fig4 = px.bar(
            sdf, x="year", y="z", color="Player", barmode="group",
            hover_data=["price_type", "Amount", "Team"],
            title="Value residual (z-score) by season — selected star players",
        )
        fig4.add_hline(y=0, line_dash="dash", line_color="gray")
        fig4.update_layout(height=480)
        st.plotly_chart(fig4, use_container_width=True)
        st.dataframe(
            sdf[["Player", "year", "price_type", "Team", "Amount", "performance", "z"]],
            hide_index=True, use_container_width=True,
        )
    else:
        st.info("Pick at least one player above.")

# ── Methodology tab ───────────────────────────────────────────────────────
with tab_method:
    st.markdown("""
### What this project does
Applies the Moneyball/sabermetrics thesis to IPL auctions: regress performance on
`log(auction price)`, take the residual (actual − expected), z-score it, and flag
players who outperform/underperform what their price predicted.

### Build history (v1 → v4)
- **v1** (`01_value_outliers.py`) — synthetic data, proved the methodology end-to-end.
- **v2** (`02_real_data_value_outliers.py`) — single self-contained real dataset (IMB381, IIM Bangalore case study, 2008–2011).
- **v3** (`03_real_merged_value_outliers.py`) — two genuinely independent real sources merged:
  performance from **Cricsheet.org** ball-by-ball data (295,732 deliveries, 2007–2026), price from
  **Kaggle's IPL Player Auction Dataset** (970 records, 2013–2022).
- **v4** (`04_retained_players_value.py`, this dashboard's data) — adds real **retention prices**
  (Kaggle, "IPL Auction Historical Data and Retention List" — the 2025 mega-auction retention list,
  46 players) merged with real 2025-season Cricsheet performance, so retained icon players who never
  appear in auction data — **Kohli, Rohit, Dhoni** — are now in the analysis too.

### The entity-resolution problem (solved transparently)
Cricsheet names are "initials + surname" (`DA Warner`); Kaggle uses full names (`David Warner`).
No shared player-ID exists. Joined on `(surname, year)`, disambiguated by checking the auction
record's first-initial against Cricsheet's full initials.
**Match rate: 620 of 969 auction records (64.0%)** — reported honestly, not hidden.
After the sample-size floor (≥30 balls faced/bowled): **521 real player-seasons**.

### Methodology correction made mid-build
First version used a naive `performance ÷ price` ratio — mathematically dominated by the
denominator (a near-zero price inflates the ratio regardless of actual performance). Switched
to the regression-residual approach, which is how real sabermetrics value-over-replacement
metrics actually work.

### Validation against real history (not just internal stats)
- **Robin Uthappa, 2014** — 3.1σ undervalued. That was the season he won the IPL Orange Cap
  (leading run-scorer) — the model had no knowledge of that, it surfaced purely from price + ball-by-ball data.
- **Kane Williamson, 2018** (3.7σ) and **KL Rahul, 2018** — both real, recognizable standout seasons.
- **Overpriced side**: Jhye Richardson and Riley Meredith (2021–22) underdelivered relative to
  price — plausible given COVID-bubble auction years had inflated pricing.
- **Retained icons, 2025**: Virat Kohli's ₹21 Cr retention scored **+2.69σ** against the
  auction-fitted expectation curve, Rohit Sharma's ₹16.3 Cr scored **+1.27σ**, MS Dhoni's ₹4 Cr
  scored **+0.27σ** (roughly in line, consistent with his reduced 2025 playing role).

### Honest limitations
- Composite performance score is a simple standardized blend (batting runs + wickets-adjusted-for-economy),
  not a fully validated cricket impact metric.
- 64% match rate means ~a third of auction records couldn't be linked to performance data.
- Auction data covers 2013–2022 only; the retention layer covers a single cycle (2025 season only).
- **Retention price isn't set by competitive bidding the way auction price is** — a franchise
  unilaterally decides what to pay to keep an icon player, rather than other franchises bidding it
  up. The "expected performance" line retained players are scored against is still the
  auction-fitted regression — an extrapolation, not a like-for-like mechanism. Useful for "how does
  this retention price compare to what the auction market curve would predict," not a claim that
  retention and auction economics are the same thing.

### Next steps
- Build a proper player-name registry to lift the auction match rate above 64%.
- Extend auction data to 2023+, and retention data across more than one mega-auction cycle.
- Add phase splits (powerplay/death-overs) now that raw ball-by-ball data is in hand.
""")
