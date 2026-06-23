"""
IPL Moneyball — Undervalued Players (REAL DATA VERSION)
==========================================================
Data: IMB381 IPL Auction dataset (real player auction records,
2008-2011 auctions, IIM Bangalore case study, widely used in DA/DS
coursework) — sourced from a public GitHub mirror, not synthetic.

Honest scope notes, stated up front rather than discovered later:
  - This dataset is older (2008-2011 auctions) and small (130 players).
    A current (2024+) auction dataset exists on Kaggle but requires
    API authentication not set up in this environment — worth doing
    as a follow-up if a more recent dataset is needed.
  - Prices are in the dataset's native units (USD, as used in early
    IPL auctions) — not converted to crores, to avoid inventing an
    exchange-rate conversion.
  - Only 6 of 130 players have zero international (Test/ODI) caps —
    too few to build the "uncapped" leaderboard on strictly that
    definition. Reframed the lens to AUCTION PRICE TIER (low base
    price vs. high) instead of cap status — this is actually closer
    to the real Moneyball thesis anyway (cheap vs. productive, not
    "uncapped" as a cricket-specific label).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/syedalhmdhusainkazmi/Desktop/ipl-moneyball/raw/ipl_auction_2013.csv")
df.columns = [c.strip() for c in df.columns]

# ── 1. Composite IPL performance score ──────────────────────────────────────
# Simple, transparent composite: standardize IPL batting runs and IPL
# bowling output (wickets, weighted down by economy rate), then sum.
# Not a perfect impact metric — a defensible single-number starting
# point, same honesty standard as the synthetic version.
bat_z = (df["RUNS-S"] - df["RUNS-S"].mean()) / df["RUNS-S"].std()

bowlers = df["WKTS"] > 0
bowl_score = pd.Series(0.0, index=df.index)
# Reward wickets, penalize high economy — only meaningful for those who bowled
bowl_score[bowlers] = df.loc[bowlers, "WKTS"] * (10 - df.loc[bowlers, "ECON"]).clip(lower=0)
bowl_z = pd.Series(0.0, index=df.index)
bowl_z[bowlers] = (bowl_score[bowlers] - bowl_score[bowlers].mean()) / bowl_score[bowlers].std()

df["performance"] = bat_z.fillna(0) + bowl_z.fillna(0)

# ── 2. Price tier (the real low-cost-vs-output question) ───────────────────
df["price_tier"] = pd.cut(
    df["BASE PRICE"], bins=[0, 100000, 200000, np.inf],
    labels=["low (<=$100k)", "mid ($100k-$200k)", "high (>$200k)"]
)

# ── 3. Sample-size honesty: flag players with almost no IPL track record ──
MIN_RUNS_OR_WKTS = 5
eligible = df[(df["RUNS-S"] >= MIN_RUNS_OR_WKTS) | (df["WKTS"] >= 1)].copy()
excluded = df[~df.index.isin(eligible.index)]
print(f"Total players: {len(df)} | Eligible (some real IPL output): {len(eligible)} "
      f"| Excluded (negligible IPL sample): {len(excluded)}")

# ── 4. Regression residual: performance vs. log(SOLD PRICE) ────────────────
log_price = np.log(eligible["SOLD PRICE"].clip(lower=1))
slope, intercept = np.polyfit(log_price, eligible["performance"], 1)
eligible["expected_performance"] = slope * log_price + intercept
eligible["value_residual"] = eligible["performance"] - eligible["expected_performance"]

mu, sd = eligible["value_residual"].mean(), eligible["value_residual"].std()
eligible["z"] = (eligible["value_residual"] - mu) / sd

print(f"\nperformance ~ {slope:.3f} * log(SOLD PRICE) + {intercept:.3f}")

# ── 5. Outliers ───────────────────────────────────────────────────────────
top_value = eligible.sort_values("z", ascending=False).head(8)
worst_value = eligible.sort_values("z").head(8)

cols = ["PLAYER NAME", "COUNTRY", "PLAYING ROLE", "BASE PRICE", "SOLD PRICE",
        "price_tier", "performance", "value_residual", "z"]

print("\n=== Most undervalued (beat price-implied expectation most) ===")
print(top_value[cols].to_string(index=False))

print("\n=== Most overpriced (fell short of price-implied expectation most) ===")
print(worst_value[cols].to_string(index=False))

# Specifically: low-price-tier players who still rank well
low_tier_value = eligible[eligible["price_tier"] == "low (<=$100k)"].sort_values("z", ascending=False)
print(f"\n=== Best value among LOW base-price tier specifically (n={len(low_tier_value)}) ===")
print(low_tier_value.head(6)[cols].to_string(index=False))

best = top_value.iloc[0]
print(f"\n'{best['PLAYER NAME']}' ({best['COUNTRY']}, sold for ${best['SOLD PRICE']:,.0f}) "
      f"outperforms their price-implied expectation by {best['z']:.1f}σ.")

# ── 6. Visualize ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

ax = axes[0]
colors_map = {"low (<=$100k)": "#16a34a", "mid ($100k-$200k)": "#d97706", "high (>$200k)": "#dc2626"}
for tier, c in colors_map.items():
    sub = eligible[eligible["price_tier"] == tier]
    ax.scatter(sub["SOLD PRICE"], sub["performance"], label=tier, color=c, alpha=0.6, s=30)
price_range = np.linspace(eligible["SOLD PRICE"].min(), eligible["SOLD PRICE"].max(), 200)
ax.plot(price_range, slope * np.log(price_range.clip(min=1)) + intercept,
        color="#1a3a5c", linewidth=2, label="Price-implied expectation")
ax.set_xscale("log")
ax.set_xlabel("Sold price (USD, log scale)")
ax.set_ylabel("Composite performance score")
ax.set_title("Real IPL Auction Data — Performance vs. Price")
ax.legend(fontsize=7)

ax = axes[1]
ax.hist(eligible["value_residual"], bins=20, color="#444444", alpha=0.7, edgecolor="black")
ax.axvline(mu, color="white", linestyle="--", linewidth=1)
for _, row in top_value.head(3).iterrows():
    ax.annotate(f"{row['PLAYER NAME']}\n{row['z']:.1f}σ", xy=(row["value_residual"], 1),
                xytext=(row["value_residual"], 6), fontsize=7, color="#16a34a",
                ha="center", arrowprops=dict(arrowstyle="->", color="#16a34a"))
for _, row in worst_value.head(2).iterrows():
    ax.annotate(f"{row['PLAYER NAME']}\n{row['z']:.1f}σ", xy=(row["value_residual"], 1),
                xytext=(row["value_residual"], 4), fontsize=7, color="#dc2626",
                ha="center", arrowprops=dict(arrowstyle="->", color="#dc2626"))
ax.set_title("Residual distribution (real auction data, n=%d)" % len(eligible))
ax.set_xlabel("Performance above/below price-implied expectation")
ax.set_ylabel("Number of players")

fig.tight_layout()
fig.savefig("/Users/syedalhmdhusainkazmi/Desktop/ipl-moneyball/real_data_value_outliers.png", dpi=150)
print("\nSaved chart -> ipl-moneyball/real_data_value_outliers.png")

eligible.to_csv("/Users/syedalhmdhusainkazmi/Desktop/ipl-moneyball/real_data_ranked.csv", index=False)
print("Saved data -> ipl-moneyball/real_data_ranked.csv")
