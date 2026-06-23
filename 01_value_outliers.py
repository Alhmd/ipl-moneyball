"""
IPL Moneyball — Undervalued Uncapped Players
==============================================
Stand-in synthetic dataset: models realistic IPL auction-price and
performance distributions. Swap in real Cricsheet (ball-by-ball) +
public IPL auction-price data once sourced — pipeline below stays
the same.

Technique: the same z-score outlier method used to show Bradman is
~6 sigma above every other Test batsman, or Messi is ~5.9 sigma above
every other attacker. Here it's applied to "performance per crore
spent" instead of a raw career stat — the actual Moneyball question:
who is winning you more than they cost?
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ── 1. Build the synthetic player pool ──────────────────────────────────────
N = 180

# Auction price (INR crore). Real IPL auctions are heavily right-skewed:
# a handful of marquee names go for 15-25 cr, most squad players go for
# well under 2 cr, and uncapped players are often a fraction of a crore.
is_marquee   = rng.random(N) < 0.08                          # ~8% marquee
is_uncapped  = (~is_marquee) & (rng.random(N) < 0.35)         # ~35% of the rest
is_capped_squad = ~(is_marquee | is_uncapped)

price = np.empty(N)
price[is_marquee]      = rng.uniform(8, 24, is_marquee.sum())
price[is_capped_squad] = rng.uniform(1, 7, is_capped_squad.sum())
price[is_uncapped]     = rng.uniform(0.2, 1.5, is_uncapped.sum())

# Innings/overs played this season — uncapped players genuinely get
# fewer chances, which is the small-sample risk this project has to
# handle deliberately rather than ignore.
innings = np.empty(N, dtype=int)
innings[is_marquee]      = rng.integers(10, 16, is_marquee.sum())
innings[is_capped_squad] = rng.integers(6, 15, is_capped_squad.sum())
innings[is_uncapped]     = rng.integers(1, 9, is_uncapped.sum())

# Composite performance score (batting + bowling impact, arbitrary
# 0-100 scale standing in for a real combined metric). Performance is
# mostly independent of price — that's the whole point: price tracks
# reputation more than it tracks output, so some cheap players will
# randomly land high performance and some expensive ones will land low.
base_skill = rng.normal(50, 15, N)
# Marquee players get a *mild* skill premium (they're not picked for
# no reason) but with wide variance — some marquee picks underperform.
base_skill[is_marquee] += rng.normal(8, 12, is_marquee.sum())
performance = np.clip(base_skill, 5, 95)

names = [f"Player_{i:03d}" for i in range(N)]
status = np.where(is_marquee, "marquee",
          np.where(is_uncapped, "uncapped", "capped_squad"))

df = pd.DataFrame({
    "player": names,
    "status": status,
    "price_crore": price.round(2),
    "innings": innings,
    "performance": performance.round(1),
})

# ── 2. Value-for-money metric — regression residual, not a raw ratio ───────
# A naive performance/price ratio is dominated by the denominator: a cheap
# player with mediocre performance still gets a huge ratio just because the
# price is tiny. That's an artifact of division, not a genuine "underpriced
# skill" finding. The defensible version: regress performance on price (log,
# since price is right-skewed) to get each player's EXPECTED performance at
# their price point, then look at who beats that expectation the most.
# Outperforming-for-your-price-tier is the actual Moneyball question.
log_price = np.log(df["price_crore"])
slope, intercept = np.polyfit(log_price, df["performance"], 1)
df["expected_performance"] = slope * log_price + intercept
df["value_residual"] = (df["performance"] - df["expected_performance"]).round(2)

# ── 3. Sample-size cutoff — the honest part, not the optional part ─────────
MIN_INNINGS = 6
eligible = df[df["innings"] >= MIN_INNINGS].copy()
excluded = df[df["innings"] < MIN_INNINGS].copy()

print(f"Total players: {len(df)}")
print(f"Eligible for ranking (>= {MIN_INNINGS} innings): {len(eligible)}")
print(f"Excluded as small-sample (< {MIN_INNINGS} innings): {len(excluded)} "
      f"— mostly uncapped: {excluded['status'].eq('uncapped').sum()}/{len(excluded)}")

# ── 4. Z-score the RESIDUAL — same technique as the Bradman/Messi reels,
# now applied to "how far above/below your price-implied expectation" ──────
mu, sd = eligible["value_residual"].mean(), eligible["value_residual"].std()
eligible["z"] = (eligible["value_residual"] - mu) / sd

print(f"\nPrice-performance regression: performance ~ {slope:.2f}*log(price) + {intercept:.2f}")
print(f"Population: value_residual mean={mu:.2f}, std={sd:.2f}")

# ── 5. Surface the outliers ──────────────────────────────────────────────────
top_value = eligible.sort_values("z", ascending=False).head(8)
worst_value = eligible.sort_values("z").head(8)

print("\n=== Most undervalued players (most above price-implied expectation, z-score) ===")
print(top_value[["player", "status", "price_crore", "performance",
                  "expected_performance", "value_residual", "z", "innings"]].to_string(index=False))

print("\n=== Most overpriced players (most below price-implied expectation, z-score) ===")
print(worst_value[["player", "status", "price_crore", "performance",
                    "expected_performance", "value_residual", "z", "innings"]].to_string(index=False))

best = top_value.iloc[0]
print(f"\n'{best.player}' ({best.status}, ₹{best.price_crore}cr) outperforms what their "
      f"price tier predicts by {best.z:.1f}σ — on {best.innings} innings of evidence.")

# ── 5b. The actual project thesis: uncapped players specifically ───────────
# The overall z-score leaderboard above can be dominated by marquee/capped
# players who simply had a good season — that's a real result, but it's not
# the question this project asks. Surface the uncapped-specific ranking too.
uncapped_eligible = eligible[eligible["status"] == "uncapped"].sort_values("z", ascending=False)
print(f"\n=== Best-value UNCAPPED players specifically (n={len(uncapped_eligible)} eligible) ===")
print(uncapped_eligible.head(5)[["player", "price_crore", "performance",
                                  "expected_performance", "z", "innings"]].to_string(index=False))
if len(uncapped_eligible):
    top_uncapped = uncapped_eligible.iloc[0]
    print(f"\nBest-value uncapped find: '{top_uncapped.player}' (₹{top_uncapped.price_crore}cr) "
          f"at {top_uncapped.z:.1f}σ above price-implied expectation, on "
          f"{top_uncapped.innings} innings — flagged, not guaranteed, given the sample size.")

# ── 6. Visualize — bell curve with annotated outliers (reelgorithm.js style) ─
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

ax = axes[0]
ax.scatter(df["price_crore"], df["performance"], alpha=0.5, color="#888888", s=20)
price_range = np.linspace(df["price_crore"].min(), df["price_crore"].max(), 200)
ax.plot(price_range, slope * np.log(price_range) + intercept, color="#1a3a5c",
        linewidth=2, label="Price-implied expectation")
for _, row in top_value.head(3).iterrows():
    ax.scatter(row.price_crore, row.performance, color="#16a34a", s=60, zorder=5)
ax.set_xscale("log")
ax.set_xlabel("Auction price (₹crore, log scale)")
ax.set_ylabel("Performance")
ax.set_title("Performance vs. Price — who beats the trend line?")
ax.legend(fontsize=8)

ax = axes[1]
ax.hist(eligible["value_residual"], bins=30, color="#444444", alpha=0.7,
        edgecolor="black")
ax.axvline(mu, color="white", linestyle="--", linewidth=1)

for _, row in top_value.head(3).iterrows():
    ax.annotate(f"{row.player}\n{row.z:.1f}σ", xy=(row.value_residual, 1),
                xytext=(row.value_residual, 8), fontsize=8, color="#16a34a",
                ha="center", arrowprops=dict(arrowstyle="->", color="#16a34a"))

for _, row in worst_value.head(2).iterrows():
    ax.annotate(f"{row.player}\n{row.z:.1f}σ", xy=(row.value_residual, 1),
                xytext=(row.value_residual, 5), fontsize=8, color="#dc2626",
                ha="center", arrowprops=dict(arrowstyle="->", color="#dc2626"))

ax.set_title("Residual distribution (eligible players)")
ax.set_xlabel("Performance above/below price-implied expectation")
ax.set_ylabel("Number of players")
fig.tight_layout()
fig.savefig("/Users/syedalhmdhusainkazmi/Desktop/ipl-moneyball/value_outliers.png", dpi=150)
print("\nSaved chart -> ipl-moneyball/value_outliers.png")

df.to_csv("/Users/syedalhmdhusainkazmi/Desktop/ipl-moneyball/synthetic_ipl_players.csv", index=False)
eligible.to_csv("/Users/syedalhmdhusainkazmi/Desktop/ipl-moneyball/eligible_ranked.csv", index=False)
print("Saved data -> ipl-moneyball/synthetic_ipl_players.csv, eligible_ranked.csv")
