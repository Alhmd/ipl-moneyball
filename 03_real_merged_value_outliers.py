"""
IPL Moneyball — v3: Two Real Sources Properly Merged
=======================================================
v1 = synthetic (methodology proof). v2 = single real dataset
(IMB381, 2008-2011, self-contained price+performance). v3 = this
script: REAL ball-by-ball performance (Cricsheet, 2007-2026) merged
with REAL auction prices (Kaggle, "IPL Player Auction Dataset - From
Start to Now", 2013-2022) for the same player in the same season —
the genuine two-source merge that wasn't possible in v2 because the
years didn't overlap.

Name-matching problem, stated honestly: Cricsheet uses "DA Warner"
style (initials + surname); the auction dataset uses full names
("David Warner"). There's no shared player-ID registry bundled with
either source, so this script joins on (surname, year) and
disambiguates same-surname collisions by checking the auction
record's first-name initial is a prefix of Cricsheet's initials.
This is a heuristic, not a guaranteed-correct entity match — real
sports-data engineering problem, reported transparently below
(match rate + unmatched count), not hidden.
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RAW = "/Users/syedalhmdhusainkazmi/Desktop/ipl-moneyball/raw"

# ── 1. Real ball-by-ball performance (Cricsheet) ────────────────────────────
balls = pd.read_csv(f"{RAW}/ipl_csv2/all_matches.csv", low_memory=False)
balls["year"] = balls["season"].str[:4].astype(int)  # "2020/21" -> 2020

# Batting: runs and balls faced per striker per year
bat = (balls.groupby(["striker", "year"])
       .agg(runs=("runs_off_bat", "sum"), balls_faced=("ball", "count"))
       .reset_index().rename(columns={"striker": "name"}))

# Bowling: wickets and runs conceded per bowler per year
# (a "wicket_type" that isn't a run-out/obstructing etc. counts as bowler's wicket;
#  keep it simple and inclusive here — same simplification flagged as a
#  limitation, consistent with the honesty standard in v1/v2)
balls["bowler_wicket"] = balls["wicket_type"].notna() & (~balls["wicket_type"].isin(
    ["run out", "retired hurt", "retired out", "obstructing the field"]))
balls["runs_conceded"] = balls["runs_off_bat"].fillna(0) + balls[["wides", "noballs"]].fillna(0).sum(axis=1)

bowl = (balls.groupby(["bowler", "year"])
        .agg(wickets=("bowler_wicket", "sum"),
             balls_bowled=("ball", "count"),
             runs_conceded=("runs_conceded", "sum"))
        .reset_index().rename(columns={"bowler": "name"}))
bowl["economy"] = bowl["runs_conceded"] / (bowl["balls_bowled"] / 6)

perf = pd.merge(bat, bowl, on=["name", "year"], how="outer")
for c in ["runs", "balls_faced", "wickets", "balls_bowled", "runs_conceded"]:
    perf[c] = perf[c].fillna(0)

# Extract Cricsheet-style surname + initials: "DA Warner" -> initials="DA", surname="warner"
def split_cricsheet_name(n):
    parts = n.strip().split()
    if len(parts) < 2:
        return None, None
    initials, surname = parts[0], " ".join(parts[1:])
    return initials, surname.lower()

perf[["initials", "surname"]] = perf["name"].apply(lambda n: pd.Series(split_cricsheet_name(n)))
perf = perf.dropna(subset=["surname"])

print(f"Cricsheet player-seasons computed: {len(perf)}")

# ── 2. Real auction prices (Kaggle) ─────────────────────────────────────────
auction = pd.read_csv(f"{RAW}/IPLPlayerAuctionData.csv")
auction = auction.dropna(subset=["Year"])
auction["year"] = auction["Year"].astype(int)

def split_auction_name(n):
    parts = re.sub(r"[^A-Za-z ]", "", n).strip().split()
    if len(parts) < 2:
        return None, None
    first_initial, surname = parts[0][0], parts[-1]
    return first_initial.upper(), surname.lower()

auction[["first_initial", "surname"]] = auction["Player"].apply(lambda n: pd.Series(split_auction_name(n)))
auction = auction.dropna(subset=["surname"])

# ── 3. Merge — surname + year, disambiguate by initial prefix ──────────────
merged = pd.merge(auction, perf, on=["surname", "year"], how="inner", suffixes=("_auction", "_perf"))
# Keep only rows where the auction's first initial is a prefix of Cricsheet's initials
matched = merged[merged.apply(lambda r: str(r["initials"]).startswith(str(r["first_initial"])), axis=1)].copy()
# Drop surname+year collisions that matched more than one Cricsheet identity
matched = matched.drop_duplicates(subset=["Player", "year"])

match_rate = len(matched) / len(auction) * 100
print(f"Auction records: {len(auction)} | Matched to real Cricsheet performance: {len(matched)} "
      f"({match_rate:.1f}%) — the rest are name-matching misses (different name formats, "
      f"data-entry variants, or players who didn't bat/bowl a ball that season) or simply "
      f"not present in the surname+year merge.")

# ── 4. Composite performance + value-for-money (same technique as v1/v2) ──
bat_z = (matched["runs"] - matched["runs"].mean()) / matched["runs"].std()
bowlers_mask = matched["wickets"] > 0
bowl_score = pd.Series(0.0, index=matched.index)
bowl_score[bowlers_mask] = matched.loc[bowlers_mask, "wickets"] * (12 - matched.loc[bowlers_mask, "economy"]).clip(lower=0)
bowl_z = pd.Series(0.0, index=matched.index)
if bowlers_mask.sum() > 1:
    bowl_z[bowlers_mask] = (bowl_score[bowlers_mask] - bowl_score[bowlers_mask].mean()) / bowl_score[bowlers_mask].std()

matched["performance"] = bat_z.fillna(0) + bowl_z.fillna(0)

MIN_BALLS = 30  # faced or bowled — sample-size floor, same honesty standard as v1/v2
eligible = matched[(matched["balls_faced"] >= MIN_BALLS) | (matched["balls_bowled"] >= MIN_BALLS)].copy()
print(f"Eligible after sample-size floor (>= {MIN_BALLS} balls faced or bowled): {len(eligible)}")

log_price = np.log(eligible["Amount"].clip(lower=1))
slope, intercept = np.polyfit(log_price, eligible["performance"], 1)
eligible["expected_performance"] = slope * log_price + intercept
eligible["value_residual"] = eligible["performance"] - eligible["expected_performance"]

mu, sd = eligible["value_residual"].mean(), eligible["value_residual"].std()
eligible["z"] = (eligible["value_residual"] - mu) / sd

print(f"\nperformance ~ {slope:.3f} * log(price) + {intercept:.3f}")

cols = ["Player", "year", "Role", "Player Origin", "Amount", "performance", "value_residual", "z"]
top_value = eligible.sort_values("z", ascending=False).head(10)
worst_value = eligible.sort_values("z").head(10)

print("\n=== Most undervalued (real prices, real performance, real years) ===")
print(top_value[cols].to_string(index=False))

print("\n=== Most overpriced ===")
print(worst_value[cols].to_string(index=False))

best = top_value.iloc[0]
print(f"\n'{best.Player}' ({best.year}, ₹{best.Amount:,.0f}) outperformed price-implied "
      f"expectation by {best.z:.1f}σ that season.")

eligible.to_csv("/Users/syedalhmdhusainkazmi/Desktop/ipl-moneyball/real_merged_ranked.csv", index=False)

# ── 5. Visualize ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
ax = axes[0]
ax.scatter(eligible["Amount"], eligible["performance"], alpha=0.4, color="#888888", s=20)
pr = np.linspace(eligible["Amount"].min(), eligible["Amount"].max(), 200)
ax.plot(pr, slope * np.log(pr.clip(min=1)) + intercept, color="#1a3a5c", linewidth=2)
ax.set_xscale("log")
ax.set_xlabel("Auction price (log scale)")
ax.set_ylabel("Composite performance (that season)")
ax.set_title(f"Real merged data — n={len(eligible)} player-seasons")

ax = axes[1]
ax.hist(eligible["value_residual"], bins=25, color="#444444", alpha=0.7, edgecolor="black")
ax.axvline(mu, color="white", linestyle="--", linewidth=1)
for _, row in top_value.head(3).iterrows():
    ax.annotate(f"{row['Player']} ({row['year']})\n{row['z']:.1f}σ", xy=(row["value_residual"], 1),
                xytext=(row["value_residual"], 6), fontsize=7, color="#16a34a",
                ha="center", arrowprops=dict(arrowstyle="->", color="#16a34a"))
for _, row in worst_value.head(2).iterrows():
    ax.annotate(f"{row['Player']} ({row['year']})\n{row['z']:.1f}σ", xy=(row["value_residual"], 1),
                xytext=(row["value_residual"], 4), fontsize=7, color="#dc2626",
                ha="center", arrowprops=dict(arrowstyle="->", color="#dc2626"))
ax.set_title("Residual distribution")
ax.set_xlabel("Performance above/below price-implied expectation")
ax.set_ylabel("Player-seasons")

fig.tight_layout()
fig.savefig("/Users/syedalhmdhusainkazmi/Desktop/ipl-moneyball/real_merged_value_outliers.png", dpi=150)
print("\nSaved chart -> ipl-moneyball/real_merged_value_outliers.png")
print("Saved data -> ipl-moneyball/real_merged_ranked.csv")
