"""
IPL Moneyball — v4: Adding Retained Icon Players
====================================================
v3 (03_real_merged_value_outliers.py) only covers players who went
through the open auction — which structurally excludes long-tenured
retained icon players (Kohli, Rohit, Dhoni) since their teams set
their price by unilateral retention, not competitive bidding.

This script adds them back in: real retention prices (Kaggle,
"IPL Auction Historical Data and Retention List" — the 2025
mega-auction retention list, 46 players) merged with their real
2025-season Cricsheet performance, scored on the *same* standardized
scale and against the *same* auction-derived expected-performance
curve as v3 — so retained and auctioned players land on one
comparable chart.

Honest caveat, stated here and in the dashboard: retention price is
not set by competitive bidding the way auction price is, so "expected
performance" for a retained player is an extrapolation of the
auction-fitted regression line, not a like-for-like mechanism. Useful
for "how does this retained price compare to what the market curve
would predict," not a claim that retention and auction are the same
economic process.
"""

import re
import numpy as np
import pandas as pd

RAW = "/Users/syedalhmdhusainkazmi/Desktop/ipl-moneyball/raw"

# ── 1. Rebuild the same Cricsheet performance table as v3 ──────────────────
balls = pd.read_csv(f"{RAW}/ipl_csv2/all_matches.csv", low_memory=False)
balls["year"] = balls["season"].str[:4].astype(int)

bat = (balls.groupby(["striker", "year"])
       .agg(runs=("runs_off_bat", "sum"), balls_faced=("ball", "count"))
       .reset_index().rename(columns={"striker": "name"}))

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

def split_cricsheet_name(n):
    parts = n.strip().split()
    if len(parts) < 2:
        return None, None
    initials, surname = parts[0], " ".join(parts[1:])
    return initials, surname.lower()

perf[["initials", "surname"]] = perf["name"].apply(lambda n: pd.Series(split_cricsheet_name(n)))
perf = perf.dropna(subset=["surname"])

# ── 2. Rebuild the v3 auction-matched population (needed as the SAME
#       standardization basis for bat_z/bowl_z — apples-to-apples) ─────────
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

merged = pd.merge(auction, perf, on=["surname", "year"], how="inner", suffixes=("_auction", "_perf"))
matched = merged[merged.apply(lambda r: str(r["initials"]).startswith(str(r["first_initial"])), axis=1)].copy()
matched = matched.drop_duplicates(subset=["Player", "year"])

# Same composite-performance formula and standardization basis as v3
bat_mean, bat_std = matched["runs"].mean(), matched["runs"].std()
bowlers_mask = matched["wickets"] > 0
bowl_score_matched = pd.Series(0.0, index=matched.index)
bowl_score_matched[bowlers_mask] = matched.loc[bowlers_mask, "wickets"] * (12 - matched.loc[bowlers_mask, "economy"]).clip(lower=0)
bowl_mean, bowl_std = bowl_score_matched[bowlers_mask].mean(), bowl_score_matched[bowlers_mask].std()

matched["performance"] = (
    ((matched["runs"] - bat_mean) / bat_std).fillna(0)
    + (((bowl_score_matched - bowl_mean) / bowl_std).where(bowlers_mask, 0)).fillna(0)
)

MIN_BALLS = 30
eligible = matched[(matched["balls_faced"] >= MIN_BALLS) | (matched["balls_bowled"] >= MIN_BALLS)].copy()

log_price = np.log(eligible["Amount"].clip(lower=1))
slope, intercept = np.polyfit(log_price, eligible["performance"], 1)
eligible["expected_performance"] = slope * log_price + intercept
eligible["value_residual"] = eligible["performance"] - eligible["expected_performance"]
mu, sd = eligible["value_residual"].mean(), eligible["value_residual"].std()
eligible["z"] = (eligible["value_residual"] - mu) / sd
eligible["price_type"] = "Auction"

print(f"Auction-priced population (same as v3): {len(eligible)} player-seasons")
print(f"performance ~ {slope:.3f} * log(price) + {intercept:.3f}  |  mu={mu:.3f}, sd={sd:.3f}")

# ── 3. Real retention data (Kaggle, 2025 mega-auction retention list) ──────
retained = pd.read_excel(f"{RAW}/ipl_retention/retention_players.xlsx")
retained["Amount"] = retained["Amount (in Cr)"] * 1e7  # crore -> rupees, same units as Amount above
retained["year"] = 2025  # retention is in force starting the 2025 season
retained[["first_initial", "surname"]] = retained["Player"].apply(lambda n: pd.Series(split_auction_name(n)))
retained = retained.dropna(subset=["surname"])

r_merged = pd.merge(retained, perf, on=["surname", "year"], how="inner", suffixes=("_retain", "_perf"))
r_matched = r_merged[r_merged.apply(lambda r: str(r["initials"]).startswith(str(r["first_initial"])), axis=1)].copy()
r_matched = r_matched.drop_duplicates(subset=["Player", "year"])

r_match_rate = len(r_matched) / len(retained) * 100
print(f"\nRetained players: {len(retained)} | Matched to 2025 Cricsheet performance: {len(r_matched)} ({r_match_rate:.1f}%)")

r_bowlers_mask = r_matched["wickets"] > 0
r_bowl_score = pd.Series(0.0, index=r_matched.index)
r_bowl_score[r_bowlers_mask] = r_matched.loc[r_bowlers_mask, "wickets"] * (12 - r_matched.loc[r_bowlers_mask, "economy"]).clip(lower=0)

# Same standardization basis (bat_mean/std, bowl_mean/std) as the auction population — apples-to-apples
r_matched["performance"] = (
    ((r_matched["runs"] - bat_mean) / bat_std).fillna(0)
    + (((r_bowl_score - bowl_mean) / bowl_std).where(r_bowlers_mask, 0)).fillna(0)
)

# Same auction-fitted expected-performance curve and same mu/sd — caveat: extrapolated onto
# retention prices, which aren't set by competitive bidding the way auction prices are.
r_log_price = np.log(r_matched["Amount"].clip(lower=1))
r_matched["expected_performance"] = slope * r_log_price + intercept
r_matched["value_residual"] = r_matched["performance"] - r_matched["expected_performance"]
r_matched["z"] = (r_matched["value_residual"] - mu) / sd
r_matched["price_type"] = "Retention"
r_matched["Role"] = None
r_matched["Player Origin"] = None

cols = ["Player", "Team", "year", "Role", "Player Origin", "Amount", "runs", "balls_faced",
        "wickets", "balls_bowled", "runs_conceded", "economy", "performance",
        "expected_performance", "value_residual", "z", "price_type"]

combined = pd.concat([eligible.assign(Role=eligible.get("Role"))[cols], r_matched[cols]], ignore_index=True)
combined.to_csv("/Users/syedalhmdhusainkazmi/Desktop/ipl-moneyball/all_players_value.csv", index=False)

print("\n=== Retained icon players — price vs. price-implied expectation (2025) ===")
icons = r_matched[r_matched["Player"].isin(["Virat Kohli", "Rohit Sharma", "MS Dhoni"])]
print(icons[["Player", "Team", "Amount", "performance", "expected_performance", "value_residual", "z"]].to_string(index=False))

print("\n=== All retained players, ranked by z ===")
print(r_matched.sort_values("z", ascending=False)[["Player", "Team", "Amount", "performance", "z"]].to_string(index=False))

print(f"\nSaved combined data -> all_players_value.csv ({len(combined)} rows: "
      f"{len(eligible)} auction + {len(r_matched)} retention)")
