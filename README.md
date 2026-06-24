# IPL Moneyball — Finding Undervalued Players in Auction Data

## Project Overview

Applies the Moneyball/sabermetrics thesis to IPL auctions: does auction price track actual on-field output, or does it track reputation more than performance? Built a regression-residual model that merges **real IPL auction prices** with **real ball-by-ball performance data** to flag players who outperform what their price predicts (undervalued) and players who underdeliver relative to theirs (overpriced) — then validated the result against independently-verifiable IPL history.

## Business Understanding

IPL franchises spend tens of millions of rupees per season bidding for players, but auction prices are set by reputation, recent form, and bidding-war dynamics as much as by underlying productivity. A franchise that can systematically identify players who are *underpriced relative to their output* gains a real competitive edge — exactly the data-driven scouting advantage made famous by the Oakland A's in baseball (the original "Moneyball" story). This project asks the same question of IPL auction data: who is winning teams more than they cost?

## Data Understanding

Two genuinely independent real sources, merged:

- **Performance — Cricsheet.org** ball-by-ball IPL data (`raw/ipl_csv2/all_matches.csv`, gitignored due to size — 295,732 real deliveries, 2007–2026). Per-player-per-season batting runs and bowling wickets/economy computed directly from this.
- **Price — Kaggle, "IPL Player Auction Dataset – From Start to Now"** (`raw/IPLPlayerAuctionData.csv`). 970 real auction records, 2013–2022, real prices, role, team, and player origin (Indian/Overseas).
- **The real entity-resolution problem, solved transparently:** Cricsheet names are "initials + surname" (`DA Warner`); the Kaggle data uses full names (`David Warner`). No shared player-ID registry exists in either source. Joined on (surname, year), disambiguating collisions by checking the auction record's first-name initial against Cricsheet's full initials. **Match rate: 620 of 969 auction records (64.0%)** — reported honestly rather than hidden; the rest are name-format misses or players with no recorded deliveries that season.
- After a sample-size floor (≥30 balls faced or bowled, the same standard applied throughout): **521 real player-seasons** of price + performance, properly time-aligned (no cross-era mismatches).
- An earlier version of this project used a single self-contained dataset (the IMB381 IIM Bangalore case study, 2008–2011) before this real-source merge was built — kept in the repo history as `02_real_data_value_outliers.py` for reference.

## Modeling and Evaluation

1. Built a composite IPL performance score per player-season: standardized batting runs + standardized bowling output (wickets, weighted down by economy rate).
2. Regressed performance on `log(auction price)` to get each player-season's price-implied *expected* performance.
3. Computed the residual (actual − expected) and z-scored it across all eligible player-seasons — the same outlier-detection technique used to show how far a true statistical anomaly sits from the norm (`z = (x − mean) / std`).
4. Applied the sample-size floor before ranking, excluding negligible-output player-seasons.

**A methodology correction made mid-build, not after:** the first version of this project used a naive `performance ÷ price` ratio. That metric is mathematically dominated by the denominator — dividing by a near-zero price inflates the ratio almost regardless of actual performance, surfacing "cheap players" rather than "genuinely underpriced skill." Switched to the regression-residual approach above, which is the actual mechanism real sabermetrics value-over-replacement metrics use.

## Conclusion

**Two independent, checkable sanity checks, not one:**
- **Robin Uthappa's 2014 season** surfaced as a top undervalued pick (3.1σ) — that was the season Uthappa actually won the IPL Orange Cap as the league's leading run-scorer, a fact the model had no knowledge of, arrived at purely from price and ball-by-ball data.
- **Kane Williamson's 2018 season** (3.7σ) and KL Rahul's 2018 season also surfaced as strongly undervalued — both real, recognizable standout seasons.
- On the overpriced side, several expensive 2021–2022 bowler signings (Jhye Richardson, Riley Meredith) underdelivered relative to price — plausible given those were COVID-bubble auction years with notably inflated pricing.

**Honest limitations:** the composite performance score is a simple, transparent standardized blend (batting runs + wickets-adjusted-for-economy), not a fully validated cricket impact metric. The 64% name-match rate means roughly a third of auction records couldn't be confidently linked to ball-by-ball performance — a real entity-resolution gap, not swept under the rug. The auction dataset covers 2013–2022; a 2023+ dataset would extend the analysis closer to the present. **Auction-only data also structurally excludes long-tenured retained icon players** — Virat Kohli, Rohit Sharma, and MS Dhoni appear nowhere in this analysis, not because of a matching failure but because RCB, MI, and CSK held them as permanent retained/icon players across most of these years rather than putting them through the open bidding auction this dataset records. Any "most undervalued/overpriced" ranking here is implicitly scoped to players who actually went through the auction.

**Next steps:** build a proper player-name registry (or source one) to lift the match rate above 64%; extend the auction data to 2023+; replace the composite score with richer per-phase features (powerplay/death-overs splits) now that the raw ball-by-ball data is already in hand.
