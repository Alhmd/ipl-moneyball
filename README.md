# IPL Moneyball — Finding Undervalued Players in Auction Data

## Project Overview

Applies the Moneyball/sabermetrics thesis to IPL auctions: does auction price track actual on-field output, or does it track reputation more than performance? Built a regression-residual model to flag players who outperform what their auction price predicts (undervalued) and players who underdeliver relative to theirs (overpriced) — then validated the result against real, independently-verifiable cricket history.

## Business Understanding

IPL franchises spend tens of millions of dollars per season bidding for players, but auction prices are set by reputation, recent form, and bidding-war dynamics as much as by underlying productivity. A franchise that can systematically identify players who are *underpriced relative to their output* gains a real competitive edge — exactly the data-driven scouting advantage made famous by the Oakland A's in baseball (the original "Moneyball" story). This project asks the same question of IPL auction data: who is winning teams more than they cost?

## Data Understanding

- **Source:** the IMB381 IPL Auction dataset — a real, well-known IIM Bangalore case study dataset, sourced from a public GitHub mirror (`raw/ipl_auction_2013.csv`). 130 real players, real `SOLD PRICE` / `BASE PRICE` (native USD units), real Test/ODI/IPL performance statistics.
- **Timeframe:** auction years 2008–2011. This is an older, smaller dataset — a current (2024+) auction dataset exists on Kaggle but requires API credentials not yet set up in this environment. Flagged as a natural extension, not hidden as a limitation.
- **A second real source was also downloaded but not yet merged:** Cricsheet.org's full IPL ball-by-ball match history (1,244 real matches, 2017–2026, `raw/ipl_csv2.zip` — gitignored here due to size, ~95MB extracted). It doesn't overlap in time with the auction dataset, so rather than forcing a fabricated cross-era join, this version uses the self-contained auction dataset alone, which already pairs price and performance for each player. Cricsheet remains available for a future, more recent iteration once a matching current auction-price source is sourced.
- **Reframing note:** the original framing targeted "uncapped" players specifically, but only 6 of 130 players in this dataset have zero international caps — too few to support that exact lens. Reframed to **auction price tier** (low/mid/high base price) instead, which is arguably closer to the real Moneyball question (cheap vs. productive) than the cricket-specific "capped" label.

## Modeling and Evaluation

1. Built a composite IPL performance score: standardized batting runs + standardized bowling output (wickets, weighted down by economy rate).
2. Regressed performance on `log(SOLD PRICE)` to get each player's price-implied *expected* performance.
3. Computed the residual (actual − expected) and z-scored it across all eligible players — the same outlier-detection technique used to show how far a true statistical anomaly sits from the norm (`z = (x − mean) / std`).
4. Applied a sample-size floor before ranking, excluding players with negligible real IPL output.

**A methodology correction made mid-build, not after:** the first version used a naive `performance ÷ price` ratio. That metric is mathematically dominated by the denominator — dividing by a near-zero price inflates the ratio almost regardless of actual performance, surfacing "cheap players" rather than "genuinely underpriced skill." Switched to the regression-residual approach above, which is the actual mechanism real sabermetrics value-over-replacement metrics use.

## Conclusion

The model's top "undervalued" picks on real data were **Malinga, Raina, Kallis, Gambhir, and Yusuf Pathan** — all genuinely strong, recognizable performers relative to their price. The model's most "overpriced" pick was **Andrew Flintoff**, whose $1.55M IPL deal is independently well known as one of the auction's most criticized signings — a real, checkable sanity check the model passed, not just an internally-consistent result on its own data.

**Honest limitations:** the dataset is small (130 players) and dated (2008–2011); a larger, more current dataset would strengthen the result and is a natural next step once Kaggle API access is set up. The composite performance score is a simple, transparent standardized blend, not a fully validated cricket impact metric — a reasonable v1, not a finished sabermetric formula.

**Next steps:** source a 2024+ auction dataset to repeat the analysis on current prices; merge in Cricsheet's ball-by-ball data (already downloaded) to build richer per-player performance features (powerplay/death-overs splits, matchup-specific stats) instead of the current single composite score.
