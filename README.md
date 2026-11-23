# Portfolio GBM

A Monte Carlo of a rebalanced book. The length is the first `Time to Maturity` on the model sheet, in years. Each path is a correlated geometric Brownian motion whose drift and volatility come from a three-state regime. A momentum sleeve, when the sheet marks one, rotates on its own clock. Allocation bands snap a whole path back to target weights when a trigger asset leaves its band.

The step engine is `gbm/gbm_step.py`. It reads `portfolio_gbm_calculator.xlsx` from the working directory. Correlation factors and the momentum-drift cache live next to the Python package, not next to the workbook.

```
poetry install
poetry run python -m gbm.gbm_step
poetry run python -m gbm.gbm_step --sheet-name "Sector Momentum Playbook" --dynamic-rebalance --top_num 3 --no-beat_cash
poetry run python -m unittest
```

Python 3.11 or newer. The workbook `portfolio_gbm_calculator.xlsx` is in the repository and is read from the working directory.

The run is not seeded. Warmup, daily shocks, and regime draws share NumPy's global generator, in that order. Changing the longest SMA window changes the valuation shocks, because the warmup draws more normals before the first priced day.

## Run settings that are not flags

| Setting | Value | Where |
| --- | --- | --- |
| Horizon | First `Time to Maturity`, in years. Asset Parameters is 20. Sector Momentum Playbook is 10. A missing cell stays at 10. | `main` |
| Step | 1/252 of a year | `main` |
| Paths | 5,000 | `main` |
| Starting value | 171,655 | `main` |
| Priced days | `int(T / dt)`, plus the opening row. 20 years is 5,040 priced days. 10 years is 2,520. | `main` |
| Workbook | `portfolio_gbm_calculator.xlsx` in the working directory | `main` |

## Command-line options

| Flag | Default | What it changes |
| --- | --- | --- |
| `--sheet-name` | `Asset Parameters` | Which model sheet supplies labels, bands, targets, the horizon, SMA windows, asset type, and buckets. The other sheet in the workbook is `Sector Momentum Playbook`. |
| `--dynamic-rebalance` | off | Band checks every trading day. Off means every 21 steps. |
| `--top_num` | 3 | How many momentum candidates share the sleeve. Each receives an equal share. A bucket counts as one candidate. |
| `--beat_cash` / `--no-beat_cash` | on | On: a candidate must sit further above its SMA than the annual cash drift. Off: any candidate above its SMA can be held. Empty shares stay in SGOV either way. |

The cash hurdle is a level gap. The score is `(price - average) / average`. It is compared with SGOV's annual drift, about 0.04 after the expense haircut. A name 3% above its average loses to a 4% cash drift. It is not an annualized excess return.

## Workbook

`main` reads four sheets. `Sheet3` is not read. `Asset Class` is not read. The first `Time to Maturity` is the horizon in years. Later cells in that column are not read. A literal `Net Expense Ratio` on the model sheet replaces the fee table for that ticker. A formula keeps the `Net Expense Ratios` lookup. Asset Parameters now stores the same ratios the run charges: SGOV is 0.003 and VXUS is 0.0003.

### Model sheet

`Asset Parameters` is the core book. `Sector Momentum Playbook` is the sector rotation. Both use the same columns.

| Column | Used for |
| --- | --- |
| `Ticker` | Label. `VOO/VIIX` takes the VOO correlation and behavior row. `BTC` and `IBIT` share the BTC correlation row. |
| `Time to Maturity` | Horizon of the run, in years. Only the first row is read. |
| `Target Allocation` | See below. This column does not have to sum to 1. |
| `Lower Band`, `Upper Band` | A weight at or outside the band can snap the path. Comparison is inclusive. |
| `SMA Exit`, `SMA Enter` | Window lengths, in trading days. A missing pair falls back to 200 and 200. |
| `SMA-exclusion` | Newest closes left out of that average. Blank or non-positive means none. The current sheets use 25. |
| `Asset Type` | `Momentum`, `Alternate` (also `Alternative`), or `Gated`. Blank is none of those. |
| `Bucket` | Momentum names that share a name are one candidate. The bucket's gap is the allocation-weighted average of its members. Member targets must sum to the momentum portion of the book. |
| `Net Expense Ratio` | A typed number replaces the fee table for that row. A formula keeps the lookup. |
| `Tracked Index` | Optional. A ticker in this column follows that row's SMA. `none` or `non` never liquidates. The current sheets do not have this column. |

`Asset Type` decides which machine runs.

- Any `Momentum` row turns on the sleeve. Dollars in those columns, including a Momentum-typed SGOV, are what a rotation can see.
- `Alternate` rows sit outside the sleeve. They keep their target weight, grow on their own, and can trip a band when that target is positive.
- `Gated` is an alternate that parks in SGOV while the path is in cash, and is split back out of that pile on the trade that leaves. Asset Parameters types BTC as `Alternate`, so BTC stays invested while the sleeve is in cash. The sector sheet types BTC as `Gated`, so there it parks.
- A blank type is not in the sleeve and not an alternate. Once any row is Momentum, that row's target is ignored, it receives no weight, and it cannot trip a band. VOO on the sector sheet is in this state.

Sleeve weight is `1 - (alternate targets + gated targets)`. Solo momentum targets are not live weights. Each of the top `top_num` names receives an equal fraction of the sleeve, and unfilled slots stay in SGOV. On Asset Parameters the alternates are DBMF 0.20 and BTC 0.20, so the sleeve is 0.60. The 0.60 printed on VOO and on SGOV does not size the book. VXUS is an alternate with target 0, so it is held at zero and its band cannot force a trade.

On the sector sheet the alternates are DBMF 0.20 and gated BTC 0.10, so the sleeve is 0.70. The 0.70 printed on every sector is the same kind of placeholder. The top 3 of those sectors share that 0.70.

### Asset State Behaviors

One row per ticker. Three column pairs: State 1 is Expansion, State 2 is Churn, State 3 is Liquidation. Drift and volatility are annual, and the row is the drift the run uses. Every path opens in Churn, not in the stationary mix of the transition matrix.

The BTC row is the mature-asset vector. The fee is subtracted after the row is read:

| Regime | Drift | Volatility |
| --- | --- | --- |
| Expansion | 0.54 | 0.35 |
| Churn | 0.43 | 0.45 |
| Liquidation | -0.40 | 0.60 |

`Momentum Lambda` and `Momentum Center` on the same sheet replace the cache. BTC is lambda 0 and a center of `exp(0.54) - 1`. DBMF and SGOV are zero. Editing either cell changes the add-on.

The transition sheet names the cash-flight tickers, currently SPMO and DWAS. When a book contains one, its liquidation row is replaced with SGOV's gross State 3 drift and volatility from this sheet, after fees have been applied to everyone else. That copy does not include SGOV's fee. A blank cash-flight cell keeps every liquidation row. The current model sheets do not hold those two tickers, so the rule is idle there. TQQQ keeps its own liquidation row. The Churn row is what `main` passes as the fallback drift. The regime book replaces it on every step.

### Net Expense Ratios

Annual drift, in every regime, is reduced by this ratio unless the model sheet types a different number on that row. Volatility is not. A label with no row and no model-sheet number keeps its full drift. `BTC` looks up `IBIT` and the reverse. `VIIIX` looks up `VOO`.

### Transition Matrix

Rows are Bull, Churn, Bear and must each sum to 1. The current sheet is:

| From | Bull | Churn | Bear |
| --- | --- | --- | --- |
| Bull | 0.88 | 0.07 | 0.05 |
| Churn | 0.15 | 0.70 | 0.15 |
| Bear | 0.20 | 0.10 | 0.70 |

`Trading Days Between Transitions` is 21. That is 21 steps of `dt`, which is 21/252 of a year. `Probability per Standard Deviation` is 0.08 on the sheet (the code default, used only when the cell is missing, is 0.10). `Standard Deviation Cap` is 3.

Below the tilted matrices, the sheet holds the stress block the run uses. A missing label falls back to the same number in code. A present label wins:

| Cell | Value the run uses |
| --- | --- |
| Churn Correlation Scale | 0.75 |
| Liquidation Blend | 0.70 |
| Liquidation Correlation | 0.85 |
| Bitcoin Stress Correlation | 0.45 |
| Bitcoin Stress Blend | 0.60 |
| DBMF Expansion Equity Correlation | -0.10 |
| DBMF Churn Equity Correlation | -0.075 |
| DBMF Liquidation Equity Correlation | -0.40 |
| Churn Volatility Grace Months | 3 |
| Churn Volatility Monthly Step | 0.10 |
| Cash Flight Tickers | SPMO, DWAS |

The trigger is VOO, or the first asset that is not SGOV when the sheet has no VOO. Its market growth is compounded from 1. Once that history covers the trigger's SMA Enter length, the close is scored in standard deviations of that window. The window does not drop `SMA-exclusion` closes. The portfolio SMA does. A positive score moves probability from Churn and Bear onto Bull, in proportion to the probability those two already hold. A negative score does the reverse. The move stops at zero and at the cap. Until the window is full, draws use the untilted matrix. The warmup that fills the portfolio SMA does not fill this window.

While a path is in Churn, the trigger's volatility increases by the monthly step for each month after the grace. Both numbers are on the transition sheet: 0.10 after three months. A month is 21 steps. Other assets do not get the add-on. The add-on uses days already accumulated, so the first day of a stay has none.

## What one day does

1. Read drift and volatility for the path's current regime. Add the Churn volatility creep on the trigger. Add the momentum drift adjustment when a full 12-month shadow history exists.
2. Draw one standard normal per asset, correlate it with the factor for that regime, and turn it into a growth factor.
3. Mark the book. A momentum sleeve marks every name at its own drift, then rotates on a trade day. A book without a sleeve pays cash drift and cash volatility on sleeves the signal has turned off, and the signal sees the unheld market factor from the same shock.
4. On a band-check day, snap any path whose trigger asset is outside its band. The whole row is replaced with target weights. On a day that is both a trade and a breach, the breach wins. The target book is already the post-trade book.
5. Record the portfolio log return against the holding that was in place at the start of the day.
6. Advance the regime trigger. When the interval elapses, draw the next state. That state applies tomorrow.

A momentum sleeve with an empty history is warmed up before step 1. The warmup is `lead_days` of shadow prices under the opening regime: the longest SMA Enter or SMA Exit window, including the closes `SMA-exclusion` leaves out of the average. The current sheets need 275 days (250 exit + 25 exclusion). The opening value is preserved and reassigned to the warmed-up targets. The regime clock does not move.

## Clocks

| Clock | Period | Advances | Does not touch |
| --- | --- | --- | --- |
| Price | every step | valuation, shadow prices | regime state, until the end of the step |
| Regime draw | 21 steps | state for the next step | portfolio weights |
| Momentum trade | 25 steps | which sleeve names are held | alternates |
| Band check | 21 steps, or every step | every weight on a path that breached | shadow prices, signal ranking |

21 and 25 meet every 525 steps. A regime month and a rebalance month fall on the same day only because both default to 21. They are separate counters.

## Correlation

The default factor is `gbm/asset_cholesky.csv`, built from up to 20 years of daily simple returns if the file is missing or the wrong shape. Pearson correlation, pairwise, then eigenvalues floored at 1e-8 and the diagonal restored before the Cholesky factor. BTC is spot history scaled onto the first ETF print, then the ETF itself. A ticker with no row is uncorrelated with the rest of the book.

Each path then uses one of three factors. The levels below are the transition-sheet cells:

- Expansion keeps that historical matrix. A book that contains DBMF rewrites DBMF's equity correlation to the expansion cell, -0.10, first.
- Churn multiplies every off-diagonal by the churn scale, 0.75. DBMF's equity correlation is the churn cell, -0.075.
- Liquidation blends the historical matrix with a flat correlation using the blend and the flat level, 70% of 0.85. BTC then walks the stress blend, 60%, of the way from its own history toward the stress correlation, 0.45, against risky assets, and keeps its historical correlation with SGOV. DBMF's equity correlation is the liquidation cell, -0.40, and its other pairs are put back to history. KMLM does not get the DBMF override.

Churn and Liquidation are projected onto a positive-definite correlation before the factor exists. That projection can move the 0.85 blend and the prescribed DBMF entries. Shocks are multiplied by L transpose, so their covariance is L @ L.T.

An equity, for the DBMF override, is a known equity ticker or any name whose historical correlation with VOO is at least 0.50. Cash, gold, bitcoin, and the other managed-futures fund stay off that list.

## Momentum drift

`gbm/momentum_drift.csv` stores, per ticker, the slope of the next 12-month simple return on the past 12-month simple return, and the mean of those past returns. Yearly returns outside [-80%, +200%] are pulled in before the slope is fit. Fewer than 24 month-end pairs produces a zero slope.

Each day, once 252 shadow days exist (12 times 21), the asset's drift becomes the regime drift plus `lambda * (trailing return - center)`, and that add-on is clipped to [-0.40, +0.35]. The shadow price compounds market growth only. Rebalance and rotation do not enter it. With no stored center, the trailing return is compared with the return today's drift itself would earn over the lookback, so a path that realizes its own drift keeps that drift.

BTC's lambda and center on the behavior sheet are 0 and `exp(0.54) - 1`, which replaces the cache. DBMF's sheet cells are zero, which replaces its cached reversal. SGOV's sheet cells are zero.

## Functions

### Correlation and shocks

| Function | Role |
| --- | --- |
| `correlation_for_labels` | Rebuild L @ L.T and copy it into the sheet's ticker order. Unknown labels stay on the identity. |
| `simulation_cholesky` | Asset Parameters uses the saved multi-asset factor. Any other sheet uses, or builds, a factor for that sheet's tickers. |
| `verify_and_generate_cholesky` | Factor a correlation. A matrix that is not positive definite gets 1e-8 on the diagonal and is tried again. |
| `nearest_correlation` | Higham-style alternating projection onto a correlation matrix. Used after the regime blends. |
| `regime_correlation_matrices` | Expansion, Churn, and Liquidation correlations, including the BTC and DBMF rewrites. |
| `regime_cholesky_factors` | One lower-triangular factor per regime, stacked in state order. |
| `apply_cholesky` | Multiply shocks by L transpose. Accepts one factor, one factor per path, or one factor per regime selected by `states`. |

### Growth

| Function | Role |
| --- | --- |
| `gbm_growth` | One lognormal step, `exp((mu - 0.5 sigma^2) dt + sigma sqrt(dt) Z)`. A zero holding still gets a shadow factor. |
| `growth_factors` | One shared shock and two drifts. The market factor feeds the SMA. The held factor is what a non-sleeve book earns. |
| `total_gbm` | Shadow closes for the warmup, oldest first, starting at 1. |
| `open_from_total_gbm` | Load that history, start every path in cash, then rank. The hurdle is SMA Enter. |
| `gbm_step_vectorized_3d` | Mark a book one day. The simulator inlines the same product. |

### Trading

| Function | Role |
| --- | --- |
| `dynamic_rebalance` | If any allowed asset is outside its band, replace the path with target weights. Otherwise keep the drifted dollars. |
| `rotate_momentum_sleeve` | Sum the momentum columns, zero them, and write that sum back onto the selection or onto sleeve weights. |
| `settle_momentum_book` | On the way out of cash, restore gated dollars first, then rotate. On the way in, rotate, then sweep gated dollars onto SGOV. |
| `momentum_shift` | Mark, record the close in the SMA, and rotate when the 25-day clock says so. Does not look at bands. |
| `monthly_momentum` | The no-sleeve day: cash parameters on unheld sleeves, targets only on a band-check day. |
| `simulate_with_rebalance` | The daily loop. Returns prices shaped `(days, paths, assets)`, plus lower and upper band counts. |
| `trading_rebalance_interval` | 1 if dynamic rebalance is on, else 21. |

### Reporting

| Function | Role |
| --- | --- |
| `summarize_valuation_metrics` | Starting value, 90th percentile, mean, median, and 5th percentile, each with the CAGR that compounds the start into that ending value. |
| `summarize_regime_time` | Years and share of time in each regime, plus the mean and median length of one finished visit. Row 0 of the state grid is not an extra interval. A visit still open on the last step is not a finished cycle. |
| `summarize_regime_cagr` | Mean CAGR pools every day spent in the regime. Median CAGR is the median of each visiting path's own rate. |
| `summarize_momentum_time` | The same two CAGRs for each momentum bucket, plus years held. |
| `candidate_momentum_days` | Collapse a bucket's members with the max day count, since they are held together. |
| `candidate_momentum_log` | Collapse the same members by taking the first member's log return. |
| `summarize_trigger_buckets` | Which names caused band snaps, and on how many paths. |
| `ending_cagr` | The rate on the valuation printout. A non-positive start or a zero horizon has no rate. |

### Entry

| Function | Role |
| --- | --- |
| `horizon_years` | The first `Time to Maturity`, in years. A missing cell stays at 10. |
| `parse_args` | The four flags above. |
| `main` | Load the workbook. The first Time to Maturity sets the length. Sheet cells replace the fee table, the stress defaults, and the momentum cache. Run the loop and print the tables. |
| `_regime_trigger` | VOO, or the first non-cash asset, with a printed note when VOO is absent. |

Supporting modules, not the step file: `regime.py` holds the Markov book and the SMA tilt, `signals.py` holds SMA state and the ranking, `momentum.py` holds the trailing-return add-on, `asset_simulation_matrix.py` builds the factor, `bitcoin.py` and `dbmf.py` apply the crisis correlations the transition sheet names, `expenses.py` subtracts fees.

## Assumptions

These are choices the output depends on.

1. Drift is the instantaneous expected return, not the expected log return. The median path grows at `mu - 0.5 sigma^2`. A 60% volatility costs about 18 percentage points of median growth per year before regimes are mixed.
2. One year is 252 steps. A "month" of 21 steps is 21/252 of a year, not a calendar month. The momentum lookback of 12 such months is exactly one simulated year. The trade clock is 25 steps and is not a month.
3. Every path starts in Churn. The chain has not been run to its stationary distribution. The first 210 trigger days, on the current sheets, draw from the untilted matrix.
4. There are no transaction costs, taxes, slippage, or borrow fees. A snap and a rotation both trade the full book at the close that was just simulated.
5. Leveraged and factor funds are their own diffusions. TQQQ is not restated as three times VOO each day. Its behavior row is the daily-reset Nasdaq-100 history since 1996, scaled into VOO's scenario. Drift comes from monthly returns in each market state. Volatility comes from the daily closes in those same months, because the step is one day. Daily-reset decay is the half-variance term on that volatility.
6. SGOV is a GBM with a yield and, in the behavior sheet, zero volatility. It is not a Treasury bill. In Liquidation the flat 0.85 blend raises most assets' correlation with SGOV. BTC and DBMF are the exceptions, by construction. Flight-to-quality is not what the blend does.
7. The band rule is path-wide. One trigger asset outside its band resets every weight. Momentum names are not triggers. A zero target is not a trigger.
8. Shadow prices ignore rebalance. The SMA and the 12-month add-on can keep trending through a snap that moved the dollars.
9. The regime average and the portfolio average of the same trigger are different series. Only the portfolio average drops the newest `SMA-exclusion` closes.
10. Mean CAGR and median CAGR answer different questions. The mean pools log returns across paths and days. The median is the median of per-path rates, so a one-day visit can move it.
11. A missing workbook replaces only the first sheet with a dummy frame. The next read, Asset State Behaviors, still requires the file. A missing workbook does not finish a run.
12. Importing `gbm_step` loads the default Cholesky factor, and will download prices if that file is missing or the wrong shape.

## License

[MIT](LICENSE). Copyright (c) 2026 Evan Voelker.
