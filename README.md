# Representative Month Selection for Tidal Range Modelling

This project identifies **representative tidal months** for multiple UK tide-gauge locations. The approach reconstructs long-term synthetic tidal water-level signals using harmonic constituents, then slides a 29.5-day window (lunar month) over the full nodal cycle (~18.61 years) and finds the lunar onth whose statistical behaviour best matches the long-term tidal record.

Representative months are determined for:

* **Tidal Ranges** (median, interquartile range, Hm0)
* **Potential Energy** (per-cycle energy distribution and total energy)

This allows hydrodynamic or energy simulations to be performed over short windows while preserving long-period tidal variability.

---

## Required Python Packages

Ensure the following libraries are installed:

```
numpy
pandas
scipy
```

### Input description

| Input                               | Description                                                                |
| ----------------------------------- | -------------------------------------------------------------------------- |
| `dictionary_tidal_gauge_data_UK.py` | Contains gauge file paths and start times                                  |
| `functions.py`                      | Must include peak detection, tidal reconstruction, PE, Hm0, RMSE functions |
| Tide gauge files                    | Raw elevation or water-level data                                          |

---

## Main Modules

### `representative_month.py`

Functions included:

| Function                   | Purpose                                                          |
| -------------------------- | ---------------------------------------------------------------- |
| `average_metrics_ranges()` | Computes tidal range statistics for all windows                  |
| `average_metrics_energy()` | Computes energy metrics for all windows                          |
| `process_locations()`      | Automates analysis for multiple locations and constituent counts |

---

## Methodology Summary

### 1. Constituent Extraction

* Harmonic constituents are extracted from observed sea-level data
* Synthetic signals reconstructed over **18.61 years** using top **2, 4, 8, 12, and 16** constituents

### 2. Sliding Window (Representative Month)

* A window of ~30 days (**58 tidal cycles**) moves through the full record
* For each window, compute two metric families:

#### A. Range-Based Metrics

* Median range
* Interquartile range (IQR)
* Sea-state: `Hm0`
* RMSE relative to nodal baseline

#### B. Energy-Based Metrics

* Per-cycle potential energy: `0.5 * ρ * g * R²`
* Median and IQR of energy distribution
* Total potential energy for window
* RMSE relative to baseline

### 3. Ranking

* Each window receives two ranks
* Final score = average of both ranks
* Best (lowest-ranked) window = Representative Month

### 4. Output

* Representative start time in seconds and datetime
* Summary table for all locations and constituent truncations
* Optional CSV export

---

## Running the Script

```
python representative_months_refactor.py
```

This will:

* Process default 46 UK gauge locations
* Use 30-day windows
* Evaluate 2, 4, 8, 12, and 16 constituents
* Print summary of representative months

To save results:

```python
summary_df.to_csv("representative_month_summary.csv", index=False)
```

---

## Outputs

* Window-by-window statistics
* Representative month per location and constituent count
* Sorted summary by tidal amplitude

Example summary structure:

| LOCATION  | K_CONS | START DATETIME RANGES | START DATETIME ENERGY |
| --------- | ------ | --------------------- | --------------------- |
| Avonmouth | 16     | 2007-04-08 00:00:00   | 2007-04-12 00:00:00   |
| Newport   | 12     | 2011-06-03 00:00:00   | 2011-06-04 00:00:00   |

---

## Applications

* Run short hydrodynamic simulations with nodal-scale reliability
* CAPEX, LCOE, or environmental studies requiring representative forcing
* Model sensitivity testing

---

## Future Enhancements

* Automated plots for range and energy distributions
* GPU-accelerated window iteration
* Region-based representative windows
* Uncertainty or sensitivity analysis

---

This README describes the Representative Month Selection workflow for tidal range power plant modelling and resource assessment.

