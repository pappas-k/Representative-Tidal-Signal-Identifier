# Representative Month Identifier for UK Tide Gauges

A quantitative pipeline to identify a single “representative month” window that best matches the statistical and energetic characteristics of the full 18.61 year lunar nodal cycle at a tide gauge location. The selected window is intended for computationally efficient modelling studies where simulating the full nodal cycle is impractical.

---

## Scientific objective

Tidal statistics at a site vary over the 18.61 year lunar nodal cycle. Many coastal engineering and energy studies only simulate weeks to months of boundary forcing, which can bias estimates of:

- typical tidal range (median)
- variability (interquartile range)
- sea level oscillation energy proxies

This project reconstructs a long tidal elevation signal from gauge derived harmonic constituents, then performs an exhaustive sliding window search to find the window whose distributional properties best represent the nodal cycle targets.

---

## Method summary

### 1) Constituent extraction from tide gauge records

For each gauge CSV:

1. Load tidal elevation time-series.
2. Filter out flagged or implausible values.
3. Perform harmonic analysis on a predefined set of tidal constituents using `uptide`.

Output: a dataframe of constituents sorted by amplitude:

- `Constituents`
- `Amplitude`
- `Phase`

### 2) Nodal cycle reconstruction

Using the top $$k$$ constituents (e.g. 2, 4, 8, 12, 16), reconstruct a synthetic elevation series over a full nodal cycle.

Core parameters used by the main script:

- Nodal cycle length: **18.61 years**
- Reference tidal period (M2): **T = 12.42 h**
- Reconstruction time step: **dt = 108 s**
- Samples per tidal cycle:

$$
N_c = \frac{T \cdot 3600}{dt} = \frac{12.42 \cdot 3600}{108} = 414
$$

- Tidal cycles per nodal cycle:

$$
N_{nodal} = \left\lfloor \frac{18.61 \cdot 365.25 \cdot 24}{12.42} \right\rfloor = 13134
$$

The reconstructed signal is stored as an `Nx2` array:

- column 0: time in seconds from a reference origin (`2002-01-01 00:00:00`)
- column 1: reconstructed elevation (m)

### 3) Sliding window definition

A representative month is defined as a fixed number of tidal cycles:

- Default window: **58 cycles**
- Duration: $$58 \times 12.42 = 720.36~\mathrm{hours} = 30.015~\mathrm{days}$$

Windows are evaluated at cycle aligned offsets (step = 1 tidal cycle).

Total number of candidate windows (default):

$$
N_{win} = N_{nodal} - 58 + 1 = 13077
$$

### 4) Metrics and ranking

Two parallel selection methods are implemented.

#### A) Tidal range representativeness

Compute high waters and low waters using peak detection, then form tidal ranges.

**Nodal targets:**

- $$P50_{nodal}$$: median tidal range  
- $$IQR_{nodal}$$: $$(P75 - P25)$$ of tidal range  
- $$Hm0_{nodal}$$: significant sea level oscillation height proxy  

$$
Hm0 = 4\ \times \sigma(\eta)
$$

**For each window:**

- $$P50$$, $$IQR$$, $$Hm0$$ for the window  

**Metric 1 (distribution match):**

$$
M1 = 0.5 \times |(P50_{window} - P50_{nodal}| + 0.5 \times |IQR_{window} - IQR_{nodal}|
$$

**Metric 2 (variability proxy match):**

$$
M2 = |Hm0_{wimdow} - Hm0_{nodal}|
$$

**Scalar RMSE (for scalar arguments):**

$$
RMSE(a,b)=\sqrt{(a-b)^2}=|a-b|
$$

Each window is ranked by $$M1$$ and $$M2$$ (lower is better). The selected representative window minimises the mean rank:

$$
R_{avg} = \frac{rank(M1) + rank(M2)}{2}
$$

#### B) Energy based representativeness

Using the same tidal ranges, define an “available head energy” proxy per tidal cycle:

$$
E_{max} = \frac{1}{2}\rho g (\Delta h)^2
$$

with:

- $$\rho = 1021~\mathrm{kg/m^3}$$
- $$g = 9.81~\mathrm{m/s^2}$$

In `representative_month.py`, this is converted to **MWh** via $$3.6\times10^6~\mathrm{J/MWh}$$.

Nodal targets:

- $$P50_{E,nodal}$$: median of $$E_{max}$$
- $$IQR_{E,nodal}$$: IQR of $$E_{max}$$
- $$PE_{nodal}$$: power density style proxy computed from the elevation series:

$$
PE = \frac{\rho g}{\Delta t} \int \eta(t)^2\,dt
$$

(as implemented in `functions.PE`, returned in $$\mathrm{Wh/m^2}$$ units)

For each window:

- $$P50_E$$, $$IQR_E$$
- $$PE_{window}$$
- Metric 1:

$$
M1 = 0.5 \times |P50_E_{window} - P50_{E,nodal}| + 0.5 \times |IQR_E_{window} - IQR_{E,nodal}|
$$

- Metric 2:

$$
M2 = |PE_{window} - PE_{nodal}|
$$

Ranking and selection uses the same average rank criterion as above.

---

## Inputs

### Tide gauge file registry

`inputs/dictionary_tidal_gauge_data_UK.py` defines a dictionary:

- key: location name
- value: `(csv_path, start_date)`

This allows batch processing of multiple UK gauges with consistent metadata.

### Tide gauge CSV format assumption

The extraction routine expects:

- elevation in column index 11 (0 based in `np.loadtxt` usage)
- QC flag in column index 12
- 2 header rows
- 15 min sampling implied by:
  \[
  t = 0, 900, 1800, \dots
  \]

If your CSV differs, edit `extract_constituents_from_tidegauge_file` accordingly.

---

## Outputs

Running the main script produces a summary table with one row per (location, k) pair:

- `LOCATION`
- `MAX AMPLITUDE` (proxy: sum of top 12 constituent amplitudes)
- `K_CONS` (number of constituents used in reconstruction)
- `START TIME RANGES` (seconds from origin)
- `START DATETIME RANGES` (absolute datetime)
- `START TIME ENERGY`
- `START DATETIME ENERGY`

Optionally, save to CSV (commented in the script).

---

## Repository structure (expected)

The main script imports using package style paths:

```text
project_root/
  representative_month.py
  modules/
    functions.py
  inputs/
    dictionary_tidal_gauge_data_UK.py
    <tide gauge csv files...>
