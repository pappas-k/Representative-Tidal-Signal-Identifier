"""
Representative Month Identifier based on the custom metric approach
"""

import datetime as dt
import numpy as np
import pandas as pd
import itertools

from modules import functions
from inputs.dictionary_tidal_gauge_data_UK import dictionary_gauge_data

# ===========================
# Configuration
# ===========================
NODAL_CYCLE = 18.61 # years of nodal cycle
T = 12.42 # M2 tidal period in hrs
DT = 108  # sampling interval (s) used in reconstructions
POINTS_PER_CYCLE = int(T * 3600 / DT)  # 414
CYCLES_PER_NODAL = int(NODAL_CYCLE * 365.25 * 24 / T)  # 13134
DEFAULT_WINDOW_CYCLES = 58 #tidal cycles
DEFAULT_START_DATE = dt.datetime(2002, 1, 1, 0, 0, 0)
RHO = 1021.0 # water density
G   = 9.81 # gravitational acceleration
MWH_CONVERSION = 3.6e6  # J -> MWh

def seconds_to_datetime(seconds, origin=DEFAULT_START_DATE):
    """Convert seconds-from-origin to absolute datetime."""
    return origin + dt.timedelta(seconds=float(seconds))

def cycles_to_seconds(cycles) :
    """Convert a number of tidal cycles to seconds using sampling grid."""
    return cycles * POINTS_PER_CYCLE * DT

def compute_nodal_quantities_ranges(signal):
    """Compute tidal range-related nodal target representative quantities.

    Returns
    -------
    IQR_nodal, P50_nodal, Hm0_nodal
    """
    rel_time, tide_elevs = signal[:, 0], signal[:, 1]
    hw_t, hw_e = functions.find_tidal_peaks(rel_time, tide_elevs, peak_type="HW")
    lw_t, lw_e = functions.find_tidal_peaks(rel_time, tide_elevs, peak_type="LW")
    tidal_ranges_nodal, _ = functions.tidal_ranges_from_peaks(hw_t, lw_t, hw_e, lw_e)

    IQR_nodal = np.percentile(tidal_ranges_nodal, 75) - np.percentile(tidal_ranges_nodal, 25)
    P50_nodal = np.percentile(tidal_ranges_nodal, 50)
    Hm0_nodal = functions.Hm0(signal)
    return IQR_nodal, P50_nodal, Hm0_nodal


def compute_nodal_quantities_energy(signal):
    """Compute energy-related nodal target representative quantities.

    Returns
    -------
    IQR_E_nodal, P50_E_nodal, PE_nodal
    """
    rel_time, tide_elevs = signal[:, 0], signal[:, 1]
    hw_t, hw_e = functions.find_tidal_peaks(rel_time, tide_elevs, peak_type="HW")
    lw_t, lw_e = functions.find_tidal_peaks(rel_time, tide_elevs, peak_type="LW")
    tidal_ranges_nodal, _ = functions.tidal_ranges_from_peaks(hw_t, lw_t, hw_e, lw_e)

    emax_nodal = 0.5 * RHO * G * np.square(tidal_ranges_nodal) / MWH_CONVERSION
    IQR_E_nodal = np.percentile(emax_nodal, 75) - np.percentile(emax_nodal, 25)
    P50_E_nodal = np.percentile(emax_nodal, 50)
    PE_nodal = functions.PE(signal)
    return IQR_E_nodal, P50_E_nodal, PE_nodal

def evaluate_window_ranges(signal, start_idx, window_cycles,
                           IQR_nodal, P50_nodal, Hm0_nodal):
    """Evaluate tidal range-based metrics for one window starting at `start_idx`.

    Returns a dict with keys:
    - TIME STEP, START TIME, METRIC_1, METRIC_2, IQR, P50, IQR %, P50 %, Hm0, HM0 %
    """
    span = POINTS_PER_CYCLE * window_cycles
    rel_time = signal[:, 0][start_idx:start_idx + span]
    tide_elevs = signal[:, 1][start_idx:start_idx + span]

    hw_t, hw_e = functions.find_tidal_peaks(rel_time, tide_elevs, peak_type="HW")
    lw_t, lw_e = functions.find_tidal_peaks(rel_time, tide_elevs, peak_type="LW")
    tidal_ranges, _ = functions.tidal_ranges_from_peaks(hw_t, lw_t, hw_e, lw_e)

    P50 = float(np.percentile(tidal_ranges, 50))
    IQR = float(np.percentile(tidal_ranges, 75) - np.percentile(tidal_ranges, 25))

    metric1 = float(0.5 * functions.rmse(P50, P50_nodal) + 0.5 * functions.rmse(IQR, IQR_nodal))

    Hm0_window = float(functions.Hm0(signal[start_idx:start_idx + span]))
    metric2 = float(functions.rmse(Hm0_window, Hm0_nodal))

    IQR_pct = float((IQR - IQR_nodal) * 100.0 / IQR_nodal) if IQR_nodal != 0 else np.nan
    P50_pct = float((P50 - P50_nodal) * 100.0 / P50_nodal) if P50_nodal != 0 else np.nan
    Hm0_pct = float((Hm0_window - Hm0_nodal) * 100.0 / Hm0_nodal) if Hm0_nodal != 0 else np.nan

    return {
        "TIME STEP": int(start_idx),
        "START TIME": float(rel_time[0]),
        "METRIC_1": metric1,
        "METRIC_2": metric2,
        "IQR": IQR,
        "P50": P50,
        "IQR %": IQR_pct,
        "P50 %": P50_pct,
        "Hm0": Hm0_window,
        "HM0 %": Hm0_pct,
    }


def evaluate_window_energy(signal, start_idx, window_cycles,
                           IQR_E_nodal, P50_E_nodal, PE_nodal ):
    """Evaluate ENERGY-based metrics for one window starting at `start_idx`.

    Returns a dict with keys:
    - TIME STEP, START TIME, METRIC_1, METRIC_2, IQR(E), P50(E), IQR(E) %, P50(E) %, Emax_sum, PE, PE %
    """
    span = POINTS_PER_CYCLE * window_cycles
    rel_time = signal[:, 0][start_idx:start_idx + span]
    tide_elevs = signal[:, 1][start_idx:start_idx + span]

    hw_t, hw_e = functions.find_tidal_peaks(rel_time, tide_elevs, peak_type="HW")
    lw_t, lw_e = functions.find_tidal_peaks(rel_time, tide_elevs, peak_type="LW")
    tidal_ranges, _ = functions.tidal_ranges_from_peaks(hw_t, lw_t, hw_e, lw_e)

    emax = 0.5 * RHO * G * np.square(tidal_ranges) / MWH_CONVERSION
    P50_E = float(np.percentile(emax, 50))
    IQR_E = float(np.percentile(emax, 75) - np.percentile(emax, 25))

    metric1 = float(0.5 * functions.rmse(P50_E, P50_E_nodal) + 0.5 * functions.rmse(IQR_E, IQR_E_nodal))

    PE_window = float(functions.PE(signal[start_idx:start_idx + span]))
    metric2 = float(functions.rmse(PE_window, PE_nodal))

    IQR_E_pct = float((IQR_E - IQR_E_nodal) * 100.0 / IQR_E_nodal) if IQR_E_nodal != 0 else np.nan
    P50_E_pct = float((P50_E - P50_E_nodal) * 100.0 / P50_E_nodal) if P50_E_nodal != 0 else np.nan
    PE_pct = float((PE_window - PE_nodal) * 100.0 / PE_nodal) if PE_nodal != 0 else np.nan

    return {
        "TIME STEP": int(start_idx),
        "START TIME": float(rel_time[0]),
        "METRIC_1": metric1,
        "METRIC_2": metric2,
        "IQR(E)": IQR_E,
        "P50(E)": P50_E,
        "IQR(E) %": IQR_E_pct,
        "P50(E) %": P50_E_pct,
        "Emax_sum": float(np.sum(emax)),
        "PE": PE_window,
        "PE %": PE_pct,
    }


def average_metrics_ranges(signal, window_cycles= DEFAULT_WINDOW_CYCLES):
    """Compute tidal range-based metrics across all sliding windows and return a tidy DataFrame.

    Columns include TIME STEP (start index), START TIME (seconds), START DATETIME,
    METRIC_1, METRIC_2, IQR, IQR %, P50, P50 %, Hm0, Hm0 %, plus ranks and average rank.
    """
    IQR_nodal, P50_nodal, Hm0_nodal = compute_nodal_quantities_ranges(signal)

    rows= []
    for j in range(0, CYCLES_PER_NODAL - window_cycles + 1):
        i = j * POINTS_PER_CYCLE
        rows.append(evaluate_window_ranges(signal, i, window_cycles, IQR_nodal, P50_nodal, Hm0_nodal))

    df = pd.DataFrame(rows)

    df["START DATETIME"] = df["START TIME"].apply(lambda s: seconds_to_datetime(s))

    # Ranking (ascending is better)
    df["RANK M1"] = df["METRIC_1"].rank(ascending=True)
    df["RANK M2"] = df["METRIC_2"].rank(ascending=True)
    df["AVERAGE RANK"] = df[["RANK M1", "RANK M2"]].mean(axis=1)

    # Add baselines for reference
    df.attrs["nodal_IQR"] = IQR_nodal
    df.attrs["nodal_P50"] = P50_nodal
    df.attrs["nodal_Hm0"] = Hm0_nodal

    return df


def average_metrics_energy(signal, window_cycles = DEFAULT_WINDOW_CYCLES) :
    """Compute ENERGY-based metrics across windows and return DataFrame.

    Columns include TIME STEP, START TIME, START DATETIME, METRIC_1, METRIC_2,
    IQR(E), IQR(E) %, P50(E), P50(E) %, Emax_sum, PE, PE %, ranks and average rank.
    """
    IQR_E_nodal, P50_E_nodal, PE_nodal = compute_nodal_quantities_energy(signal)

    rows = []
    for j in range(0, CYCLES_PER_NODAL - window_cycles + 1):
        i = j * POINTS_PER_CYCLE
        rows.append(evaluate_window_energy(signal, i, window_cycles, IQR_E_nodal, P50_E_nodal, PE_nodal))

    df = pd.DataFrame(rows)

    df["START DATETIME"] = df["START TIME"].apply(lambda s: seconds_to_datetime(s))

    # Ranking (ascending is better)
    df["RANK M1"] = df["METRIC_1"].rank(ascending=True)
    df["RANK M2"] = df["METRIC_2"].rank(ascending=True)
    df["AVERAGE RANK"] = df[["RANK M1", "RANK M2"]].mean(axis=1)

    # Store baselines in attrs
    df.attrs["nodal_IQR_E"] = IQR_E_nodal
    df.attrs["nodal_P50_E"] = P50_E_nodal
    df.attrs["nodal_PE"] = PE_nodal

    return df

def process_gauges(n_locations = 46, window_cycles = DEFAULT_WINDOW_CYCLES,
                   constituent_counts = (2, 4, 8, 12, 16)):
    """Process the first `n_locations` in `dictionary_gauge_data` and return a summary DataFrame.

    The output table includes, for each location and each constituent count, the
    start time of the representative month for both
    tidal ranges and energy criteria.
    """
    locations = list(dict(itertools.islice(dictionary_gauge_data.items(), n_locations)).keys())

    rows = []
    for location in locations:
        print(f"----- {location} -----")
        tg_file, start_date = dictionary_gauge_data.get(location)

        pd1, t, eta = functions.extract_constituents_from_tidegauge_file(
            tidegauge_file=tg_file, start_date=start_date)

        # Max amplitude proxy: sum of top 12 amplitudes
        max_amp = float(pd1["Amplitude"][0:12].sum())

        for k in constituent_counts:
            print(f"  > Reconstructing signal with top {k} constituents")
            signal = functions.signal_reconstruction(
                pd1,
                constituents=(pd1["Constituents"][:k]),
                signal_duration=NODAL_CYCLE * 365.25 * 24 * 3600,
                start_date=DEFAULT_START_DATE)

            # RANGES
            df_ranges = average_metrics_ranges(signal, window_cycles=window_cycles)
            best_row_ranges = df_ranges.loc[df_ranges["AVERAGE RANK"].idxmin()]

            # ENERGY
            df_energy = average_metrics_energy(signal, window_cycles=window_cycles)
            best_row_energy = df_energy.loc[df_energy["AVERAGE RANK"].idxmin()]

            rows.append({
                "LOCATION": location,
                "MAX AMPLITUDE": max_amp,
                "K_CONS": k,
                "START TIME RANGES": float(best_row_ranges["START TIME"]),
                "START DATETIME RANGES": best_row_ranges["START DATETIME"],
                "START TIME ENERGY": float(best_row_energy["START TIME"]),
                "START DATETIME ENERGY": best_row_energy["START DATETIME"],
            })

    summary = pd.DataFrame(rows)
    # Sort by amplitude desc, then by K
    summary.sort_values(by=["MAX AMPLITUDE", "K_CONS"], ascending=[False, True], inplace=True, ignore_index=True)
    return summary


if __name__ == "__main__":
    N = 46                     # BODC gauges
    M = DEFAULT_WINDOW_CYCLES  # tidal cycles

    summary_df = process_gauges(n_locations=N, window_cycles=M, constituent_counts=(2, 4, 8, 12, 16))
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 160)
    print(summary_df.to_string(index=False))

    # Optionally save:
    # summary_df.to_csv(f"start_time_of_rep_months_AVERAGE_RANK_METHOD_refactored_{M}cycles.csv", index=False)

