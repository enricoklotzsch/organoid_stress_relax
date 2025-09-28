#!/usr/bin/env python3
"""
Organoid relaxation analysis

This script:
- Reads multiple CSV files with organoid area over time
- Normalizes the area (A/A0)
- Extracts relaxation time constant tau via exponential fit
- Computes maximal strain (max(A/A0) - 1)
- Saves results into a CSV table
- Generates multi-page PDF visualizations (trace + fit)

Author: Enrico Klotzsch
License: MIT
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages

# ----------------------------
# Parameters
# ----------------------------
FPS = 60                 # frames per second
FRAMES_PER_CYCLE = 60    # stimulation cycle length in frames

# ----------------------------
# Exponential decay function
# ----------------------------
def exp_decay(t, A, tau, C):
    """Single exponential decay."""
    return A * np.exp(-t / tau) + C

# ----------------------------
# File analysis
# ----------------------------
def analyze_file(filepath):
    """Analyze one CSV file, return tau and max strain."""
    df = pd.read_csv(filepath, delimiter="\t|,", engine="python")

    # Ensure Frame column exists
    if "Frame" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Frame"})

    if "Area" not in df.columns:
        raise ValueError(f"'Area' column not found in {filepath}")

    # Normalize
    df["NormalizedArea"] = df["Area"] / df["Area"].iloc[0]

    # Filter unrealistic drift
    df = df[(df["NormalizedArea"] >= 0.9) & (df["NormalizedArea"] <= 1.1)]
    if df.empty:
        return None

    # Max strain
    max_strain = df["NormalizedArea"].max() - 1.0

    # Collect right flanks
    right_flanks = []
    n_cycles = int(df["Frame"].max() / FRAMES_PER_CYCLE)
    for i in range(5, n_cycles - 5):
        start = i * FRAMES_PER_CYCLE
        segment = df[(df["Frame"] >= start) & (df["Frame"] < start + FRAMES_PER_CYCLE)]
        if segment.empty:
            continue
        t = (segment["Frame"] - start).values / FPS
        y = segment["NormalizedArea"].values
        peak_idx = np.argmax(y)
        tR, yR = t[peak_idx+2:], y[peak_idx+2:]
        if len(yR) >= 10:
            tR = tR - tR[0]
            right_flanks.append((tR, yR))

    # Interpolate onto common grid
    tR_common = np.arange(0, 0.5, 1 / FPS)
    R_mat = []
    for tR, yR in right_flanks:
        if tR.max() >= tR_common.max():
            R_mat.append(np.interp(tR_common, tR, yR))
    R_mat = np.array(R_mat)

    tau, fit_curve = np.nan, None
    if R_mat.size:
        R_mean = np.nanmean(R_mat, axis=0)
        try:
            popt, _ = curve_fit(exp_decay, tR_common, R_mean,
                                p0=[R_mean[0] - R_mean[-1], 0.2, R_mean[-1]], maxfev=5000)
            tau = popt[1]
            fit_curve = exp_decay(tR_common, *popt)
        except RuntimeError:
            pass

    return {
        "File": os.path.basename(filepath),
        "Tau_relaxation_s": tau,
        "Max_strain": max_strain
    }, df, tR_common, R_mat, fit_curve

# ----------------------------
# Visualization
# ----------------------------
def plot_analysis(filepath, df, tR_common, R_mat, fit_curve, tau):
    """Return matplotlib figure with 2 subplots: trace and relaxation fit."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: full trace
    axes[0].plot(df["Frame"] / FPS, df["NormalizedArea"], color="black")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("A/A0")
    axes[0].set_title(os.path.basename(filepath))

    # Right: relaxation
    if R_mat is not None and R_mat.size:
        for row in R_mat[:50]:
            axes[1].plot(tR_common, row, color="gray", alpha=0.25)
        R_mean = np.nanmean(R_mat, axis=0)
        axes[1].plot(tR_common, R_mean, color="blue", linewidth=2, label="Average right flank")
        if fit_curve is not None and not np.isnan(tau):
            axes[1].plot(tR_common, fit_curve, "r--", linewidth=2,
                         label=f"Fit τ = {tau:.3f} s")
        axes[1].legend()
    axes[1].set_xlabel("Time after peak (s)")
    axes[1].set_ylabel("A/A0")
    axes[1].set_title("Relaxation fit")
    axes[1].grid(True)

    plt.tight_layout()
    return fig

# ----------------------------
# Batch analysis
# ----------------------------
def analyze_folder(folder, output_pdf="analysis.pdf", output_csv="results.csv"):
    """Analyze all CSV files in a folder and save results."""
    files = glob.glob(os.path.join(folder, "*.csv"))
    results = []

    with PdfPages(output_pdf) as pdf:
        for f in files:
            try:
                res, df, tR_common, R_mat, fit_curve = analyze_file(f)
                if res is None:
                    continue
                results.append(res)
                fig = plot_analysis(f, df, tR_common, R_mat, fit_curve, res["Tau_relaxation_s"])
                pdf.savefig(fig)
                plt.close(fig)
            except Exception as e:
                print(f"Error with {f}: {e}")

    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Analysis complete. Saved {len(results)} results to {output_csv} and plots to {output_pdf}")

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Organoid relaxation analysis")
    parser.add_argument("folder", help="Folder containing .csv files")
    parser.add_argument("--pdf", default="analysis.pdf", help="Output PDF file")
    parser.add_argument("--csv", default="results.csv", help="Output CSV file")
    args = parser.parse_args()

    analyze_folder(args.folder, output_pdf=args.pdf, output_csv=args.csv)
