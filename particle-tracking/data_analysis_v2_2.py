from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Settings
FRAME_TIME = 1 / 16.07        # seconds per frame (16.07 Hz)
BEAD_RADIUS = 65e-9           # m (130 nm diameter beads)
TEMPERATURE = 22 + 273.15     # K (room temperature)
K_B = 1.380649e-23            # Boltzmann constant (J/K)

# Lag definition: How many frames apart two positions of the same bead are
MAX_LAG = 10      # longest lag (in frames) to calculate
FIT_LAGS = 4      # fit D using only the first few lags
MIN_COUNT = 10    # skip lags with fewer displacements than this

folder = Path(__file__).parent   # folder this script is in (file paths below are relative to it)

# Data files for each condition
files = {
    "Water": [
        "water/130nmBead_water_100pctPower_561_RFP_Confocal_iXon EMCCD_2026-09-25_RFP_Confocal_iXon_TimelapseFast_14.40.11_allspots.csv",
        "water/130nmBead_water_561_RFP_Confocal_iXon EMCCD_2026-09-25_RFP_Confocal_iXon_TimelapseFast_14.35.38_allspots.csv",
        "water/130nmBeads_water_561_RFP_Confocal_iXon EMCCD_2026-09-25_RFP_Confocal_iXon_TimelapseFast_14.32.14_allspots.csv",
    ],
    "50% Glycerol": [
        "50glycerol/130nmBead_50pctGlycerol_100pctPower_561_RFP_Confocal_iXon EMCCD_2026-09-25_RFP_Confocal_iXon_TimelapseFast_14.49.23_allspots.csv",
        "50glycerol/130nmBead_50pctGlycerol_100pctPower_561_RFP_Confocal_iXon EMCCD_2026-09-25_RFP_Confocal_iXon_TimelapseFast_14.54.24_allspots.csv",
        "50glycerol/130nmBead_50pctGlycerol_100pctPower_RFP_Confocal_iXon EMCCD_2026-09-25_RFP_Confocal_iXon_TimelapseFast_14.53.43_allspots.csv",
    ],
}


# Read a TrackMate spots file and split it into a list of tracks
def load_tracks(filename):
    data = pd.read_csv(folder / filename, skiprows=[1, 2, 3])   # skip TrackMate's 3 extra header rows
    data = data.dropna(subset=["TRACK_ID"])                      # drop spots that aren't part of a track
    return [track for _, track in data.groupby("TRACK_ID")]      # one table of spots per track


# For every lag, collect the squared displacement of every pair of spots that many frames apart
def squared_displacements(tracks):
    result = {lag: [] for lag in range(1, MAX_LAG + 1)}   # empty list for each lag: {1: [], 2: [], ...}

    for track in tracks:
        # look up a bead's position by frame number (this way gaps in a track are skipped properly)
        positions = {}
        for frame, x, y in zip(track["FRAME"], track["POSITION_X"], track["POSITION_Y"]):
            positions[int(frame)] = (x, y)               # ex) positions[12] = (x, y) at frame 12

        for frame, (x, y) in positions.items():          # go through each spot in the track
            for lag in range(1, MAX_LAG + 1):            # and each lag
                if frame + lag in positions:             # only if the bead was found `lag` frames later
                    x2, y2 = positions[frame + lag]      # position `lag` frames later
                    result[lag].append((x2 - x) ** 2 + (y2 - y) ** 2)   # squared distance moved: dx² + dy²

    return result


fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(11, 4.5))   # two side-by-side plots: linear and log-log

for name, file_list in files.items():       # run the analysis once per condition
    tracks = []
    for filename in file_list:
        tracks += load_tracks(filename)     # combine tracks from all files of this condition

    # Average the squared displacements at each lag to get the MSD
    lags, msd, err = [], [], []
    for lag, values in squared_displacements(tracks).items():
        if len(values) >= MIN_COUNT:                                     # skip lags with too little data
            lags.append(lag)
            msd.append(np.mean(values))                                  # mean squared displacement
            err.append(np.std(values, ddof=1) / np.sqrt(len(values)))   # standard error of the mean

    tau = np.array(lags) * FRAME_TIME   # lag in seconds
    msd = np.array(msd)                 # convert lists to arrays so we can do math on them
    err = np.array(err)

    # Fit a line MSD = 4 D tau + offset to the first few lags (points with smaller error count more)
    # Weight by 1/error
    # Offset is the y-intercept
    (slope, offset), cov = np.polyfit(tau[:FIT_LAGS], msd[:FIT_LAGS], 1, w=1 / err[:FIT_LAGS], cov="unscaled")
    D = slope / 4                       # µm²/s (in 2D, slope = 4D)
    D_err = np.sqrt(cov[0, 0]) / 4      # uncertainty in D from the fit

    # Stokes-Einstein: viscosity = kT / (6 pi r D)
    viscosity = K_B * TEMPERATURE / (6 * np.pi * BEAD_RADIUS * D * 1e-12)   # 1e-12 converts µm² to m²
    viscosity_err = viscosity * D_err / D                                 # same % error as D

    # Slope on a log-log plot: 1 means normal diffusion
    alpha = np.polyfit(np.log(tau), np.log(msd), 1)[0]   # [0] keeps just the slope

    # Print results for this condition
    print(f"\n{name}  ({len(tracks)} tracks)")
    print(f"  D         = {D:.3f} ± {D_err:.3f} µm²/s")
    print(f"  viscosity = {viscosity * 1e3:.2f} ± {viscosity_err * 1e3:.2f} mPa·s")   # * 1e3 converts Pa·s to mPa·s
    print(f"  alpha     = {alpha:.2f}")

    # Plot the MSD with its fit line, and the same data on log-log axes
    points = ax_lin.errorbar(tau, msd, yerr=err, fmt="o", capsize=3, label=f"{name}: D = {D:.3f} µm²/s, η = {viscosity * 1e3:.2f} mPa·s")   # MSD points with error bars
    color = points[0].get_color()                                   # reuse the same color for this condition's fit line
    fit_t = np.linspace(0, tau[FIT_LAGS - 1], 50)                   # x values for the fit line (only over the fitted lags)
    ax_lin.plot(fit_t, slope * fit_t + offset, "-", color=color)    # draw the fit line
    ax_log.loglog(tau, msd, "o", color=color, label=f"{name}: α = {alpha:.2f}")   # same MSD points on log-log axes

# Axis labels and titles
ax_lin.set_xlabel("Lag time τ (s)")
ax_lin.set_ylabel("MSD (µm²)")
ax_lin.set_title("MSD vs lag time")
ax_lin.legend()

ax_log.set_xlabel("Lag time τ (s)")
ax_log.set_ylabel("MSD (µm²)")
ax_log.set_title("Log-log (slope 1 = normal diffusion)")
ax_log.legend()

plt.tight_layout()                                    # keep labels from overlapping
plt.savefig(folder / "msd_v2.png", dpi=200)           # save the figure as an image
plt.show()                                            # open the figure in a window
