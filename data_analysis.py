import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

# PROCESS DATA

# Load the TrackMate spots file
df = pd.read_csv("130nmBead_50pctGlycerol_100pctPower_561_RFP_Confocal_iXon EMCCD_2026-09-25_RFP_Confocal_iXon_TimelapseFast_14.49.23_allspots.csv")

# TrackMate adds 3 extra header rows; skip them
df = df.iloc[3:]

# Keep only the columns we need and make them numbers
df = df[["TRACK_ID", "POSITION_X", "POSITION_Y", "FRAME"]].apply(pd.to_numeric)

# Drop spots that weren't part of a track, and sort each track by time
df = df.dropna().sort_values(["TRACK_ID", "FRAME"])

# PLOT TRACKS

tracks = list(df.groupby("TRACK_ID"))
ncols = 5
nrows = math.ceil(len(tracks) / ncols)

# shared axis range: big enough for the track that wandered farthest
limit = 0
for track_id, track in tracks:
    dx = track["POSITION_X"] - track["POSITION_X"].iloc[0]
    dy = track["POSITION_Y"] - track["POSITION_Y"].iloc[0]
    limit = max(limit, dx.abs().max(), dy.abs().max())
limit = limit * 1.1   # a little padding

fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
axes = axes.flatten()

for ax, (track_id, track) in zip(axes, tracks):
    dx = track["POSITION_X"] - track["POSITION_X"].iloc[0]
    dy = track["POSITION_Y"] - track["POSITION_Y"].iloc[0]
    ax.plot(dx, dy, "-o", markersize=3)
    ax.plot(0, 0, "k+", markersize=10)          # starting point
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.set_title(f"Track {int(track_id)} ({len(track)} spots)", fontsize=9)

for ax in axes[len(tracks):]:                   # hide unused panels
    ax.axis("off")

fig.supxlabel("Δx (µm)")
fig.supylabel("Δy (µm)")
plt.tight_layout()
# plt.show()

# EXTRACT MSD 

# max_lag - how many frames apart two positions are
# dt = 1 / 16.07    # seconds per frame
def diffusion_coefficient(track, dt, max_lag=5):
    """Return D (µm²/s) for ONE track, from the slope of MSD vs time (<r²> = 4Dt)."""
    x = track["POSITION_X"].values
    y = track["POSITION_Y"].values

    # can't use lags as long as the track itself
    max_lag = min(max_lag, len(x) - 1)

    times = []
    msds = []
    for lag in range(1, max_lag + 1):
        dx = x[lag:] - x[:-lag]
        dy = y[lag:] - y[:-lag]
        times.append(lag * dt)
        msds.append(np.mean(dx**2 + dy**2))

    slope, intercept = np.polyfit(times, msds, 1)
    D = slope / 4
    return D, times, msds

# VISCOSITY CALCULATION 

def viscosity(D, radius_nm, temp_C):
    """Viscosity (Pa·s) from the Stokes-Einstein relation: eta = kT / (6 pi a D).

    D         : diffusion coefficient in µm²/s (from diffusion_coefficient)
    radius_nm : bead RADIUS in nm (half the diameter)
    temp_C    : temperature in °C
    """
    k = 1.380649e-23          # Boltzmann constant, J/K
    T = temp_C + 273.15       # °C → K
    a = radius_nm * 1e-9      # nm → m
    D_SI = D * 1e-12          # µm²/s → m²/s

    return k * T / (6 * np.pi * a * D_SI)

# Calculations below 

# Type here