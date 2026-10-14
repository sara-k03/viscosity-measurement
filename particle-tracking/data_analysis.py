import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import math

# PROCESS DATA

# Load the TrackMate spots file

# 50% Glycerol Files
# df = pd.read_csv("50glycerol/130nmBead_50pctGlycerol_100pctPower_561_RFP_Confocal_iXon EMCCD_2026-09-25_RFP_Confocal_iXon_TimelapseFast_14.49.23_allspots.csv")
# df = pd.read_csv("50glycerol/130nmBead_50pctGlycerol_100pctPower_561_RFP_Confocal_iXon EMCCD_2026-09-25_RFP_Confocal_iXon_TimelapseFast_14.54.24_allspots.csv")
# df = pd.read_csv("50glycerol/130nmBead_50pctGlycerol_100pctPower_RFP_Confocal_iXon EMCCD_2026-09-25_RFP_Confocal_iXon_TimelapseFast_14.53.43_allspots.csv")

# Water files
# df = pd.read_csv("water/130nmBead_water_100pctPower_561_RFP_Confocal_iXon EMCCD_2026-09-25_RFP_Confocal_iXon_TimelapseFast_14.40.11_allspots.csv")
# df = pd.read_csv("water/130nmBead_water_561_RFP_Confocal_iXon EMCCD_2026-09-25_RFP_Confocal_iXon_TimelapseFast_14.35.38_allspots.csv")
df = pd.read_csv("water/130nmBeads_water_561_RFP_Confocal_iXon EMCCD_2026-09-25_RFP_Confocal_iXon_TimelapseFast_14.32.14_allspots.csv")

# TrackMate adds 3 extra header rows; skip them
df = df.iloc[3:]

# Keep only the columns we need and make them numbers
df = df[["TRACK_ID", "POSITION_X", "POSITION_Y", "FRAME"]].apply(pd.to_numeric)

# Drop spots that weren't part of a track, and sort each track by time
df = df.dropna().sort_values(["TRACK_ID", "FRAME"])

# PLOT TRACKS

tracks = list(df.groupby("TRACK_ID"))
ncols, nrows = 6, 2
per_page = ncols * nrows
n_pages = math.ceil(len(tracks) / per_page)

# shared axis range across ALL tracks
limit = 0
for track_id, track in tracks:
    dx = track["POSITION_X"] - track["POSITION_X"].iloc[0]
    dy = track["POSITION_Y"] - track["POSITION_Y"].iloc[0]
    limit = max(limit, dx.abs().max(), dy.abs().max())
limit *= 1.1

fig, axes = plt.subplots(nrows, ncols, figsize=(13, 8.5))
axes = axes.flatten()
state = {"page": 0}

def draw_page():
    start = state["page"] * per_page
    page = tracks[start:start + per_page]
    for ax in axes:
        ax.clear()
        ax.axis("off")
    for ax, (track_id, track) in zip(axes, page):
        ax.axis("on")
        dx = track["POSITION_X"] - track["POSITION_X"].iloc[0]
        dy = track["POSITION_Y"] - track["POSITION_Y"].iloc[0]
        ax.plot(dx, dy, "-o", markersize=3)
        ax.plot(0, 0, "k+", markersize=10)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal")
        ax.set_title(f"Track {int(track_id)} ({len(track)} spots)", fontsize=9)
    fig.suptitle(f"Page {state['page'] + 1} of {n_pages}  (← → or scroll to change page)")
    fig.canvas.draw_idle()

def change_page(step):
    state["page"] = (state["page"] + step) % n_pages
    draw_page()

def on_key(event):
    if event.key in ("right", "down", "pagedown"):
        change_page(1)
    elif event.key in ("left", "up", "pageup"):
        change_page(-1)

def on_scroll(event):
    change_page(1 if event.button == "down" else -1)

fig.canvas.mpl_connect("key_press_event", on_key)
fig.canvas.mpl_connect("scroll_event", on_scroll)

fig.supxlabel("Δx (µm)")
fig.supylabel("Δy (µm)")
draw_page()
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# EXTRACT MSD 

# max_lag - how many frames apart two positions are
# dt = 1 / 16.07    # seconds per frame
def diffusion_coefficient(track, dt = 1 / 16.07, max_lag=5):
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
# Currently just using room temperature as the default, might have to change this
# Diameter = 0.130 microns which is 130 nanometers, which is 65 nm radius
def viscosity(D, radius_nm=65, temp_C=22):
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

"""50% Glycerol, Sample 1"""

# track_4 = df[df["TRACK_ID"] == 4]
# track_5 = df[df["TRACK_ID"] == 5]

# D_4, _, _ = diffusion_coefficient(track=track_4)
# D_5, _, _ = diffusion_coefficient(track=track_5)

# print("50% Glycerol, Sample 1:")
# print("D for Track 4:", D_4)
# print("D for Track 5:", D_5)
# print("Viscosity for Track 4:", viscosity(D=D_4))
# print("Viscosity for Track 5:", viscosity(D=D_5))

# print("")

"""50% Glycerol, Sample 2"""

# track_0 = df[df["TRACK_ID"] == 0]
# track_1 = df[df["TRACK_ID"] == 1]
# track_2 = df[df["TRACK_ID"] == 2]
# track_3 = df[df["TRACK_ID"] == 3]
# track_4 = df[df["TRACK_ID"] == 4]
# track_5 = df[df["TRACK_ID"] == 5]

# D_0, _, _ = diffusion_coefficient(track=track_0)
# D_1, _, _ = diffusion_coefficient(track=track_1)
# D_2, _, _ = diffusion_coefficient(track=track_2)
# D_3, _, _ = diffusion_coefficient(track=track_3)
# D_4, _, _ = diffusion_coefficient(track=track_2)
# D_5, _, _ = diffusion_coefficient(track=track_3)

# print("50% Glycerol, Sample 2:")
# print("D for Track 0:", D_0)
# print("D for Track 1:", D_1)
# print("D for Track 2:", D_2)
# print("D for Track 3:", D_3)
# print("D for Track 4:", D_4)
# print("D for Track 5:", D_5)
# print("")
# print("Viscosity for Track 0:", viscosity(D=D_0))
# print("Viscosity for Track 1:", viscosity(D=D_1))
# print("Viscosity for Track 2:", viscosity(D=D_2))
# print("Viscosity for Track 3:", viscosity(D=D_3))
# print("Viscosity for Track 4:", viscosity(D=D_4))
# print("Viscosity for Track 5:", viscosity(D=D_5))

"""50% Glycerol, Sample 3"""

# track_4 = df[df["TRACK_ID"] == 4]

# D_4, _, _ = diffusion_coefficient(track=track_4)

# print("50% Glycerol, Sample 3:")
# print("D for Track 4:", D_4)
# print("Viscosity for Track 4:", viscosity(D=D_4))

"""Water, Sample 1"""

# track_10 = df[df["TRACK_ID"] == 10]
# track_26 = df[df["TRACK_ID"] == 26]
# track_72 = df[df["TRACK_ID"] == 72]

# D_10, _, _ = diffusion_coefficient(track=track_10)
# D_26, _, _ = diffusion_coefficient(track=track_26)
# D_72, _, _ = diffusion_coefficient(track=track_72)

# print("Water, Sample 1:")
# print("D for Track 10:", D_10)
# print("D for Track 26:", D_26)
# print("D for Track 72:", D_72)
# print("")
# print("Viscosity for Track 10:", viscosity(D=D_10))
# print("Viscosity for Track 26:", viscosity(D=D_26))
# print("Viscosity for Track 72:", viscosity(D=D_72))

"""Water, Sample 2"""

# track_16 = df[df["TRACK_ID"] == 16]
# track_25 = df[df["TRACK_ID"] == 25]
# track_34 = df[df["TRACK_ID"] == 34]
# track_97 = df[df["TRACK_ID"] == 97]
# track_123 = df[df["TRACK_ID"] == 123]

# D_16, _, _ = diffusion_coefficient(track=track_16)
# D_25, _, _ = diffusion_coefficient(track=track_25)
# D_34, _, _ = diffusion_coefficient(track=track_34)
# D_97, _, _ = diffusion_coefficient(track=track_97)
# D_123, _, _ = diffusion_coefficient(track=track_123)

# print("Water, Sample 2:")
# print("D for Track 16:", D_16)
# print("D for Track 25:", D_25)
# print("D for Track 34:", D_34)
# print("D for Track 97:", D_97)
# print("D for Track 123:", D_123)
# print("")
# print("Viscosity for Track 16:", viscosity(D=D_16))
# print("Viscosity for Track 25:", viscosity(D=D_25))
# print("Viscosity for Track 34:", viscosity(D=D_34))
# print("Viscosity for Track 97:", viscosity(D=D_97))
# print("Viscosity for Track 123:", viscosity(D=D_123))

"""Water, Sample 3"""

# track_19 = df[df["TRACK_ID"] == 19]
# track_20 = df[df["TRACK_ID"] == 20]
# track_25 = df[df["TRACK_ID"] == 25]
# track_34 = df[df["TRACK_ID"] == 34]
# track_65 = df[df["TRACK_ID"] == 65]
# track_70 = df[df["TRACK_ID"] == 70]
# track_88 = df[df["TRACK_ID"] == 88]
# track_94 = df[df["TRACK_ID"] == 94]
# track_97 = df[df["TRACK_ID"] == 97]

# D_19, _, _ = diffusion_coefficient(track=track_19)
# D_20, _, _ = diffusion_coefficient(track=track_20)
# D_25, _, _ = diffusion_coefficient(track=track_25)
# D_34, _, _ = diffusion_coefficient(track=track_34)
# D_65, _, _ = diffusion_coefficient(track=track_65)
# D_70, _, _ = diffusion_coefficient(track=track_70)
# D_88, _, _ = diffusion_coefficient(track=track_88)
# D_94, _, _ = diffusion_coefficient(track=track_94)
# D_97, _, _ = diffusion_coefficient(track=track_97)

# print("Water, Sample 3:")
# print("D for Track 19:", D_19)
# print("D for Track 20:", D_20)
# print("D for Track 25:", D_25)
# print("D for Track 34:", D_34)
# print("D for Track 65:", D_65)
# print("D for Track 70:", D_70)
# print("D for Track 88:", D_88)
# print("D for Track 94:", D_94)
# print("D for Track 97:", D_97)
# print("")
# print("Viscosity for Track 19:", viscosity(D=D_19))
# print("Viscosity for Track 20:", viscosity(D=D_20))
# print("Viscosity for Track 25:", viscosity(D=D_25))
# print("Viscosity for Track 34:", viscosity(D=D_34))
# print("Viscosity for Track 65:", viscosity(D=D_65))
# print("Viscosity for Track 70:", viscosity(D=D_70))
# print("Viscosity for Track 88:", viscosity(D=D_88))
# print("Viscosity for Track 94:", viscosity(D=D_94))
# print("Viscosity for Track 97:", viscosity(D=D_97))
