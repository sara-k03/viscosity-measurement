"""
Per-frame displacement calculator for single-particle tracking.

Given a particle diameter, computes the Stokes-Einstein diffusion
coefficient D in each medium, then the expected root-mean-square
per-frame displacement at each frame rate:

    D  = kT / (6 * pi * eta * a)          Stokes-Einstein
    dr = sqrt(4 * D * dt)                 2D per-frame RMS displacement

Edit MEDIA and FRAME_RATES_HZ below to match your lab setup.

Usage:
    python3 particle_displacement.py
    (then enter the diameter in microns when prompted)
"""

import math

k_B = 1.380649e-23  # Boltzmann constant, J/K


# name -> dynamic viscosity in Pa*s at the given temperature
MEDIA = {
    "Water":    0.89e-3,   # Pa*s at 25 C
    "Glycerol": 0.934,     # Pa*s at 25 C (~99% glycerol)
}

TEMPERATURE_C = 25.0
FRAME_RATES_HZ = [10, 50]


def diffusion_coefficient(radius_m: float, eta_pa_s: float, temp_k: float) -> float:
    """Stokes-Einstein diffusion coefficient, m^2/s."""
    return (k_B * temp_k) / (6 * math.pi * eta_pa_s * radius_m)


def rms_displacement(D_m2_s: float, dt_s: float) -> float:
    """2D root-mean-square per-frame displacement, meters."""
    return math.sqrt(4 * D_m2_s * dt_s)


def report(diameter_um: float):
    temp_k = TEMPERATURE_C + 273.15
    radius_m = (diameter_um * 1e-6) / 2

    print(f"\nParticle diameter: {diameter_um} um   (T = {TEMPERATURE_C} C)")
    for medium_name, eta in MEDIA.items():
        D = diffusion_coefficient(radius_m, eta, temp_k)
        print(f"\n  {medium_name}  (eta = {eta:g} Pa*s)")
        print(f"    D = {D * 1e12:.4g} um^2/s")
        for hz in FRAME_RATES_HZ:
            dt = 1.0 / hz
            dr = rms_displacement(D, dt)
            print(f"    dr @ {hz:>3} Hz (dt={dt*1000:.0f} ms): {dr * 1e6:.4g} um")
    print()


if __name__ == "__main__":
    diameter_um = float(input("Particle diameter (um): "))
    report(diameter_um)