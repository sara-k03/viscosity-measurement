import math

dr_10Hz = [
    (2.86, 0.08828),     # 0.024 um
    (2.215, 0.06838),    # 0.040 um
    (1.229, 0.03793),    # 0.130 um
    (0.9044, 0.02792),   # 0.240 um
    (0.6144, 0.01897),   # 0.520 um
    (0.5017, 0.01549),   # 0.780 um
    (0.4778, 0.01475),   # 0.860 um
    (0.4723, 0.01458),   # 0.880 um
    (0.4431, 0.01368),   # 1.0 um
    (0.2558, 0.007896),  # 3.0 um
    (0.1401, 0.004325),  # 10.0 um
]

dr_50Hz = [
    (1.279, 0.03948),    # 0.024 um
    (0.9907, 0.03058),   # 0.040 um
    (0.5495, 0.01696),   # 0.130 um
    (0.4045, 0.01249),   # 0.240 um
    (0.2748, 0.008482),  # 0.520 um
    (0.2244, 0.006925),  # 0.780 um
    (0.2137, 0.006595),  # 0.860 um
    (0.2112, 0.00652),   # 0.880 um
    (0.1981, 0.006116),  # 1.0 um
    (0.1144, 0.003531),  # 3.0 um
    (0.06266, 0.001934), # 10.0 um
]

sizes_um = [0.024, 0.040, 0.130, 0.240, 0.520, 0.780, 0.860, 0.880, 1.0, 3.0, 10.0]


def total_displacement(dr, frame_rate, total_frames=500):
    dr_total = math.sqrt(total_frames) * dr
    total_time = total_frames / frame_rate
    return dr_total, total_time


def main():
    fr = 10
    print("10 Hz Total Displacements")
    for size, (dr_water, dr_gly) in zip(sizes_um, dr_10Hz):
        water_total, total_time = total_displacement(dr_water, fr)
        gly_total, _ = total_displacement(dr_gly, fr)

        print(f"\n{size} um  (t = {total_time:.4g} s)")
        print(f"  water total displacement:    {water_total:.4g} um")
        print(f"  glycerol total displacement:  {gly_total:.4g} um")

    fr = 50
    print("\n50 Hz Total Displacements")
    for size, (dr_water, dr_gly) in zip(sizes_um, dr_50Hz):
        water_total, total_time = total_displacement(dr_water, fr)
        gly_total, _ = total_displacement(dr_gly, fr)

        print(f"\n{size} um  (t = {total_time:.4g} s)")
        print(f"  water total displacement:    {water_total:.4g} um")
        print(f"  glycerol total displacement:  {gly_total:.4g} um")


if __name__ == "__main__":
    main()