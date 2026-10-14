import matplotlib.pyplot as plt

# 50% Glycerol, 50% Water
glycerol_viscosities = [
    0.005008242316297389,  # Sample 1, Track 4
    0.008934250282301536,  # Sample 1, Track 5
    0.006638083626127223,  # Sample 2, Track 0
    0.006936502859237333,  # Sample 2, Track 1
    0.007516588677079124,  # Sample 2, Track 2
    0.0077455406940222914, # Sample 2, Track 3
    0.007516588677079124,  # Sample 2, Track 4
    0.0077455406940222914, # Sample 2, Track 5
    0.006082591561960694   # Sample 3, Track 4
]

# Water
water_viscosities = [
    0.0006163774465219932,  # Sample 1, Track 10
    0.00013097203916347369, # Sample 1, Track 26
    0.0011011783414707984,  # Sample 1, Track 72
    0.00033438918661162604, # Sample 2, Track 16
    0.00042838408977179327, # Sample 2, Track 25
    0.00037905316043862914, # Sample 2, Track 34
    0.0027330080024911265,  # Sample 2, Track 97
    0.00029316867745616496, # Sample 2, Track 123
    0.0004635652899303567,  # Sample 3, Track 19
    0.000730033544928861,   # Sample 3, Track 20
    0.00016015923193069798, # Sample 3, Track 25
    0.0005780740064944712,  # Sample 3, Track 34
    0.0006274481105899689,  # Sample 3, Track 65
    0.0009517699890423791,  # Sample 3, Track 70
    0.00035680349330152256, # Sample 3, Track 88
    0.0008403964215663103,  # Sample 3, Track 94
    0.0008376576558334841   # Sample 3, Track 97
]

# 50% Glycerol, 50% Water
plt.figure()
plt.scatter(range(len(glycerol_viscosities)), glycerol_viscosities)
plt.ylabel("Viscosity")
plt.title("Viscosity of 50% Glycerol, 50% Water")
plt.xticks([])
plt.xlabel("")
plt.ylim(bottom=0)
plt.grid(True, axis="y")
plt.show()

# Water
plt.figure()
plt.scatter(range(len(water_viscosities)), water_viscosities)
plt.ylabel("Viscosity")
plt.title("Viscosity of Water")
plt.xticks([])
plt.xlabel("")
plt.ylim(bottom=0)
plt.grid(True, axis="y")
plt.show()