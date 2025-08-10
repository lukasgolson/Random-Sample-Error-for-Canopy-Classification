import numpy as np
import matplotlib.pyplot as plt
from canopy_model import generate_canopy_map

# Configuration
width = height = 1490
sample_points = 100
num_trials = 10

clustering_levels = range(0, 101, 5)
canopy_covers = [0.10, 0.20, 0.30]  # 10%, 20%, 30%

results_standard = {cover: [] for cover in canopy_covers}
results_neff = {cover: [] for cover in canopy_covers}

for cover in canopy_covers:
    for clustering in clustering_levels:

        canopy_map = generate_canopy_map(width, height, clustering=clustering, canopy_cover=cover)
        true_cover = np.mean(canopy_map)

        misses_standard = 0
        misses_neff = 0

        rho = clustering / 100  # Approximate Moran's I
        neff = sample_points / (1 + (sample_points - 1) * rho)

        for _ in range(num_trials):
            sample_x = np.random.randint(0, width, sample_points)
            sample_y = np.random.randint(0, height, sample_points)
            sample_vals = canopy_map[sample_y, sample_x]

            p = np.mean(sample_vals)

            # Standard SE
            se_standard = np.sqrt(p * (1 - p) / sample_points)
            margin_standard = 1.96 * se_standard
            lb_standard = max(0, p - margin_standard)
            ub_standard = min(1, p + margin_standard)

            if not (lb_standard <= true_cover <= ub_standard):
                misses_standard += 1

            # Neff-adjusted SE
            se_neff = np.sqrt(p * (1 - p) / neff)
            margin_neff = 1.96 * se_neff
            lb_neff = max(0, p - margin_neff)
            ub_neff = min(1, p + margin_neff)

            if not (lb_neff <= true_cover <= ub_neff):
                misses_neff += 1

        results_standard[cover].append(misses_standard)
        results_neff[cover].append(misses_neff)

# Plotting
plt.figure(figsize=(10, 7))

for cover in canopy_covers:
    label_std = f"{int(cover*100)}% Cover (Standard SE)"
    label_neff = f"{int(cover*100)}% Cover (Neff Adjusted)"
    plt.plot(results_standard[cover], clustering_levels, linestyle='--', marker='o', label=label_std)
    plt.plot(results_neff[cover], clustering_levels, linestyle='-', marker='o', label=label_neff)

plt.xlabel("Number of Trials Outside 95% CI", fontsize=14)
plt.ylabel("Clustering Level", fontsize=14)
plt.grid(True)
plt.legend(title="Canopy Cover & Method")
plt.tight_layout()
plt.gca().invert_yaxis()
plt.show()
#endregion
