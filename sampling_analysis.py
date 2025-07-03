import numpy as np
import matplotlib.pyplot as plt
from canopy_model import generate_canopy_map
from canopy_sampling import calculate_direct_cover, simulate_random_sampling

# --- Main Simulation and Visualization ---

# Generate a base map to analyze
canopy_map = generate_canopy_map(width=1024, height=1024, clustering=70, canopy_cover=0.55, seed=101)

num_samples = 200

# Calculate the true canopy cover using direct calculation and simulate random sampling
true_cover = calculate_direct_cover(canopy_map)
estimated_cover, sample_x, sample_y = simulate_random_sampling(canopy_map, num_samples)

# Visualize the results
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(canopy_map, cmap='Greens', interpolation='nearest')
ax.scatter(sample_x, sample_y, c='red', marker='+', s=50, label=f'{num_samples} Sample Points')

ax.legend()
ax.set_xticks([])
ax.set_yticks([])
title_text = (
    f"Direct Calculation vs. Random Sampling\n"
    f"True Cover: {true_cover:.2f}% | "
    f"Estimated Cover: {estimated_cover:.2f}%"
)
ax.set_title(title_text)

plt.show()