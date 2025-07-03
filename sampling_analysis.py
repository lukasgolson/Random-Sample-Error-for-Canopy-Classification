import numpy as np
import matplotlib.pyplot as plt
from canopy_model import generate_canopy_map


def calculate_direct_cover(canopy_map):

    # The mean of a binary (0/1) array is the proportion of 1s.
    true_proportion = np.mean(canopy_map)
    return true_proportion * 100


def simulate_random_sampling(canopy_map, num_samples):
    height, width = canopy_map.shape

    # Generate random coordinates for each sample point
    sample_x = np.random.randint(0, width, num_samples)
    sample_y = np.random.randint(0, height, num_samples)

    # Check for canopy presence at each sample point and sum the "hits"
    canopy_hits = np.sum(canopy_map[sample_y, sample_x])

    # Calculate the estimated proportion of cover
    estimated_proportion = canopy_hits / num_samples

    return estimated_proportion * 100, sample_x, sample_y


# --- Main Simulation and Visualization ---

# 1. Generate a base map to analyze
canopy_map = generate_canopy_map(clustering=70, canopy_cover=0.55, seed=101)

# 2. Set the number of samples for our simulation
num_samples = 200

# 3. Perform the calculations
true_cover = calculate_direct_cover(canopy_map)
estimated_cover, sample_x, sample_y = simulate_random_sampling(canopy_map, num_samples)

# 4. Visualize the results
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(canopy_map, cmap='Greens', interpolation='nearest')

# Overlay the random sample points
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