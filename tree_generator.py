import random
import matplotlib.pyplot as plt
import noise
import numpy as np
from skimage.feature import peak_local_max

## -------------------------------------------------- DEFINE FUNCTIONS -------------------------------------------------
#region

# Original function using a single threshold on Perlin noise. Results in large, blob-like canopy shapes
def generate_canopy_map(width=100, height=100, clustering=50, canopy_cover=0.5, seed=None):

    if clustering == 0:
        if seed is not None:
            np.random.seed(seed)
        random_grid = np.random.rand(height, width)
        return (random_grid < canopy_cover).astype(int)

    base_seed = seed if seed is not None else np.random.randint(0, np.iinfo(np.int32).max)
    world = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            world[i][j] = noise.pnoise2(i / clustering, j / clustering,
                                        octaves=6, persistence=0.5, lacunarity=2.0,
                                        base=base_seed)
    threshold = np.percentile(world, (1 - canopy_cover) * 100)
    return (world > threshold).astype(int)

# Generates a canopy map by layering multiple Perlin noise fields and placing circular trees
def generate_canopy_map_layered_noise(width=100, height=100, base_clustering=80, detail_clustering=20,
        detail_weight=0.3, density=0.6, tree_radius_range=(5, 10), seed=None):

    rng = np.random.default_rng(seed)
    base_seed = int(rng.integers(0, np.iinfo(np.int32).max))
    base_noise = np.zeros((height, width))
    detail_noise = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            base_noise[i][j] = (noise.pnoise2(i / base_clustering, j / base_clustering, octaves=6,
                                              base=base_seed) + 1) / 2
            detail_noise[i][j] = (noise.pnoise2(i / detail_clustering, j / detail_clustering, octaves=6,
                                                base=base_seed + 1) + 1) / 2

    combined_noise = base_noise * (1 - detail_weight) + detail_noise * detail_weight

    min_rad, max_rad = tree_radius_range
    coordinates = peak_local_max(combined_noise, min_distance=min_rad, threshold_abs=density)

    canopy_map = np.zeros((height, width), dtype=np.int8)
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))

    for y_tree, x_tree in coordinates:
        radius = rng.uniform(min_rad, max_rad)
        distance_sq = (x_coords - x_tree) ** 2 + (y_coords - y_tree) ** 2
        canopy_map[distance_sq <= radius ** 2] = 1

    return canopy_map

# A simple class to hold a pre-generated canopy shape and its area
class CanopyStamp:

    def __init__(self, shape):
        self.shape = shape
        self.area = np.sum(shape)
        self.height, self.width = shape.shape

# Generates a library of pre-rendered, irregular canopy shapes (stamps)
def create_stamp_library(num_stamps, radius_range, shape_complexity, seed=None):

    print(f"Generating {num_stamps} canopy stamps for the library...")
    rng = np.random.default_rng(seed)
    base_seed = int(rng.integers(0, np.iinfo(np.int32).max))
    library = []

    for i in range(num_stamps):
        radius = rng.uniform(*radius_range)
        stamp_dim = int(radius * 2)
        if stamp_dim == 0:
            continue

        # Generate a small noise field for the stamp
        tree_noise = np.zeros((stamp_dim, stamp_dim))
        for y in range(stamp_dim):
            for x in range(stamp_dim):
                tree_noise[y, x] = (noise.pnoise2(x / shape_complexity, y / shape_complexity, octaves=4,
                                                  base=base_seed + i) + 1) / 2

        # Create a circular falloff to contain the blob
        center = stamp_dim / 2.0
        y_indices, x_indices = np.ogrid[:stamp_dim, :stamp_dim]
        dist_sq = (y_indices - center) ** 2 + (x_indices - center) ** 2

        # Create a gradient falloff
        falloff = np.maximum(0, 1 - np.sqrt(dist_sq) / radius)

        shaped_noise = tree_noise * falloff

        canopy_shape = (shaped_noise > 0.4).astype(np.int8)
        library.append(CanopyStamp(canopy_shape))

    return library

# Generates a forest with a target canopy cover by placing pre-generated tree stamps one by one until the target is met
def generate_forest_analytically(width, height, target_cover, stamp_library, generator_params, seed=None):

    rng = np.random.default_rng(seed)
    base_seed = int(rng.integers(0, np.iinfo(np.int32).max))

    # Unpack parameters
    base_clustering = generator_params.get("base_clustering", 120)
    radius_range = generator_params.get("radius_range", (15, 25))
    min_rad, _ = radius_range

    print(f"\nAnalytically generating map for target cover: {target_cover:.1%}")

    # 1. Generate the base noise field to determine potential tree locations
    location_noise = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            location_noise[i][j] = (noise.pnoise2(i / base_clustering, j / base_clustering, octaves=6,
                                                  base=base_seed) + 1) / 2

    # 2. Find all potential peaks and rank them
    # FIX: Set min_distance to half the minimum radius. This prevents extreme
    # overlap, but can probably be improved
    min_dist = int(min_rad / 2)
    all_peaks = peak_local_max(location_noise, min_distance=min_dist, threshold_rel=0.01)
    print(f"Found {len(all_peaks)} potential tree locations with min_distance={min_dist}.")

    peak_values = location_noise[all_peaks[:, 0], all_peaks[:, 1]]
    sorted_peak_indices = np.argsort(peak_values)[::-1]
    ranked_peaks = all_peaks[sorted_peak_indices]

    # 3. Place trees from the stamp library one by one
    canopy_map = np.zeros((height, width), dtype=np.int8)
    total_pixels = width * height
    covered_pixels = 0

    for y_center, x_center in ranked_peaks:
        current_cover = covered_pixels / total_pixels
        if current_cover >= target_cover:
            print(f"Target reached. Final cover: {current_cover:.2%}")
            break

        # Select a random stamp from the library
        stamp = random.choice(stamp_library)

        # --- Robust slicing logic ---
        h, w = stamp.height, stamp.width
        stamp_cy, stamp_cx = h // 2, w // 2

        # Calculate the top-left corner of where the stamp should be placed
        tl_y = y_center - stamp_cy
        tl_x = x_center - stamp_cx

        # Determine the slice boundaries on the main map
        map_y_start = max(0, tl_y)
        map_y_end = min(height, tl_y + h)
        map_x_start = max(0, tl_x)
        map_x_end = min(width, tl_x + w)

        # Determine the corresponding slice from the stamp
        stamp_y_start = max(0, -tl_y)
        stamp_y_end = stamp_y_start + (map_y_end - map_y_start)
        stamp_x_start = max(0, -tl_x)
        stamp_x_end = stamp_x_start + (map_x_end - map_x_start)

        if map_y_start >= map_y_end or map_x_start >= map_x_end:
            continue

        map_slice = canopy_map[map_y_start:map_y_end, map_x_start:map_x_end]
        stamp_slice = stamp.shape[stamp_y_start:stamp_y_end, stamp_x_start:stamp_x_end]

        # Add the new tree to the map, counting only newly covered pixels
        new_pixels = np.sum(stamp_slice & (1 - map_slice))
        covered_pixels += new_pixels
        map_slice |= stamp_slice
    else:
        print(f"Warning: Ran out of locations. Final cover: {np.mean(canopy_map):.2%}")

    return canopy_map

#endregion

## ----------------------------------------------- VISUALIZATION EXAMPLE -----------------------------------------------
#region

if __name__ == '__main__':
    map_size = 500
    seed = 42

    # --- Define parameters for the generators ---
    # Moved radius_range here to be accessible by the analytical function
    forest_style_params = {
        "base_clustering":  120,
        "radius_range":     (5, 50),
        "shape_complexity": 25
    }

    # The stamp library now references the main style parameters
    stamp_library_params = {
        "num_stamps":       100,
        "radius_range":     forest_style_params["radius_range"],
        "shape_complexity": forest_style_params["shape_complexity"]
    }

    target_canopy_percentage = 0.4

    # --- Generate the stamp library first ---
    stamp_library = create_stamp_library(**stamp_library_params, seed=seed)

    # --- Generate Maps ---
    original_map = generate_canopy_map(
        width=map_size, height=map_size, clustering=80, canopy_cover=target_canopy_percentage, seed=seed
    )

    circular_trees_map = generate_canopy_map_layered_noise(
        width=map_size, height=map_size, base_clustering=120, detail_clustering=30,
        detail_weight=0.4, density=0.55, tree_radius_range=(8, 15), seed=seed
    )

    # Use the analytical method with the pre-built stamp library
    irregular_trees_map = generate_forest_analytically(
        width=map_size, height=map_size,
        target_cover=target_canopy_percentage,
        stamp_library=stamp_library,
        generator_params=forest_style_params,
        seed=seed
    )

    # --- Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Comparison of Canopy Generation Methods", fontsize=16, fontweight='bold')

    axes[0].imshow(original_map, cmap='Greens', interpolation='nearest')
    axes[0].set_title(f"Original Method (Blob-like)\nCover: {np.mean(original_map):.2%}", fontsize=12)

    axes[1].imshow(circular_trees_map, cmap='Greens', interpolation='nearest')
    axes[1].set_title(f"Circular Trees Method\nCover: {np.mean(circular_trees_map):.2%}", fontsize=12)

    axes[2].imshow(irregular_trees_map, cmap='Greens', interpolation='nearest')
    axes[2].set_title(f"Stamp Library Method\nCover: {np.mean(irregular_trees_map):.2%}", fontsize=12)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

#endregion
