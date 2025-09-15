## Conceptual framework for code
# 1. Create a tree stamp
# 2. Generate an AOI (400, 2000, 30000) with canopy extent (0 - 100%) and Moran's i (-1 to +1). Canopy in steps of 1% (101 steps including 0%), Moran's i in steps of .05 (40 steps) (total: 4,040 steps)
# 3. Apply 100,000 random sample points (same points for all runs) and identify as canopy (1) or no canopy (0). Include meters x and y of sample points (in replace of latitude and longitude)
# 4. Export results as CSV with columns Sample Point ID, Sample Point X Location, Sample Point Y Location, then columns with code {AOI}_{Target_Extent}_{Target_Moran}_{True_Extent}_{True_Moran}. 


# Defining the column titles for the CSV
AOI = AOI in AOIs
Target_Extent = canopy_extent

True_Extent = np.mean(canopy_map) * 100 # The mean of a binary (0/1) array is the proportion of 1s


exit() # Safety stop
## EVERYTHING BELOW HERE IS COPIED FROM OTHER DRAFTS

# Tree generator
import opensimplex
import numpy as np
import matplotlib.pyplot as plt
import numba
from skimage.feature import peak_local_max
import random



def _generate_fast_noise(width, height, scale, seed, octaves, persistence, lacunarity):
    world = np.empty((height, width), dtype=np.float64)

    simplex = opensimplex.OpenSimplex(seed=seed)

    for i in range(height):
        for j in range(width):
            noise_value = 0.0
            frequency = 1.0
            amplitude = 1.0
            for _ in range(octaves):
                nx = i / scale * frequency
                ny = j / scale * frequency
                noise_value += simplex.noise2(nx, ny) * amplitude

                amplitude *= persistence
                frequency *= lacunarity

            world[i, j] = noise_value

    return world



def generate_canopy_map_fast(width=100, height=100, clustering=50, canopy_cover=0.5, seed=None):
    """
    Generates a canopy map using OpenSimplex
    This creates large, blob-like canopy shapes.
    """
    if clustering <= 0:
        rng = np.random.default_rng(seed)
        random_grid = rng.random((height, width))
        return (random_grid < canopy_cover).astype(np.int8)

    base_seed = seed if seed is not None else np.random.randint(0, np.iinfo(np.int32).max)

    world = _generate_fast_noise(
        width, height, clustering, base_seed,
        octaves=6, persistence=0.5, lacunarity=2.0
    )

    threshold = np.percentile(world, (1 - canopy_cover) * 100)
    return (world > threshold).astype(np.int8)


def generate_canopy_map_layered_noise(width=100, height=100, base_clustering=80, detail_clustering=20,
        detail_weight=0.3, density=0.6, tree_radius_range=(5, 10), seed=None):
    """
    Generates a canopy map by layering multiple OpenSimplex noise fields and placing circular trees.
    """
    rng = np.random.default_rng(seed)
    base_seed = int(rng.integers(0, np.iinfo(np.int32).max))

    # Generate both noise fields using the fast, compiled function
    base_noise = _generate_fast_noise(width, height, base_clustering, base_seed, 6, 0.5, 2.0)
    detail_noise = _generate_fast_noise(width, height, detail_clustering, base_seed + 1, 6, 0.5, 2.0)

    # Normalize noise to 0-1 range
    base_noise = (base_noise - base_noise.min()) / (base_noise.max() - base_noise.min())
    detail_noise = (detail_noise - detail_noise.min()) / (detail_noise.max() - detail_noise.min())

    combined_noise = base_noise * (1 - detail_weight) + detail_noise * detail_weight

    min_rad, max_rad = tree_radius_range
    coordinates = peak_local_max(combined_noise, min_distance=min_rad, threshold_abs=density)

    canopy_map = np.zeros((height, width), dtype=np.int8)
    y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width))

    for y_tree, x_tree in coordinates:
        radius = rng.uniform(min_rad, max_rad)
        # Corrected distance calculation for meshgrid
        distance_sq = (x_coords - x_tree) ** 2 + (y_coords - y_tree) ** 2
        canopy_map[distance_sq <= radius ** 2] = 1

    return canopy_map


class CanopyStamp:
    """A simple class to hold a pre-generated canopy shape and its area."""

    def __init__(self, shape):
        self.shape = shape
        self.area = np.sum(shape)
        self.height, self.width = shape.shape


def create_stamp_library(num_stamps, radius_range, shape_complexity, seed=None):
    """
    Generates a library of pre-rendered, irregular canopy shapes (stamps)
    using the opensimplex library.
    """
    print(f"Generating {num_stamps} canopy stamps for the library...")
    rng = np.random.default_rng(seed)
    base_seed = int(rng.integers(0, np.iinfo(np.int32).max))
    library = []

    # Prevent division by zero, which could still cause issues.
    if shape_complexity <= 0:
        shape_complexity = 1e-9

    for i in range(num_stamps):
        # Initialize OpenSimplex with a unique seed for each stamp
        simplex = opensimplex.OpenSimplex(seed=base_seed + i)
        radius = rng.uniform(*radius_range)
        stamp_dim = int(radius * 2)
        if stamp_dim == 0:
            continue

        tree_noise = np.zeros((stamp_dim, stamp_dim))
        for y in range(stamp_dim):
            for x in range(stamp_dim):
                # Manually implement octaves for detailed noise
                frequency = 1.0
                amplitude = 1.0
                noise_value = 0
                for _ in range(4):  # Number of octaves
                    nx = x / shape_complexity * frequency
                    ny = y / shape_complexity * frequency
                    noise_value += simplex.noise2(nx, ny) * amplitude
                    amplitude *= 0.5
                    frequency *= 2.0
                tree_noise[y, x] = (noise_value + 1) / 2

        center = stamp_dim / 2.0
        y_indices, x_indices = np.ogrid[:stamp_dim, :stamp_dim]
        dist_sq = (y_indices - center) ** 2 + (x_indices - center) ** 2
        falloff = np.maximum(0, 1 - np.sqrt(dist_sq) / radius)
        shaped_noise = tree_noise * falloff
        canopy_shape = (shaped_noise > 0.4).astype(np.int8)
        library.append(CanopyStamp(canopy_shape))
    return library


def generate_forest_analytically(width, height, target_cover, stamp_library, generator_params, seed=None):
    """
    Generates a forest by analytically placing tree stamps until a target cover is met.
    """
    rng = np.random.default_rng(seed)
    base_seed = int(rng.integers(0, np.iinfo(np.int32).max))
    base_clustering = generator_params.get("base_clustering", 120)
    min_rad, _ = generator_params.get("radius_range", (15, 25))

    print(f"\nAnalytically generating map for target cover: {target_cover:.1%}")

    # 1. Generate the base noise field using the fast function
    location_noise = _generate_fast_noise(width, height, base_clustering, base_seed, 6, 0.5, 2.0)
    location_noise = (location_noise - location_noise.min()) / (location_noise.max() - location_noise.min())

    # 2. Find and rank potential tree locations
    min_dist = int(min_rad / 2)
    all_peaks = peak_local_max(location_noise, min_distance=min_dist, threshold_rel=0.01)
    print(f"Found {len(all_peaks)} potential tree locations with min_distance={min_dist}.")
    peak_values = location_noise[all_peaks[:, 0], all_peaks[:, 1]]
    ranked_peaks = all_peaks[np.argsort(peak_values)[::-1]]

    # 3. Place trees from the stamp library one by one
    canopy_map = np.zeros((height, width), dtype=np.int8)
    total_pixels = width * height
    covered_pixels = 0
    for y_center, x_center in ranked_peaks:
        if covered_pixels / total_pixels >= target_cover:
            print(f"Target reached. Final cover: {covered_pixels / total_pixels:.2%}")
            break

        stamp = random.choice(stamp_library)
        h, w = stamp.height, stamp.width
        tl_y, tl_x = y_center - h // 2, x_center - w // 2
        map_y_start, map_y_end = max(0, tl_y), min(height, tl_y + h)
        map_x_start, map_x_end = max(0, tl_x), min(width, tl_x + w)
        stamp_y_start = max(0, -tl_y)
        stamp_y_end = stamp_y_start + (map_y_end - map_y_start)
        stamp_x_start = max(0, -tl_x)
        stamp_x_end = stamp_x_start + (map_x_end - map_x_start)

        if map_y_start >= map_y_end or map_x_start >= map_x_end:
            continue

        map_slice = canopy_map[map_y_start:map_y_end, map_x_start:map_x_end]
        stamp_slice = stamp.shape[stamp_y_start:stamp_y_end, stamp_x_start:stamp_x_end]

        new_pixels = np.sum(stamp_slice & (1 - map_slice))
        covered_pixels += new_pixels
        map_slice |= stamp_slice
    else:
        print(f"Warning: Ran out of locations. Final cover: {np.mean(canopy_map):.2%}")
    return canopy_map


# --- Visualization Example ---
if __name__ == '__main__':
    map_size = 500
    seed = 42
    target_canopy_percentage = 0.1

    forest_style_params = {
        "base_clustering":  60,
        "radius_range":     (10, 25),
        "shape_complexity": 5
    }
    stamp_library_params = {
        "num_stamps":       300,
        "radius_range":     forest_style_params["radius_range"],
        "shape_complexity": forest_style_params["shape_complexity"]
    }

    # --- Generate the stamp library first ---
    stamp_library = create_stamp_library(**stamp_library_params, seed=seed)

    # --- Generate Maps using the new, safe functions ---
    blob_map = generate_canopy_map_fast(
        width=map_size, height=map_size, clustering=80, canopy_cover=target_canopy_percentage, seed=seed
    )
    circular_trees_map = generate_canopy_map_layered_noise(
        width=map_size, height=map_size, base_clustering=120, detail_clustering=30,
        detail_weight=0.4, density=0.55, tree_radius_range=(8, 15), seed=seed
    )
    irregular_trees_map = generate_forest_analytically(
        width=map_size, height=map_size,
        target_cover=target_canopy_percentage,
        stamp_library=stamp_library,
        generator_params=forest_style_params,
        seed=seed
    )

    # --- Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Comparison of Canopy Generation Methods (using OpenSimplex)", fontsize=16, fontweight='bold')

    axes[0].imshow(blob_map, cmap='Greens', interpolation='nearest')
    axes[0].set_title(f"Fast Blob Method\nCover: {np.mean(blob_map):.2%}", fontsize=12)

    axes[1].imshow(circular_trees_map, cmap='Greens', interpolation='nearest')
    axes[1].set_title(f"Circular Trees Method\nCover: {np.mean(circular_trees_map):.2%}", fontsize=12)

    axes[2].imshow(irregular_trees_map, cmap='Greens', interpolation='nearest')
    axes[2].set_title(f"Stamp Library Method\nCover: {np.mean(irregular_trees_map):.2%}", fontsize=12)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# Canopy_models
import numpy as np
import noise
from matplotlib import pyplot as plt


def generate_canopy_map(width=100, height=100, clustering=50, canopy_cover=0.5, seed=None):
    """
    Generates a binary raster simulating a canopy presence map.
    """
    if not 0 <= clustering <= 100:
        raise ValueError("Clustering must be an integer between 0 and 100.")

    # --- Handle the 0 Clustering Case ---
    # If clustering is zero, create a purely random map (no spatial correlation).
    if clustering == 0:
        if seed is not None:
            np.random.seed(seed)
        # Generate a grid of random values between 0 and 1
        random_grid = np.random.rand(height, width)
        # Create a binary map based on the canopy_cover fraction
        canopy_map = (random_grid < canopy_cover).astype(int)
        return canopy_map

    base_seed = seed if seed is not None else np.random.randint(0, np.iinfo(np.int32).max)

    world = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            world[i][j] = noise.pnoise2(i / clustering,
                                        j / clustering,
                                        octaves=6,
                                        persistence=0.5,
                                        lacunarity=2.0,
                                        base=base_seed)

    # Determine the canopy cover threshold
    threshold = np.percentile(world, (1 - canopy_cover) * 100)

    # Create the binary map by applying the threshold
    canopy_map = (world > threshold).astype(int)

    return canopy_map


def calculate_morans_i(grid):
    """
    Calculates Moran's I for a 2D grid using Queen contiguity.

    Args:
        grid (numpy.ndarray): A 2D numpy array with numerical values.

    Returns:
        float: The Moran's I statistic. Returns np.nan if the grid variance is zero.
    """
    height, width = grid.shape
    n = height * width

    # Create a flattened version of the grid for easier iteration
    flat_grid = grid.flatten()
    mean_val = np.mean(flat_grid)
    variance = np.var(flat_grid)

    # If all values are the same, Moran's I is undefined.
    if variance == 0:
        return np.nan

    # Numerator and denominator for the Moran's I formula
    numerator = 0.0
    denominator = np.sum((flat_grid - mean_val) ** 2)

    # Sum of weights
    w_sum = 0.0

    # Iterate over each cell in the grid
    for r in range(height):
        for c in range(width):
            # Get the deviation from the mean for the current cell
            deviation_i = grid[r, c] - mean_val

            # Check all 8 neighboring cells (Queen contiguity)
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    # Skip the cell itself
                    if dr == 0 and dc == 0:
                        continue

                    nr, nc = r + dr, c + dc

                    # Check if the neighbor is within the grid bounds
                    if 0 <= nr < height and 0 <= nc < width:
                        # Weight is 1 for neighbors, 0 otherwise
                        w_ij = 1.0
                        w_sum += w_ij

                        deviation_j = grid[nr, nc] - mean_val
                        numerator += w_ij * deviation_i * deviation_j

    if w_sum == 0 or denominator == 0:
        return np.nan

    return (n / w_sum) * (numerator / denominator)


def run_simulation_and_plot():
    """
    Runs the simulation by generating maps with different clustering values,
    calculating Moran's I for each, and plotting the relationship.
    """
    print("Starting simulation...")

    # Define the range of clustering values to test
    # We use 21 points to include both 0 and 100.
    clustering_values = np.linspace(0, 20, 100)
    morans_i_results = []

    # Use a fixed seed for reproducibility of the entire experiment
    simulation_seed = 42

    # Loop through each clustering value with a progress bar
    for cluster_val in clustering_values:
        # Generate the map. Note: clustering=0 is a special case.
        # For clustering > 0, we need to handle potential division by zero.
        if cluster_val > 0:
            canopy_map = generate_canopy_map(
                width=500,
                height=500,
                clustering=cluster_val,
                canopy_cover=0.5,
                seed=simulation_seed
            )
        else:
            canopy_map = generate_canopy_map(
                width=500,
                height=500,
                clustering=0,
                canopy_cover=0.5,
                seed=simulation_seed
            )

        # Calculate Moran's I for the generated map
        morans_i = calculate_morans_i(canopy_map)
        morans_i_results.append(morans_i)

    print("Simulation finished. Plotting results...")

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    # Scatter plot of the results
    ax.plot(clustering_values, morans_i_results, marker='o', linestyle='-', color='royalblue')

    # Titles and labels
    ax.set_title("Relationship between Clustering Parameter and Moran's I", fontsize=16, fontweight='bold')
    ax.set_xlabel("Clustering Parameter", fontsize=12)
    ax.set_ylabel("Moran's I", fontsize=12)

    # Set axis limits for clarity
    ax.set_xlim(0, 20)
    ax.set_ylim(-0.1, 1.0)  # Moran's I typically ranges from -1 to 1

    # Add a horizontal line at I=0 for reference (random pattern)
    ax.axhline(0, color='red', linestyle='--', linewidth=1, label='Expected I for Random Pattern')

    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()

    # --- Plot example maps ---
    print("Generating example maps for visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Map for Clustering = 0 (Random)
    map_0 = generate_canopy_map(clustering=0, seed=simulation_seed)
    axes[0].imshow(map_0, cmap='Greens', interpolation='nearest')
    axes[0].set_title(f"Clustering = 0\nMoran's I: {calculate_morans_i(map_0):.3f}")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Map for Clustering = 20 (Moderate)
    map_20 = generate_canopy_map(clustering=20, seed=simulation_seed)
    axes[1].imshow(map_20, cmap='Greens', interpolation='nearest')
    axes[1].set_title(f"Clustering = 20\nMoran's I: {calculate_morans_i(map_20):.3f}")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    # Map for Clustering = 80 (High)
    map_80 = generate_canopy_map(clustering=80, seed=simulation_seed)
    axes[2].imshow(map_80, cmap='Greens', interpolation='nearest')
    axes[2].set_title(f"Clustering = 80\nMoran's I: {calculate_morans_i(map_80):.3f}")
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    fig.suptitle("Example Canopy Maps", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == '__main__':
    # To run the analysis, simply execute this script.
    # Make sure you have the required libraries installed:
    # pip install numpy noise matplotlib
    run_simulation_and_plot()

import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.features import shapes
from shapely.geometry import shape
from multiprocessing import Pool
import pandas as pd
from tqdm import tqdm

# Constants
N_CPUS = 192
OUT_DIR = "/scratch/arbmarta/Outputs/CSVs"
os.makedirs(OUT_DIR, exist_ok=True)

# Input boundaries
van_bayan = gpd.read_file('/scratch/arbmarta/Trinity/Vancouver/TVAN.shp').to_crs("EPSG:32610")
wpg_bayan = gpd.read_file('/scratch/arbmarta/Trinity/Winnipeg/TWPG.shp').to_crs("EPSG:32614")
ott_bayan = gpd.read_file('/scratch/arbmarta/Trinity/Ottawa/TOTT.shp').to_crs("EPSG:32618")

rasters = {
    "Vancouver": {"LiDAR": '/scratch/arbmarta/CHMs/LiDAR/Vancouver LiDAR.tif', "bayan": van_bayan, "epsg": "EPSG:32610"},
    "Winnipeg": {"LiDAR": '/scratch/arbmarta/CHMs/LiDAR/Winnipeg LiDAR.tif', "bayan": wpg_bayan, "epsg": "EPSG:32614"},
    "Ottawa": {"LiDAR": '/scratch/arbmarta/CHMs/LiDAR/Ottawa LiDAR.tif', "bayan": ott_bayan, "epsg": "EPSG:32618"},
}

def compute_chm_stats(grid_raster, grid_geom):
    """Compute tree height stats within a grid for heights >0 and >=2 m"""
    with rasterio.open(grid_raster) as src:
        out_image, _ = mask(src, [grid_geom], crop=True)
        data = out_image[0].astype(float)
        
        # Debug: Check if we have any data
        print(f"Data shape: {data.shape}")
        print(f"Data range: {np.min(data)} to {np.max(data)}")
        print(f"Non-zero values: {np.count_nonzero(data)}")
        print(f"Values > 0: {np.sum(data > 0)}")
        print(f"Values >= 2: {np.sum(data >= 2)}")

        # Heights > 0
        data_pos = data[data > 0]
        if data_pos.size == 0:
            print("WARNING: No positive height values found!")
            stats_pos = dict(min_height_0=0, max_height_0=0, mean_height_0=0, 
                             median_height_0=0, std_height_0=0)
        else:
            stats_pos = dict(
                min_height_0=float(np.min(data_pos)),
                max_height_0=float(np.max(data_pos)),
                mean_height_0=float(np.mean(data_pos)),
                median_height_0=float(np.median(data_pos)),
                std_height_0=float(np.std(data_pos))
            )
            print(f"Stats for heights > 0: {stats_pos}")

        # Heights >= 2
        data_2 = data[data >= 2]
        if data_2.size == 0:
            print("WARNING: No values >= 2m found!")
            stats_2 = dict(min_height_2=0, max_height_2=0, mean_height_2=0, 
                           median_height_2=0, std_height_2=0)
        else:
            stats_2 = dict(
                min_height_2=float(np.min(data_2)),
                max_height_2=float(np.max(data_2)),
                mean_height_2=float(np.mean(data_2)),
                median_height_2=float(np.median(data_2)),
                std_height_2=float(np.std(data_2))
            )
            print(f"Stats for heights >= 2: {stats_2}")

        # Combine both dictionaries
        stats_pos.update(stats_2)
        return stats_pos

def compute_clumpy(binary_array):
    """
    Compute CLUMPY index from a 2D binary array (1=canopy, 0=no canopy)
    Using adjacency counts: like adjacencies / expected like adjacencies
    """
    import numpy as np

    # 4-neighbor adjacency
    arr = binary_array
    like = 0
    total = 0

    for dx, dy in [(0,1),(1,0)]:  # right and down neighbors
        shifted = np.roll(arr, shift=-dx, axis=0)
        shifted = np.roll(shifted, shift=-dy, axis=1)
        mask = (np.arange(arr.shape[0]-dx)[:, None] < arr.shape[0]-dx) & (np.arange(arr.shape[1]-dy)[None, :] < arr.shape[1]-dy)
        like += np.sum(arr[:-dx or None, :-dy or None] == shifted[:-dx or None, :-dy or None])
        total += np.sum(mask)

    p_obs = like / total if total > 0 else 0
    p_exp = np.mean(arr)**2 if np.mean(arr) > 0 else 0

    clumpy = (p_obs - p_exp) / (1 - p_exp) if p_exp < 1 else 0
    return clumpy
    
def raster_to_polygons(masked_arr, out_transform, nodata=None):
    """Convert a binary raster (1=canopy, 0=no canopy) to polygons"""
    band = masked_arr[0]
    mask_vals = (band == 1)
    results = [(shape(geom), 1) for geom, val in shapes(band, mask=mask_vals, transform=out_transform)]

    if not results:
        return gpd.GeoDataFrame(columns=["value", "geometry"], geometry=[], crs=None)
    geoms, vals = zip(*results)
    return gpd.GeoDataFrame({"value": vals}, geometry=list(geoms), crs=None)

def compute_fragmentation_metrics(polygon_df, grid_area=14400, edge_depth=2):
    """
    Compute stable fragmentation metrics for a grid:
      - LSI: Landscape Shape Index
      - CAI_AM: Area-Weighted Mean Core Area Index (percent)
    
    Parameters:
        polygon_df: GeoDataFrame with patch polygons
        grid_area: float, area of the grid (default 120*120)
        edge_depth: float, buffer distance in meters to define "core"
    
    Returns:
        pd.Series with keys: 'LSI', 'CAI_AM'
    """
    if polygon_df.empty:
        return pd.Series({"LSI": 0, "CAI_AM": 0})

    # Compute total perimeter and area
    areas = polygon_df.geometry.area
    perimeters = polygon_df.geometry.length
    total_area = areas.sum()
    total_perimeter = perimeters.sum()

    # Landscape Shape Index (LSI)
    lsi = total_perimeter / (4 * np.sqrt(total_area)) if total_area > 0 else 0

    # Compute core areas for each patch
    cai_values = []
    weights = []
    for patch in polygon_df.geometry:
        patch_area = patch.area
        if patch_area <= 0:
            continue

        # inward buffer to define core
        core = patch.buffer(-edge_depth)
        core_area = core.area if not core.is_empty else 0.0

        cai_patch = (core_area / patch_area) * 100  # percent of patch that is core
        cai_values.append(cai_patch)
        weights.append(patch_area)

    # Area-weighted mean core area index
    cai_am = np.average(cai_values, weights=weights) if cai_values else 0

    return pd.Series({"LSI": lsi, "CAI_AM": cai_am})

def process_grid(args):
    """Process a single 120m grid for LiDAR canopy analysis"""
    city, raster_path, grid_geom, grid_id, epsg = args

    result = {
        "grid_id": grid_id,
        "city": city
    }

    # Compute centroid in lat/lon
    grid_centroid = gpd.GeoSeries([grid_geom], crs=epsg).to_crs("EPSG:4326").geometry[0].centroid
    result["Longitude"] = grid_centroid.x
    result["Latitude"] = grid_centroid.y

    try:
        with rasterio.open(raster_path) as src:
            out_image, out_transform = mask(src, [grid_geom], crop=True)

            # Compute CHM height stats for this grid
            chm_stats = compute_chm_stats(raster_path, grid_geom)
            result.update(chm_stats)

            # Create binary raster: canopy >= 2 m
            binary_canopy = (out_image[0] >= 2).astype(np.uint8)
            
            # Compute CLUMPY
            clumpy_value = compute_clumpy(binary_canopy)
            result["CLUMPY"] = clumpy_value
            
            # Convert binary raster to polygons for fragmentation metrics
            # Use the binary_canopy array, not out_image
            polygons = raster_to_polygons(binary_canopy[np.newaxis, :, :], out_transform, nodata=None)

            if polygons.empty:
                # Keep the CHM stats that were already computed
                result.update({
                    "total_m2": 0, 
                    "patch_count": 0, 
                    "total_perimeter": 0,
                    "percent_cover": 0, 
                    "mean_patch_size": 0,
                    "area_cv": 0, 
                    "perimeter_cv": 0,
                    "CAI_AM": 0, 
                    "LSI": 0
                })
                # Don't overwrite CLUMPY and height stats - they're already set
            else:
                polygons.set_crs(src.crs, inplace=True)
                polygons = polygons.to_crs(epsg)
                clipped = gpd.overlay(polygons, gpd.GeoDataFrame(geometry=[grid_geom], crs=epsg), how="intersection")
                clipped["m2"] = clipped.geometry.area
                clipped["perimeter"] = clipped.geometry.length

                total_m2 = clipped["m2"].sum()
                poly_ct = len(clipped)

                result.update({
                    "total_m2": total_m2,
                    "patch_count": poly_ct,
                    "total_perimeter": clipped["perimeter"].sum(),
                    "percent_cover": (total_m2 / 14400) * 100,
                    "mean_patch_size": total_m2 / poly_ct if poly_ct else 0,
                    "area_cv": clipped["m2"].std() / clipped["m2"].mean() if clipped["m2"].mean() > 0 else 0,
                    "perimeter_cv": clipped["perimeter"].std() / clipped["perimeter"].mean() if clipped["perimeter"].mean() > 0 else 0
                })
                
                # Compute fragmentation metrics
                frag_metrics = compute_fragmentation_metrics(clipped, grid_area=14400)
                result.update(frag_metrics)

    except Exception as e:
        print(f"Error processing {city} grid {grid_id}: {e}")
        # Set all metrics to 0 on error
        result.update({
            "total_m2": 0, "patch_count": 0, "total_perimeter": 0,
            "percent_cover": 0, "mean_patch_size": 0,
            "area_cv": 0, "perimeter_cv": 0,
            "CAI_AM": 0, "LSI": 0,
            "CLUMPY": 0,
            "min_height_0": 0, "max_height_0": 0, "mean_height_0": 0, 
            "median_height_0": 0, "std_height_0": 0,
            "min_height_2": 0, "max_height_2": 0, "mean_height_2": 0, 
            "median_height_2": 0, "std_height_2": 0
        })
    
    return result

def main():
    print(f"Building processing tasks for 120m grids...")
    tasks = []

    for city, config in rasters.items():
        epsg = config["epsg"]
        raster = config["LiDAR"]
        bayan = config["bayan"].to_crs(epsg)
        bayan["grid_id"] = ((bayan.geometry.centroid.x // 120).astype(int).astype(str) + "_" +
                            (bayan.geometry.centroid.y // 120).astype(int).astype(str))

        for _, row in bayan.iterrows():
            grid_geom = row.geometry
            grid_id = row.grid_id
            tasks.append((city, raster, grid_geom, grid_id, epsg))

    print(f"Processing {len(tasks)} tasks using {N_CPUS} CPUs...")

    with Pool(processes=N_CPUS) as pool:
        results = list(tqdm(pool.imap_unordered(process_grid, tasks), total=len(tasks), desc="Processing LiDAR 120m grids"))

    print("Saving results...")
    df = pd.DataFrame(results)
    cols = ["city", "grid_id", "Latitude", "Longitude",
            "total_m2", "percent_cover", "patch_count",
            "mean_patch_size", "total_perimeter",
            "area_cv", "perimeter_cv", "CAI_AM", "LSI",
            "CLUMPY",
            "min_height_0","max_height_0","mean_height_0","median_height_0","std_height_0",
            "min_height_2","max_height_2","mean_height_2","median_height_2","std_height_2"]
    df = df[cols]

    output_path = os.path.join(OUT_DIR, "LiDAR_120m_Grid_Canopy_Metrics.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    print(f"Total records: {len(df)}")

if __name__ == "__main__":
    main()

import opensimplex
import numpy as np
import matplotlib.pyplot as plt
import numba
from skimage.feature import peak_local_max
import random



def _generate_fast_noise(width, height, scale, seed, octaves, persistence, lacunarity):
    world = np.empty((height, width), dtype=np.float64)

    simplex = opensimplex.OpenSimplex(seed=seed)

    for i in range(height):
        for j in range(width):
            noise_value = 0.0
            frequency = 1.0
            amplitude = 1.0
            for _ in range(octaves):
                nx = i / scale * frequency
                ny = j / scale * frequency
                noise_value += simplex.noise2(nx, ny) * amplitude

                amplitude *= persistence
                frequency *= lacunarity

            world[i, j] = noise_value

    return world



def generate_canopy_map_fast(width=100, height=100, clustering=50, canopy_cover=0.5, seed=None):
    """
    Generates a canopy map using OpenSimplex
    This creates large, blob-like canopy shapes.
    """
    if clustering <= 0:
        rng = np.random.default_rng(seed)
        random_grid = rng.random((height, width))
        return (random_grid < canopy_cover).astype(np.int8)

    base_seed = seed if seed is not None else np.random.randint(0, np.iinfo(np.int32).max)

    world = _generate_fast_noise(
        width, height, clustering, base_seed,
        octaves=6, persistence=0.5, lacunarity=2.0
    )

    threshold = np.percentile(world, (1 - canopy_cover) * 100)
    return (world > threshold).astype(np.int8)


def generate_canopy_map_layered_noise(width=100, height=100, base_clustering=80, detail_clustering=20,
        detail_weight=0.3, density=0.6, tree_radius_range=(5, 10), seed=None):
    """
    Generates a canopy map by layering multiple OpenSimplex noise fields and placing circular trees.
    """
    rng = np.random.default_rng(seed)
    base_seed = int(rng.integers(0, np.iinfo(np.int32).max))

    # Generate both noise fields using the fast, compiled function
    base_noise = _generate_fast_noise(width, height, base_clustering, base_seed, 6, 0.5, 2.0)
    detail_noise = _generate_fast_noise(width, height, detail_clustering, base_seed + 1, 6, 0.5, 2.0)

    # Normalize noise to 0-1 range
    base_noise = (base_noise - base_noise.min()) / (base_noise.max() - base_noise.min())
    detail_noise = (detail_noise - detail_noise.min()) / (detail_noise.max() - detail_noise.min())

    combined_noise = base_noise * (1 - detail_weight) + detail_noise * detail_weight

    min_rad, max_rad = tree_radius_range
    coordinates = peak_local_max(combined_noise, min_distance=min_rad, threshold_abs=density)

    canopy_map = np.zeros((height, width), dtype=np.int8)
    y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width))

    for y_tree, x_tree in coordinates:
        radius = rng.uniform(min_rad, max_rad)
        # Corrected distance calculation for meshgrid
        distance_sq = (x_coords - x_tree) ** 2 + (y_coords - y_tree) ** 2
        canopy_map[distance_sq <= radius ** 2] = 1

    return canopy_map


class CanopyStamp:
    """A simple class to hold a pre-generated canopy shape and its area."""

    def __init__(self, shape):
        self.shape = shape
        self.area = np.sum(shape)
        self.height, self.width = shape.shape


def create_stamp_library(num_stamps, radius_range, shape_complexity, seed=None):
    """
    Generates a library of pre-rendered, irregular canopy shapes (stamps)
    using the opensimplex library.
    """
    print(f"Generating {num_stamps} canopy stamps for the library...")
    rng = np.random.default_rng(seed)
    base_seed = int(rng.integers(0, np.iinfo(np.int32).max))
    library = []

    # Prevent division by zero, which could still cause issues.
    if shape_complexity <= 0:
        shape_complexity = 1e-9

    for i in range(num_stamps):
        # Initialize OpenSimplex with a unique seed for each stamp
        simplex = opensimplex.OpenSimplex(seed=base_seed + i)
        radius = rng.uniform(*radius_range)
        stamp_dim = int(radius * 2)
        if stamp_dim == 0:
            continue

        tree_noise = np.zeros((stamp_dim, stamp_dim))
        for y in range(stamp_dim):
            for x in range(stamp_dim):
                # Manually implement octaves for detailed noise
                frequency = 1.0
                amplitude = 1.0
                noise_value = 0
                for _ in range(4):  # Number of octaves
                    nx = x / shape_complexity * frequency
                    ny = y / shape_complexity * frequency
                    noise_value += simplex.noise2(nx, ny) * amplitude
                    amplitude *= 0.5
                    frequency *= 2.0
                tree_noise[y, x] = (noise_value + 1) / 2

        center = stamp_dim / 2.0
        y_indices, x_indices = np.ogrid[:stamp_dim, :stamp_dim]
        dist_sq = (y_indices - center) ** 2 + (x_indices - center) ** 2
        falloff = np.maximum(0, 1 - np.sqrt(dist_sq) / radius)
        shaped_noise = tree_noise * falloff
        canopy_shape = (shaped_noise > 0.4).astype(np.int8)
        library.append(CanopyStamp(canopy_shape))
    return library


def generate_forest_analytically(width, height, target_cover, stamp_library, generator_params, seed=None):
    """
    Generates a forest by analytically placing tree stamps until a target cover is met.
    """
    rng = np.random.default_rng(seed)
    base_seed = int(rng.integers(0, np.iinfo(np.int32).max))
    base_clustering = generator_params.get("base_clustering", 120)
    min_rad, _ = generator_params.get("radius_range", (15, 25))

    print(f"\nAnalytically generating map for target cover: {target_cover:.1%}")

    # 1. Generate the base noise field using the fast function
    location_noise = _generate_fast_noise(width, height, base_clustering, base_seed, 6, 0.5, 2.0)
    location_noise = (location_noise - location_noise.min()) / (location_noise.max() - location_noise.min())

    # 2. Find and rank potential tree locations
    min_dist = int(min_rad / 2)
    all_peaks = peak_local_max(location_noise, min_distance=min_dist, threshold_rel=0.01)
    print(f"Found {len(all_peaks)} potential tree locations with min_distance={min_dist}.")
    peak_values = location_noise[all_peaks[:, 0], all_peaks[:, 1]]
    ranked_peaks = all_peaks[np.argsort(peak_values)[::-1]]

    # 3. Place trees from the stamp library one by one
    canopy_map = np.zeros((height, width), dtype=np.int8)
    total_pixels = width * height
    covered_pixels = 0
    for y_center, x_center in ranked_peaks:
        if covered_pixels / total_pixels >= target_cover:
            print(f"Target reached. Final cover: {covered_pixels / total_pixels:.2%}")
            break

        stamp = random.choice(stamp_library)
        h, w = stamp.height, stamp.width
        tl_y, tl_x = y_center - h // 2, x_center - w // 2
        map_y_start, map_y_end = max(0, tl_y), min(height, tl_y + h)
        map_x_start, map_x_end = max(0, tl_x), min(width, tl_x + w)
        stamp_y_start = max(0, -tl_y)
        stamp_y_end = stamp_y_start + (map_y_end - map_y_start)
        stamp_x_start = max(0, -tl_x)
        stamp_x_end = stamp_x_start + (map_x_end - map_x_start)

        if map_y_start >= map_y_end or map_x_start >= map_x_end:
            continue

        map_slice = canopy_map[map_y_start:map_y_end, map_x_start:map_x_end]
        stamp_slice = stamp.shape[stamp_y_start:stamp_y_end, stamp_x_start:stamp_x_end]

        new_pixels = np.sum(stamp_slice & (1 - map_slice))
        covered_pixels += new_pixels
        map_slice |= stamp_slice
    else:
        print(f"Warning: Ran out of locations. Final cover: {np.mean(canopy_map):.2%}")
    return canopy_map


# --- Visualization Example ---
if __name__ == '__main__':
    map_size = 500
    seed = 42
    target_canopy_percentage = 0.1

    forest_style_params = {
        "base_clustering":  60,
        "radius_range":     (10, 25),
        "shape_complexity": 5
    }
    stamp_library_params = {
        "num_stamps":       300,
        "radius_range":     forest_style_params["radius_range"],
        "shape_complexity": forest_style_params["shape_complexity"]
    }

    # --- Generate the stamp library first ---
    stamp_library = create_stamp_library(**stamp_library_params, seed=seed)

    # --- Generate Maps using the new, safe functions ---
    blob_map = generate_canopy_map_fast(
        width=map_size, height=map_size, clustering=80, canopy_cover=target_canopy_percentage, seed=seed
    )
    circular_trees_map = generate_canopy_map_layered_noise(
        width=map_size, height=map_size, base_clustering=120, detail_clustering=30,
        detail_weight=0.4, density=0.55, tree_radius_range=(8, 15), seed=seed
    )
    irregular_trees_map = generate_forest_analytically(
        width=map_size, height=map_size,
        target_cover=target_canopy_percentage,
        stamp_library=stamp_library,
        generator_params=forest_style_params,
        seed=seed
    )

    # --- Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Comparison of Canopy Generation Methods (using OpenSimplex)", fontsize=16, fontweight='bold')

    axes[0].imshow(blob_map, cmap='Greens', interpolation='nearest')
    axes[0].set_title(f"Fast Blob Method\nCover: {np.mean(blob_map):.2%}", fontsize=12)

    axes[1].imshow(circular_trees_map, cmap='Greens', interpolation='nearest')
    axes[1].set_title(f"Circular Trees Method\nCover: {np.mean(circular_trees_map):.2%}", fontsize=12)

    axes[2].imshow(irregular_trees_map, cmap='Greens', interpolation='nearest')
    axes[2].set_title(f"Stamp Library Method\nCover: {np.mean(irregular_trees_map):.2%}", fontsize=12)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# This script answers the question:
# How does the reliability of my estimate change as I increase the number of sample points?
# Results: Paragraph 1, Figure 1

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from canopy_model import generate_canopy_map
from matplotlib.font_manager import FontProperties

## ------------------------------------------------- DEFINE SIMULATION -------------------------------------------------
#region

# Analysis configuration
MIN_SAMPLE_SIZE = 50 # Minimum number of sample points
MAX_SAMPLE_SIZE = 2000 # Maximum number of sample points
INCREMENT_SIZE = 50 # Increment between the number of sample points

MIN_CANOPY_COVER = 0 # Minimum total canopy cover percentages
MAX_CANOPY_COVER = 60 # Maximum total canopy cover percentages
INCREMENT_CANOPY_COVER = 5 # Increment between the canopy cover percentages

MIN_CLUSTERING = 0 # Minimum level of clustering
MAX_CLUSTERING = 100 # Maximum level of clustering
INCREMENT_CLUSTERING = 25 # Increment between the clustering levels

trials = 10000 # Number of trials (bootstraps)
agreement_tolerance = 1 # Proximity to true canopy cover considered acceptable (%)

# AOI definitions (in pixels, assuming 30 cm resolution)
AOIS = {
    "Neighbourhood (20 km²)": (14907, 14907), # Neighbourhood (20 km²) AOI
    "City (400 km²)": (66667, 66667), # City (400 km²) AOI
    "Region (3,000 km²)": (182574, 182574), # Region / County (3,000 km²) AOI
}

#endregion

## -------------------------------------------------- DEFINE FUNCTIONS -------------------------------------------------
#region

# Performs a random sample run and returns the single estimated cover
def get_single_estimate(canopy_map, num_samples):

    # Handle edge case where a map has 0% cover to avoid divide-by-zero if num_samples is 0
    if num_samples == 0:
        return 0
    height, width = canopy_map.shape
    sample_x = np.random.randint(0, width, num_samples)
    sample_y = np.random.randint(0, height, num_samples)
    canopy_hits = np.sum(canopy_map[sample_y, sample_x])
    return (canopy_hits / num_samples) * 100

#  Analyzes the relationship between sample size and agreement for a given canopy cover
def analyze_sample_size_agreement(config, target_canopy_cover, width, height):

    sample_sizes = config['SAMPLE_SIZES_TO_TEST']
    num_trials = config['NUM_TRIALS_PER_SIZE']
    tolerance = config['AGREEMENT_TOLERANCE']

    # Generate a representative map with the specified target canopy cover
    canopy_map = generate_canopy_map(
        width=width, height=height, clustering=65, canopy_cover=target_canopy_cover, seed=42
    )

    true_cover = np.mean(canopy_map) * 100

    agreement_results = []

    with multiprocessing.Pool(processes=4) as pool:
        for n_samples in sample_sizes:
            task_args = [(canopy_map, n_samples)] * num_trials
            estimates = pool.starmap(get_single_estimate, task_args)

            in_agreement_count = np.sum(np.abs(np.array(estimates) - true_cover) <= tolerance)
            agreement_percent = (in_agreement_count / num_trials) * 100
            agreement_results.append(agreement_percent)

    return sample_sizes, agreement_results, true_cover

#endregion

## ---------------------------------------------- MAIN EXECUTION: Figure 1 ---------------------------------------------
#region

if __name__ == "__main__":

    # Configuration
    canopy_cover_levels_to_test = np.arange(MIN_CANOPY_COVER, MAX_CANOPY_COVER + 1, INCREMENT_CANOPY_COVER) / 100
    sample_sizes_to_test = np.arange(MIN_SAMPLE_SIZE, MAX_SAMPLE_SIZE + 1, INCREMENT_SIZE)
    clustering_levels = range(MIN_CLUSTERING, MAX_CLUSTERING + 1, INCREMENT_CLUSTERING)

    CONFIG = {
        "SAMPLE_SIZES_TO_TEST": sample_sizes_to_test,
        "NUM_TRIALS_PER_SIZE": trials,
        "AGREEMENT_TOLERANCE": agreement_tolerance,
    }

    # Prepare 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    plt.style.use('seaborn-v0_8-whitegrid')

    # Flatten axes for easy indexing
    axes = axes.flatten()

    print(f"\nRunning simulations for:")

    for i, (aoi_name, (width, height)) in enumerate(AOIS.items()):
        ax = axes[i]
        print(f" - {aoi_name}")

        for clustering in clustering_levels:
            required_sample_sizes = []

            for cover_level in canopy_cover_levels_to_test:
                canopy_map = generate_canopy_map(
                    width=width,
                    height=height,
                    clustering=clustering,
                    canopy_cover=cover_level,
                    seed=42
                )

                true_cover = np.mean(canopy_map) * 100
                found = False

                for n_samples in sample_sizes_to_test:
                    task_args = [(canopy_map, n_samples)] * trials
                    estimates = [get_single_estimate(canopy_map, n_samples) for _ in range(trials)]

                    in_agreement = np.sum(np.abs(np.array(estimates) - true_cover) <= agreement_tolerance)
                    agreement = in_agreement / trials * 100

                    if agreement >= 95:
                        required_sample_sizes.append(n_samples)
                        found = True
                        break

                if not found:
                    required_sample_sizes.append(np.nan)

            # Switch axes: X = sample size, Y = canopy cover
            ax.plot(required_sample_sizes, canopy_cover_levels_to_test * 100, marker='o',
                    label=f"Clustering {clustering}")

        ax.set_title(aoi_name, fontsize=19, fontweight='bold')
        ax.set_xlabel('Sample Points to Reach 95% Agreeance', fontsize=17, labelpad=10)
        ax.set_ylabel('Canopy Cover (%)', fontsize=17)
        ax.tick_params(axis='both', labelsize=15)
        ax.minorticks_on()
        ax.tick_params(axis='both', which='major', length=5, width=1.5)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 65)
        ax.grid(False)

        # Make the plot border (spines) black
        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(1)

        # Remove 0 from x-axis tick labels
        xticks = ax.get_xticks()
        ax.set_xticks([tick for tick in xticks if tick != 0])

    # Remove y-axis label from top-right
    axes[1].set_ylabel('')

    # Use bottom-right for legend
    axes[3].axis('off')
    handles, labels = axes[0].get_legend_handles_labels()
    bold_font = FontProperties(weight='bold', size=18)
    legend = axes[3].legend(handles, labels, title="Clustering Level", loc='center',
                            fontsize=16, title_fontproperties=bold_font, frameon=True, ncol=2)

    # Adjust layout and spacing
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.25, wspace=0.08)  # Increased column spacing with wspace

    plt.show()

#endregion

# Canopy Sampling
import numpy as np





def simulate_random_sampling(canopy_map, num_samples, with_coordinates=False):
    height, width = canopy_map.shape

    # Generate random coordinates for each sample point
    sample_x = np.random.randint(0, width, num_samples)
    sample_y = np.random.randint(0, height, num_samples)

    # Check for canopy presence at each sample point and sum the "hits"
    canopy_hits = np.sum(canopy_map[sample_y, sample_x])

    # Calculate the estimated proportion of cover
    estimated_proportion = canopy_hits / num_samples

    if with_coordinates:
        return estimated_proportion * 100, sample_x, sample_y
    else:
        return estimated_proportion * 100


def get_single_estimate(canopy_map, num_samples):
    """Performs a random sample run and returns the single estimated cover."""
    if num_samples <= 0:
        return 0
    height, width = canopy_map.shape
    sample_x = np.random.randint(0, width, num_samples)
    sample_y = np.random.randint(0, height, num_samples)
    canopy_hits = np.sum(canopy_map[sample_y, sample_x])
    return (canopy_hits / num_samples) * 100
