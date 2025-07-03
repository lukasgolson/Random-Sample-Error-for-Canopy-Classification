import numpy as np
import noise
from matplotlib import pyplot as plt
from tqdm import tqdm


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
    for cluster_val in tqdm(clustering_values, desc="Calculating Moran's I"):
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
    # pip install numpy noise matplotlib tqdm
    run_simulation_and_plot()
