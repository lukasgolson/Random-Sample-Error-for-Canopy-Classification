import noise
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing

## -------------------------------------------------- DEFINE FUNCTIONS -------------------------------------------------
#region

def calculate_direct_cover(canopy_map):
    true_proportion = np.mean(canopy_map)
    return true_proportion * 100

def simulate_random_sampling(canopy_map, num_samples, with_coordinates=False):
    height, width = canopy_map.shape

    # Generate random coordinates for each sample point
    sample_x = np.random.randint(0, width, num_samples)
    sample_y = np.random.randint(0, height, num_samples)

    # Check for canopy presence at each sample point and sum the number of trees
    canopy_hits = np.sum(canopy_map[sample_y, sample_x])

    # Calculate the estimated proportion of cover
    estimated_proportion = canopy_hits / num_samples

    if with_coordinates:
        return estimated_proportion * 100, sample_x, sample_y
    else:
        return estimated_proportion * 100

# Generates a binary raster simulating a canopy presence map
def generate_canopy_map(width=100, height=100, clustering=50, canopy_cover=0.5, seed=None):

    if not 0 <= clustering <= 100:
        raise ValueError("Clustering must be an integer between 0 and 100.")

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

# Calculates Moran's I for a 2D grid using Queen contiguity
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

# Calculates the proportion of hits, standard error, and confidence intervals
def calculate_itree_stats(n_hits, n_total):
    if n_total == 0: return 0, 0, 0, 0
    p = n_hits / n_total
    q = 1 - p
    if n_hits < 10:
        se = (n_hits ** 0.5) / n_total
    else:
        se = (p * q / n_total) ** 0.5
    lower_bound, upper_bound = None, None
    if n_total >= 30:
        margin = 1.96 * se
        lower_bound = max(0, p - margin)
        upper_bound = min(1, p + margin)
    return p, se, lower_bound, upper_bound

# Processes a single map with the given configuration and seed
def process_one_map(config, seed):
    clustering_level = config['CLUSTERING']
    map_width = config['MAP_WIDTH']
    map_height = config['MAP_HEIGHT']
    num_sample_sets = config['NUM_SAMPLE_SETS_PER_MAP']
    samples_per_set = config['NUM_SAMPLES_PER_SET']
    target_cover = np.random.uniform(0.45, 0.65)

    canopy_map = generate_canopy_map(width=map_width, height=map_height, clustering=clustering_level,
                                     canopy_cover=target_cover, seed=seed)
    true_cover = calculate_direct_cover(canopy_map)
    estimated_covers = [simulate_random_sampling(canopy_map, samples_per_set) for _ in range(num_sample_sets)]
    return true_cover, estimated_covers

#  Runs the Monte Carlo simulation for multiple maps with multiprocessing
def run_monte_carlo(config):
    num_maps = config['NUM_MAPS']
    task_args = [(config, i) for i in range(num_maps)]
    with multiprocessing.Pool(processes=4) as pool:
        results = pool.starmap(process_one_map, task_args)
    true_covers = []
    estimated_covers = []
    for true_cover, est_list in results:
        true_covers.append(true_cover)
        estimated_covers.extend(est_list)
    return np.array(true_covers), np.array(estimated_covers)

# Analyzes the relationship between sample size and agreement for a given canopy cover
def analyze_sample_size_agreement(config, target_canopy_cover):

    sample_sizes = config['SAMPLE_SIZES_TO_TEST']
    num_trials = config['NUM_TRIALS_PER_SIZE']
    tolerance = config['AGREEMENT_TOLERANCE']

    # Generate a representative map with the specified target canopy cover
    canopy_map = generate_canopy_map(
        width=200, height=200, clustering=65, canopy_cover=target_canopy_cover, seed=42
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

# endregion

## ------------------------------------------------- EXECUTE SIMULATION ------------------------------------------------
#region

# Generate a base map to analyze
canopy_map = generate_canopy_map(width=1024, height=1024, clustering=70, canopy_cover=0.55, seed=101)

num_samples = 200

# Calculate the true canopy cover using direct calculation and simulate random sampling
true_cover = calculate_direct_cover(canopy_map)
estimated_cover, sample_x, sample_y = simulate_random_sampling(canopy_map, num_samples, True)

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

#endregion

## ------------------------------------------------- TEST VISUALIZATION ------------------------------------------------
#region

# Fraction of the area that should be covered by canopies
canopy_fraction = 0.5
model_seed = None

low_cluster_map = generate_canopy_map(clustering=5, canopy_cover=canopy_fraction, seed=model_seed)
medium_cluster_map = generate_canopy_map(clustering=25, canopy_cover=canopy_fraction, seed=model_seed)
high_cluster_map = generate_canopy_map(clustering=50, canopy_cover=canopy_fraction, seed=model_seed)

# Plotting logic
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Simulated Canopy Maps with Varying Clustering', fontsize=16)

# Colormap for visualization
cmap = 'Greens'

axes[0].imshow(low_cluster_map, cmap=cmap, interpolation='nearest')
axes[0].set_title('Clustering: 5')

axes[1].imshow(medium_cluster_map, cmap=cmap, interpolation='nearest')
axes[1].set_title('Clustering: 25')

axes[2].imshow(high_cluster_map, cmap=cmap, interpolation='nearest')
axes[2].set_title('Clustering: 50')

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# endregion

# Simulation execution
if __name__ == '__main__':
    run_simulation_and_plot()

# Monte Carlos
if __name__ == "__main__":
    CONFIG = {
        "MAP_WIDTH": 200,
        "MAP_HEIGHT": 200,
        "NUM_SAMPLES_PER_SET": 500,
        "NUM_MAPS": 2000,
        "NUM_SAMPLE_SETS_PER_MAP": 100,
        "CLUSTERING": -1,  # Placeholder for clustering level
    }

    clustering_levels_to_test = np.arange(0, 101, 5)  # We are getting an artifact at clustering = 0
    all_results = []

    # --- 1. Run all simulations first to gather data ---
    print("Step 1: Running all simulations to gather data...")
    for cluster_level in tqdm(clustering_levels_to_test, desc="Analyzing Clustering Levels"):
        CONFIG['CLUSTERING'] = cluster_level
        true_dist, estimated_dist = run_monte_carlo(CONFIG)
        all_results.append({
            'level': cluster_level,
            'true_dist': true_dist,
            'estimated_dist': estimated_dist
        })

    # --- 2. Find the global maximum Y-value for consistent scaling ---
    print("Step 2: Calculating global plot scale...")
    global_ymax = 0
    for result in all_results:
        # Calculate histogram densities, and update global_ymax if larger
        true_heights, _ = np.histogram(result['true_dist'], bins=50, density=True)
        est_heights, _ = np.histogram(result['estimated_dist'], bins=50, density=True)
        global_ymax = max(global_ymax, true_heights.max(), est_heights.max())

    # --- 3. Create the distribution plots with a consistent scale ---
    print("Step 3: Generating plots...")
    n_levels = len(all_results)
    n_cols = 3
    n_rows = (n_levels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4), constrained_layout=True)
    axes = axes.flatten()

    for idx, result in enumerate(all_results):
        ax = axes[idx]
        # Use the same bins for both histograms in a pair for direct comparison
        bins = np.histogram_bin_edges(np.concatenate([result['true_dist'], result['estimated_dist']]), bins=50)

        ax.hist(result['estimated_dist'], bins=bins, alpha=0.7, label='Estimated', density=True, color='skyblue')
        ax.hist(result['true_dist'], bins=bins, alpha=0.7, label='True', density=True, color='salmon')
        ax.axvline(np.mean(result['true_dist']), color='red', linestyle='--', lw=2)

        ax.set_title(f'Clustering Level: {result["level"]}')
        ax.set_xlim(20, 90)
        # Apply the global y-axis limit, with 10% padding
        ax.set_ylim(0, global_ymax * 1.1)

    for i in range(n_levels, len(axes)):
        axes[i].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    fig.suptitle('Distribution of Estimates per Clustering Level (Consistent Scale)', fontsize=18)
    print("\nDisplaying distribution plots...")
    plt.show()

    # --- 4. Create the summary plot ---
    summary_std_devs = [np.std(res['estimated_dist']) for res in all_results]
    plt.figure(figsize=(10, 6))
    plt.plot(clustering_levels_to_test, summary_std_devs, marker='o', linestyle='-', color='crimson')
    plt.title('Effect of Canopy Clustering on Sampling Precision', fontsize=16)
    plt.xlabel('Clustering Level', fontsize=12)
    plt.ylabel('Standard Deviation of Estimated Cover (%)', fontsize=12)
    plt.grid(True)
    print("Displaying summary plot...")
    plt.show()

if __name__ == "__main__":
    # --- ANALYSIS CONFIGURATION ---
    MIN_SAMPLE_SIZE = 10
    MAX_SAMPLE_SIZE = 750
    INCREMENT_SIZE = 25

    sample_sizes_to_test = np.arange(MIN_SAMPLE_SIZE, MAX_SAMPLE_SIZE + 1, INCREMENT_SIZE)

    CONFIG = {
        "SAMPLE_SIZES_TO_TEST": sample_sizes_to_test,
        "NUM_TRIALS_PER_SIZE": 10000,  # Lowered for faster multi-run execution
        "AGREEMENT_TOLERANCE": 2.5,
    }

    # --- Plotting Setup ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    # --- Run Analysis for Each Canopy Cover Level ---
    canopy_cover_levels_to_test = np.arange(0, 41, 5) / 100.0  # 0% to 40%

    for cover_level in tqdm(canopy_cover_levels_to_test, desc="Overall Progress"):
        print(f"\n--- Running analysis for Canopy Cover: {cover_level:.0%} ---")

        sample_sizes, agreement_percents, actual_cover = analyze_sample_size_agreement(CONFIG, cover_level)

        # Plot the result for this canopy cover level
        ax.plot(sample_sizes, agreement_percents, marker='o', linestyle='-', markersize=4,
                label=f'True Cover: {actual_cover:.1f}%')

    # --- Finalize Visualization ---
    ax.axhline(y=95, color='r', linestyle='--', label='95% Confidence Target')
    ax.set_title('Sample Size vs. Reliability for Varying Canopy Covers', fontsize=16)
    ax.set_xlabel('Number of Sample Points', fontsize=12)
    ax.set_ylabel(f'Agreement with True Cover (±{CONFIG["AGREEMENT_TOLERANCE"]}%)', fontsize=12)
    ax.set_ylim(0, 105)
    ax.legend(title="Canopy Cover Level", bbox_to_anchor=(1.04, 1), loc="upper left")

    fig.tight_layout()
    plt.show()
