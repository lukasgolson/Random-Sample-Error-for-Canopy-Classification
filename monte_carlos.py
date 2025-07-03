# This script answers the question:
# For a fixed sample size, how does the distribution of my estimates change
# as the spatial clustering of the landscape changes?


import numpy as np
import matplotlib.pyplot as plt
from canopy_model import generate_canopy_map
import multiprocessing
from tqdm import tqdm

from canopy_sampling import calculate_direct_cover, simulate_random_sampling


def calculate_itree_stats(n_hits, n_total):
    """ Calculates the proportion of hits, standard error, and confidence intervals. Code from ArbMarta"""
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


def process_one_map(config, seed):
    """ Processes a single map with the given configuration and seed."""
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


def run_monte_carlo(config):
    """ Runs the Monte Carlo simulation for multiple maps with multiprocessing."""
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


if __name__ == "__main__":
    CONFIG = {
        "MAP_WIDTH": 100,
        "MAP_HEIGHT": 100,
        "NUM_SAMPLES_PER_SET": 100,
        "NUM_MAPS": 2000,
        "NUM_SAMPLE_SETS_PER_MAP": 100,
        "CLUSTERING": 50,
    }

    clustering_levels_to_test = np.arange(0, 101, 10)
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