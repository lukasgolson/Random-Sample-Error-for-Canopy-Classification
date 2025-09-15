import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the csv
df = pd.read_csv('sample_points.csv')

def_calculations
  # Error calculations
  se_standard = np.sqrt(p * (1 - p) / sample_points) # Standard (i-Tree) SE calculation
  se_neff = np.sqrt(p * (1 - p) / neff) # N effective (Moran's i) SE calculation
  
  # Margin calculations
  margin_standard = 1.96 * se_standard # Margin of error derived from standard SE calculation
  margin_neff = 1.96 * se_neff # Margin of error derived from N effective SE calculation
  
  # LB and UB standards
  lb_standard = max(0, p - margin_standard)
  ub_standard = min(1, p + margin_standard)
  lb_neff = max(0, p - margin_neff)
  ub_neff = min(1, p + margin_neff)

  return(se_standard, se_neff, margin_standard, margin_neff, lb_standard, ub_standard, lb_neff, ub_neff)

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

# figure3_generator.py
#
# This script generates Figure 3 from the manuscript, comparing the standard
# standard error (SE) with the spatially adjusted standard error (SE_adjusted)
# across varying levels of spatial clustering. The simulation is run for three
# different Area of Interest (AOI) sizes: Neighbourhood, City, and Region.

import math
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

# --- Assuming these functions are in their respective files ---
# It's good practice to have your project structured as a package,
# but for a standalone script, we ensure the necessary functions are available.
from canopy_model import generate_canopy_map, calculate_morans_i
from canopy_sampling import get_single_estimate

# --- Helper Functions for SE Calculation ---

def calculate_se(p, n_samples):
    """
    Calculates the standard formula for Standard Error (SE).
    Args:
        p (float): The proportion of canopy cover (0 to 1).
        n_samples (int): The total number of sample points.
    Returns:
        float: The calculated standard error.
    """
    if n_samples == 0 or p < 0 or p > 1:
        return 0

    value = p * (1 - p) / n_samples
    return math.sqrt(max(0, value))

def calculate_se_adjusted(p, n_samples, morans_i):
    """
    Calculates the spatially adjusted Standard Error using Moran's I.
    This adjustment accounts for the lack of independence in spatial data.
    Args:
        p (float): The proportion of canopy cover (0 to 1).
        n_samples (int): The total number of sample points.
        morans_i (float): The calculated Moran's I for the landscape.
    Returns:
        float: The spatially adjusted standard error.
    """
    if n_samples <= 1 or morans_i is None or np.isnan(morans_i):
        # Fallback to standard SE if adjustment is not possible
        return calculate_se(p, n_samples)

    # Calculate the effective sample size (N_effective)
    # Clamp Moran's I to prevent N_eff from becoming negative or zero with noisy I values.
    # A rho >= 1/(N-1) is mathematically problematic.
    max_rho = 1.0 / (n_samples - 1)
    clamped_rho = max(morans_i, -max_rho) # Ensure denominator is positive

    n_effective = n_samples / (1 + (n_samples - 1) * clamped_rho)

    if n_effective <= 0:
        # If N_eff is still non-positive, adjustment is unstable; return standard SE.
        return calculate_se(p, n_samples)

    # Calculate adjusted SE using the effective sample size
    value = p * (1 - p) / n_effective
    return math.sqrt(max(0.0, value))


# --- Simulation Runner ---

def run_simulation_for_aoi(aoi_config):
    aoi_name, config = aoi_config
    print(f"--- Starting simulation for: {aoi_name} ---")

    # Unpack configuration
    width, height = config["DIMENSIONS"]
    clustering_levels = config["CLUSTERING_LEVELS"]
    n_samples = config["NUM_SAMPLES"]
    canopy_cover_target = config["CANOPY_COVER"]
    num_runs_per_level = config["NUM_RUNS_PER_LEVEL"]

    results = []

    for cluster_val in clustering_levels:
        # Store metrics for each run at this clustering level to average later
        run_morans_i = []
        run_se = []
        run_se_adjusted = []

        for i in range(num_runs_per_level):
            # 1. Generate a unique map for each run
            canopy_map = generate_canopy_map(
                width=width,
                height=height,
                clustering=cluster_val,
                canopy_cover=canopy_cover_target,
                seed=None # Use a different seed for each map
            )

            # 2. Calculate the map's spatial autocorrelation
            morans_i = calculate_morans_i(canopy_map)
            if morans_i is None or np.isnan(morans_i):
                continue # Skip if Moran's I is undefined

            # 3. Get a single sample-based estimate of canopy proportion
            p_estimated = get_single_estimate(canopy_map, n_samples) / 100.0

            # 4. Calculate both SE values based on the estimate
            se_standard = calculate_se(p_estimated, n_samples)
            se_adj = calculate_se_adjusted(p_estimated, n_samples, morans_i)

            # Store results for this run
            run_morans_i.append(morans_i)
            run_se.append(se_standard)
            run_se_adjusted.append(se_adj)

        # Average the results for the current clustering level
        if run_morans_i:
            avg_morans_i = np.mean(run_morans_i)
            avg_se = np.mean(run_se)
            avg_se_adj = np.mean(run_se_adjusted)
            results.append((avg_morans_i, avg_se, avg_se_adj))

    # Sort results by Moran's I for clean plotting
    results.sort()
    return aoi_name, results


# --- Main Execution Block ---

if __name__ == "__main__":
    # --- GLOBAL SIMULATION CONFIGURATION ---
    # Using smaller dimensions for faster execution. The relative effect holds.
    # The original pixel dimensions from sample_size_agreement.py are very large
    # and would make map generation extremely slow.
    AOI_DEFINITIONS = {
        "Neighbourhood": (500, 500),
        "City": (1000, 1000),
        "Region/County": (1500, 1500),
    }

    SIM_CONFIG = {
        "CLUSTERING_LEVELS": np.linspace(1, 80, 25), # Range of clustering to test
        "NUM_SAMPLES": 10000, # Fixed number of samples as per manuscript
        "CANOPY_COVER": 0.4, # Fixed canopy cover for consistency
        "NUM_RUNS_PER_LEVEL": 10 # Number of maps to average per clustering level
    }

    # Prepare configurations for multiprocessing
    tasks = []
    for name, dims in AOI_DEFINITIONS.items():
        aoi_task_config = SIM_CONFIG.copy()
        aoi_task_config["DIMENSIONS"] = dims
        tasks.append((name, aoi_task_config))

    # Run simulations in parallel
    with multiprocessing.Pool(processes=len(tasks)) as pool:
        all_results = pool.map(run_simulation_for_aoi, tasks)

    # --- PLOTTING ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(21, 6), sharey=True)
    fig.suptitle("Comparison of Standard and Adjusted Standard Error (SE)", fontsize=18, fontweight='bold')

    # Plot results for each AOI
    for i, (aoi_name, results_list) in enumerate(all_results):
        if not results_list:
            print(f"No results to plot for {aoi_name}")
            continue

        # Unzip the results for plotting
        morans_i_vals, se_vals, se_adjusted_vals = zip(*results_list)

        ax = axes[i]
        ax.plot(morans_i_vals, se_vals, 'o-', label='Standard SE', color='cornflowerblue', markersize=5)
        ax.plot(morans_i_vals, se_adjusted_vals, 's-', label='Adjusted SE', color='crimson', markersize=5)

        ax.set_title(f"AOI: {aoi_name}", fontsize=14)
        ax.set_xlabel("Spatial Autocorrelation (Moran's I)", fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend()
        ax.set_xlim(0, 1.0) # Moran's I for canopy is typically positive

    # Set a shared Y-axis label
    axes[0].set_ylabel("Standard Error", fontsize=12)

    # Adjust layout and show the plot
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust for suptitle
    plt.show()
