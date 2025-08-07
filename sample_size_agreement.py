# This script answers the question:
# How does the reliability of my estimate change as I increase the number of sample points?
# Results: Paragraph 1, Figure 1

# Notes
# Each pixel represents a 30 cm x 30 cm cell

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
from canopy_model import generate_canopy_map

## ------------------------------------------------- DEFINE SIMULATION -------------------------------------------------
#region

# Analysis configuration
MIN_SAMPLE_SIZE = 10 # Minimum number of sample points
MAX_SAMPLE_SIZE = 50 # Maximum number of sample points
INCREMENT_SIZE = 25 # Increment between the number of sample points

MIN_CANOPY_COVER = 0 # Minimum total canopy cover percentages
MAX_CANOPY_COVER = 60 # Maximum total canopy cover percentages
INCREMENT_CANOPY_COVER = 5 # Increment between the canopy cover percentages

trials = 10000 # Number of trials (bootstraps)
agreement_tolerance = 2.5 # Proximity to true canopy cover considered acceptable (%)

# AOI definitions (in pixels, assuming 30 cm resolution)
AOIS = {
    "Neighbourhood": (14907, 14907), # Neighbourhood (20 km²) AOI
    "City": (66667, 66667), # City (400 km²) AOI
    "Region": (182574, 182574), # Region / County (3,000 km²) AOI
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

## -------------------------------- MAIN EXECUTION: Run Sample Size Reliability Analysis -------------------------------
#region

if __name__ == "__main__":
    # Configuration from Define Simulation block
    canopy_cover_levels_to_test = np.arange(MIN_CANOPY_COVER, MAX_CANOPY_COVER + 1, INCREMENT_CANOPY_COVER) / 100.0
    sample_sizes_to_test = np.arange(MIN_SAMPLE_SIZE, MAX_SAMPLE_SIZE + 1, INCREMENT_SIZE)

    CONFIG = {
        "SAMPLE_SIZES_TO_TEST": sample_sizes_to_test,
        "NUM_TRIALS_PER_SIZE": trials,
        "AGREEMENT_TOLERANCE": agreement_tolerance,
    }

    for aoi_name, (width, height) in AOIS.items():
        print(f"\n\n=== Running simulations for {aoi_name} ===")

        fig, ax = plt.subplots(figsize=(12, 8))
        plt.style.use('seaborn-v0_8-whitegrid')

        for cover_level in tqdm(canopy_cover_levels_to_test, desc=f"{aoi_name} Progress"):
            print(f"--- Canopy Cover: {cover_level:.0%} ---")
            sample_sizes, agreement_percents, actual_cover = analyze_sample_size_agreement(
                CONFIG, cover_level, width, height
            )

            ax.plot(sample_sizes, agreement_percents, marker='o', linestyle='-', markersize=4,
                    label=f'True Cover: {actual_cover:.1f}%')

        # Finalize visualization for this AOI
        ax.axhline(y=95, color='r', linestyle='--', label='95% Confidence Target')
        ax.set_xlabel('Number of Sample Points', fontsize=12)
        ax.set_ylabel(f'Proportion of Estimates within ±{CONFIG["AGREEMENT_TOLERANCE"]}% of the True Canopy Cover', fontsize=12)
        ax.set_ylim(0, 105)
        ax.grid(False)
        ax.legend(title="Canopy Cover Level", bbox_to_anchor=(1.04, 1), loc="upper left")
        ax.set_title(f"Sampling Agreement for {aoi_name}", fontsize=14, fontweight='bold')

        fig.tight_layout()
        plt.show()

#endregion
