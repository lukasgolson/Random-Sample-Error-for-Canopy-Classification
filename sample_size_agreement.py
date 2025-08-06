# This script answers the question:
# How does the reliability of my estimate change as I increase the number of sample points?
# Results: Paragraph 1, Figure 1

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
from canopy_model import generate_canopy_map

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

## -------------------------------- MAIN EXECUTION: Run Sample Size Reliability Analysis -------------------------------
#region

if __name__ == "__main__":
    # --- ANALYSIS CONFIGURATION ---
    MIN_SAMPLE_SIZE = 10
    MAX_SAMPLE_SIZE = 50
    INCREMENT_SIZE = 25

    sample_sizes_to_test = np.arange(MIN_SAMPLE_SIZE, MAX_SAMPLE_SIZE + 1, INCREMENT_SIZE)

    CONFIG = {
        "SAMPLE_SIZES_TO_TEST": sample_sizes_to_test,
        "NUM_TRIALS_PER_SIZE": 10000,  # Number of bootstraps
        "AGREEMENT_TOLERANCE": 1.5,
    }

    # --- Plotting Setup ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    # --- Run Analysis for Each Canopy Cover Level ---
    canopy_cover_levels_to_test = np.arange(0, 61, 5) / 100.0  # 0% to 60% in increments of 5%

    for cover_level in tqdm(canopy_cover_levels_to_test, desc="Overall Progress"):
        print(f"\n--- Running analysis for Canopy Cover: {cover_level:.0%} ---")

        sample_sizes, agreement_percents, actual_cover = analyze_sample_size_agreement(CONFIG, cover_level)

        # Plot the result for this canopy cover level
        ax.plot(sample_sizes, agreement_percents, marker='o', linestyle='-', markersize=4,
                label=f'True Cover: {actual_cover:.1f}%')

    # --- Finalize Visualization ---
    ax.axhline(y=95, color='r', linestyle='--', label='95% Confidence Target')
    ax.set_xlabel('Number of Sample Points', fontsize=12)
    ax.set_ylabel(f'Agreement with True Cover (±{CONFIG["AGREEMENT_TOLERANCE"]}%)', fontsize=12)
    ax.set_ylim(0, 105)
    ax.grid(False)
    ax.legend(title="Canopy Cover Level", bbox_to_anchor=(1.04, 1), loc="upper left")

    fig.tight_layout()
    plt.show()

#endregion
