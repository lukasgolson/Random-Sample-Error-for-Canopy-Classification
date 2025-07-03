# This script answers the question:
# How does the reliability of my estimate change as I increase the number of sample points?


import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
import os
from canopy_model import generate_canopy_map
from canopy_sampling import get_single_estimate
from sampling_analysis import calculate_direct_cover





# --- Main Analysis Function ---
def analyze_sample_size_agreement(config, target_canopy_cover):
    """
    Analyzes the relationship between sample size and agreement for a given canopy cover.
    """
    sample_sizes = config['SAMPLE_SIZES_TO_TEST']
    num_trials = config['NUM_TRIALS_PER_SIZE']
    tolerance = config['AGREEMENT_TOLERANCE']

    # Generate a representative map with the specified target canopy cover
    canopy_map = generate_canopy_map(
        width=200, height=200, clustering=65, canopy_cover=target_canopy_cover, seed=42
    )
    true_cover = calculate_direct_cover(canopy_map)

    agreement_results = []

    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        for n_samples in sample_sizes:
            task_args = [(canopy_map, n_samples)] * num_trials
            estimates = pool.starmap(get_single_estimate, task_args)

            in_agreement_count = np.sum(np.abs(np.array(estimates) - true_cover) <= tolerance)
            agreement_percent = (in_agreement_count / num_trials) * 100
            agreement_results.append(agreement_percent)

    return sample_sizes, agreement_results, true_cover


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