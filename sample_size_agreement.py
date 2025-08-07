# This script answers the question:
# How does the reliability of my estimate change as I increase the number of sample points?
# Results: Paragraph 1, Figure 1

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
from canopy_model import generate_canopy_map
from matplotlib.font_manager import FontProperties

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

## -------------------------------- MAIN EXECUTION: Run Sample Size Reliability Analysis -------------------------------
#region

if __name__ == "__main__":
    # Configuration
    canopy_cover_levels_to_test = np.arange(MIN_CANOPY_COVER, MAX_CANOPY_COVER + 1, INCREMENT_CANOPY_COVER) / 100
    sample_sizes_to_test = np.arange(MIN_SAMPLE_SIZE, MAX_SAMPLE_SIZE + 1, INCREMENT_SIZE)

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

    # Store canopy maps to show later
    canopy_maps_to_show = []

    for i, (aoi_name, (width, height)) in enumerate(AOIS.items()):
        ax = axes[i]
        print(f"\n\n=== Running simulations for {aoi_name} ===")

        # Generate the canopy map for visualization (30% cover)
        sample_cover = 0.3
        canopy_map = generate_canopy_map(
            width=width,
            height=height,
            clustering=65,
            canopy_cover=sample_cover,
            seed=42
        )
        canopy_maps_to_show.append((aoi_name, canopy_map))

        # Run analysis and plot agreement curves
        for cover_level in tqdm(canopy_cover_levels_to_test, desc=f"{aoi_name} Progress"):
            print(f"--- Canopy Cover: {cover_level:.0%} ---")
            sample_sizes, agreement_percents, actual_cover = analyze_sample_size_agreement(
                CONFIG, cover_level, width, height
            )

            ax.plot(sample_sizes, agreement_percents, marker='o', linestyle='-', markersize=6,
                    label=f'{actual_cover:.1f}%')

        ax.axhline(y=95, color='r', linestyle='--', label='95% CI Target')
        ax.set_title(aoi_name, fontsize=19, fontweight='bold')
        ax.set_xlabel('Number of Sample Points', fontsize=17)
        ax.set_ylabel('Agreement Within Tolerance (%)', fontsize=17)
        ax.set_ylim(0, 105)
        ax.tick_params(axis='both', labelsize=15)
        ax.grid(False)

    # Remove y-axis label from top-right subplot
    axes[1].set_ylabel('')

    # Remove unused bottom-right plot and use for legend
    axes[3].axis('off')
    handles, labels = axes[0].get_legend_handles_labels()
    bold_font = FontProperties(weight='bold', size=18)
    legend = axes[3].legend(handles, labels, title="Canopy Cover Extent", loc='center',
                            fontsize=16, title_fontproperties=bold_font, frameon=True)

    # Adjust layout and spacing
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.25)

    plt.show()

    # Print standalone agreement plots for each AOI
    for i, (aoi_name, (width, height)) in enumerate(AOIS.items()):
        print(f"\n\n=== Re-plotting standalone figure for {aoi_name} ===")

        fig_single, ax_single = plt.subplots(figsize=(8, 6))
        sample_cover = 0.3
        canopy_map = generate_canopy_map(
            width=width,
            height=height,
            clustering=65,
            canopy_cover=sample_cover,
            seed=42
        )

        for cover_level in canopy_cover_levels_to_test:
            sample_sizes, agreement_percents, actual_cover = analyze_sample_size_agreement(
                CONFIG, cover_level, width, height
            )
            ax_single.plot(sample_sizes, agreement_percents, marker='o', linestyle='-', markersize=6,
                           label=f'{actual_cover:.1f}%')

        ax_single.axhline(y=95, color='r', linestyle='--', label='95% CI Target')
        ax_single.set_title(aoi_name, fontsize=15, fontweight='bold')
        ax_single.set_xlabel('Number of Sample Points', fontsize=13)
        ax_single.set_ylabel('Agreement Within Tolerance (%)', fontsize=13)
        ax_single.set_ylim(0, 105)
        ax_single.tick_params(axis='both', labelsize=13)
        ax_single.grid(False)

        bold_font = FontProperties(weight='bold', size=12)
        ax_single.legend(title="Canopy Cover Extent", loc='center left',
                         bbox_to_anchor=(1.05, 0.5), fontsize=10,
                         title_fontproperties=bold_font, frameon=False)

        fig_single.tight_layout()
        plt.show()

    # Show each stored canopy map
    for aoi_name, canopy_map in canopy_maps_to_show:
        fig_map, ax_map = plt.subplots(figsize=(6, 6))
        ax_map.imshow(canopy_map, cmap='Greens', interpolation='nearest')
        ax_map.set_title(f"{aoi_name} Canopy Map (30% Cover)", fontsize=17)
        ax_map.set_xticks([])
        ax_map.set_yticks([])
        plt.tight_layout()
        plt.show()

#endregion
