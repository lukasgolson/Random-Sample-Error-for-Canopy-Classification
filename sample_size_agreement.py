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
    "Neighbourhood (20 km²)": (14907, 14907), # Neighbourhood (20 km²) AOI
    "City (400 km²)": (66667, 66667), # City (400 km²) AOI
    "Region (3,000 km²)": (182574, 182574), # Region / County (3,000 km²) AOI
}

#endregion

## -------------------------------------------------- DEFINE FUNCTIONS -------------------------------------------------
#region

if __name__ == "__main__":
    # Configuration from Define Simulation block
    canopy_cover_levels_to_test = np.arange(MIN_CANOPY_COVER, MAX_CANOPY_COVER + 1, INCREMENT_CANOPY_COVER) / 100
    sample_sizes_to_test = np.arange(MIN_SAMPLE_SIZE, MAX_SAMPLE_SIZE + 1, INCREMENT_SIZE)

    CONFIG = {
        "SAMPLE_SIZES_TO_TEST": sample_sizes_to_test,
        "NUM_TRIALS_PER_SIZE": trials,
        "AGREEMENT_TOLERANCE": agreement_tolerance,
    }

    # Prepare a 1-row, 3-column figure
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    plt.style.use('seaborn-v0_8-whitegrid')

    for ax, (aoi_name, (width, height)) in zip(axes, AOIS.items()):
        print(f"\n\n=== Running simulations for {aoi_name} ===")

        for cover_level in tqdm(canopy_cover_levels_to_test, desc=f"{aoi_name} Progress"):
            print(f"--- Canopy Cover: {cover_level:%} ---")
            sample_sizes, agreement_percents, actual_cover = analyze_sample_size_agreement(
                CONFIG, cover_level, width, height
            )

            ax.plot(sample_sizes, agreement_percents, marker='o', linestyle='-', markersize=4,
                    label=f'{actual_cover:.1f}%')

        # Finalize each subplot
        ax.axhline(y=95, color='r', linestyle='--', label='95% Target')
        ax.set_title(aoi_name, fontsize=13, fontweight='bold')
        ax.set_xlabel('Number of Sample Points', fontsize=11)
        ax.set_ylim(0, 105)
        ax.grid(False)

    # Adjust bottom spacing to make room for the legend
    plt.subplots_adjust(bottom=0.22)

    # Shared legend below the figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Canopy Cover",
               loc='lower center', bbox_to_anchor=(0.5, -0.05),
               ncol=12, fontsize=10, title_fontsize=12, frameon=False)

    plt.show()

    # Generate and show the canopy map for this AOI
    sample_cover = 0.3 # Use mid canopy cover (e.g., 30%) just for visualization

    canopy_map = generate_canopy_map(
        width=width,
        height=height,
        clustering=65,
        canopy_cover=sample_cover,
        seed=42
    )

    fig_map, ax_map = plt.subplots(figsize=(6, 6))
    ax_map.imshow(canopy_map, cmap='Greens', interpolation='nearest')
    ax_map.set_title(f"{aoi_name} Canopy Map (30% Cover)", fontsize=13)
    ax_map.set_xticks([])
    ax_map.set_yticks([])

    plt.tight_layout()
    plt.show()

#endregion
