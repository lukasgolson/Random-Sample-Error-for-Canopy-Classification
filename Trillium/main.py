## Conceptual framework for code
# 1. Create a tree stamp
# 2. Generate an AOI (400, 2000, 30000) with canopy extent (0 - 100%) and Moran's i (-1 to +1). Canopy in steps of 1% (101 steps including 0%), Moran's i in steps of .05 (40 steps) (total: 4,040 steps)
# 3. Apply 100,000 random sample points (same points for all runs) and identify as canopy (1) or no canopy (0). Include meters x and y of sample points (in replace of latitude and longitude)
# 4. Export results as CSV with columns Sample Point ID, Sample Point X Location, Sample Point Y Location, then columns with code {AOI}_{Target_Extent}_{Target_Moran}_{True_Extent}_{True_Moran}. 


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
