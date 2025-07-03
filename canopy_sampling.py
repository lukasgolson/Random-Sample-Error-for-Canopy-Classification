import numpy as np


def calculate_direct_cover(canopy_map):

    # The mean of a binary (0/1) array is the proportion of 1s.
    true_proportion = np.mean(canopy_map)
    return true_proportion * 100


def simulate_random_sampling(canopy_map, num_samples):
    height, width = canopy_map.shape

    # Generate random coordinates for each sample point
    sample_x = np.random.randint(0, width, num_samples)
    sample_y = np.random.randint(0, height, num_samples)

    # Check for canopy presence at each sample point and sum the "hits"
    canopy_hits = np.sum(canopy_map[sample_y, sample_x])

    # Calculate the estimated proportion of cover
    estimated_proportion = canopy_hits / num_samples

    return estimated_proportion * 100, sample_x, sample_y


def get_single_estimate(canopy_map, num_samples):
    """Performs a random sample run and returns the single estimated cover."""
    if num_samples == 0:
        return 0
    height, width = canopy_map.shape
    sample_x = np.random.randint(0, width, num_samples)
    sample_y = np.random.randint(0, height, num_samples)
    canopy_hits = np.sum(canopy_map[sample_y, sample_x])
    return (canopy_hits / num_samples) * 100