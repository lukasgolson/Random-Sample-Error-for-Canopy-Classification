import numpy as np
import noise

def generate_canopy_map(width=100, height=100, clustering=50, canopy_cover=0.5, seed=None):
    """
    Generates a binary raster simulating a canopy presence map.
    """
    if not 0 <= clustering <= 100:
        raise ValueError("Clustering must be an integer between 0 and 100.")

    # --- Handle the 0 Clustering Case ---
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