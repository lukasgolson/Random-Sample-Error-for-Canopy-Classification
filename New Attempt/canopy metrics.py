import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import ndimage
from scipy.spatial.distance import pdist
from skimage.measure import label, regionprops
from sklearn.neighbors import NearestNeighbors
import warnings

USE_TEST_SETTINGS = True

# Import the grids
grid_1 = gpd.read_file('AOI/grid_1km.gpkg')
grid_20 = gpd.read_file('AOI/grid_20km.gpkg')
grid_40 = gpd.read_file('AOI/grid_40km.gpkg')
grid_100 = gpd.read_file('AOI/grid_100km.gpkg')

grids = [grid_100] if USE_TEST_SETTINGS else [grid_1, grid_20, grid_40]

def calculate_landscape_metrics(raster_data, cell_size=None):
    """
    Calculate comprehensive landscape metrics for a raster dataset using pure raster approach.
    This follows standard landscape ecology methodology (similar to FRAGSTATS).

    Parameters:
    -----------
    raster_data : numpy.ndarray
        2D array representing the raster data (binary or classified)
        For canopy analysis, typically 1 = canopy, 0 = non-canopy
    cell_size : float, optional
        Size of each raster cell in appropriate units (e.g., meters)
        If None, calculations will be in pixel units

    Returns:
    --------
    dict : Dictionary containing all calculated metrics
    """

    # Ensure binary data for most calculations
    if np.unique(raster_data).size > 2:
        warnings.warn("Raster has more than 2 unique values. Using > 0 as presence.")
        binary_raster = (raster_data > 0).astype(int)
    else:
        binary_raster = raster_data.astype(int)

    # Handle cell size
    if cell_size is None:
        cell_size = 1.0
        area_multiplier = 1.0
    else:
        area_multiplier = cell_size ** 2

    results = {}

    # 1. CANOPY EXTENT
    total_cells = binary_raster.size
    canopy_cells = np.sum(binary_raster)
    results['canopy_extent'] = (canopy_cells / total_cells) * 100  # Percentage

    # 2. MORAN'S I (Spatial Autocorrelation)
    results['morans_i'] = calculate_morans_i(binary_raster)

    # 3. EDGE DENSITY
    results['edge_density'] = calculate_edge_density(binary_raster, cell_size)

    # 4. CLUMPY INDEX
    results['clumpy'] = calculate_clumpy_index(binary_raster)

    # Patch-based metrics using consistent raster approach
    patch_metrics = calculate_patch_metrics_raster(binary_raster, area_multiplier, cell_size)
    results.update(patch_metrics)

    return results


def calculate_patch_metrics_raster(binary_raster, area_multiplier, cell_size):
    """Calculate all patch metrics using consistent raster approach."""
    # Label connected components (patches) using 8-connectivity
    labeled_patches, num_patches = label(binary_raster, connectivity=2, return_num=True)

    results = {'number_of_patches': num_patches}

    if num_patches > 0:
        # Get patch properties
        patch_props = regionprops(labeled_patches)
        patch_areas = [prop.area * area_multiplier for prop in patch_props]

        # Basic patch statistics
        results.update({
            'avg_patch_size': np.mean(patch_areas),
            'patch_size_std': np.std(patch_areas, ddof=1) if len(patch_areas) > 1 else 0,
            'patch_size_median': np.median(patch_areas),
            'patch_size_min': np.min(patch_areas),
            'patch_size_max': np.max(patch_areas)
        })

        # Calculate normalized LSI using raster approach
        results['normalized_lsi'] = calculate_normalized_lsi_raster(labeled_patches, patch_props, cell_size)

    else:
        # No patches found
        results.update({
            'avg_patch_size': 0,
            'patch_size_std': 0,
            'patch_size_median': 0,
            'patch_size_min': 0,
            'patch_size_max': 0,
            'normalized_lsi': 0
        })

    return results


def calculate_normalized_lsi_raster(labeled_patches, patch_props, cell_size):
    """Calculate Normalized Landscape Shape Index using raster approach."""
    try:
        total_area = 0
        total_perimeter = 0

        for prop in patch_props:
            # Area in real units
            area = prop.area * (cell_size ** 2)

            # Calculate perimeter using more accurate method
            # Count edge pixels by checking 4-connectivity boundaries
            patch_mask = (labeled_patches == prop.label)

            # Use convolution to count edge cells
            kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])  # 4-connectivity
            edges = ndimage.convolve(patch_mask.astype(int), kernel, mode='constant', cval=0)
            # Edge cells are those with at least one non-patch neighbor
            edge_cells = np.sum((patch_mask) & (edges < 4))

            # Convert to actual perimeter length
            perimeter = edge_cells * cell_size

            total_area += area
            total_perimeter += perimeter

        if total_area > 0:
            # Standard LSI calculation
            lsi = total_perimeter / (2 * np.sqrt(np.pi * total_area))

            # Normalize by landscape area (optional normalization)
            # This makes LSI comparable across different landscape sizes
            return lsi
        else:
            return 0

    except Exception as e:
        warnings.warn(f"Could not calculate normalized LSI: {e}")
        return 0


def calculate_morans_i(raster_data):
    """Calculate Moran's I spatial autocorrelation index."""
    try:
        rows, cols = raster_data.shape
        n = rows * cols

        # Flatten the raster
        values = raster_data.flatten()

        # Create spatial weights matrix (8-connectivity)
        coordinates = np.array([[i, j] for i in range(rows) for j in range(cols)])

        # Use a subset for large rasters to avoid memory issues
        if n > 10000:
            sample_size = 5000
            indices = np.random.choice(n, sample_size, replace=False)
            coordinates = coordinates[indices]
            values = values[indices]
            n = sample_size

        # Calculate distances and create weights
        distances = pdist(coordinates, metric='chebyshev')  # Max distance (8-connectivity)
        weights = (distances <= 1.5).astype(float)  # Adjacent cells

        # Convert to square matrix
        from scipy.spatial.distance import squareform
        W = squareform(weights)
        np.fill_diagonal(W, 0)  # No self-weights

        # Calculate Moran's I
        W_sum = np.sum(W)
        if W_sum == 0:
            return 0

        mean_val = np.mean(values)
        numerator = 0
        denominator = 0

        for i in range(n):
            for j in range(n):
                if i != j and W[i, j] > 0:
                    numerator += W[i, j] * (values[i] - mean_val) * (values[j] - mean_val)
            denominator += (values[i] - mean_val) ** 2

        if denominator == 0:
            return 0

        morans_i = (n / W_sum) * (numerator / denominator)
        return morans_i

    except Exception as e:
        warnings.warn(f"Could not calculate Moran's I: {e}")
        return 0


def calculate_edge_density(binary_raster, cell_size):
    """Calculate edge density (total edge length per unit area)."""
    try:
        # Create edge map using morphological operations
        from scipy.ndimage import binary_erosion

        # Find edges by subtracting eroded image from original
        eroded = binary_erosion(binary_raster)
        edges = binary_raster - eroded

        # Calculate total edge length
        edge_cells = np.sum(edges)
        edge_length = edge_cells * cell_size

        # Total landscape area
        total_area = binary_raster.size * (cell_size ** 2)

        # Edge density (length per unit area)
        edge_density = edge_length / total_area if total_area > 0 else 0

        return edge_density

    except Exception as e:
        warnings.warn(f"Could not calculate edge density: {e}")
        return 0


def calculate_clumpy_index(binary_raster):
    """Calculate CLUMPY index (measure of aggregation)."""
    try:
        # Calculate proportion of landscape
        P = np.sum(binary_raster) / binary_raster.size

        if P == 0 or P == 1:
            return 0

        # Calculate observed proportion of like adjacencies
        rows, cols = binary_raster.shape
        like_adjacencies = 0
        total_adjacencies = 0

        # Check horizontal adjacencies
        for i in range(rows):
            for j in range(cols - 1):
                if binary_raster[i, j] == binary_raster[i, j + 1] == 1:
                    like_adjacencies += 1
                total_adjacencies += 1

        # Check vertical adjacencies
        for i in range(rows - 1):
            for j in range(cols):
                if binary_raster[i, j] == binary_raster[i + 1, j] == 1:
                    like_adjacencies += 1
                total_adjacencies += 1

        if total_adjacencies == 0:
            return 0

        # Observed proportion of like adjacencies
        G_observed = like_adjacencies / total_adjacencies

        # Expected proportion under random distribution
        G_expected = P

        # CLUMPY index
        if P < 0.5:
            clumpy = (G_observed - G_expected) / (P)
        else:
            clumpy = (G_observed - G_expected) / (1 - P)

        return clumpy

    except Exception as e:
        warnings.warn(f"Could not calculate CLUMPY index: {e}")
        return 0


def calculate_normalized_lsi(binary_raster, cell_size):
    """Calculate Normalized Landscape Shape Index using improved raster method."""
    return calculate_normalized_lsi_raster(None, None, cell_size)  # Placeholder for compatibility


def process_grid_cells(grid_gdf, raster_path, aoi_size, output_dir='.'):
    """
    Process grid cells and calculate landscape metrics for each.

    Parameters:
    -----------
    grid_gdf : geopandas.GeoDataFrame
        Grid with cell_id column and geometry
    raster_path : str
        Path to the raster dataset
    aoi_size : str
        AOI size identifier for CSV naming
    output_dir : str
        Output directory for CSV files

    Returns:
    --------
    pandas.DataFrame : Results dataframe
    """
    import rasterio
    import rasterio.mask
    import geopandas as gpd

    results_list = []

    with rasterio.open(raster_path) as src:
        for idx, row in grid_gdf.iterrows():
            cell_id = row['cell_id']
            geometry = [row.geometry]

            try:
                # Extract raster data for this grid cell
                # Using intersects approach as specified
                out_image, out_transform = rasterio.mask.mask(
                    src, geometry, crop=True, all_touched=True
                )

                # Get the first band (assuming single band or using first band)
                raster_data = out_image[0]

                # Skip if no data
                if raster_data.size == 0 or np.all(raster_data == src.nodata):
                    continue

                # Calculate cell size from transform
                cell_size = abs(out_transform.a)  # Pixel width

                # Calculate metrics using pure raster approach
                metrics = calculate_landscape_metrics(raster_data, cell_size=cell_size)
                metrics['cell_id'] = cell_id

                results_list.append(metrics)

            except Exception as e:
                print(f"Error processing cell_id {cell_id}: {e}")
                continue

    # Create DataFrame
    results_df = pd.DataFrame(results_list)

    # Save to CSV
    output_filename = f"landscape_metrics_{aoi_size}.csv"
    output_path = f"{output_dir}/{output_filename}"
    results_df.to_csv(output_path, index=False)

    print(f"Results saved to: {output_path}")
    return results_df


# Example usage:
if __name__ == "__main__":
    # For testing individual function
    # Create sample binary raster (canopy/no-canopy)
    sample_raster = np.random.choice([0, 1], size=(100, 100), p=[0.6, 0.4])

    # Calculate metrics
    metrics = calculate_landscape_metrics(sample_raster, cell_size=30.0)

    print("Calculated Landscape Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
