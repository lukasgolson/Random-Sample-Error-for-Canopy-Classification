import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import ndimage
from scipy.spatial.distance import pdist
from skimage.measure import label, regionprops
import rasterio
import rasterio.mask
from rasterio.warp import reproject, Resampling
import os
import glob
from pathlib import Path
import warnings
from tqdm import tqdm

USE_TEST_SETTINGS = True

# Import the grids
grid_1 = gpd.read_file('AOI/grid_1km.gpkg')
grid_20 = gpd.read_file('AOI/grid_20km.gpkg')
grid_40 = gpd.read_file('AOI/grid_40km.gpkg')
grid_100 = gpd.read_file('AOI/grid_100km.gpkg')

grids = [grid_100] if USE_TEST_SETTINGS else [grid_1, grid_20, grid_40]


def build_raster_spatial_index(raster_folder):
    """
    Build a spatial index of all raster files for efficient intersection queries.

    Parameters:
    -----------
    raster_folder : str
        Path to folder containing raster files

    Returns:
    --------
    tuple : (rtree_index, raster_bounds_dict)
        - rtree_index: R-tree spatial index
        - raster_bounds_dict: Dictionary mapping index IDs to raster info
    """
    try:
        from rtree import index
    except ImportError:
        print("Warning: rtree not available. Install with: pip install rtree")
        print("Falling back to brute force intersection checking...")
        return None, None

    # Create R-tree index
    idx = index.Index()
    raster_info = {}

    # Get all GeoTIFF files
    raster_pattern = os.path.join(raster_folder, "*.tif*")
    raster_files = glob.glob(raster_pattern)

    print(f"Building spatial index for {len(raster_files)} raster files...")

    for i, raster_path in enumerate(tqdm(raster_files, desc="Indexing rasters")):
        try:
            with rasterio.open(raster_path) as src:
                bounds = src.bounds
                # Insert bounding box into spatial index
                idx.insert(i, (bounds.left, bounds.bottom, bounds.right, bounds.top))

                # Store raster information
                raster_info[i] = {
                    'path': raster_path,
                    'bounds': bounds,
                    'crs': src.crs,
                    'transform': src.transform,
                    'shape': src.shape
                }

        except Exception as e:
            print(f"Warning: Could not index {raster_path}: {e}")
            continue

    print(f"Spatial index built with {len(raster_info)} rasters")
    return idx, raster_info


def get_intersecting_rasters_indexed(grid_cell_geometry, spatial_index, raster_info, buffer_distance=0):
    """
    Find all raster files that intersect with a given grid cell geometry using spatial index.

    Parameters:
    -----------
    grid_cell_geometry : shapely.geometry
        The geometry of the grid cell
    spatial_index : rtree.index.Index
        R-tree spatial index of raster bounds
    raster_info : dict
        Dictionary mapping index IDs to raster information
    buffer_distance : float
        Buffer distance to expand grid cell for intersection (optional)

    Returns:
    --------
    list : List of raster file paths that intersect the grid cell
    """
    if spatial_index is None:
        # Fallback to brute force method
        return get_intersecting_rasters_bruteforce(grid_cell_geometry, raster_info, buffer_distance)

    intersecting_rasters = []

    # Buffer the grid cell geometry if specified
    if buffer_distance > 0:
        search_geometry = grid_cell_geometry.buffer(buffer_distance)
    else:
        search_geometry = grid_cell_geometry

    # Get bounding box of search geometry
    minx, miny, maxx, maxy = search_geometry.bounds

    # Query spatial index for potentially intersecting rasters
    candidate_ids = list(spatial_index.intersection((minx, miny, maxx, maxy)))

    # Perform detailed intersection check for candidates
    for raster_id in candidate_ids:
        try:
            raster_bounds = raster_info[raster_id]['bounds']

            # Create raster geometry
            from shapely.geometry import box
            raster_geom = box(raster_bounds.left, raster_bounds.bottom,
                              raster_bounds.right, raster_bounds.top)

            # Check detailed intersection
            if search_geometry.intersects(raster_geom):
                intersecting_rasters.append(raster_info[raster_id]['path'])

        except Exception as e:
            print(f"Warning: Error checking intersection for raster {raster_id}: {e}")
            continue

    return intersecting_rasters


def get_intersecting_rasters_bruteforce(grid_cell_geometry, raster_folder_or_info, buffer_distance=0):
    """
    Fallback brute force method for finding intersecting rasters.
    """
    intersecting_rasters = []

    # Buffer the grid cell geometry if specified
    if buffer_distance > 0:
        search_geometry = grid_cell_geometry.buffer(buffer_distance)
    else:
        search_geometry = grid_cell_geometry

    # Handle different input types
    if isinstance(raster_folder_or_info, dict):
        # Called from indexed method as fallback
        raster_files = [info['path'] for info in raster_folder_or_info.values()]
    else:
        # Called directly with folder path
        raster_pattern = os.path.join(raster_folder_or_info, "*.tif*")
        raster_files = glob.glob(raster_pattern)

    for raster_path in raster_files:
        try:
            with rasterio.open(raster_path) as src:
                # Get raster bounds and create geometry
                bounds = src.bounds
                from shapely.geometry import box
                raster_geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)

                # Check intersection
                if search_geometry.intersects(raster_geom):
                    intersecting_rasters.append(raster_path)

        except Exception as e:
            print(f"Warning: Could not check intersection for {raster_path}: {e}")
            continue

    return intersecting_rasters


def mosaic_rasters(raster_paths, target_geometry, target_crs, target_resolution=None):
    """
    Mosaic multiple rasters that intersect a target geometry.

    Parameters:
    -----------
    raster_paths : list
        List of raster file paths to mosaic
    target_geometry : shapely.geometry
        Target geometry to clip the mosaic to
    target_crs : CRS
        Target coordinate reference system
    target_resolution : float, optional
        Target pixel resolution. If None, uses the finest resolution from input rasters

    Returns:
    --------
    tuple : (mosaicked_array, transform)
    """
    if not raster_paths:
        return None, None

    if len(raster_paths) == 1:
        # Single raster case
        with rasterio.open(raster_paths[0]) as src:
            out_image, out_transform = rasterio.mask.mask(
                src, [target_geometry], crop=True, all_touched=False
            )
            return out_image[0], out_transform

    # Multiple rasters - need to mosaic
    try:
        from rasterio.merge import merge

        # Open all rasters
        src_files = [rasterio.open(path) for path in raster_paths]

        # Determine target resolution
        if target_resolution is None:
            target_resolution = min([abs(src.transform.a) for src in src_files])

        # Merge rasters
        mosaic, out_trans = merge(
            src_files,
            res=target_resolution,
            method='max'  # Use max value where rasters overlap
        )

        # Close source files
        for src in src_files:
            src.close()

        # Create temporary in-memory dataset for clipping
        from rasterio.io import MemoryFile

        with MemoryFile() as memfile:
            with memfile.open(
                    driver='GTiff',
                    height=mosaic.shape[1],
                    width=mosaic.shape[2],
                    count=1,
                    dtype=mosaic.dtype,
                    crs=src_files[0].crs,  # Assuming all rasters have same CRS
                    transform=out_trans
            ) as dataset:
                dataset.write(mosaic[0], 1)

                # Clip to target geometry
                out_image, out_transform = rasterio.mask.mask(
                    dataset, [target_geometry], crop=True, all_touched=False
                )

                return out_image[0], out_transform

    except Exception as e:
        print(f"Error in mosaicking: {e}")
        return None, None


def calculate_landscape_metrics(raster_data, cell_size=None):
    """
    Calculate comprehensive landscape metrics for a 1-bit binary raster dataset.

    METRICS CALCULATED:
    ==================

    LANDSCAPE-LEVEL METRICS:
    -----------------------
    1. canopy_extent: Percentage of landscape covered by canopy (0-100%)
    2. edge_density: Total edge length per unit area (length/area units)
    3. clumpy: Aggregation index (0-1)
    4. landscape_type: Categorical description ('mixed', 'all_canopy', 'all_non_canopy', 'no_data')

    SPATIAL AUTOCORRELATION METRICS:
    ---------------------------------------
    5. morans_i: Moran's I (NOT recommended for binary data - kept for comparison)
    6. join_count_bb: Join Count BB statistic (RECOMMENDED for binary data)
    7. hansens_uniformity: Hansen's Uniformity Index (0-1)
    8. geary_c: Geary's C (alternative to Moran's I for binary data)

    PATCH-LEVEL METRICS:
    -------------------
    9. number_of_patches: Count of discrete canopy patches
    10. avg_patch_size: Mean patch area
    11. patch_size_std: Standard deviation of patch areas
    12. patch_size_median: Median patch area
    13. patch_size_min: Smallest patch area
    14. patch_size_max: Largest patch area
    15. normalized_lsi: Normalized Landscape Shape Index (whole grid cell)

    Parameters:
    -----------
    raster_data : numpy.ndarray
        2D binary array (0=non-canopy, 1=canopy) from 1-bit raster
    cell_size : float, optional
        Size of each raster cell in appropriate units (e.g., meters)
        If None, calculations will be in pixel units

    Returns:
    --------
    dict : Dictionary containing all calculated metrics
    """
    # Handle invalid data
    if raster_data is None or raster_data.size == 0:
        return {
            'canopy_extent': 0,
            'morans_i': 0,
            'join_count_bb': 0,
            'hansens_uniformity': 0.5,
            'geary_c': 1.0,
            'edge_density': 0,
            'clumpy': 0,
            'number_of_patches': 0,
            'avg_patch_size': 0,
            'patch_size_std': 0,
            'patch_size_median': 0,
            'patch_size_min': 0,
            'patch_size_max': 0,
            'normalized_lsi': 0,
            'landscape_type': 'no_data'
        }

    # Data is already binary (1-bit rasters can only be 0 or 1)
    binary_raster = raster_data.astype(int)

    # Handle cell size
    if cell_size is None:
        cell_size = 1.0
        area_multiplier = 1.0
    else:
        area_multiplier = cell_size ** 2

    results = {}

    # Check for uniform landscapes (all 0s or all 1s)
    unique_values = np.unique(binary_raster)
    total_cells = binary_raster.size
    canopy_cells = np.sum(binary_raster)

    # 1. CANOPY EXTENT
    results['canopy_extent'] = (canopy_cells / total_cells) * 100  # Percentage

    # Handle special cases for uniform landscapes
    if len(unique_values) == 1:
        if unique_values[0] == 0:
            # All zeros - no canopy
            results.update({
                'morans_i': 0,  # Not meaningful for uniform data
                'join_count_bb': 0,  # No BB joins possible
                'hansens_uniformity': 1.0,  # Perfect uniformity (all same)
                'geary_c': 1.0,  # No spatial autocorrelation possible
                'edge_density': 0,  # No edges
                'clumpy': 0,  # No aggregation possible
                'number_of_patches': 0,  # No patches
                'avg_patch_size': 0,
                'patch_size_std': 0,
                'patch_size_median': 0,
                'patch_size_min': 0,
                'patch_size_max': 0,
                'normalized_lsi': 0,  # No shape complexity
                'landscape_type': 'all_non_canopy'
            })
        else:
            # All ones - complete canopy coverage
            total_area = total_cells * area_multiplier
            results.update({
                'morans_i': 1.0,  # Perfect positive autocorrelation
                'join_count_bb': float('inf'),  # All possible joins are BB
                'hansens_uniformity': 1.0,  # Perfect uniformity (all same)
                'geary_c': 0.0,  # Perfect clustering (no differences)
                'edge_density': 0,  # No internal edges in uniform landscape
                'clumpy': 1.0,  # Maximum aggregation
                'number_of_patches': 1,  # Single large patch
                'avg_patch_size': total_area,
                'patch_size_std': 0,  # No variation in patch size
                'patch_size_median': total_area,
                'patch_size_min': total_area,
                'patch_size_max': total_area,
                'normalized_lsi': 1.0,  # Minimum shape complexity (circular)
                'landscape_type': 'all_canopy'
            })
        return results

    # Mixed landscape - calculate all metrics normally
    results['landscape_type'] = 'mixed'

    # 2. SPATIAL AUTOCORRELATION METRICS
    results['morans_i'] = calculate_morans_i(binary_raster)  # Keep for comparison but not recommended
    results['join_count_bb'] = calculate_join_count_bb(binary_raster)  # RECOMMENDED for binary data
    results['hansens_uniformity'] = calculate_hansens_uniformity(binary_raster)  # Hansen's measure
    results['geary_c'] = calculate_geary_c_binary(binary_raster)  # Alternative to Moran's I

    # 3. EDGE DENSITY
    results['edge_density'] = calculate_edge_density(binary_raster, cell_size)

    # 4. CLUMPY INDEX
    results['clumpy'] = calculate_clumpy_index(binary_raster)

    # Patch-based metrics using consistent raster approach
    patch_metrics = calculate_patch_metrics_raster(binary_raster, area_multiplier, cell_size)
    results.update(patch_metrics)

    return results


def calculate_patch_metrics_raster(binary_raster, area_multiplier, cell_size):
    """
    Calculate patch-based landscape metrics using consistent raster approach.

    Patch Metrics Calculated:
    ------------------------
    1. number_of_patches: Total count of discrete canopy patches (8-connected)
    2. avg_patch_size: Mean area of all patches
    3. patch_size_std: Standard deviation of patch areas
    4. patch_size_median: Median patch area
    5. patch_size_min: Smallest patch area
    6. patch_size_max: Largest patch area
    7. normalized_lsi: Normalized Landscape Shape Index (shape complexity)

    Parameters:
    -----------
    binary_raster : numpy.ndarray
        Binary raster (0=non-canopy, 1=canopy)
    area_multiplier : float
        Multiplier to convert pixel count to real area units
    cell_size : float
        Size of each pixel in real units

    Returns:
    --------
    dict : Dictionary of patch metrics
    """
    # Label connected components (patches) using 8-connectivity
    # 8-connectivity means diagonal neighbors are considered connected
    labeled_patches, num_patches = label(binary_raster, connectivity=2, return_num=True)

    results = {'number_of_patches': num_patches}

    if num_patches > 0:
        # Get patch properties using scikit-image regionprops
        patch_props = regionprops(labeled_patches)

        # Calculate patch areas in real units
        patch_areas = [prop.area * area_multiplier for prop in patch_props]

        # Basic patch area statistics
        results.update({
            'avg_patch_size': np.mean(patch_areas),
            'patch_size_std': np.std(patch_areas, ddof=1) if len(patch_areas) > 1 else 0,
            'patch_size_median': np.median(patch_areas),
            'patch_size_min': np.min(patch_areas),
            'patch_size_max': np.max(patch_areas)
        })

        # Calculate normalized LSI (shape complexity measure)
        results['normalized_lsi'] = calculate_normalized_lsi_raster(labeled_patches, patch_props, cell_size)

    else:
        # No patches found (all zeros)
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
            return lsi
        else:
            return 0

    except Exception as e:
        warnings.warn(f"Could not calculate normalized LSI: {e}")
        return 0


def calculate_morans_i(raster_data):
    """
    Calculate Moran's I spatial autocorrelation index.
    NOTE: This is not recommended for binary data - kept for comparison purposes only.
    """
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


def calculate_join_count_bb(binary_raster):
    """
    Calculate Join Count BB statistic for binary spatial autocorrelation.
    This is the appropriate measure for binary data instead of Moran's I.

    Join Count BB measures the number of Black-Black joins (canopy-canopy adjacencies)
    relative to what would be expected under spatial randomness.

    Returns:
    --------
    float : Standardized Join Count BB statistic
        - Positive values = clustering (more BB joins than expected)
        - Negative values = dispersion (fewer BB joins than expected)
        - Zero = random spatial pattern
    """
    try:
        rows, cols = binary_raster.shape

        # Count actual Black-Black (1-1) joins using 4-connectivity
        bb_joins = 0
        total_joins = 0

        # Horizontal adjacencies
        for i in range(rows):
            for j in range(cols - 1):
                if binary_raster[i, j] == 1 and binary_raster[i, j + 1] == 1:
                    bb_joins += 1
                total_joins += 1

        # Vertical adjacencies
        for i in range(rows - 1):
            for j in range(cols):
                if binary_raster[i, j] == 1 and binary_raster[i + 1, j] == 1:
                    bb_joins += 1
                total_joins += 1

        if total_joins == 0:
            return 0

        # Calculate expected BB joins under random distribution
        p = np.sum(binary_raster) / binary_raster.size  # Proportion of 1s
        expected_bb = total_joins * p * p

        # Calculate variance (simplified version)
        variance_bb = total_joins * p * p * (1 - p * p)

        if variance_bb <= 0:
            return 0

        # Standardized Join Count BB
        standardized_bb = (bb_joins - expected_bb) / np.sqrt(variance_bb)

        return standardized_bb

    except Exception as e:
        warnings.warn(f"Could not calculate Join Count BB: {e}")
        return 0


def calculate_hansens_uniformity(binary_raster):
    """
    Calculate Hansen's Uniformity Index for binary spatial patterns.

    This measures the uniformity of the spatial distribution of binary values.
    Based on the variance in local density across the landscape.

    Returns:
    --------
    float : Hansen's Uniformity Index (0-1)
        - 0 = maximum clustering/segregation
        - 1 = maximum uniformity/mixing
        - 0.5 = random distribution
    """
    try:
        rows, cols = binary_raster.shape

        # Use a moving window to calculate local densities
        window_size = min(5, min(rows, cols) // 3)  # Adaptive window size
        if window_size < 3:
            window_size = 3

        local_densities = []

        # Calculate local density for each possible window position
        for i in range(rows - window_size + 1):
            for j in range(cols - window_size + 1):
                window = binary_raster[i:i + window_size, j:j + window_size]
                local_density = np.mean(window)
                local_densities.append(local_density)

        if len(local_densities) == 0:
            return 0.5  # Default to random

        local_densities = np.array(local_densities)

        # Calculate observed variance in local densities
        observed_variance = np.var(local_densities)

        # Calculate global proportion
        global_proportion = np.mean(binary_raster)

        # Calculate maximum possible variance (occurs with maximum segregation)
        # This happens when windows are either all 0s or all 1s
        max_variance = global_proportion * (1 - global_proportion)

        if max_variance <= 0:
            return 1.0  # Uniform (all same value)

        # Hansen's Uniformity Index
        uniformity = 1 - (observed_variance / max_variance)

        # Ensure bounds [0, 1]
        uniformity = max(0, min(1, uniformity))

        return uniformity

    except Exception as e:
        warnings.warn(f"Could not calculate Hansen's Uniformity: {e}")
        return 0.5


def calculate_geary_c_binary(binary_raster):
    """
    Calculate Geary's C for binary data (alternative to Moran's I).

    Geary's C is more appropriate for binary data than Moran's I because
    it focuses on differences between neighboring values rather than covariance.

    Returns:
    --------
    float : Geary's C statistic
        - C < 1 = positive spatial autocorrelation (clustering)
        - C = 1 = no spatial autocorrelation (random)
        - C > 1 = negative spatial autocorrelation (dispersion)
    """
    try:
        rows, cols = binary_raster.shape
        n = rows * cols

        # Calculate mean
        mean_val = np.mean(binary_raster)

        # Calculate numerator: sum of squared differences between neighbors
        numerator = 0
        w_sum = 0  # Sum of weights

        # 4-connectivity neighbors
        for i in range(rows):
            for j in range(cols):
                # Check right neighbor
                if j < cols - 1:
                    diff = (binary_raster[i, j] - binary_raster[i, j + 1]) ** 2
                    numerator += diff
                    w_sum += 1

                # Check bottom neighbor
                if i < rows - 1:
                    diff = (binary_raster[i, j] - binary_raster[i + 1, j]) ** 2
                    numerator += diff
                    w_sum += 1

        # Calculate denominator: sum of squared deviations from mean
        denominator = 0
        for i in range(rows):
            for j in range(cols):
                denominator += (binary_raster[i, j] - mean_val) ** 2

        if denominator == 0 or w_sum == 0:
            return 1.0  # No variation or no neighbors

        # Geary's C formula
        geary_c = ((n - 1) / (2 * w_sum)) * (numerator / denominator)

        return geary_c

    except Exception as e:
        warnings.warn(f"Could not calculate Geary's C: {e}")
        return 1.0


def process_grid_cells_with_raster_association(grid_gdf, raster_folder, aoi_size, output_dir='.',
                                               intersection_method='mosaic'):
    """
    Process grid cells and calculate landscape metrics for each, handling multiple intersecting rasters.

    Parameters:
    -----------
    grid_gdf : geopandas.GeoDataFrame
        Grid with cell_id column and geometry
    raster_folder : str
        Path to folder containing raster files
    aoi_size : str
        AOI size identifier for CSV naming
    output_dir : str
        Output directory for CSV files
    intersection_method : str
        Method to handle multiple intersecting rasters:
        - 'mosaic': Mosaic all intersecting rasters
        - 'largest': Use the raster with largest intersection area
        - 'first': Use the first intersecting raster found

    Returns:
    --------
    pandas.DataFrame : Results dataframe
    """
    results_list = []

    print(f"Processing {len(grid_gdf)} grid cells for AOI size: {aoi_size}")

    # Build spatial index for efficient raster lookup
    print("Building spatial index for raster files...")
    spatial_index, raster_info = build_raster_spatial_index(raster_folder)

    if spatial_index is not None:
        print(f"Using spatial index with {len(raster_info)} rasters")
    else:
        print("Using brute force intersection checking")

    for idx, row in tqdm(grid_gdf.iterrows(), total=len(grid_gdf), desc=f"Processing {aoi_size}"):
        cell_id = row['cell_id']
        geometry = row.geometry

        try:
            # Find intersecting rasters using spatial index
            if spatial_index is not None:
                intersecting_rasters = get_intersecting_rasters_indexed(
                    geometry, spatial_index, raster_info
                )
            else:
                intersecting_rasters = get_intersecting_rasters_bruteforce(
                    geometry, raster_folder
                )

            if not intersecting_rasters:
                print(f"No intersecting rasters found for cell_id {cell_id}")
                # Add a record with zero values
                metrics = {
                    'canopy_extent': 0,
                    'morans_i': 0,
                    'join_count_bb': 0,
                    'hansens_uniformity': 0.5,
                    'geary_c': 1.0,
                    'edge_density': 0,
                    'clumpy': 0,
                    'number_of_patches': 0,
                    'avg_patch_size': 0,
                    'patch_size_std': 0,
                    'patch_size_median': 0,
                    'patch_size_min': 0,
                    'patch_size_max': 0,
                    'normalized_lsi': 0,
                    'landscape_type': 'no_data',
                    'cell_id': cell_id,
                    'num_intersecting_rasters': 0
                }
                results_list.append(metrics)
                continue

            # Handle intersection method
            if intersection_method == 'mosaic' or len(intersecting_rasters) > 1:
                # Mosaic intersecting rasters
                raster_data, transform = mosaic_rasters(
                    intersecting_rasters,
                    geometry,
                    grid_gdf.crs
                )
            else:
                # Single raster case
                with rasterio.open(intersecting_rasters[0]) as src:
                    out_image, transform = rasterio.mask.mask(
                        src, [geometry], crop=True, all_touched=False
                    )
                    raster_data = out_image[0]

            # Skip if no valid data
            if raster_data is None or raster_data.size == 0:
                print(f"No valid raster data for cell_id {cell_id}")
                continue

            # Calculate cell size from transform
            cell_size = abs(transform.a)  # Pixel width

            # Calculate metrics
            metrics = calculate_landscape_metrics(raster_data, cell_size=cell_size)
            metrics['cell_id'] = cell_id
            metrics['num_intersecting_rasters'] = len(intersecting_rasters)

            results_list.append(metrics)

        except Exception as e:
            print(f"Error processing cell_id {cell_id}: {e}")
            continue

    # Create DataFrame
    results_df = pd.DataFrame(results_list)

    # Save to CSV
    output_filename = f"canopy_metrics_{aoi_size}.csv"
    output_path = os.path.join(output_dir, output_filename)
    results_df.to_csv(output_path, index=False)

    print(f"Results saved to: {output_path}")
    print(f"Processed {len(results_df)} grid cells successfully")

    return results_df


# Main execution
if __name__ == "__main__":
    # Configuration
    raster_folder = "Meta CHM Binary"
    output_directory = "output"

    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Process each grid size
    for grid in grids:
        # Determine AOI size from grid
        if 'grid_1km' in str(grid):
            aoi_size = "1km"
        elif 'grid_20km' in str(grid):
            aoi_size = "20km"
        elif 'grid_40km' in str(grid):
            aoi_size = "40km"
        elif 'grid_100km' in str(grid):
            aoi_size = "100km"
        else:
            aoi_size = "unknown"

        print(f"\n=== Processing {aoi_size} grid ===")

        # Process grid cells
        results_df = process_grid_cells_with_raster_association(
            grid,
            raster_folder,
            aoi_size,
            output_directory,
            intersection_method='mosaic'  # Change this if needed
        )

        # Print summary statistics
        print(f"\nSummary for {aoi_size}:")
        print(f"Total cells processed: {len(results_df)}")
        print(f"Average canopy extent: {results_df['canopy_extent'].mean():.2f}%")
        print(f"Cells with no intersecting rasters: {sum(results_df['num_intersecting_rasters'] == 0)}")
        print(f"Cells with multiple intersecting rasters: {sum(results_df['num_intersecting_rasters'] > 1)}")
