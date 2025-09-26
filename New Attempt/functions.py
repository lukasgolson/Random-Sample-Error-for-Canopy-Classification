# This is the file that stores functions

"""
Functions:
- list_s3_directories(): Explore S3 bucket structure
- download_tiles_geojson(): Download and filter tiles from S3
- create_grid(): Generate square grid cells of specified size
- spatial_filter_grid(): Filter grid cells to those within tile boundaries
- save_grid(): Save grid to GeoPackage format
- run_grid_generation(): Main execution function (called from pre-processing.py)
"""

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
from tqdm import tqdm
import pandas as pd
import io
import rasterio

def list_s3_directories(bucket_name, prefix=''):
    """
    List directories and files in an S3 bucket with folder-like structure.

    Parameters:
    bucket_name (str): Name of the S3 bucket
    prefix (str): S3 prefix/folder path to explore

    Returns:
    None: Prints directory structure to console
    """
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    try:
        # Use delimiter='/' to get folder-like structure
        response = s3.list_objects_v2(
            Bucket=bucket_name,
            Prefix=prefix,
            Delimiter='/'
        )

        print(f"Directories in s3://{bucket_name}/{prefix}")
        print("-" * 60)

        # Show subdirectories
        if 'CommonPrefixes' in response:
            for prefix_info in response['CommonPrefixes']:
                folder_name = prefix_info['Prefix']
                print(f"📁 {folder_name}")

        # Show files in current directory (not in subdirectories)
        if 'Contents' in response:
            for obj in response['Contents']:
                # Skip the prefix itself if it's a directory marker
                if obj['Key'] != prefix:
                    size = obj['Size']
                    key = obj['Key']
                    modified = obj['LastModified'].strftime('%Y-%m-%d %H:%M:%S')
                    print(f"📄 {modified} {size:>12} {key}")

        if 'CommonPrefixes' not in response and 'Contents' not in response:
            print("No directories or files found")

    except Exception as e:
        print(f"Error: {e}")


def download_tiles_geojson(bucket_name, key, bbox=None, show_plot=False):
    """
    Download and filter the tiles.geojson file from Meta's forest data.

    Parameters:
    bucket_name (str): S3 bucket name
    key (str): S3 object key for the geojson file
    bbox (list): Bounding box as [min_lon, min_lat, max_lon, max_lat]
    show_plot (bool): If True, display a map of the tiles

    Returns:
    gpd.GeoDataFrame: Filtered tiles geodataframe
    """
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    try:
        print(f"Downloading {key}...")
        response = s3.get_object(Bucket=bucket_name, Key=key)
        geojson_data = response['Body'].read()

        # Read with geopandas
        gdf = gpd.read_file(io.BytesIO(geojson_data))
        print(f"  Successfully loaded {key}")
        print(f"    Original shape: {gdf.shape}")
        print(f"    CRS: {gdf.crs}")
        print(f"    Columns: {list(gdf.columns)}")

        # Apply bounding box filter if provided
        if bbox is not None:
            min_lon, min_lat, max_lon, max_lat = bbox
            print(f"  Applying bounding box filter: {bbox}")
            gdf = gdf.cx[min_lon:max_lon, min_lat:max_lat]
            print(f"  Filtered to {len(gdf)} tiles in AOI")

            # Calculate average polygon size in sq. metres
            gdf_projected = gdf.to_crs(epsg=5070)  # Albers projection for North America
            avg_area = gdf_projected.area.mean()
            print(f"  Average tile size: {avg_area / 1e6:,.2f} km²")

        # Optional visualization
        if show_plot:
            _plot_tiles(gdf, bbox)

        return gdf

    except Exception as e:
        print(f"Error downloading tiles: {e}")
        return None


def _plot_tiles(gdf, bbox=None):
    """
    Helper function to plot tiles with optional bounding box.

    Parameters:
    gdf (gpd.GeoDataFrame): Tiles geodataframe to plot
    bbox (list): Optional bounding box to highlight
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    gdf.plot(ax=ax, facecolor='none', edgecolor='blue', alpha=0.7, linewidth=0.5)

    # Add bounding box rectangle if specified
    if bbox is not None:
        min_lon, min_lat, max_lon, max_lat = bbox
        rect = Rectangle((min_lon, min_lat), max_lon - min_lon, max_lat - min_lat,
                         linewidth=3, edgecolor='red', facecolor='none', linestyle='-')
        ax.add_patch(rect)
        ax.text(min_lon + 5, min_lat + .5, "AOI Bounding Box",
                fontsize=20, color='red', ha='left', va='bottom')
        ax.set_xlim(min_lon - 5, max_lon + 5)
        ax.set_ylim(min_lat - 2, max_lat + 2)

    ax.set_xlabel('Longitude', fontsize=20)
    ax.set_ylabel('Latitude', fontsize=20)
    ax.xaxis.set_major_locator(plt.MultipleLocator(10))
    ax.yaxis.set_major_locator(plt.MultipleLocator(5))
    ax.tick_params(axis='both', labelsize=18)
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()


def create_grid(bounds, cell_size_meters, crs='EPSG:4326'):
    """
    Create a square grid of specified cell size.

    Parameters:
    bounds (tuple): Bounding box as (minx, miny, maxx, maxy) in geographic coordinates
    cell_size_meters (int): Size of each grid cell in meters
    crs (str): Coordinate reference system for the bounds

    Returns:
    gpd.GeoDataFrame: Grid cells as polygons with metadata
    """
    print(f"Creating {cell_size_meters}m x {cell_size_meters}m grid...")

    # Convert to a projected CRS for accurate distance measurements
    # Using Albers Equal Area Conic for North America (EPSG:5070)
    temp_gdf = gpd.GeoDataFrame([1], geometry=[Polygon.from_bounds(*bounds)], crs=crs)
    temp_projected = temp_gdf.to_crs('EPSG:5070')
    projected_bounds = temp_projected.total_bounds

    minx, miny, maxx, maxy = projected_bounds

    # Calculate number of cells in each direction
    n_cells_x = int(np.ceil((maxx - minx) / cell_size_meters))
    n_cells_y = int(np.ceil((maxy - miny) / cell_size_meters))

    print(f"Grid dimensions: {n_cells_x} x {n_cells_y} = {n_cells_x * n_cells_y:,} cells")

    # Create grid coordinates
    x_coords = np.linspace(minx, minx + n_cells_x * cell_size_meters, n_cells_x + 1)
    y_coords = np.linspace(miny, miny + n_cells_y * cell_size_meters, n_cells_y + 1)

    # Create grid polygons
    polygons = []
    grid_ids = []

    print("Generating grid polygons...")
    for i in tqdm(range(n_cells_x)):
        for j in range(n_cells_y):
            # Create polygon for this cell
            poly = Polygon([
                (x_coords[i], y_coords[j]),
                (x_coords[i + 1], y_coords[j]),
                (x_coords[i + 1], y_coords[j + 1]),
                (x_coords[i], y_coords[j + 1])
            ])
            polygons.append(poly)
            grid_ids.append(f"{cell_size_meters // 1000}km_{i}_{j}")

    # Create GeoDataFrame in projected CRS
    grid_gdf = gpd.GeoDataFrame({
        'grid_id': grid_ids,
        'cell_size_m': cell_size_meters,
        'area_m2': cell_size_meters ** 2
    }, geometry=polygons, crs='EPSG:5070')

    # Convert back to geographic coordinates
    grid_gdf = grid_gdf.to_crs(crs)

    return grid_gdf


def spatial_filter_grid(grid_gdf, tiles_gdf):
    """
    Keep only grid cells that are entirely within tile boundaries.

    Parameters:
    grid_gdf (gpd.GeoDataFrame): Grid cells to filter
    tiles_gdf (gpd.GeoDataFrame): Tiles defining the valid area

    Returns:
    gpd.GeoDataFrame: Filtered grid cells
    """
    print(f"Spatial filtering {len(grid_gdf):,} grid cells...")
    print("This may take a few minutes for large grids...")

    # Ensure both datasets are in the same CRS
    if grid_gdf.crs != tiles_gdf.crs:
        tiles_gdf = tiles_gdf.to_crs(grid_gdf.crs)

    # Create a single geometry representing all tiles combined
    print("Creating union of all tiles...")
    tiles_union = tiles_gdf.unary_union

    # Find grid cells that are entirely within the tiles
    print("Checking which grid cells are within tiles...")
    within_mask = grid_gdf.geometry.within(tiles_union)

    filtered_grid = grid_gdf[within_mask].copy()

    print(f"Kept {len(filtered_grid):,} grid cells ({len(filtered_grid) / len(grid_gdf) * 100:.1f}%)")
    print(f"Removed {len(grid_gdf) - len(filtered_grid):,} grid cells that extended outside tile boundaries")

    return filtered_grid


def save_grid(grid_gdf, filename, cell_size):
    """
    Save grid to GeoPackage format with metadata.

    Parameters:
    grid_gdf (gpd.GeoDataFrame): Grid to save
    filename (str): Output filename
    cell_size (int): Cell size in meters for summary statistics
    """
    # Add some useful metadata
    grid_gdf = grid_gdf.copy()
    grid_gdf['created'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    grid_gdf['total_area_km2'] = (grid_gdf['area_m2'] * len(grid_gdf)) / 1e6

    print(f"Saving {len(grid_gdf):,} grid cells to {filename}")
    grid_gdf.to_file(filename, driver='GPKG')

    # Print summary statistics
    total_area_km2 = (grid_gdf['area_m2'].sum()) / 1e6
    print(f"Summary for {cell_size // 1000}km grid:")
    print(f"  Total cells: {len(grid_gdf):,}")
    print(f"  Total area: {total_area_km2:,.1f} km²")
    print(f"  File saved: {filename}")
    print()


def _get_filename_from_km(cell_size_km):
    """Generate appropriate filename based on cell size in km"""
    if cell_size_km == 0.6:
        return "grid_0.6km.gpkg"
    elif cell_size_km == 1:
        return "grid_1km.gpkg"
    elif cell_size_km == 10:
        return "grid_10km.gpkg"
    else:
        # Fallback for other sizes
        if cell_size_km < 1:
            return f"grid_{cell_size_km:.1f}km.gpkg"
        else:
            return f"grid_{int(cell_size_km)}km.gpkg"


def _get_filename(cell_size):
    """Generate appropriate filename based on cell size in meters (legacy function)"""
    if cell_size == 600:
        return "grid_0.6km.gpkg"
    elif cell_size == 1000:
        return "grid_1km.gpkg"
    elif cell_size == 10000:
        return "grid_10km.gpkg"
    else:
        # Fallback for other sizes
        if cell_size < 1000:
            return f"grid_{cell_size/1000:.1f}km.gpkg"
        else:
            return f"grid_{cell_size//1000}km.gpkg"


def _get_grid_label(cell_size):
    """Generate appropriate grid label for display based on meters (legacy function)"""
    if cell_size == 600:
        return "0.6km"
    elif cell_size == 1000:
        return "1km"
    elif cell_size == 10000:
        return "10km"
    else:
        # Fallback for other sizes
        if cell_size < 1000:
            return f"{cell_size/1000:.1f}km"
        else:
            return f"{cell_size//1000}km"


def run_grid_generation(config):
    """
    Main function to generate all grids using provided configuration.

    Parameters:
    config (dict): Configuration dictionary with all settings
                   - grid_sizes should be in kilometers (e.g., [0.6, 1, 10])

    Returns:
    bool: True if successful, False otherwise
    """

    print("Grid Cell Generation Script")
    print("=" * 50)
    print(f"Target area: USA")
    print(f"Bounding box: {config['bbox']}")
    print(f"Grid sizes: {config['grid_sizes']} km")
    print("=" * 50)

    # Optional: Explore S3 bucket structure
    if config['explore_s3_structure']:
        print("\n🔍 EXPLORING S3 BUCKET STRUCTURE...")
        list_s3_directories(config['bucket_name'], 'forests/v1/alsgedi_global_v6_float/')
        print("\n" + "=" * 60 + "\n")

    # Step 1: Download and filter tiles
    print("\n📥 STEP 1: LOADING TILES DATA...")
    tiles_gdf = download_tiles_geojson(
        config['bucket_name'],
        config['tiles_key'],
        bbox=config['bbox'],
        show_plot=config['show_tiles_plot']
    )

    if tiles_gdf is None:
        print("❌ Failed to load tiles data. Exiting.")
        return False

    # Get overall bounds for grid creation
    bounds = tiles_gdf.total_bounds
    print(f"\n✅ Tiles loaded successfully")
    print(f"   Tiles extent: {bounds}")
    print(f"   Number of tiles: {len(tiles_gdf)}")

    # Step 2-4: Process each grid size
    successful_grids = []

    for cell_size_km in config['grid_sizes']:
        # Convert km to meters for internal calculations
        cell_size_meters = int(cell_size_km * 1000)
        grid_label = f"{cell_size_km}km"

        print(f"\n{'🔲 PROCESSING ' + grid_label + ' GRID':=^60}")

        try:
            # Create grid
            print(f"\n⚙️  Step 2: Creating {grid_label} grid...")
            grid = create_grid(bounds, cell_size_meters, crs=tiles_gdf.crs)

            # Spatial filter
            print(f"\n🔍 Step 3: Spatially filtering {grid_label} grid...")
            filtered_grid = spatial_filter_grid(grid, tiles_gdf)

            # Save grid
            print(f"\n💾 Step 4: Saving {grid_label} grid...")
            filename = _get_filename_from_km(cell_size_km)
            save_grid(filtered_grid, filename, cell_size_meters)

            successful_grids.append(filename)
            print(f"✅ {grid_label} grid completed successfully!")

            # Clear memory
            del grid, filtered_grid

        except Exception as e:
            print(f"❌ Error processing {grid_label} grid: {e}")
            continue

    # Final summary
    print(f"\n{'🎉 GRID GENERATION COMPLETE':=^60}")
    if successful_grids:
        print("✅ Successfully created grids:")
        for filename in successful_grids:
            print(f"   📄 {filename}")
    else:
        print("❌ No grids were created successfully")
        return False

    return True


def check_chm_file_sizes(tiles_gdf, bucket_name, sample_size=200):
    """
    Check average file size of CHM GeoTIFF files corresponding to tiles.

    Parameters:
    tiles_gdf (gpd.GeoDataFrame): Tiles data with quadkey/tile column
    bucket_name (str): S3 bucket name
    sample_size (int): Number of files to sample for size estimation

    Returns:
    dict: File size statistics
    """
    import random

    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    print(f"🔍 CHECKING CHM FILE SIZES...")

    # Get tile identifiers - check what column contains the quadkey
    if 'tile' in tiles_gdf.columns:
        tile_ids = tiles_gdf['tile'].tolist()
        id_column = 'tile'
    elif 'quadkey' in tiles_gdf.columns:
        tile_ids = tiles_gdf['quadkey'].tolist()
        id_column = 'quadkey'
    else:
        print("❌ Could not find 'tile' or 'quadkey' column in tiles data")
        return None

    print(f"   Found {len(tile_ids)} tiles using column '{id_column}'")

    # Sample random tiles to check
    sample_tiles = random.sample(tile_ids, min(sample_size, len(tile_ids)))
    print(f"   Sampling {len(sample_tiles)} tiles for file size analysis...")

    file_sizes = []
    found_files = 0
    missing_files = 0

    for i, tile_id in enumerate(sample_tiles):
        if i % 10 == 0:  # Progress indicator
            print(f"   Progress: {i + 1}/{len(sample_tiles)}")

        # Construct S3 key for CHM file
        chm_key = f'forests/v1/alsgedi_global_v6_float/chm/{tile_id}.tif'

        try:
            # Get object metadata (HEAD request - doesn't download file)
            response = s3.head_object(Bucket=bucket_name, Key=chm_key)
            file_size_bytes = response['ContentLength']
            file_sizes.append(file_size_bytes)
            found_files += 1

        except Exception as e:
            missing_files += 1
            if missing_files <= 5:  # Only show first few missing files
                print(f"   ⚠️  Missing: {chm_key}")

    if not file_sizes:
        print("❌ No CHM files found")
        return None

    # Calculate statistics
    avg_size_bytes = sum(file_sizes) / len(file_sizes)
    min_size_bytes = min(file_sizes)
    max_size_bytes = max(file_sizes)
    total_estimated_gb = (avg_size_bytes * len(tile_ids)) / (1024 ** 3)

    # Convert to human readable
    def bytes_to_human(bytes_val):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_val < 1024.0:
                return f"{bytes_val:.1f} {unit}"
            bytes_val /= 1024.0
        return f"{bytes_val:.1f} TB"

    stats = {
        'sample_size': len(sample_tiles),
        'found_files': found_files,
        'missing_files': missing_files,
        'avg_size_bytes': avg_size_bytes,
        'avg_size_human': bytes_to_human(avg_size_bytes),
        'min_size_human': bytes_to_human(min_size_bytes),
        'max_size_human': bytes_to_human(max_size_bytes),
        'total_tiles': len(tile_ids),
        'estimated_total_size_gb': total_estimated_gb
    }

    # Print results
    print(f"\n📊 CHM FILE SIZE ANALYSIS:")
    print(f"   Sample analyzed: {found_files}/{len(sample_tiles)} files")
    print(f"   Missing files: {missing_files}")
    print(f"   Average file size: {stats['avg_size_human']}")
    print(f"   Size range: {stats['min_size_human']} - {stats['max_size_human']}")
    print(f"   Total tiles in dataset: {stats['total_tiles']:,}")
    print(f"   Estimated total size: {stats['estimated_total_size_gb']:.1f} GB")
    print(f"   Storage space needed: ~{stats['estimated_total_size_gb'] * 1.2:.1f} GB (with 20% buffer)")

    return stats


def generate_systematic_sample_points(grid_gdf, points_per_cell):
    """
    Generate systematic sample points within each grid cell.
    Points have the same relative positions within each cell.

    Parameters:
    grid_gdf (gpd.GeoDataFrame): Grid cells
    points_per_cell (int): Number of sample points per grid cell

    Returns:
    gpd.GeoDataFrame: Sample points with grid_id reference
    """
    import numpy as np
    from shapely.geometry import Point
    import math

    print(f"🎯 GENERATING {points_per_cell:,} SYSTEMATIC SAMPLE POINTS PER GRID CELL...")
    print(f"   Processing {len(grid_gdf):,} grid cells...")

    # Calculate grid layout for points (as square as possible)
    n_points = points_per_cell
    cols = int(math.ceil(math.sqrt(n_points)))
    rows = int(math.ceil(n_points / cols))

    print(f"   Point pattern: {rows} rows × {cols} columns = {rows * cols} points per cell")

    # Generate relative positions (0 to 1) within a unit square
    x_positions = np.linspace(0.1, 0.9, cols)  # 10% buffer from edges
    y_positions = np.linspace(0.1, 0.9, rows)  # 10% buffer from edges

    # Create all combinations of x,y positions
    relative_points = []
    for y in y_positions:
        for x in x_positions:
            relative_points.append((x, y))

    # Only take the number we need
    relative_points = relative_points[:n_points]

    print(f"   Using {len(relative_points)} points per cell (some cells may have fewer due to grid layout)")

    # Generate points for each grid cell
    all_points = []
    all_grid_ids = []

    for idx, row in tqdm(grid_gdf.iterrows(), total=len(grid_gdf), desc="Generating points"):
        grid_cell = row.geometry
        grid_id = row['grid_id']

        # Get bounding box of this grid cell
        minx, miny, maxx, maxy = grid_cell.bounds
        cell_width = maxx - minx
        cell_height = maxy - miny

        # Generate points at same relative positions within this cell
        cell_points = []
        for rel_x, rel_y in relative_points:
            # Convert relative position to absolute coordinates
            abs_x = minx + (rel_x * cell_width)
            abs_y = miny + (rel_y * cell_height)

            point = Point(abs_x, abs_y)

            # Ensure point is actually inside the grid cell (for irregular shapes)
            if grid_cell.contains(point):
                cell_points.append(point)
                all_grid_ids.append(grid_id)

        all_points.extend(cell_points)

    # Create GeoDataFrame of sample points
    points_gdf = gpd.GeoDataFrame({
        'point_id': range(len(all_points)),
        'grid_id': all_grid_ids,
        'sample_type': 'systematic'
    }, geometry=all_points, crs=grid_gdf.crs)

    print(f"✅ Generated {len(points_gdf):,} total sample points")
    print(f"   Average points per grid cell: {len(points_gdf) / len(grid_gdf):.1f}")

    return points_gdf


def generate_sample_points_for_grids(grid_sizes_dict, active_bbox):
    """
    Generate sample points for multiple grid files.

    Parameters:
    grid_sizes_dict (dict): {grid_size_km: points_per_cell}
    active_bbox (list): Bounding box for naming

    Returns:
    dict: {grid_size: points_gdf}
    """
    results = {}

    for grid_size_km, points_per_cell in grid_sizes_dict.items():
        print(f"\n{'=' * 60}")
        print(f"PROCESSING {grid_size_km}km GRID")
        print(f"{'=' * 60}")

        # Load the grid file
        grid_filename = _get_filename_from_km(grid_size_km)

        try:
            grid_gdf = gpd.read_file(grid_filename)
            print(f"📂 Loaded grid: {grid_filename}")
            print(f"   Grid cells: {len(grid_gdf):,}")

            # Generate sample points
            points_gdf = generate_systematic_sample_points(grid_gdf, points_per_cell)

            # Save sample points
            points_filename = f"sample_points_{grid_size_km}km.gpkg"
            points_gdf.to_file(points_filename, driver='GPKG')
            print(f"💾 Saved sample points: {points_filename}")

            results[grid_size_km] = points_gdf

        except Exception as e:
            print(f"❌ Error processing {grid_size_km}km grid: {e}")
            continue

    return results


def download_chm(tiles_gdf, bucket_name, output_dir='chm_binary', binary_threshold=2.0):
    """
    Download CHM tiles from S3 and convert to binary rasters.

    Parameters:
    tiles_gdf (gpd.GeoDataFrame): Tiles data with tile/quadkey identifiers
    bucket_name (str): S3 bucket name containing CHM data
    output_dir (str): Local directory to save binary CHM files
    binary_threshold (float): Threshold value - values >= threshold become 1, others become 0

    Returns:
    dict: Summary of download and processing results
    """
    import rasterio
    import numpy as np
    from pathlib import Path

    print(f"🌲 DOWNLOADING AND PROCESSING CHM TILES...")
    print(f"   Tiles to process: {len(tiles_gdf):,}")
    print(f"   Binary threshold: values >= {binary_threshold} → 1, values < {binary_threshold} → 0")
    print(f"   Output directory: {output_dir}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Initialize S3 client
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    # Get tile identifiers
    if 'tile' in tiles_gdf.columns:
        tile_ids = tiles_gdf['tile'].tolist()
        id_column = 'tile'
    elif 'quadkey' in tiles_gdf.columns:
        tile_ids = tiles_gdf['quadkey'].tolist()
        id_column = 'quadkey'
    else:
        print("❌ Could not find 'tile' or 'quadkey' column in tiles data")
        return None

    print(f"   Using column '{id_column}' for tile identifiers")

    # Track processing results
    results = {
        'total_tiles': len(tile_ids),
        'downloaded': 0,
        'processed': 0,
        'failed': 0,
        'errors': [],
        'output_files': []
    }

    # Process each tile
    for i, tile_id in enumerate(tqdm(tile_ids, desc="Processing CHM tiles")):
        try:
            # Construct S3 key for CHM file
            chm_key = f'forests/v1/alsgedi_global_v6_float/chm/{tile_id}.tif'

            # Download CHM file to memory
            try:
                response = s3.get_object(Bucket=bucket_name, Key=chm_key)
                chm_data = response['Body'].read()
                results['downloaded'] += 1
            except Exception as e:
                results['failed'] += 1
                results['errors'].append(f"Download failed for {tile_id}: {str(e)}")
                continue

            # Read raster data using rasterio
            with rasterio.io.MemoryFile(chm_data) as memfile:
                with memfile.open() as src:
                    # Read the raster data
                    chm_array = src.read(1)  # Read first (and typically only) band
                    profile = src.profile.copy()

                    # Convert to binary based on threshold
                    # Values >= threshold become 1, others become 0
                    binary_array = np.where(chm_array >= binary_threshold, 1, 0).astype(np.uint8)

                    # Handle NoData values (keep them as NoData)
                    if src.nodata is not None:
                        nodata_mask = chm_array == src.nodata
                        binary_array[nodata_mask] = 255  # Use 255 as NoData for uint8

                    # Update profile for binary output
                    profile.update({
                        'dtype': 'uint8',
                        'nodata': 255,
                        'compress': 'lzw'
                    })

                    # Save binary raster
                    output_file = output_path / f"{tile_id}_binary.tif"
                    with rasterio.open(output_file, 'w', **profile) as dst:
                        dst.write(binary_array, 1)

                    results['processed'] += 1
                    results['output_files'].append(str(output_file))

        except Exception as e:
            results['failed'] += 1
            results['errors'].append(f"Processing failed for {tile_id}: {str(e)}")
            continue

        # Progress update every 10 files
        if (i + 1) % 10 == 0:
            print(f"   Progress: {i + 1}/{len(tile_ids)} tiles processed")

    # Print summary
    print(f"\n📊 CHM PROCESSING SUMMARY:")
    print(f"   Total tiles: {results['total_tiles']:,}")
    print(f"   Successfully downloaded: {results['downloaded']:,}")
    print(f"   Successfully processed: {results['processed']:,}")
    print(f"   Failed: {results['failed']:,}")
    print(f"   Success rate: {results['processed'] / results['total_tiles'] * 100:.1f}%")
    print(f"   Output files saved to: {output_path.absolute()}")

    # Show first few errors if any
    if results['errors']:
        print(f"\n⚠️  First few errors:")
        for error in results['errors'][:5]:
            print(f"     {error}")
        if len(results['errors']) > 5:
            print(f"     ... and {len(results['errors']) - 5} more errors")

    return results


def create_chm_mosaic(binary_files_dir, output_mosaic_path, tiles_gdf=None):
    """
    Create a mosaic from binary CHM tiles.

    Parameters:
    binary_files_dir (str): Directory containing binary CHM files
    output_mosaic_path (str): Output path for the mosaic
    tiles_gdf (gpd.GeoDataFrame): Optional tiles geodataframe for spatial reference

    Returns:
    str: Path to created mosaic file
    """
    import rasterio
    from rasterio.merge import merge
    from pathlib import Path
    import glob

    print(f"🗺️  CREATING CHM MOSAIC...")

    # Find all binary CHM files
    binary_files = glob.glob(str(Path(binary_files_dir) / "*_binary.tif"))

    if not binary_files:
        print(f"❌ No binary CHM files found in {binary_files_dir}")
        return None

    print(f"   Found {len(binary_files)} binary CHM files")

    # Open all files
    src_files = []
    try:
        for file_path in binary_files:
            src = rasterio.open(file_path)
            src_files.append(src)

        # Create mosaic
        print("   Merging tiles into mosaic...")
        mosaic, out_transform = merge(src_files, nodata=255)

        # Update metadata
        out_meta = src_files[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_transform,
            "compress": "lzw",
            "nodata": 255
        })

        # Write mosaic
        print(f"   Saving mosaic to: {output_mosaic_path}")
        with rasterio.open(output_mosaic_path, "w", **out_meta) as dest:
            dest.write(mosaic)

        print(f"✅ Mosaic created successfully!")
        print(f"   Dimensions: {mosaic.shape[2]} x {mosaic.shape[1]} pixels")
        print(f"   File size: {Path(output_mosaic_path).stat().st_size / 1024 / 1024:.1f} MB")

        return output_mosaic_path

    finally:
        # Close all source files
        for src in src_files:
            src.close()


def plot_sample_points_map(points_gdf, tiles_gdf, bbox, grid_size_km):
    """
    Plot sample points overlaid on tiles and grid.

    Parameters:
    points_gdf (gpd.GeoDataFrame): Sample points to plot
    tiles_gdf (gpd.GeoDataFrame): Tiles geodataframe
    bbox (list): Bounding box for map extent
    grid_size_km (int): Grid size in kilometers for title

    Returns:
    None: Displays the plot
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import numpy as np

    print(f"   Creating sample points map for {grid_size_km}km grid...")

    fig, ax = plt.subplots(1, 1, figsize=(15, 12))

    # Plot tiles as background
    if tiles_gdf is not None:
        tiles_gdf.plot(ax=ax, facecolor='lightgreen', edgecolor='darkgreen',
                       alpha=0.3, linewidth=0.5)

    # Sample a subset of points if there are too many (for performance)
    if len(points_gdf) > 5000:
        sample_size = 5000
        points_sample = points_gdf.sample(n=sample_size, random_state=42)
        print(f"   Showing {sample_size:,} of {len(points_gdf):,} points for visualization")
    else:
        points_sample = points_gdf
        print(f"   Showing all {len(points_gdf):,} sample points")

    # Plot sample points
    points_sample.plot(ax=ax, color='red', markersize=1, alpha=0.7)

    # Add bounding box if provided
    if bbox is not None:
        min_lon, min_lat, max_lon, max_lat = bbox
        rect = Rectangle((min_lon, min_lat), max_lon - min_lon, max_lat - min_lat,
                         linewidth=2, edgecolor='blue', facecolor='none', linestyle='--')
        ax.add_patch(rect)
        ax.set_xlim(min_lon - 1, max_lon + 1)
        ax.set_ylim(min_lat - 0.5, max_lat + 0.5)

    # Formatting
    ax.set_xlabel('Longitude', fontsize=14)
    ax.set_ylabel('Latitude', fontsize=14)
    ax.set_title(f'Sample Points Distribution - {grid_size_km}km Grid\n{len(points_gdf):,} total points',
                 fontsize=16, pad=20)

    # Create custom legend (fixes the warning)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='lightgreen',
               markersize=10, alpha=0.3, label='Forest Tiles'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markersize=5, alpha=0.7, label='Sample Points'),
        Line2D([0], [0], color='blue', linestyle='--', linewidth=2, label='AOI Boundary')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

    ax.grid(True, alpha=0.3)

    # Add statistics text box
    stats_text = f'Grid Size: {grid_size_km}km\nTotal Points: {len(points_gdf):,}\nUnique Grids: {points_gdf["grid_id"].nunique():,}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()


def placeholder_analysis(tiles_gdf, config):
    print("☀️ Hello, World!")
    return True
