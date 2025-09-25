# This is the file that stores functions

"""
Functions:
- list_s3_directories(): Explore S3 bucket structure
- download_tiles_geojson(): Download and filter tiles from S3
- create_grid(): Generate square grid cells of specified size
- spatial_filter_grid(): Filter grid cells to those within tile boundaries
- save_grid(): Save grid to GeoPackage format
- run_grid_generation(): Main execution function (called from main.py)
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
            print(f"  Average tile size: {avg_area:,.2f} m²")

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


def placeholder_analysis(tiles_gdf, config):
    print("☀️ Hello, World!")
    return True
