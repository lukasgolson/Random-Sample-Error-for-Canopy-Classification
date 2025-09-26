# This is the file that processes the CHM files

# =============================================================================
# SELECT WHICH PROCESSES TO RUN
# =============================================================================
#region

# Run through pre-processing with test settings (low complexity, fast run)
USE_TEST_SETTINGS = True        # Set to True to run a test of the code using a larger BBOX and AOI

# Explore the CHM tiles
EXPLORE_S3_STRUCTURE = False    # Set to True to explore S3 bucket structure
CHECK_CHM_FILE_SIZES = False    # Set to True to analyze CHM file sizes
SHOW_TILES_PLOT = True         # Set to True to visualize the tiles

# Grid generation and mapping
GENERATE_GRIDS = False          # Set to True to generate grids

# Sample point generation and mapping
GENERATE_SAMPLE_POINTS = True   # Set to True to generate systematic sample points
SHOW_SAMPLE_POINTS_MAP = True   # Set to True to show a map of the generated sample points

# Download the CHMs and merge
DOWNLOAD_CHM = False             # Set to True to downlaod the Meta CHM tiles, converted to binary
CREATE_CHM_MOSAIC = False       # Set to True to merge the Meta CHM tiles

#endregion

# =============================================================================
# CONFIGURATION OPTIONS - Modify these as needed
# =============================================================================
#region

# Geographic settings
BBOX = [-127, 24, -66.9, 49]   # Bounding box: [min_lon, min_lat, max_lon, max_lat]
TEST_BBOX = [-80, 40, -70, 45]      # Small NY/New England region

# Grid specifications
GRID_SIZES = [1, 20, 40]    # Neighbourhood (1 km), city (20 km), and region (35 km) grids
TEST_GRID_SIZES = [100]     # Large 100km grids for speed

# Sample point specifications
base_points = 1000 # Number of points per 1 sq km area
SAMPLE_POINTS_CONFIG = {1: base_points, 20: (20*20*base_points), 40: (40*40*base_points), 100: base_points}

# S3 data source configuration
BUCKET_NAME = 'dataforgood-fb-data'
TILES_KEY = 'forests/v1/alsgedi_global_v6_float/tiles.geojson'

# CHM download configuration
CHM_OUTPUT_DIR = "chm_binary"   # Output directory
CHM_BINARY_THRESHOLD = 2.0      # Binary threshold value

#endregion

from functions import *

if __name__ == "__main__":

    # Determine which settings to use
    active_bbox = TEST_BBOX if USE_TEST_SETTINGS else BBOX
    active_grid_sizes = TEST_GRID_SIZES if USE_TEST_SETTINGS else GRID_SIZES
    mode = "TESTING" if USE_TEST_SETTINGS else "PRODUCTION"

    print("Access and Process Meta CHM Script")
    print("=" * 50)
    print(f"Target area: contiguous USA")
    print(f"Bounding box: {active_bbox}")
    print(f"Configuration:")
    print(f"  - Test run: {USE_TEST_SETTINGS}")
    print(f"  - Explore S3: {EXPLORE_S3_STRUCTURE}")
    print(f"  - Show tiles plot: {SHOW_TILES_PLOT}")
    print(f"  - Generate grids: {GENERATE_GRIDS}")
    print(f"  - Generate sample points: {GENERATE_SAMPLE_POINTS}")
    print(f"  - Download CHM: {DOWNLOAD_CHM}")
    print(f"  - Merge CHM: {CREATE_CHM_MOSAIC}")
    print(f"  - Grid sizes: {active_grid_sizes} km")
    print("=" * 50)
    print("\n")

    success = True
    tiles_gdf = None

    # Step 1: Optional S3 exploration
    if EXPLORE_S3_STRUCTURE:
        print("\n🔍 EXPLORING S3 BUCKET STRUCTURE...")
        list_s3_directories(BUCKET_NAME, 'forests/v1/alsgedi_global_v6_float/')
        print("\n" + "=" * 60 + "\n")

    # Step 2: Load tiles data (needed for grids, plots, or analysis)
    if GENERATE_GRIDS or SHOW_TILES_PLOT or DOWNLOAD_CHM or GENERATE_SAMPLE_POINTS or (
            not GENERATE_GRIDS and not EXPLORE_S3_STRUCTURE and not SHOW_TILES_PLOT):
        print("📥 LOADING TILES DATA...")
        tiles_gdf = download_tiles_geojson(
            BUCKET_NAME,
            TILES_KEY,
            bbox=active_bbox,
            show_plot=SHOW_TILES_PLOT
        )

        if tiles_gdf is None:
            print("❌ Failed to load tiles data. Exiting.")
            success = False
        else:
            print(f"\n✅ Tiles loaded successfully")
            print(f"   Tiles extent: {tiles_gdf.total_bounds}")
            print(f"   Number of tiles: {len(tiles_gdf)}")

    # Step 3: Optional CHM file size check
    if CHECK_CHM_FILE_SIZES and tiles_gdf is not None:
        print(f"\n📊 CHECKING CHM FILE SIZES...")
        from functions import check_chm_file_sizes

        file_stats = check_chm_file_sizes(
            tiles_gdf,
            BUCKET_NAME,
            sample_size=50
        )

    # Step 4: Generate grids if requested
    if GENERATE_GRIDS and success and tiles_gdf is not None:
        print(f"\n🔲 GENERATING GRIDS...")

        # Create config for grid generation
        grid_config = {
            'bbox': active_bbox,
            'grid_sizes': active_grid_sizes,
            'generate_grids': True  # Force True for this step
        }

        from functions import create_grid, spatial_filter_grid, save_grid, _get_filename_from_km

        successful_grids = []
        bounds = tiles_gdf.total_bounds

        for cell_size_km in active_grid_sizes:
            # Convert km to meters for internal calculations
            cell_size_meters = int(cell_size_km * 1000)
            grid_label = f"{cell_size_km}km"

            print(f"\n{'🔲 PROCESSING ' + grid_label + ' GRID':=^60}")

            try:
                # Create grid
                print(f"⚙️  Creating {grid_label} grid...")
                grid = create_grid(bounds, cell_size_meters, crs=tiles_gdf.crs)

                # Spatial filter
                print(f"🔍 Spatially filtering {grid_label} grid...")
                filtered_grid = spatial_filter_grid(grid, tiles_gdf)

                # Save grid
                print(f"💾 Saving {grid_label} grid...")
                filename = _get_filename_from_km(cell_size_km)
                save_grid(filtered_grid, filename, cell_size_meters)

                successful_grids.append(filename)
                print(f"✅ {grid_label} grid completed successfully!")

                # Clear memory
                del grid, filtered_grid

            except Exception as e:
                print(f"❌ Error processing {grid_label} grid: {e}")
                success = False
                continue

        # Grid generation summary
        if successful_grids:
            print(f"\n{'🎉 GRID GENERATION COMPLETE':=^60}")
            print("✅ Successfully created grids:")
            for filename in successful_grids:
                print(f"   📄 {filename}")
        else:
            print("❌ No grids were created successfully")
            success = False

    # Step 5: Run analysis if grids are disabled
    if not GENERATE_GRIDS:
        # If we haven't loaded tiles yet (when all flags are False), load them now for analysis
        if tiles_gdf is None:
            print("📥 LOADING TILES DATA FOR ANALYSIS...")
            tiles_gdf = download_tiles_geojson(
                BUCKET_NAME,
                TILES_KEY,
                bbox=active_bbox,
                show_plot=False  # Don't show plot here since SHOW_TILES_PLOT was False
            )

        if tiles_gdf is not None:
            print(f"\n🔬 RUNNING ANALYSIS...")
            config = {
                'bbox': active_bbox,
                'grid_sizes': active_grid_sizes,
                'generate_grids': GENERATE_GRIDS
            }
            analysis_success = placeholder_analysis(tiles_gdf, config)
            success = success and analysis_success
        else:
            print("❌ Could not load tiles for analysis")
            success = False

    # Step 6: Generate sample points if requested
    if GENERATE_SAMPLE_POINTS:
        print(f"\n🎯 GENERATING SAMPLE POINTS...")

        # Only process grids that exist for current mode
        active_sample_config = {size: SAMPLE_POINTS_CONFIG[size]
                                for size in active_grid_sizes
                                if size in SAMPLE_POINTS_CONFIG}

        if active_sample_config:
            # Generate sample points for all active grid sizes
            sample_results = generate_sample_points_for_grids(active_sample_config, active_bbox)

            # Optional: Show sample points map
            if SHOW_SAMPLE_POINTS_MAP and sample_results:
                print(f"\n🗺️  CREATING SAMPLE POINTS MAP...")
                from functions import plot_sample_points_map

                for grid_size_km, points_gdf in sample_results.items():
                    if points_gdf is not None and len(points_gdf) > 0:
                        # Use the new function that plots all points
                        plot_sample_points_map(points_gdf, tiles_gdf=tiles_gdf,
                                               bbox=active_bbox, grid_size_km=grid_size_km)
        else:
            print("   No sample point configuration found for active grid sizes")

    # Step 7: Download and process CHM tiles if requested
    if DOWNLOAD_CHM and tiles_gdf is not None:
        print(f"\n🌲 DOWNLOADING CHM TILES...")

        from functions import download_chm, create_chm_mosaic

        # Download and process CHM tiles
        chm_results = download_chm(
            tiles_gdf,
            BUCKET_NAME,
            output_dir=CHM_OUTPUT_DIR,
            binary_threshold=CHM_BINARY_THRESHOLD
        )

        if chm_results and chm_results['processed'] > 0:
            print(f"✅ CHM processing completed: {chm_results['processed']} tiles processed")

            # Optionally create mosaic
            if CREATE_CHM_MOSAIC:
                print(f"\n🗺️  CREATING CHM MOSAIC...")
                mosaic_path = f"{CHM_OUTPUT_DIR}/chm_binary_mosaic.tif"
                mosaic_result = create_chm_mosaic(CHM_OUTPUT_DIR, mosaic_path, tiles_gdf)

                if mosaic_result:
                    print(f"✅ CHM mosaic created: {mosaic_result}")
                else:
                    print("❌ Failed to create CHM mosaic")
        else:
            print("❌ CHM processing failed or no tiles were processed")
            success = False

    # Final summary
    print(f"\n{'📋 EXECUTION SUMMARY':=^60}")
    operations = []
    if EXPLORE_S3_STRUCTURE:
        operations.append("✅ S3 exploration")
    if SHOW_TILES_PLOT:
        operations.append("✅ Tiles visualization")
    if GENERATE_GRIDS:
        operations.append("✅ Grid generation")
    if not GENERATE_GRIDS and tiles_gdf is not None:
        operations.append("✅ Analysis placeholder")
    if DOWNLOAD_CHM:
        operations.append("✅ CHM tile download and processing")

    if operations:
        print("Completed operations:")
        for op in operations:
            print(f"   {op}")
    else:
        print("No operations were configured to run")

    print(f"\nOverall result: {'SUCCESS' if success else 'FAILED'}")

    # Exit with appropriate code
    exit(0 if success else 1)
