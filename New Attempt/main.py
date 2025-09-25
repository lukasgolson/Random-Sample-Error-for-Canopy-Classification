# Main script to generate analysis grids for Meta's forest canopy height data.
# Creates 600m, 1km, and 10km square grids filtered to tile boundaries.

from functions import *

# =============================================================================
# CONFIGURATION OPTIONS - Modify these as needed
# =============================================================================

# Select which processes to run
EXPLORE_S3_STRUCTURE = False    # Set to True to explore S3 bucket structure
SHOW_TILES_PLOT = False         # Set to True to visualize the tiles
GENERATE_GRIDS = False          # Set to True to generate grids

# Geographic settings
BBOX = [-127, 24, -66.9, 49]   # Bounding box: [min_lon, min_lat, max_lon, max_lat]

# Grid specifications
GRID_SIZES = [1, 20, 35]
# 1 km: Neighbourhood-sized grid
# 20 km: City-sized grid
# 35 km: Region-sized grid (counties, regions, upper tier municipalities)

# S3 data source configuration
BUCKET_NAME = 'dataforgood-fb-data'
TILES_KEY = 'forests/v1/alsgedi_global_v6_float/tiles.geojson'

if __name__ == "__main__":
    print("Access and Process Meta CHM Script")
    print("=" * 50)
    print(f"Target area: contiguous USA")
    print(f"Configuration:")
    print(f"  - Explore S3: {EXPLORE_S3_STRUCTURE}")
    print(f"  - Show tiles plot: {SHOW_TILES_PLOT}")
    print(f"  - Generate grids: {GENERATE_GRIDS}")
    print(f"  - Grid sizes: {GRID_SIZES} km")
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
    if GENERATE_GRIDS or SHOW_TILES_PLOT or (not GENERATE_GRIDS and not EXPLORE_S3_STRUCTURE and not SHOW_TILES_PLOT):
        print("📥 LOADING TILES DATA...")
        tiles_gdf = download_tiles_geojson(
            BUCKET_NAME,
            TILES_KEY,
            bbox=BBOX,
            show_plot=SHOW_TILES_PLOT
        )

        if tiles_gdf is None:
            print("❌ Failed to load tiles data. Exiting.")
            success = False
        else:
            print(f"✅ Tiles loaded successfully")
            print(f"   Tiles extent: {tiles_gdf.total_bounds}")
            print(f"   Number of tiles: {len(tiles_gdf)}")

    # Step 3: Generate grids if requested
    if GENERATE_GRIDS and success and tiles_gdf is not None:
        print(f"\n🔲 GENERATING GRIDS...")

        # Create config for grid generation
        grid_config = {
            'bbox': BBOX,
            'grid_sizes': GRID_SIZES,
            'generate_grids': True  # Force True for this step
        }

        from functions import create_grid, spatial_filter_grid, save_grid, _get_filename_from_km

        successful_grids = []
        bounds = tiles_gdf.total_bounds

        for cell_size_km in GRID_SIZES:
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

    # Step 4: Run analysis if grids are disabled
    if not GENERATE_GRIDS:
        # If we haven't loaded tiles yet (when all flags are False), load them now for analysis
        if tiles_gdf is None:
            print("📥 LOADING TILES DATA FOR ANALYSIS...")
            tiles_gdf = download_tiles_geojson(
                BUCKET_NAME,
                TILES_KEY,
                bbox=BBOX,
                show_plot=False  # Don't show plot here since SHOW_TILES_PLOT was False
            )

        if tiles_gdf is not None:
            print(f"\n🔬 RUNNING ANALYSIS...")
            config = {
                'bbox': BBOX,
                'grid_sizes': GRID_SIZES,
                'generate_grids': GENERATE_GRIDS
            }
            analysis_success = placeholder_analysis(tiles_gdf, config)
            success = success and analysis_success
        else:
            print("❌ Could not load tiles for analysis")
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

    if operations:
        print("Completed operations:")
        for op in operations:
            print(f"   {op}")
    else:
        print("No operations were configured to run")

    print(f"\nOverall result: {'SUCCESS' if success else 'FAILED'}")

    # Exit with appropriate code
    exit(0 if success else 1)
