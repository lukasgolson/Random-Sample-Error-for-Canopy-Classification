from functions import *

USE_TEST_SETTINGS = True

# Import the grids
grid_1 = gpd.read_file('AOI/grid_1km.gpkg')
grid_20 = gpd.read_file('AOI/grid_20km.gpkg')
grid_40 = gpd.read_file('AOI/grid_40km.gpkg')
grid_100 = gpd.read_file('AOI/grid_100km.gpkg')

grids = [grid_100] if USE_TEST_SETTINGS else [grid_1, grid_20, grid_40]

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
