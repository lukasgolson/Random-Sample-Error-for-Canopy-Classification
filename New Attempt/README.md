# Notes between Lukas and Alex
The AOI folder is done. Download and save to PyCharm project. Needs to run grid_generator.py to create the grid_1km.gpkg as it is 1.8 gb.

Next steps:
1. Download the Meta CHMs to Lukas' computer using the download Meta.py (already set up to run)
1. Need to calculate canopy statistics for each grid cell. The grids already have a unique cell_id, so we can connect them to this. The variables of interest are canopy extent, moran's i, edge density, number of patches, average patch size, patch size standard deviation, patch size median, patch size minimum, patch size maximum, CLUMPY, normalized landscape shape index. This should save to a CSV with column cell_id and CSV naming convention based on AOI size. The raster should be handled in a way that preserves raster cells consistently. Best way is probably to keep the cells if they intersect the grid, regardless of proportion within the grid. We will repeat the same for the random sample points.
2. Need to generate the random sample points as a sticker that can be applied across all grid cells. Need different generator for different AOI sizes. Should run 10,000 sample points at 1 km, 100,000 at 20 and 40 km. This should save as a csv with columns cell_id and point_id with each point having its own column (because they will have the same position x,y on each grid cell. Generate one csv for each AOI size.
3. Statistical Analysis after all this is done.


# Files
## Folders
### AOI
This folder contains the scripts and datasets for generating grids over the contiguous United States (CONUS), which is our area of interest.

**.py Files:**
1. **CONUS.py:** Loads the US states shapefile from TIGER (US Census Bureau), removes states and territories outside the contiguous United States (AK, HI, PR, GU, VI, MP, AS), merges the remaining states, and exports conus.gpkg. This file provides the AOI polygon used for further processing.
2. **grid_generator.py:** Uses conus.gpkg to generate square grid datasets across the contiguous United States with cell sizes of 1 km, 20 km, and 40 km. Each grid includes a unique cell_id and centroid coordinates in both projected (EPSG:5070) and geographic (EPSG:4326) coordinates. Only cells fully contained within the AOI are retained.

**Datasets:**
1. **conus.gpkg:** Geopackage of the contiguous United States AOI.
2. **tiles_in_aoi.geojson:** AOI-filtered Meta CHM tiles used to define grid extent.
3. **grid_1km.gpkg:** Generated grid cells with length and width of 1 km (must be created locally).
4. **grid_20km.gpkg:** Generated grid cells with length and width of 20 km.
5. **grid_40km.gpkg:** Generated grid cells with length and width of 40 km.
6. **tiles_in_aoi.txt:** List of tile IDs contained within the AOI.
7. **Tiles in AOI.pdf:** Map showing tiles and AOI boundaries.

## Scripts
1. **download Meta.py:** Automates the download and processing of the Meta CHM tiles from the S3 bucket, first validating that each file exists in the cloud. If the tiles are not already locally stored, it downloads them in parallel, then converts them into 1-bit binary rasters (0/1) based on a height threshold (heights above 2 m coded as 1, else 0) and deletes the raw files after successful conversion.
