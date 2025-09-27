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
3. **grid_1km.gpkg:** Generated grid cells with length and width of 1 km.
4. **grid_20km.gpkg:** Generated grid cells with length and width of 20 km.
5. **grid_40km.gpkg:** Generated grid cells with length and width of 40 km.
6. **tiles_in_aoi.txt:** List of tile IDs contained within the AOI.
7. **Tiles in AOI.pdf:** Map showing tiles and AOI boundaries.

## Scripts
