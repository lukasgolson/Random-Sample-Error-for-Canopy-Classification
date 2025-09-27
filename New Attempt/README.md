# Files

## Folders
### CONUS
This folder includes two .py files and one datasets for processing across the contiguous United States, which we use are our area of interest. 

**.py Files:**
1. CONUS.py: Using a shapefile of the US states from TIGER (US Census Bureau), states outside the contiguous United States are removed and the remaining states are merged. This file exports conus.gpkg, a geopackage file that is used to access the Meta dataset.
2. grid generator.py: Using conus.gpkg, this script generates a grid dataset across the contiguous United States with grid cell sizes of length and width 1, 10, and 40 km<sup>2<sup>.

**Dataset:**
1. conus.gpkg: The geopackage file of the contigous United States.

## Scripts
