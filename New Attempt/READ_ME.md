# Files

## Folders
###CONUS
This folder includes two .py files and one datasets for processing across the contiguous United States, which we use are our area of interest. 

**.py Files:**
1. CONUS.py: Using a shapefile of the US states from TIGER (US Census Bureau), states outside the contiguous United States are removed and the remaining states are merged. This file exports conus.gpkg, a geopackage file that is used to access the Meta dataset.
2. grid generator.py: Using conus.gpkg, this script generates a grid dataset across the contiguous United States with grid cell sizes of length and width 1, 10, and 40 km^2


: The geopackage file of the contigous United States. The python file (CONUS.py) generates the file, CONUS.gpkg.

Files:
download Meta.py: downloads the Meta CHM for the expressed area of interest (in this case, the contiguous United States). The files are converted from unsigned 8-bit integers into 1-bit (canopy (1) or not (0))
