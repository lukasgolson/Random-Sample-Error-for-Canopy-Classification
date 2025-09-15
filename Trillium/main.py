## Conceptual framework for code
# 1. Create a tree stamp
# 2. Generate an AOI (400, 2000, 30000) with canopy extent (0 - 100%) and Moran's i (-1 to +1). Canopy in steps of 1% (101 steps including 0%), Moran's i in steps of .05 (40 steps) (total: 4,040 steps)
# 3. Apply 100,000 random sample points (same points for all runs) and identify as canopy (1) or no canopy (0). Include meters x and y of sample points (in replace of latitude and longitude)
# 4. Export results as CSV with columns Sample Point ID, Sample Point X Location, Sample Point Y Location, then columns with code {AOI}_{Extent}_{Moran}. 
