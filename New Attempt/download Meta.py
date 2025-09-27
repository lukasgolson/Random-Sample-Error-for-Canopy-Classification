# This is the file that downloads the Meta CHMs

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tqdm import tqdm

USE_TEST_SETTINGS = True  # Test this code using a low complexity, fast run by setting this value to True

## ------------------------------ EXPLORE THE S3 BUCKET STRUCTURE ------------------------------
# region

s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

# Use delimiter='/' to get folder-like structure
response = s3.list_objects_v2(
    Bucket='dataforgood-fb-data',
    Prefix='forests/v1/alsgedi_global_v6_float/',
    Delimiter='/'
)

print(f"Directories in s3://{'dataforgood-fb-data'}/{'forests/v1/alsgedi_global_v6_float/'}")
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
        if obj['Key'] != '':
            size = obj['Size']
            key = obj['Key']
            modified = obj['LastModified'].strftime('%Y-%m-%d %H:%M:%S')
            print(f"📄 {modified} {size:>12} {key}")

if 'CommonPrefixes' not in response and 'Contents' not in response:
    print("No directories or files found")

print("-" * 60)
print("\n")
# endregion

## ------------------------------ IDENTIFY META CHM TILES IN AOI ------------------------------
# region

# Identify which AOI to use
if USE_TEST_SETTINGS is False:
    AOI = [-127, 24, -66.9, 49]  # Bounding box: [min_lon, min_lat, max_lon, max_lat]
else:
    AOI = [-80, 40, -70, 45]  # Small NY/New England region

# File paths
bucket_name = 'dataforgood-fb-data'
key = 'forests/v1/alsgedi_global_v6_float/tiles.geojson'
local_file = 'tiles.geojson'
reprojected_file = 'tiles_us_albers.geojson'

# Check if reprojected file already exists to avoid redundant work
if Path(reprojected_file).exists():
    print(f"Loading existing reprojected tiles from {reprojected_file}")
    tiles_us_albers = gpd.read_file(reprojected_file)
else:
    # Check if original file exists
    if not Path(local_file).exists():
        print(f"Downloading {local_file} from S3...")
        s3.download_file(bucket_name, key, local_file)
        print(f"Downloaded {local_file}")
    else:
        print(f"Using existing {local_file}")

    # Load, reproject, and save in one efficient chain
    print("Reprojecting tiles to US Albers...")
    tiles_us_albers = (gpd.read_file(local_file)
                       .to_crs(epsg=5070))

    # Save reprojected version for future use
    tiles_us_albers.to_file(reprojected_file, driver="GeoJSON")
    print(f"Saved reprojected tiles to {reprojected_file}")

print(f"Loaded {len(tiles_us_albers)} tiles")

# Create AOI geometry directly in target CRS (more efficient)
# Convert AOI bounds to target CRS once
aoi_wgs84 = box(*AOI)
aoi_gdf = gpd.GeoDataFrame([1], geometry=[aoi_wgs84], crs="EPSG:4326")
aoi_geom_albers = aoi_gdf.to_crs(epsg=5070).geometry[0]

# Use spatial index for faster intersection
print("Finding tiles that intersect AOI...")
if hasattr(tiles_us_albers, 'sindex'):
    # Use spatial index for preliminary filtering
    possible_matches_index = list(tiles_us_albers.sindex.intersection(aoi_geom_albers.bounds))
    possible_matches = tiles_us_albers.iloc[possible_matches_index]
    # Then do precise intersection on smaller subset
    tiles_in_aoi = possible_matches[possible_matches.intersects(aoi_geom_albers)]
else:
    # Fallback to regular intersection if no spatial index
    tiles_in_aoi = tiles_us_albers[tiles_us_albers.intersects(aoi_geom_albers)]

print(f"Number of tiles in AOI: {len(tiles_in_aoi)}")

# Save list of QuadKeys from tiles.geojson
qk = tiles_in_aoi['tile'].tolist()
qk = [tile for tile in qk if tile]

# More efficient plotting
fig, ax = plt.subplots(figsize=(10, 8))

# Plot tiles - use plot() instead of boundary.plot() if you want filled polygons
tiles_in_aoi.boundary.plot(ax=ax, color='blue', linewidth=1, label=f'Tiles ({len(tiles_in_aoi)})')

# Create AOI geodataframe for plotting (more efficient than GeoSeries)
aoi_plot_gdf = gpd.GeoDataFrame([1], geometry=[aoi_geom_albers])
aoi_plot_gdf.boundary.plot(ax=ax, color='red', linewidth=2, label='AOI')

ax.set_xlabel("Easting (meters)")
ax.set_ylabel("Northing (meters)")
ax.set_title("Meta CHM Tiles Intersecting AOI")
ax.legend()
ax.grid(True, alpha=0.3)

# Format large coordinates
ax.ticklabel_format(style='scientific', axis='both', scilimits=(0, 0))

plt.tight_layout()
plt.show()

# Optional: Print some efficiency stats
print(f"\nEfficiency summary:")
print(f"- Used cached files: {Path(reprojected_file).exists()}")
print(f"- Spatial index available: {hasattr(tiles_us_albers, 'sindex')}")
print(f"- Total tiles: {len(tiles_us_albers)}")
print(f"- Tiles in AOI: {len(tiles_in_aoi)}")

# endregion

## ------------------------------ DOWNLOAD THE META TILES ------------------------------
#region

# Fastest download approach - parallel with tqdm progress bar
import subprocess
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tqdm import tqdm

# Create download directory
download_dir = "downloaded_tiles"
Path(download_dir).mkdir(exist_ok=True)

baseurl = "s3://dataforgood-fb-data/forests/v1/alsgedi_global_v6_float/chm"

print(f"\n\nFast downloading {len(qk)} tiles...")
start_time = time.time()


def download_tile_fast(quad):
    """Download single tile - optimized for speed"""
    downloadlink = f"{os.path.join(baseurl, quad)}.tif"
    local_file = f"{download_dir}/{quad}.tif"

    # Skip if already exists
    if Path(local_file).exists():
        return ("success", f"{quad}.tif (exists)")

    # Clean up any existing temp files first
    temp_pattern = f"{download_dir}/{quad}.tif.*"
    import glob
    for temp_file in glob.glob(temp_pattern):
        try:
            os.remove(temp_file)
        except:
            pass

    try:
        result = subprocess.run([
            "aws", "s3", "cp",
            "--no-sign-request",
            downloadlink,
            local_file,
            "--cli-write-timeout", "0",  # Disable timeout that can cause temp files
            "--cli-read-timeout", "0"
        ], capture_output=True, text=True, check=True)

        # Verify the file exists with correct name
        if Path(local_file).exists():
            return ("success", f"{quad}.tif")
        else:
            return ("failed", f"{quad}.tif: File not created properly")

    except subprocess.CalledProcessError as e:
        # Clean up any temp files on failure
        for temp_file in glob.glob(temp_pattern):
            try:
                os.remove(temp_file)
            except:
                pass
        return ("failed", f"{quad}.tif: {e.stderr.strip()}")


# Use more threads for pure downloading (no GRASS bottleneck)
max_workers = 16  # Increase for faster downloads
successful = 0
failed = 0
failed_files = []

# Progress bar with tqdm
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_to_quad = {executor.submit(download_tile_fast, quad): quad for quad in qk}

    # Create progress bar
    with tqdm(total=len(qk), desc="Downloading", unit="files") as pbar:
        for future in as_completed(future_to_quad):
            status, message = future.result()

            if status == "success":
                successful += 1
                pbar.set_postfix({"✓": successful, "✗": failed}, refresh=False)
            else:
                failed += 1
                failed_files.append(message)
                pbar.set_postfix({"✓": successful, "✗": failed}, refresh=False)

            pbar.update(1)

elapsed = time.time() - start_time
print(f"\nDownload Complete!")
print(f"  Time: {elapsed:.1f} seconds")
print(f"  Successful: {successful}")
print(f"  Failed: {failed}")
print(f"  Speed: {successful / elapsed:.1f} files/second")

# Clean up any remaining temp files
print("Cleaning up temp files...")
import glob

temp_files = glob.glob(f"{download_dir}/*.tif.*")
for temp_file in temp_files:
    try:
        os.remove(temp_file)
        print(f"Removed temp file: {Path(temp_file).name}")
    except:
        pass

# Show what we downloaded
downloaded_files = list(Path(download_dir).glob("*.tif"))
print(f"\n{len(downloaded_files)} clean .tif files in {download_dir}/")
print(f"Total size: {sum(f.stat().st_size for f in downloaded_files) / 1024 ** 2:.1f} MB")

# Show first few filenames to verify correct naming
if downloaded_files:
    print("Sample filenames:")
    for f in sorted(downloaded_files)[:5]:
        print(f"  {f.name}")
    if len(downloaded_files) > 5:
        print(f"  ... and {len(downloaded_files) - 5} more")

#endregion
