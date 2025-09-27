# Enhanced Meta CHM downloader with robust file existence checking

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt
import subprocess
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tqdm import tqdm
import logging

USE_TEST_SETTINGS = True  # Test this code using a low complexity, fast run by setting this value to True

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

## ------------------------------ EXPLORE THE S3 BUCKET STRUCTURE ------------------------------
# region

Bucket = 'dataforgood-fb-data'
Prefix = 'forests/v1/alsgedi_global_v6_float/chm/'

s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

# Use delimiter='/' to get folder-like structure
response = s3.list_objects_v2(Bucket=Bucket, Prefix=Prefix, Delimiter='/')

print(f"Directories in s3://{Bucket}/{Prefix}")
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
print(f"QuadKeys: {qk}")

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

## ------------------------------ FILE EXISTENCE CHECKING FUNCTIONS ------------------------------
# region

def check_s3_file_exists(s3_client, bucket, key):
    """
    Check if a file exists in S3 bucket without downloading it
    Returns: (exists: bool, size: int, last_modified: datetime or None)
    """
    try:
        response = s3_client.head_object(Bucket=bucket, Key=key)
        return True, response['ContentLength'], response['LastModified']
    except s3_client.exceptions.NoSuchKey:
        return False, 0, None
    except Exception as e:
        logger.warning(f"Error checking S3 file {key}: {str(e)}")
        return False, 0, None


def batch_check_s3_files(s3_client, bucket, keys, max_workers=10):
    """
    Check existence of multiple S3 files in parallel
    Returns: dict mapping key -> (exists, size, last_modified)
    """
    results = {}

    def check_single_file(key):
        exists, size, modified = check_s3_file_exists(s3_client, bucket, key)
        return key, (exists, size, modified)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(check_single_file, key): key for key in keys}

        for future in tqdm(as_completed(futures), total=len(keys), desc="Checking S3 files"):
            key, result = future.result()
            results[key] = result

    return results


def validate_quadkey_paths(quadkeys, bucket_name, base_prefix):
    """
    Validate that the expected S3 paths exist for given quadkeys
    Returns: (valid_paths, invalid_paths, file_info)
    """
    # Generate expected S3 keys
    expected_keys = []
    for qk in quadkeys:
        # Meta CHM files are typically stored as: chm/{quadkey}.tif
        key = f"{base_prefix}{qk}.tif"
        expected_keys.append(key)

    print(f"Checking existence of {len(expected_keys)} files in S3...")

    # Batch check all files
    file_check_results = batch_check_s3_files(s3, bucket_name, expected_keys)

    # Separate valid and invalid paths
    valid_paths = []
    invalid_paths = []
    file_info = {}

    for key, (exists, size, modified) in file_check_results.items():
        if exists:
            valid_paths.append(f"s3://{bucket_name}/{key}")
            file_info[key] = {
                'size_mb': size / (1024 * 1024),
                'last_modified': modified,
                'quadkey': Path(key).stem
            }
        else:
            invalid_paths.append(f"s3://{bucket_name}/{key}")

    return valid_paths, invalid_paths, file_info


def generate_alternative_paths(quadkey, bucket_name, base_prefix):
    """
    Generate alternative S3 paths for a quadkey in case the standard path doesn't exist
    Meta might organize files differently (subdirectories, different naming, etc.)
    """
    alternatives = []

    # Try different possible organizational structures
    # Option 1: Subdirectories by zoom level or geographic region
    if len(quadkey) >= 2:
        # Subdirectory by first 2 characters
        alt1 = f"{base_prefix}{quadkey[:2]}/{quadkey}.tif"
        alternatives.append(f"s3://{bucket_name}/{alt1}")

    if len(quadkey) >= 4:
        # Subdirectory by first 4 characters
        alt2 = f"{base_prefix}{quadkey[:4]}/{quadkey}.tif"
        alternatives.append(f"s3://{bucket_name}/{alt2}")

    # Option 2: Different file extensions
    for ext in ['.tiff', '.TIF', '.TIFF']:
        alt3 = f"{base_prefix}{quadkey}{ext}"
        alternatives.append(f"s3://{bucket_name}/{alt3}")

    # Option 3: Different naming conventions
    alt4 = f"{base_prefix}chm_{quadkey}.tif"
    alternatives.append(f"s3://{bucket_name}/{alt4}")

    return alternatives


def find_missing_files_alternatives(invalid_paths, bucket_name, base_prefix):
    """
    For files that don't exist at expected paths, try to find them at alternative locations
    """
    found_alternatives = {}

    print(f"Searching for alternatives for {len(invalid_paths)} missing files...")

    for s3_path in tqdm(invalid_paths, desc="Finding alternatives"):
        # Extract quadkey from path
        quadkey = Path(s3_path).stem

        # Generate alternative paths
        alternatives = generate_alternative_paths(quadkey, bucket_name, base_prefix)

        # Check each alternative
        for alt_path in alternatives:
            alt_key = alt_path.replace(f"s3://{bucket_name}/", "")
            exists, size, modified = check_s3_file_exists(s3, bucket_name, alt_key)

            if exists:
                found_alternatives[s3_path] = {
                    'alternative_path': alt_path,
                    'size_mb': size / (1024 * 1024),
                    'last_modified': modified
                }
                break

    return found_alternatives


# endregion

## ------------------------------ ENHANCED DOWNLOAD FUNCTIONS ------------------------------
# region

def download_tile_boto3(s3_client, bucket, key, local_path, max_retries=3):
    """
    Download a single tile using boto3 with pre-download existence check
    """
    # First, verify the file exists and get its info
    exists, size, modified = check_s3_file_exists(s3_client, bucket, key)
    if not exists:
        return False, f"✗ {Path(local_path).name}: File does not exist in S3"

    # Check if local file already exists and is complete
    if Path(local_path).exists():
        local_size = Path(local_path).stat().st_size
        if local_size == size:
            return True, f"✓ {Path(local_path).name} (already complete)"
        else:
            logger.warning(f"Local file size mismatch for {Path(local_path).name}: {local_size} vs {size}")
            # Remove incomplete file
            Path(local_path).unlink(missing_ok=True)

    # Create directory if it doesn't exist
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)

    # Download with retries
    for attempt in range(max_retries):
        try:
            s3_client.download_file(bucket, key, local_path)

            # Verify download completed successfully
            if check_file_integrity(local_path, size):
                return True, f"✓ {Path(local_path).name} ({size / (1024 * 1024):.1f} MB)"
            else:
                return False, f"✗ {Path(local_path).name}: Downloaded file failed integrity check"

        except Exception as e:
            if attempt == max_retries - 1:
                return False, f"✗ {Path(local_path).name}: {str(e)}"
            time.sleep(2 ** attempt)  # Exponential backoff

    return False, f"✗ {Path(local_path).name}: Max retries exceeded"


def check_file_integrity(file_path, expected_size=None, min_size_mb=0.1):
    """
    Enhanced file integrity check with optional size validation
    """
    try:
        if not Path(file_path).exists():
            return False

        actual_size = Path(file_path).stat().st_size
        size_mb = actual_size / (1024 * 1024)

        # Check minimum size
        if size_mb < min_size_mb:
            logger.warning(f"File {file_path} is too small: {size_mb:.2f} MB")
            return False

        # Check expected size if provided
        if expected_size is not None and actual_size != expected_size:
            logger.warning(f"File {file_path} size mismatch: {actual_size} vs expected {expected_size}")
            return False

        return True
    except Exception as e:
        logger.error(f"Error checking file integrity for {file_path}: {str(e)}")
        return False


# endregion

## ------------------------------ MAIN DOWNLOAD WORKFLOW WITH VALIDATION ------------------------------
# region

def main_download_workflow(quadkeys, download_dir="Meta CHM Tiles"):
    """
    Main workflow that validates files before downloading
    """
    bucket_name = 'dataforgood-fb-data'
    base_prefix = 'forests/v1/alsgedi_global_v6_float/chm/'

    # Create download directory
    Path(download_dir).mkdir(parents=True, exist_ok=True)

    print(f"Starting download workflow for {len(quadkeys)} quadkeys")
    print(f"Download directory: {Path(download_dir).absolute()}")
    print("-" * 60)

    # Step 1: Validate that files exist in S3
    print("Step 1: Validating file existence in S3...")
    valid_paths, invalid_paths, file_info = validate_quadkey_paths(
        quadkeys, bucket_name, base_prefix
    )

    print(f"✓ Found {len(valid_paths)} valid files")
    if invalid_paths:
        print(f"✗ {len(invalid_paths)} files not found at expected locations")

        # Try to find alternatives for missing files
        alternatives = find_missing_files_alternatives(invalid_paths, bucket_name, base_prefix)

        if alternatives:
            print(f"✓ Found alternatives for {len(alternatives)} missing files")
            # Add alternatives to valid paths
            for original_path, alt_info in alternatives.items():
                valid_paths.append(alt_info['alternative_path'])
                alt_key = alt_info['alternative_path'].replace(f"s3://{bucket_name}/", "")
                file_info[alt_key] = {
                    'size_mb': alt_info['size_mb'],
                    'last_modified': alt_info['last_modified'],
                    'quadkey': Path(alt_key).stem
                }

        remaining_missing = len(invalid_paths) - len(alternatives)
        if remaining_missing > 0:
            print(f"⚠ {remaining_missing} files could not be located in S3")
            print("Missing files:")
            for path in invalid_paths[:10]:  # Show first 10
                if not any(path in alt for alt in alternatives.keys()):
                    print(f"  - {path}")
            if len(invalid_paths) > 10:
                print(f"  ... and {len(invalid_paths) - 10} more")

    if not valid_paths:
        print("❌ No valid files found to download!")
        return

    # Step 2: Check what's already downloaded locally
    print(f"\nStep 2: Checking local files...")
    files_to_download = []
    already_downloaded = 0

    for s3_path in valid_paths:
        key = s3_path.replace(f"s3://{bucket_name}/", "")
        quadkey = file_info[key]['quadkey']
        local_path = Path(download_dir) / f"{quadkey}.tif"
        expected_size = int(file_info[key]['size_mb'] * 1024 * 1024)

        if check_file_integrity(local_path, expected_size):
            already_downloaded += 1
        else:
            files_to_download.append((s3_path, key, local_path, expected_size))

    print(f"✓ {already_downloaded} files already downloaded")
    print(f"📥 {len(files_to_download)} files need downloading")

    if not files_to_download:
        print("✅ All files are already downloaded!")
        return

    # Step 3: Calculate total download size
    total_size_mb = sum(info['size_mb'] for info in file_info.values()
                        if any(info['quadkey'] in path for path, _, _, _ in files_to_download))

    print(f"\nStep 3: Download summary")
    print(f"Total download size: {total_size_mb:.1f} MB ({total_size_mb / 1024:.1f} GB)")
    print("-" * 60)

    # Step 4: Download files
    print("Step 4: Downloading files...")

    successful_downloads = 0
    failed_downloads = 0

    # Use ThreadPoolExecutor for parallel downloads
    max_workers = min(5, len(files_to_download))  # Limit concurrent downloads

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_file = {}
        for s3_path, key, local_path, expected_size in files_to_download:
            future = executor.submit(
                download_tile_boto3,
                s3, bucket_name, key, str(local_path)
            )
            future_to_file[future] = (s3_path, local_path)

        # Process completed downloads
        with tqdm(total=len(files_to_download), desc="Downloading") as pbar:
            for future in as_completed(future_to_file):
                s3_path, local_path = future_to_file[future]
                success, message = future.result()

                if success:
                    successful_downloads += 1
                else:
                    failed_downloads += 1
                    logger.error(f"Download failed: {message}")

                pbar.set_postfix({
                    'Success': successful_downloads,
                    'Failed': failed_downloads
                })
                pbar.update(1)

    # Step 5: Final summary
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"✅ Successful downloads: {successful_downloads}")
    print(f"❌ Failed downloads: {failed_downloads}")
    print(f"📁 Files saved to: {Path(download_dir).absolute()}")

    if failed_downloads > 0:
        print(f"\n⚠ {failed_downloads} downloads failed. Check logs for details.")
        print("You can re-run this script to retry failed downloads.")


# Run the enhanced download workflow
if __name__ == "__main__":
    # Use the quadkeys identified from your AOI analysis
    main_download_workflow(qk)

# endregion
