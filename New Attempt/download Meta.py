# Enhanced Meta CHM downloader with robust file existence checking

import boto3
import numpy as np
import rasterio
from botocore import UNSIGNED
from botocore.config import Config
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import time
from tqdm import tqdm
import logging

USE_TEST_SETTINGS = True  # Test this code using a low complexity, fast run by setting this value to True

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

Bucket = 'dataforgood-fb-data'
Prefix = 'forests/v1/alsgedi_global_v6_float/chm/'

s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED, max_pool_connections=50))

# Use delimiter='/' to get folder-like structure
response = s3.list_objects_v2(Bucket=Bucket, Prefix=Prefix, Delimiter='/')

## ------------------------------ IDENTIFY META CHM TILES IN AOI ------------------------------
# region

# Identify which AOI to use
if USE_TEST_SETTINGS is False:
    with open('AOI/tiles_in_aoi.txt', "r") as f:
        aoi = [line.strip() for line in f]

else:
    with open('AOI/tiles_in_aoi_test.txt', "r") as f:
        aoi = [line.strip() for line in f]

print(f"Total number of tiles: {len(aoi)}")

# endregion

## ----------------------------------------------------- FUNCTIONS -----------------------------------------------------
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


def create_binary_raster(input_path, output_path, threshold, nodata_value=None):
    """
    Converts a CHM raster to a binary raster based on a height threshold.
    """
    try:
        with rasterio.open(input_path) as src:
            data = src.read(1)
            meta = src.meta.copy()
            source_nodata = src.nodata

            # Pixels > threshold become 1, all others become 0
            binary_data = (data > threshold).astype(np.uint8)

            # Preserve the original NoData values
            if source_nodata is not None:
                binary_data[data == source_nodata] = source_nodata

            meta.update(dtype='uint8', count=1, compress='lzw')
            if nodata_value is not None:
                meta['nodata'] = nodata_value

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            with rasterio.open(output_path, 'w', **meta) as dst:
                dst.write(binary_data, 1)

        return True, f"Created binary raster: {Path(output_path).name}"
    except Exception as e:
        logger.error(f"Failed to create binary raster for {input_path}: {e}")
        return False, str(e)


def conversion_worker(q, binary_dir, threshold):
    """
    A worker that pulls a file path from a queue, converts it to a binary raster,
    and deletes the original raw file upon success.
    """
    while True:
        try:
            raw_path_str = q.get()
            if raw_path_str is None:  # Sentinel value to stop the worker
                break

            raw_path = Path(raw_path_str)
            binary_output_path = Path(binary_dir) / raw_path.name

            success, message = create_binary_raster(
                input_path=str(raw_path),
                output_path=str(binary_output_path),
                threshold=threshold
            )

            if success:
                try:
                    os.remove(raw_path)
                    logger.info(f"✓ Converted and removed raw file: {raw_path.name}")
                except OSError as e:
                    logger.error(f"Error removing raw file {raw_path}: {e}")
            else:
                logger.error(f"✗ Conversion failed for {raw_path.name}: {message}")
        finally:
            q.task_done()


def main_download_workflow(quadkeys, raw_dir="Meta CHM Raw", binary_dir="Meta CHM Binary"):
    """
    Main workflow that validates files in S3, then downloads and converts them in parallel.
    Raw downloaded files are deleted after successful conversion.
    """
    bucket_name = 'dataforgood-fb-data'
    base_prefix = 'forests/v1/alsgedi_global_v6_float/chm/'

    # --- Step 1: Validate that files exist in S3 ---
    print("Step 1: Validating file existence in S3...")
    valid_paths, invalid_paths, file_info = validate_quadkey_paths(
        quadkeys, bucket_name, base_prefix
    )

    print(f"✓ Found {len(valid_paths)} valid files")
    if invalid_paths:
        print(f"✗ {len(invalid_paths)} files not found at expected locations")
        # (Optional: Add logic here to find alternatives for missing files if needed)

    if not valid_paths:
        print("❌ No valid files found to download!")
        return

    # --- Step 2: Check what's already downloaded locally ---
    print(f"\nStep 2: Checking local files...")
    files_to_download = []
    already_converted = 0

    # We check the FINAL destination (binary directory) to see if work is already done
    for s3_path in valid_paths:
        key = s3_path.replace(f"s3://{bucket_name}/", "")
        quadkey = file_info[key]['quadkey']

        # Check if the FINAL binary file already exists and is valid
        final_binary_path = Path(binary_dir) / f"{quadkey}.tif"
        if final_binary_path.exists() and final_binary_path.stat().st_size > 1024:  # Basic integrity check
            already_converted += 1
        else:
            files_to_download.append((s3_path, key))

    print(f"✓ {already_converted} files already converted and exist in '{binary_dir}'")
    print(f"📥 {len(files_to_download)} files need downloading and conversion")

    if not files_to_download:
        print("✅ All files are already processed!")
        return

    # --- Step 3: Calculate total download size ---
    total_size_mb = sum(file_info[key]['size_mb'] for _, key in files_to_download)

    print(f"\nStep 3: Download summary")
    print(f"Total download size: {total_size_mb:.1f} MB ({total_size_mb / 1024:.1f} GB)")

    # --- Step 4: Setup directories and parallel processing ---
    print("\nStep 4: Starting parallel download and conversion...")
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    Path(binary_dir).mkdir(parents=True, exist_ok=True)

    conversion_queue = queue.Queue()
    HEIGHT_THRESHOLD = 5  # meters

    # Define number of parallel workers for each task
    num_download_workers = 5
    num_conversion_workers = max(1, os.cpu_count() - 1)  # Use most CPU cores for conversion

    successful_downloads = 0
    failed_downloads = 0

    with ThreadPoolExecutor(max_workers=num_conversion_workers, thread_name_prefix='Converter') as conversion_executor, \
            ThreadPoolExecutor(max_workers=num_download_workers, thread_name_prefix='Downloader') as download_executor:

        # 1. Start the conversion workers. They will wait for items in the queue.
        for _ in range(num_conversion_workers):
            conversion_executor.submit(conversion_worker, conversion_queue, binary_dir, HEIGHT_THRESHOLD)

        # 2. Submit all download tasks
        future_to_file = {}
        for s3_path, key in files_to_download:
            quadkey = Path(key).stem
            local_path = Path(raw_dir) / f"{quadkey}.tif"
            future = download_executor.submit(
                download_tile_boto3, s3, bucket_name, key, str(local_path)
            )
            future_to_file[future] = local_path

        # 3. Process downloads as they complete
        with tqdm(total=len(files_to_download), desc="Downloading & Converting") as pbar:
            for future in as_completed(future_to_file):
                local_path = future_to_file[future]
                success, message = future.result()

                if success:
                    successful_downloads += 1
                    # Add the successfully downloaded file path to the queue for conversion
                    conversion_queue.put(str(local_path))
                else:
                    failed_downloads += 1
                    logger.error(f"Download failed: {message}")

                pbar.update(1)

        # 4. All downloads are done. Wait for the conversion queue to be empty.
        print("\nAll downloads complete. Waiting for conversions to finish...")
        conversion_queue.join()

        # 5. Signal the conversion workers to stop by sending 'None'
        for _ in range(num_conversion_workers):
            conversion_queue.put(None)

    # --- Step 5: Final summary ---
    print("\n" + "=" * 60)
    print("WORKFLOW COMPLETE")
    print("=" * 60)
    print(f"✅ Successful downloads: {successful_downloads}")
    print(f"❌ Failed downloads: {failed_downloads}")
    print(f"📁 Binary rasters saved to: {Path(binary_dir).absolute()}")
    print(f"🧹 Raw files directory '{Path(raw_dir).absolute()}' should now be empty.")

    if failed_downloads > 0:
        print(f"\n⚠ {failed_downloads} downloads failed. Check logs for details.")

# endregion

## ---------------------------------------------------- RUN SCRIPT -----------------------------------------------------
# Run the enhanced download workflow
if __name__ == "__main__":
    # Use the quadkeys identified from the AOI analysis
    main_download_workflow(aoi)
