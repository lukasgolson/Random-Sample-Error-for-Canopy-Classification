# Code to access Meta files using AWS CLI. Requires installation of AWS CLI

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import geopandas as gpd
import matplotlib.pyplot as plt
import io

"""

This first code segment lists the available files in the folder, which contains:
1. tiles.geojson: A vector layer with polygons showing tile locations
2. Three sub-folders: chm, metadata, and msk

The tiles.geojson file includes QuadKey identifiers in the attribute table that correspond to the geotiff files stored in the chm sub-folder.

"""

def list_s3_directories(bucket_name, prefix=''):
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    try:
        # Use delimiter='/' to get folder-like structure
        response = s3.list_objects_v2(
            Bucket=bucket_name,
            Prefix=prefix,
            Delimiter='/'
        )

        print(f"Directories in s3://{bucket_name}/{prefix}")
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
                if obj['Key'] != prefix:
                    size = obj['Size']
                    key = obj['Key']
                    modified = obj['LastModified'].strftime('%Y-%m-%d %H:%M:%S')
                    print(f"📄 {modified} {size:>12} {key}")

        if 'CommonPrefixes' not in response and 'Contents' not in response:
            print("No directories or files found")

    except Exception as e:
        print(f"Error: {e}")


# Usage - see what's inside alsgedi_global_v6_float/
list_s3_directories('dataforgood-fb-data', 'forests/v1/alsgedi_global_v6_float/')

"""

The above code results in a print out that appears as follows:

Directories in s3://dataforgood-fb-data/forests/v1/alsgedi_global_v6_float/
------------------------------------------------------------
📁 forests/v1/alsgedi_global_v6_float/chm/
📁 forests/v1/alsgedi_global_v6_float/metadata/
📁 forests/v1/alsgedi_global_v6_float/msk/
📄 2024-05-07 21:34:03   2014008198 forests/v1/alsgedi_global_v6_float/CHM_acquisition_date.tif
📄 2024-04-09 17:26:19     15167629 forests/v1/alsgedi_global_v6_float/tiles.geojson

We are interested in the tiles.geojson so that we can first identify what tiles cover Canada and the United States.

The next code segment downloads and displays the geojson file. A limiting latitude and longitude bounding
box is used to show tiles only covering Canada and the United States (see in-line comment).

"""

def download_and_display_geojson(bucket_name, key):
    # Create S3 client for unsigned requests
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    
    try:
        # Download the file to memory
        response = s3.get_object(Bucket=bucket_name, Key=key)
        geojson_data = response['Body'].read()
        
        # Read with geopandas directly from memory
        gdf = gpd.read_file(io.BytesIO(geojson_data))
        
        print(f"Successfully loaded {key}")
        print(f"Shape: {gdf.shape}")
        print(f"CRS: {gdf.crs}")
        print(f"Columns: {list(gdf.columns)}")
        
        # Display basic info
        print("\nFirst few rows:")
        print(gdf.head())
        
        # Plot the tiles
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        gdf.plot(ax=ax, facecolor='none', edgecolor='blue', alpha=0.7)
        plt.title('Forest Canopy Height Model Tiles')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return gdf
        
    except Exception as e:
        print(f"Error downloading/displaying file: {e}")
        return None

# Download and display the tiles
tiles_gdf = download_and_display_geojson(
    'dataforgood-fb-data', 
    'forests/v1/alsgedi_global_v6_float/tiles.geojson'
)

# If you want to save it locally
if tiles_gdf is not None:
    tiles_gdf.to_file('tiles.geojson', driver='GeoJSON')
    print("Saved tiles.geojson locally")
