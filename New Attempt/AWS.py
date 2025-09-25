# Code to access Meta files using AWS CLI. Requires installation of AWS CLI

"""

This code lists the available files in the folder, which contains:
1. tiles.geojson: A vector layer with polygons showing tile locations
2. Three sub-folders: chm, metadata, and msk

The tiles.geojson file includes QuadKey identifiers in the attribute table that correspond to the geotiff files stored in the chm sub-folder.

"""
import boto3
from botocore import UNSIGNED
from botocore.config import Config


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
