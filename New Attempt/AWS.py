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


def list_s3_bucket(bucket_name, prefix=''):
    # Create S3 client for unsigned requests (public buckets)
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    try:
        # List bucket contents
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

        if 'Contents' in response:
            print(f"Contents of s3://{bucket_name}/{prefix}")
            print("-" * 60)
            for obj in response['Contents']:
                size = obj['Size']
                key = obj['Key']
                modified = obj['LastModified'].strftime('%Y-%m-%d %H:%M:%S')
                print(f"{modified} {size:>12} {key}")
        else:
            print("No objects found")

    except Exception as e:
        print(f"Error: {e}")


# Usage
list_s3_bucket('dataforgood-fb-data', 'forests/v1/alsgedi_global_v6_float/')
