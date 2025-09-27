# This is the file that accesses and downloads the CHM tiles

from functions import *
from tqdm import tqdm

# Run through pre-processing with test settings (low complexity, fast run)
USE_TEST_SETTINGS = True       # Set to True to run a test of the code using a larger BBOX and AOI

# Geographic settings
BBOX = [-127, 24, -66.9, 49]   # Bounding box: [min_lon, min_lat, max_lon, max_lat]
TEST_BBOX = [-80, 40, -70, 45]      # Small NY/New England region

if __name__ == "__main__":
    # Pick AOI based on setting
    bbox = TEST_BBOX if USE_TEST_SETTINGS else BBOX

    print("📥 LOADING META TILES...")
    tiles_gdf = download_tiles_geojson(
        "dataforgood-fb-data",
        "forests/v1/alsgedi_global_v6_float/tiles.geojson",
        bbox=bbox,
        show_plot=False
    )

    if tiles_gdf is None or len(tiles_gdf) == 0:
        print("❌ No tiles found in AOI")
        exit(1)

    print(f"✅ Loaded {len(tiles_gdf)} tiles in AOI")

    # Download tiles with tqdm progress bar
    print("\n🌲 DOWNLOADING CHM TILES...")
    chm_results = {"processed": 0}
    for _, tile in tqdm(tiles_gdf.iterrows(), total=len(tiles_gdf), desc="Downloading tiles"):
        result = download_chm(
            tiles_gdf.loc[[_]],  # single-tile GeoDataFrame
            "dataforgood-fb-data",
            output_dir="chm_binary",
            binary_threshold=2.0   # hard-coded threshold
        )
        if result and result['processed'] > 0:
            chm_results['processed'] += result['processed']

    if chm_results['processed'] == 0:
        print("❌ CHM download/processing failed")
        exit(1)

    print(f"✅ {chm_results['processed']} CHM tiles processed")

    # Always create mosaic (hard-coded path)
    print("\n🗺️  CREATING MOSAIC...")
    mosaic_result = create_chm_mosaic("chm_binary", "chm_binary/chm_binary_mosaic.tif", tiles_gdf)
    if mosaic_result:
        print(f"✅ Mosaic saved: {mosaic_result}")
    else:
        print("❌ Mosaic creation failed")
