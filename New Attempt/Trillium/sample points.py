import glob
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import rasterio
import rasterio.mask
from shapely.geometry import Point, box
from tqdm import tqdm
import geopandas as gpd
from multiprocessing import Pool, Manager
from functools import partial
import json
import pandas as pd
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Hard-coded for 192 CPU HPC nodes
N_CPUS = 192


# ============================================
# HPC-OPTIMIZED PARALLEL PROCESSING FUNCTIONS
# ============================================

def process_grid_cell_chunk(chunk_data, relative_points, raster_info=None):
    """
    Process a chunk of grid cells in parallel.

    Parameters:
    -----------
    chunk_data : tuple
        (chunk_index, grid_cells_subset, crs)
    relative_points : list
        List of (x, y) relative positions for points
    raster_info : dict, optional
        Shared dictionary of raster information

    Returns:
    --------
    dict : Results containing points, grid_ids, point_ids, has_raster
    """
    chunk_idx, grid_cells, crs = chunk_data

    chunk_points = []
    chunk_grid_ids = []
    chunk_point_ids = []
    chunk_has_raster = []

    for idx, row in grid_cells.iterrows():
        grid_cell = row.geometry
        grid_id = row['grid_id']

        # Check raster coverage
        has_raster = False
        if raster_info:
            has_raster = check_raster_coverage_fast(grid_cell, raster_info)

        minx, miny, maxx, maxy = grid_cell.bounds
        cell_width = maxx - minx
        cell_height = maxy - miny

        for point_idx, (rel_x, rel_y) in enumerate(relative_points):
            abs_x = minx + (rel_x * cell_width)
            abs_y = miny + (rel_y * cell_height)
            point = Point(abs_x, abs_y)

            if grid_cell.contains(point) or grid_cell.intersects(point):
                chunk_points.append(point)
                chunk_grid_ids.append(grid_id)
                chunk_point_ids.append(f"{grid_id}_p{point_idx}")
                chunk_has_raster.append(has_raster)

    return {
        'points': chunk_points,
        'grid_ids': chunk_grid_ids,
        'point_ids': chunk_point_ids,
        'has_raster': chunk_has_raster,
        'chunk_idx': chunk_idx
    }


def check_raster_coverage_fast(grid_cell, raster_info):
    """
    Fast check if grid cell intersects any raster using serialized bounds.

    Parameters:
    -----------
    grid_cell : shapely.geometry
        Grid cell geometry
    raster_info : dict
        Dictionary of raster information with bounds

    Returns:
    --------
    bool : True if grid cell intersects any raster
    """
    cell_bounds = grid_cell.bounds
    minx, miny, maxx, maxy = cell_bounds

    for raster_data in raster_info.values():
        bounds = raster_data['bounds']
        # Quick bounding box intersection check
        if not (bounds['right'] < minx or bounds['left'] > maxx or
                bounds['top'] < miny or bounds['bottom'] > maxy):
            return True

    return False


def calculate_optimal_chunk_size(n_cells, n_workers=N_CPUS, target_chunks_per_worker=4):
    """
    Calculate optimal chunk size for load balancing.
    
    Parameters:
    -----------
    n_cells : int
        Total number of grid cells
    n_workers : int
        Number of parallel workers (default: 192)
    target_chunks_per_worker : int
        Target number of chunks per worker for good load balancing
    
    Returns:
    --------
    int : Optimal chunk size
    """
    target_chunks = n_workers * target_chunks_per_worker
    chunk_size = max(1, n_cells // target_chunks)
    
    # Ensure minimum chunk size of 10 for efficiency
    chunk_size = max(10, chunk_size)
    
    return chunk_size


def generate_systematic_sample_points_parallel(grid_gdf, points_per_cell,
                                               raster_folder=None,
                                               chunk_size=None):
    """
    Generate systematic sample points using parallel processing on 192 CPUs.

    Parameters:
    -----------
    grid_gdf : GeoDataFrame
        Grid cells to sample
    points_per_cell : int
        Number of sample points per grid cell
    raster_folder : str, optional
        Path to raster folder for coverage validation
    chunk_size : int, optional
        Number of grid cells per chunk (auto-calculated if None)

    Returns:
    --------
    GeoDataFrame : Sample points with metadata
    """
    start_time = time.time()
    
    print(f"🚀 HPC MODE: GENERATING {points_per_cell:,} SYSTEMATIC SAMPLE POINTS PER GRID CELL...")
    print(f"   Processing {len(grid_gdf):,} grid cells...")
    print(f"   Using {N_CPUS} parallel workers (hard-coded)")

    # Calculate optimal chunk size if not provided
    if chunk_size is None:
        chunk_size = calculate_optimal_chunk_size(len(grid_gdf), N_CPUS)
    
    n_chunks = math.ceil(len(grid_gdf) / chunk_size)
    print(f"   Chunk size: {chunk_size} cells")
    print(f"   Total chunks: {n_chunks}")
    print(f"   Chunks per worker: {n_chunks / N_CPUS:.1f}")
    
    if n_chunks < N_CPUS:
        logger.warning(f"⚠️  Only {n_chunks} chunks for {N_CPUS} workers - some CPUs will be idle!")
        logger.warning(f"   Consider reducing chunk_size to {max(1, len(grid_gdf) // (N_CPUS * 3))}")

    # Calculate grid layout for points
    n_points = points_per_cell
    cols = int(math.ceil(math.sqrt(n_points)))
    rows = int(math.ceil(n_points / cols))
    print(f"   Point pattern: {rows} rows × {cols} columns")

    x_positions = np.linspace(0, 1, cols)
    y_positions = np.linspace(0, 1, rows)

    relative_points = []
    point_count = 0
    for y in y_positions:
        for x in x_positions:
            if point_count < n_points:
                relative_points.append((x, y))
                point_count += 1

    print(f"   Using {len(relative_points)} points per cell")

    # Build raster info and share via Manager (no file I/O)
    raster_info = None
    if raster_folder:
        print(f"\n   Building spatial index for raster validation...")
        raster_info_dict = build_raster_info_dict(raster_folder)
        
        # Use Manager for shared memory across processes (no temp files!)
        manager = Manager()
        raster_info = manager.dict(raster_info_dict)
        print(f"   Raster info loaded into shared memory")

    # Split grid into chunks
    chunks = []
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(grid_gdf))
        chunk_gdf = grid_gdf.iloc[start_idx:end_idx].copy()
        chunks.append((i, chunk_gdf, grid_gdf.crs))

    # Estimate processing time
    estimated_time = (len(grid_gdf) * len(relative_points)) / (N_CPUS * 1000)  # Rough estimate
    print(f"\n   Estimated processing time: {estimated_time:.1f} seconds")
    print(f"   Processing chunks in parallel...")

    # Process chunks in parallel
    process_func = partial(
        process_grid_cell_chunk,
        relative_points=relative_points,
        raster_info=raster_info
    )

    with Pool(processes=N_CPUS) as pool:
        results = list(tqdm(
            pool.imap(process_func, chunks),
            total=n_chunks,
            desc="Processing chunks"
        ))

    # Combine results
    print(f"\n   Combining results from {len(results)} chunks...")
    all_points = []
    all_grid_ids = []
    all_point_ids = []
    all_has_raster = []

    for result in results:
        all_points.extend(result['points'])
        all_grid_ids.extend(result['grid_ids'])
        all_point_ids.extend(result['point_ids'])
        all_has_raster.extend(result['has_raster'])

    # Create GeoDataFrame
    points_gdf = gpd.GeoDataFrame({
        'point_id': all_point_ids,
        'grid_id': all_grid_ids,
        'sample_type': 'systematic',
        'points_per_cell': points_per_cell,
        'has_raster_data': all_has_raster
    }, geometry=all_points, crs=grid_gdf.crs)

    elapsed_time = time.time() - start_time
    points_per_second = len(points_gdf) / elapsed_time
    
    print(f"\n✅ Generated {len(points_gdf):,} total sample points")
    print(f"   Processing time: {elapsed_time:.1f} seconds")
    print(f"   Throughput: {points_per_second:,.0f} points/second")
    print(f"   Points per cell: {len(points_gdf) / len(grid_gdf):.1f} (target: {points_per_cell})")
    
    if raster_folder:
        points_with_data = sum(all_has_raster)
        print(f"   Points with raster data: {points_with_data:,} ({100 * points_with_data / len(points_gdf):.1f}%)")

    return points_gdf


def build_raster_info_dict(raster_folder):
    """
    Build a serializable dictionary of raster information.

    Parameters:
    -----------
    raster_folder : str
        Path to folder containing raster files

    Returns:
    --------
    dict : Raster information with serializable bounds
    """
    raster_info = {}
    raster_pattern = os.path.join(raster_folder, "*.tif*")
    raster_files = glob.glob(raster_pattern)

    print(f"   Indexing {len(raster_files)} raster files...")

    for i, raster_path in enumerate(tqdm(raster_files, desc="Building index")):
        try:
            with rasterio.open(raster_path) as src:
                bounds = src.bounds

                raster_info[str(i)] = {
                    'path': raster_path,
                    'bounds': {
                        'left': bounds.left,
                        'bottom': bounds.bottom,
                        'right': bounds.right,
                        'top': bounds.top
                    }
                }
        except Exception as e:
            logger.warning(f"Could not index {raster_path}: {e}")
            continue

    print(f"   Successfully indexed {len(raster_info)} rasters")
    return raster_info


# ============================================
# ARRAY JOB SUPPORT FOR HPC CLUSTERS
# ============================================

def save_grid_chunks_for_array_jobs(grid_gdf, output_dir, cells_per_job=None):
    """
    Save grid chunks for SLURM/PBS array jobs.
    Each job will use all 192 CPUs.

    Parameters:
    -----------
    grid_gdf : GeoDataFrame
        Grid to split into chunks
    output_dir : str
        Directory to save chunks
    cells_per_job : int, optional
        Number of grid cells per array job (auto-calculated if None)

    Returns:
    --------
    int : Number of array jobs needed
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Auto-calculate cells per job for reasonable job count
    if cells_per_job is None:
        # Target 50-200 array jobs for good cluster utilization
        target_jobs = min(200, max(50, len(grid_gdf) // 1000))
        cells_per_job = max(100, len(grid_gdf) // target_jobs)
    
    n_jobs = math.ceil(len(grid_gdf) / cells_per_job)

    print(f"📦 Preparing {n_jobs} array jobs...")
    print(f"   Cells per job: {cells_per_job}")
    print(f"   Each job will use {N_CPUS} CPUs")

    for job_id in range(n_jobs):
        start_idx = job_id * cells_per_job
        end_idx = min((job_id + 1) * cells_per_job, len(grid_gdf))

        chunk_gdf = grid_gdf.iloc[start_idx:end_idx].copy()
        chunk_file = output_path / f"grid_chunk_{job_id:04d}.gpkg"
        chunk_gdf.to_file(chunk_file, driver="GPKG")

    print(f"✅ Saved {n_jobs} grid chunks to {output_dir}")
    print(f"\nTo run as SLURM array job:")
    print(f"   #SBATCH --array=0-{n_jobs - 1}")
    print(f"   #SBATCH --cpus-per-task={N_CPUS}")
    print(f"   python process_chunk.py --chunk-id $SLURM_ARRAY_TASK_ID")

    return n_jobs


def process_single_chunk_for_array_job(chunk_id, chunks_dir, output_dir,
                                       points_per_cell, raster_folder=None):
    """
    Process a single chunk using all 192 CPUs (for use in array jobs).

    Parameters:
    -----------
    chunk_id : int
        Array job task ID
    chunks_dir : str
        Directory containing grid chunks
    output_dir : str
        Directory to save results
    points_per_cell : int
        Number of points per grid cell
    raster_folder : str, optional
        Path to raster folder
    """
    chunk_file = Path(chunks_dir) / f"grid_chunk_{chunk_id:04d}.gpkg"

    if not chunk_file.exists():
        logger.error(f"Chunk file not found: {chunk_file}")
        return

    print(f"Processing chunk {chunk_id}: {chunk_file}")
    print(f"Using all {N_CPUS} CPUs for this job")
    
    grid_chunk = gpd.read_file(chunk_file)

    # Process this chunk using ALL 192 CPUs (not just 1!)
    points = generate_systematic_sample_points_parallel(
        grid_chunk,
        points_per_cell=points_per_cell,
        raster_folder=raster_folder,
        chunk_size=None  # Auto-calculate for 192 CPUs
    )

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"points_chunk_{chunk_id:04d}.gpkg"
    points.to_file(output_file, driver="GPKG")

    print(f"✅ Saved {len(points)} points to {output_file}")


def merge_chunk_results(chunks_dir, output_file):
    """
    Merge results from array jobs.

    Parameters:
    -----------
    chunks_dir : str
        Directory containing chunk results
    output_file : str
        Output file path for merged results
    """
    chunk_pattern = os.path.join(chunks_dir, "points_chunk_*.gpkg")
    chunk_files = sorted(glob.glob(chunk_pattern))

    print(f"Merging {len(chunk_files)} chunk results...")

    all_points = []
    for chunk_file in tqdm(chunk_files, desc="Loading chunks"):
        chunk_gdf = gpd.read_file(chunk_file)
        all_points.append(chunk_gdf)

    merged = gpd.GeoDataFrame(pd.concat(all_points, ignore_index=True))
    merged.to_file(output_file, driver="GPKG")

    print(f"✅ Merged {len(merged)} points to {output_file}")
    return merged


# ============================================
# USAGE
# ============================================

def process_multiple_grids(grid_sizes, aoi_dir="AOI", raster_folder=None, 
                          points_per_cell=10, output_dir="results"):
    """
    Process multiple grid sizes in sequence.
    
    Parameters:
    -----------
    grid_sizes : list
        List of grid size strings (e.g., ['3km', '24km', '54km'])
    aoi_dir : str
        Directory containing grid files
    raster_folder : str, optional
        Path to raster folder for coverage validation
    points_per_cell : int
        Number of points per grid cell
    output_dir : str
        Directory to save results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    total_start = time.time()
    results_summary = []
    
    print("=" * 80)
    print(f"🔄 PROCESSING {len(grid_sizes)} GRID SIZES")
    print(f"   Grid sizes: {', '.join(grid_sizes)}")
    print(f"   Points per cell: {points_per_cell}")
    print(f"   Using {N_CPUS} CPUs per grid")
    print("=" * 80)
    
    for i, grid_size in enumerate(grid_sizes, 1):
        print(f"\n{'=' * 80}")
        print(f"📍 GRID {i}/{len(grid_sizes)}: {grid_size.upper()}")
        print(f"{'=' * 80}")
        
        # Construct file paths
        grid_file = Path(aoi_dir) / f"grid_{grid_size}.gpkg"
        output_file = output_path / f"sample_points_{grid_size}.gpkg"
        
        # Check if grid file exists
        if not grid_file.exists():
            logger.error(f"❌ Grid file not found: {grid_file}")
            results_summary.append({
                'grid_size': grid_size,
                'status': 'FAILED',
                'reason': 'File not found'
            })
            continue
        
        try:
            # Load grid
            print(f"📂 Loading grid from: {grid_file}")
            grid_gdf = gpd.read_file(grid_file)
            print(f"   Loaded {len(grid_gdf):,} grid cells")
            
            # Process grid
            sample_points = generate_systematic_sample_points_parallel(
                grid_gdf,
                points_per_cell=points_per_cell,
                raster_folder=raster_folder,
                chunk_size=None  # Auto-calculate
            )
            
            # Save results
            print(f"💾 Saving results to: {output_file}")
            sample_points.to_file(output_file, driver="GPKG")
            
            # Record success
            results_summary.append({
                'grid_size': grid_size,
                'status': 'SUCCESS',
                'n_cells': len(grid_gdf),
                'n_points': len(sample_points),
                'output_file': str(output_file)
            })
            
            print(f"✅ Completed {grid_size}: {len(sample_points):,} points saved")
            
        except Exception as e:
            logger.error(f"❌ Error processing {grid_size}: {e}")
            results_summary.append({
                'grid_size': grid_size,
                'status': 'FAILED',
                'reason': str(e)
            })
            continue
    
    # Print summary
    total_time = time.time() - total_start
    print(f"\n{'=' * 80}")
    print(f"🏁 PROCESSING COMPLETE")
    print(f"{'=' * 80}")
    print(f"Total processing time: {total_time / 60:.1f} minutes")
    print(f"\nResults Summary:")
    print("-" * 80)
    
    for result in results_summary:
        if result['status'] == 'SUCCESS':
            print(f"✅ {result['grid_size']:>6s}: {result['n_points']:>10,} points "
                  f"from {result['n_cells']:>6,} cells")
        else:
            print(f"❌ {result['grid_size']:>6s}: FAILED - {result.get('reason', 'Unknown error')}")
    
    print("-" * 80)
    
    # Save summary to JSON
    summary_file = output_path / "processing_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'total_time_seconds': total_time,
            'n_cpus': N_CPUS,
            'points_per_cell': points_per_cell,
            'results': results_summary
        }, f, indent=2)
    print(f"\n📋 Summary saved to: {summary_file}")
    
    return results_summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=f"HPC-optimized sample point generation ({N_CPUS} CPUs)"
    )
    parser.add_argument("--mode", choices=["parallel", "multi-grid", "array-prep", "array-process", "array-merge"],
                        default="multi-grid", help="Processing mode")
    parser.add_argument("--grid-file", required=False,
                        help="Single grid file (for parallel mode)")
    parser.add_argument("--grid-sizes", nargs='+', default=['3km', '24km', '54km'],
                        help="Grid sizes to process (for multi-grid mode)")
    parser.add_argument("--aoi-dir", default="AOI",
                        help="Directory containing grid files (for multi-grid mode)")
    parser.add_argument("--raster-folder", default="Meta CHM Binary")
    parser.add_argument("--points-per-cell", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=None,
                        help="Cells per chunk (auto-calculated if not specified)")
    parser.add_argument("--cells-per-job", type=int, default=None,
                        help="Cells per array job (for array-prep mode)")
    parser.add_argument("--chunk-id", type=int, default=None)
    parser.add_argument("--chunks-dir", default="grid_chunks")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--output-file", required=False,
                        help="Output file (for parallel mode)")

    args = parser.parse_args()

    if args.mode == "multi-grid":
        # Process multiple grid sizes (3km, 24km, 54km)
        process_multiple_grids(
            grid_sizes=args.grid_sizes,
            aoi_dir=args.aoi_dir,
            raster_folder=args.raster_folder,
            points_per_cell=args.points_per_cell,
            output_dir=args.output_dir
        )

    elif args.mode == "parallel":
        # Standard parallel processing on single node with 192 CPUs
        if not args.grid_file:
            raise ValueError("--grid-file required for parallel mode")
        if not args.output_file:
            raise ValueError("--output-file required for parallel mode")
        
        grid_gdf = gpd.read_file(args.grid_file)
        sample_points = generate_systematic_sample_points_parallel(
            grid_gdf,
            points_per_cell=args.points_per_cell,
            raster_folder=args.raster_folder,
            chunk_size=args.chunk_size
        )
        sample_points.to_file(args.output_file, driver="GPKG")
        print(f"\n✅ Saved {len(sample_points)} points to {args.output_file}")

    elif args.mode == "array-prep":
        # Prepare chunks for array jobs
        grid_gdf = gpd.read_file(args.grid_file)
        save_grid_chunks_for_array_jobs(
            grid_gdf,
            output_dir=args.chunks_dir,
            cells_per_job=args.cells_per_job
        )

    elif args.mode == "array-process":
        # Process single chunk using all 192 CPUs
        if args.chunk_id is None:
            raise ValueError("--chunk-id required for array-process mode")
        process_single_chunk_for_array_job(
            args.chunk_id,
            chunks_dir=args.chunks_dir,
            output_dir=args.output_dir,
            points_per_cell=args.points_per_cell,
            raster_folder=args.raster_folder
        )

    elif args.mode == "array-merge":
        # Merge results from array jobs
        merge_chunk_results(args.output_dir, args.output_file)
