# Neutral Landscape Generator with Bootstrap Sampling Analysis
# Designed for HPC environments with configurable parameters

import numpy as np
import pandas as pd
import nlmpy
from pysal.lib import weights
from pysal.explore import esda
from scipy.optimize import minimize_scalar
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import time
import warnings
import argparse
import os
import sys

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION SETTINGS - MODIFY THESE AS NEEDED
# =============================================================================

# Landscape Parameters
AOI_VALUES = [200, 600, 4000]  # Area of interest in m²
CANOPY_EXTENTS = [0.2, 0.4, 0.6, 0.8]  # Target canopy coverage proportions
MORANS_I_VALUES = [-0.5, -0.2, 0.0, 0.2, 0.5, 0.8]  # Target Moran's I values

# Sampling Parameters
N_SAMPLE_POINTS = 100000  # Number of random sample points to generate
CELL_SIZE = 1  # Size of each cell in meters
RANDOM_SEED = 42  # For reproducible results

# Replication and Processing
N_REPLICATES = 3  # Number of replicates per parameter combination (increased for 30min run)
SAVE_LANDSCAPES = True  # Whether to save landscape arrays (required for sample points export)

# Quality Control Tolerances
MORANS_TOLERANCE = 0.025  # Acceptable difference from target Moran's I (±0.025)
CANOPY_TOLERANCE = 0.001  # Acceptable difference from target canopy extent (±0.1%)
FILTER_UNSUCCESSFUL = True  # Only keep landscapes meeting tolerance criteria

# Optimization Parameters
MAX_ITERATIONS = 75  # Increased for better success rates with 192 CPUs
N_BOOTSTRAP_SAMPLES = 1000  # Number of bootstrap samples for confidence intervals
CONFIDENCE_LEVEL = 0.95  # Confidence level for intervals

# Algorithm Selection (None = auto-select based on target Moran's I)
FORCE_ALGORITHM = None  # Options: None, 'mpd', 'randomClusterNN', 'random'

# Output Settings - Configured for your scratch directory
OUTPUT_DIR = '/scratch/arbmarta/NLM/results'  # Output directory on scratch
OUTPUT_PREFIX = 'NLM_analysis'  # Prefix for output files
GENERATE_PLOTS = False  # Set to False for HPC environments without display

# HPC Settings - Optimized for your 192-CPU allocation
VERBOSE = True  # Print progress messages
SAVE_MEMORY = False  # With 192 CPUs, we can afford more memory usage for speed
PARALLEL_SAFE = True  # Ensure thread safety for parallel execution

# =============================================================================
# MAIN GENERATOR CLASS
# =============================================================================

class NeutralLandscapeGenerator:
    """
    Generate neutral landscapes with specified AOI, canopy extent, and Moran's I values
    Apply consistent random sampling across all landscapes for comparison studies
    Includes comprehensive bootstrap analysis for confidence intervals
    """
    
    def __init__(self, cell_size=CELL_SIZE, n_sample_points=N_SAMPLE_POINTS, 
                 random_seed=RANDOM_SEED, verbose=VERBOSE):
        """
        Initialize generator with configuration parameters
        """
        self.cell_size = cell_size
        self.n_sample_points = n_sample_points
        self.random_seed = random_seed
        self.verbose = verbose
        self.results = []
        self.sample_points_cache = {}  # Cache sample points by AOI size
        
        if self.verbose:
            print(f"Initialized NeutralLandscapeGenerator:")
            print(f"  Cell size: {cell_size}m")
            print(f"  Sample points: {n_sample_points:,}")
            print(f"  Random seed: {random_seed}")
        
    def log(self, message):
        """Thread-safe logging function"""
        if self.verbose:
            print(f"[{time.strftime('%H:%M:%S')}] {message}")
            sys.stdout.flush()
        
    def generate_sample_points(self, aoi_m2):
        """
        Generate consistent random sample points for a given AOI
        """
        if aoi_m2 in self.sample_points_cache:
            return self.sample_points_cache[aoi_m2]
        
        side_length = np.sqrt(aoi_m2)
        np.random.seed(self.random_seed)
        
        x_coords = np.random.uniform(0, side_length, self.n_sample_points)
        y_coords = np.random.uniform(0, side_length, self.n_sample_points)
        
        self.sample_points_cache[aoi_m2] = (x_coords, y_coords)
        return x_coords, y_coords
    
    def sample_landscape(self, landscape, x_coords, y_coords):
        """
        Sample landscape values at given coordinates
        """
        nrows, ncols = landscape.shape
        
        col_indices = np.floor(x_coords / self.cell_size).astype(int)
        row_indices = np.floor(y_coords / self.cell_size).astype(int)
        
        col_indices = np.clip(col_indices, 0, ncols - 1)
        row_indices = np.clip(row_indices, 0, nrows - 1)
        
        sampled_values = landscape[row_indices, col_indices]
        return sampled_values
    
    def calculate_bootstrap_statistics(self, sampled_values, n_bootstrap=N_BOOTSTRAP_SAMPLES, 
                                     confidence_level=CONFIDENCE_LEVEL):
        """
        Calculate bootstrap statistics for sampling results
        """
        n_samples = len(sampled_values)
        bootstrap_proportions = []
        
        np.random.seed(self.random_seed + 1000)
        
        for i in range(n_bootstrap):
            bootstrap_sample = np.random.choice(sampled_values, size=n_samples, replace=True)
            bootstrap_proportion = np.mean(bootstrap_sample)
            bootstrap_proportions.append(bootstrap_proportion)
        
        bootstrap_proportions = np.array(bootstrap_proportions)
        
        bootstrap_mean = np.mean(bootstrap_proportions)
        bootstrap_std = np.std(bootstrap_proportions)
        
        alpha = 1 - confidence_level
        lower_percentile = (alpha/2) * 100
        upper_percentile = (1 - alpha/2) * 100
        
        ci_lower = np.percentile(bootstrap_proportions, lower_percentile)
        ci_upper = np.percentile(bootstrap_proportions, upper_percentile)
        
        # BCa confidence intervals
        original_proportion = np.mean(sampled_values)
        bias_correction = np.sum(bootstrap_proportions < original_proportion) / n_bootstrap
        bias_correction = 2 * bias_correction - 1
        
        # Jackknife for acceleration
        jackknife_proportions = []
        for i in range(n_samples):
            jackknife_sample = np.concatenate([sampled_values[:i], sampled_values[i+1:]])
            jackknife_proportions.append(np.mean(jackknife_sample))
        
        jackknife_proportions = np.array(jackknife_proportions)
        jackknife_mean = np.mean(jackknife_proportions)
        
        if np.sum((jackknife_mean - jackknife_proportions)**2) > 0:
            acceleration = np.sum((jackknife_mean - jackknife_proportions)**3) / (6 * (np.sum((jackknife_mean - jackknife_proportions)**2))**1.5)
        else:
            acceleration = 0
        
        z_alpha_2 = norm.ppf(alpha/2)
        z_1_alpha_2 = norm.ppf(1 - alpha/2)
        
        bca_lower_z = bias_correction + (bias_correction + z_alpha_2) / (1 - acceleration * (bias_correction + z_alpha_2))
        bca_upper_z = bias_correction + (bias_correction + z_1_alpha_2) / (1 - acceleration * (bias_correction + z_1_alpha_2))
        
        bca_lower_percentile = np.clip(norm.cdf(bca_lower_z) * 100, 0, 100)
        bca_upper_percentile = np.clip(norm.cdf(bca_upper_z) * 100, 0, 100)
        
        bca_ci_lower = np.percentile(bootstrap_proportions, bca_lower_percentile)
        bca_ci_upper = np.percentile(bootstrap_proportions, bca_upper_percentile)
        
        return {
            'bootstrap_mean': bootstrap_mean,
            'bootstrap_std': bootstrap_std,
            'bootstrap_samples': n_bootstrap,
            'percentile_ci_lower': ci_lower,
            'percentile_ci_upper': ci_upper,
            'percentile_ci_width': ci_upper - ci_lower,
            'bca_ci_lower': bca_ci_lower,
            'bca_ci_upper': bca_ci_upper,
            'bca_ci_width': bca_ci_upper - bca_ci_lower,
            'bias_correction': bias_correction,
            'acceleration': acceleration
        }
    
    def calculate_sampling_statistics(self, sampled_values, true_canopy_proportion):
        """
        Calculate comprehensive sampling statistics including bootstrap analysis
        """
        n_canopy_hits = np.sum(sampled_values)
        estimated_proportion = n_canopy_hits / len(sampled_values)
        bias = estimated_proportion - true_canopy_proportion
        absolute_error = abs(bias)
        
        std_error = np.sqrt(estimated_proportion * (1 - estimated_proportion) / len(sampled_values))
        ci_lower = estimated_proportion - 1.96 * std_error
        ci_upper = estimated_proportion + 1.96 * std_error
        ci_width = ci_upper - ci_lower
        
        bootstrap_stats = self.calculate_bootstrap_statistics(sampled_values)
        
        ci_contains_true = ci_lower <= true_canopy_proportion <= ci_upper
        bootstrap_percentile_contains_true = bootstrap_stats['percentile_ci_lower'] <= true_canopy_proportion <= bootstrap_stats['percentile_ci_upper']
        bootstrap_bca_contains_true = bootstrap_stats['bca_ci_lower'] <= true_canopy_proportion <= bootstrap_stats['bca_ci_upper']
        
        return {
            'n_sample_points': len(sampled_values),
            'n_canopy_hits': n_canopy_hits,
            'estimated_canopy_proportion': estimated_proportion,
            'true_canopy_proportion': true_canopy_proportion,
            'bias': bias,
            'absolute_error': absolute_error,
            'relative_error': absolute_error / true_canopy_proportion if true_canopy_proportion > 0 else 0,
            'standard_error': std_error,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_width': ci_width,
            'ci_contains_true': ci_contains_true,
            'bootstrap_mean': bootstrap_stats['bootstrap_mean'],
            'bootstrap_std': bootstrap_stats['bootstrap_std'],
            'bootstrap_samples': bootstrap_stats['bootstrap_samples'],
            'bootstrap_percentile_ci_lower': bootstrap_stats['percentile_ci_lower'],
            'bootstrap_percentile_ci_upper': bootstrap_stats['percentile_ci_upper'],
            'bootstrap_percentile_ci_width': bootstrap_stats['percentile_ci_width'],
            'bootstrap_percentile_contains_true': bootstrap_percentile_contains_true,
            'bootstrap_bca_ci_lower': bootstrap_stats['bca_ci_lower'],
            'bootstrap_bca_ci_upper': bootstrap_stats['bca_ci_upper'],
            'bootstrap_bca_ci_width': bootstrap_stats['bca_ci_width'],
            'bootstrap_bca_contains_true': bootstrap_bca_contains_true,
            'bias_correction': bootstrap_stats['bias_correction'],
            'acceleration': bootstrap_stats['acceleration']
        }
        
    def calculate_dimensions(self, aoi_m2):
        """Calculate grid dimensions for given area of interest"""
        side_length = np.sqrt(aoi_m2)
        cells_per_side = int(side_length / self.cell_size)
        return cells_per_side, cells_per_side
    
    def calculate_morans_i(self, binary_landscape):
        """Calculate Moran's I for a binary landscape"""
        nrows, ncols = binary_landscape.shape
        w = weights.lat2W(nrows, ncols, rook=False)
        moran = esda.Moran(binary_landscape.flatten(), w)
        return moran.I
    
    def select_optimal_algorithm(self, target_morans_i):
        """Select the best algorithm based on target Moran's I value"""
        if target_morans_i < 0.2:
            return 'random'
        elif target_morans_i > 0.8:
            return 'randomClusterNN'
        else:
            return 'mpd'
    
    def generate_landscape_with_target_morans(self, nrows, ncols, canopy_percent, 
                                            target_morans_i, algorithm=None, 
                                            morans_tolerance=MORANS_TOLERANCE, 
                                            canopy_tolerance=CANOPY_TOLERANCE, 
                                            max_iterations=MAX_ITERATIONS):
        """
        Generate landscape with target Moran's I value using optimal algorithm selection
        """
        
        if algorithm is None:
            algorithm = self.select_optimal_algorithm(target_morans_i)
            if self.verbose:
                self.log(f"Auto-selected algorithm: {algorithm} for target Moran's I = {target_morans_i:.2f}")
        
        def objective_function(param):
            try:
                if algorithm == 'mpd':
                    landscape = nlmpy.mpd(nrows, ncols, h=param)
                elif algorithm == 'randomClusterNN':
                    landscape = nlmpy.randomClusterNN(nrows, ncols, p=param, n=4)
                elif algorithm == 'random':
                    landscape = nlmpy.random(nrows, ncols)
                    if param > 0.1:
                        from scipy import ndimage
                        landscape = ndimage.gaussian_filter(landscape, sigma=param)
                else:
                    raise ValueError(f"Unknown algorithm: {algorithm}")
                
                threshold = np.percentile(landscape, (1-canopy_percent)*100)
                binary_landscape = (landscape > threshold).astype(int)
                achieved_morans_i = self.calculate_morans_i(binary_landscape)
                
                return abs(achieved_morans_i - target_morans_i)
                
            except Exception as e:
                return 999.0
        
        def check_success(landscape, achieved_morans_i, target_morans_i, canopy_percent):
            actual_canopy = np.mean(landscape)
            morans_success = abs(achieved_morans_i - target_morans_i) <= morans_tolerance
            canopy_success = abs(actual_canopy - canopy_percent) <= canopy_tolerance
            return morans_success and canopy_success, actual_canopy
        
        # Algorithm-specific optimization
        if algorithm == 'random':
            try:
                landscape = nlmpy.random(nrows, ncols)
                threshold = np.percentile(landscape, (1-canopy_percent)*100)
                binary_landscape = (landscape > threshold).astype(int)
                achieved_morans_i = self.calculate_morans_i(binary_landscape)
                
                success, actual_canopy = check_success(binary_landscape, achieved_morans_i, target_morans_i, canopy_percent)
                if success:
                    return {
                        'landscape': binary_landscape,
                        'continuous_landscape': landscape,
                        'target_morans_i': target_morans_i,
                        'achieved_morans_i': achieved_morans_i,
                        'difference': abs(achieved_morans_i - target_morans_i),
                        'target_canopy': canopy_percent,
                        'actual_canopy': actual_canopy,
                        'canopy_difference': abs(actual_canopy - canopy_percent),
                        'optimal_parameter': 0.0,
                        'algorithm': algorithm,
                        'success': True,
                        'morans_tolerance': morans_tolerance,
                        'canopy_tolerance': canopy_tolerance
                    }
                bounds = (0.0, 2.0)
            except Exception:
                bounds = (0.0, 2.0)
        elif algorithm == 'mpd':
            if target_morans_i < 0.3:
                bounds = (0.7, 0.99)
            elif target_morans_i > 0.7:
                bounds = (0.01, 0.3)
            else:
                bounds = (0.01, 0.99)
        else:  # randomClusterNN
            if target_morans_i > 0.7:
                bounds = (0.3, 0.8)
            else:
                bounds = (0.01, 0.5)
        
        try:
            result = minimize_scalar(objective_function, bounds=bounds, 
                                   method='bounded', options={'maxiter': max_iterations})
            
            optimal_param = result.x
            
            if algorithm == 'mpd':
                final_continuous = nlmpy.mpd(nrows, ncols, h=optimal_param)
            elif algorithm == 'randomClusterNN':
                final_continuous = nlmpy.randomClusterNN(nrows, ncols, p=optimal_param, n=4)
            elif algorithm == 'random':
                final_continuous = nlmpy.random(nrows, ncols)
                if optimal_param > 0.1:
                    from scipy import ndimage
                    final_continuous = ndimage.gaussian_filter(final_continuous, sigma=optimal_param)
            
            threshold = np.percentile(final_continuous, (1-canopy_percent)*100)
            final_binary = (final_continuous > threshold).astype(int)
            achieved_morans_i = self.calculate_morans_i(final_binary)
            success, actual_canopy = check_success(final_binary, achieved_morans_i, target_morans_i, canopy_percent)
            
            return {
                'landscape': final_binary,
                'continuous_landscape': final_continuous,
                'target_morans_i': target_morans_i,
                'achieved_morans_i': achieved_morans_i,
                'difference': abs(achieved_morans_i - target_morans_i),
                'target_canopy': canopy_percent,
                'actual_canopy': actual_canopy,
                'canopy_difference': abs(actual_canopy - canopy_percent),
                'optimal_parameter': optimal_param,
                'algorithm': algorithm,
                'success': success,
                'morans_tolerance': morans_tolerance,
                'canopy_tolerance': canopy_tolerance
            }
            
        except Exception as e:
            if self.verbose:
                self.log(f"Error in optimization: {e}")
            
            # Fallback strategy
            fallback_algorithms = ['random', 'mpd', 'randomClusterNN']
            fallback_algorithms = [alg for alg in fallback_algorithms if alg != algorithm]
            
            for fallback_alg in fallback_algorithms:
                try:
                    if self.verbose:
                        self.log(f"Trying fallback algorithm: {fallback_alg}")
                    fallback_result = self.generate_landscape_with_target_morans(
                        nrows, ncols, canopy_percent, target_morans_i, 
                        algorithm=fallback_alg, morans_tolerance=morans_tolerance*1.5, 
                        canopy_tolerance=canopy_tolerance*1.5, max_iterations=max_iterations//2
                    )
                    fallback_result['algorithm'] = f"{algorithm}_fallback_{fallback_alg}"
                    return fallback_result
                except Exception:
                    continue
            
            # Final fallback
            if self.verbose:
                self.log("Using final random fallback")
            fallback = nlmpy.random(nrows, ncols)
            threshold = np.percentile(fallback, (1-canopy_percent)*100)
            fallback_binary = (fallback > threshold).astype(int)
            fallback_morans = self.calculate_morans_i(fallback_binary)
            fallback_success, fallback_actual_canopy = check_success(fallback_binary, fallback_morans, target_morans_i, canopy_percent)
            
            return {
                'landscape': fallback_binary,
                'continuous_landscape': fallback,
                'target_morans_i': target_morans_i,
                'achieved_morans_i': fallback_morans,
                'difference': abs(fallback_morans - target_morans_i),
                'target_canopy': canopy_percent,
                'actual_canopy': fallback_actual_canopy,
                'canopy_difference': abs(fallback_actual_canopy - canopy_percent),
                'optimal_parameter': None,
                'algorithm': 'final_random_fallback',
                'success': fallback_success,
                'morans_tolerance': morans_tolerance,
                'canopy_tolerance': canopy_tolerance
            }
    
    def run_parameter_sweep(self, aoi_values, canopy_extents, morans_i_values, 
                          algorithm=FORCE_ALGORITHM, n_replicates=N_REPLICATES, 
                          save_landscapes=SAVE_LANDSCAPES, morans_tolerance=MORANS_TOLERANCE, 
                          canopy_tolerance=CANOPY_TOLERANCE, filter_unsuccessful=FILTER_UNSUCCESSFUL):
        """
        Run complete parameter sweep across all combinations
        """
        
        if algorithm is None:
            self.log("Starting parameter sweep with AUTO-SELECTED algorithms...")
            self.log("Algorithm selection strategy:")
            self.log("  Moran's I < 0.2: 'random' algorithm")
            self.log("  Moran's I 0.2-0.8: 'mpd' algorithm") 
            self.log("  Moran's I > 0.8: 'randomClusterNN' algorithm")
        else:
            self.log(f"Starting parameter sweep with {algorithm} algorithm...")
        
        self.log(f"AOI values: {aoi_values}")
        self.log(f"Canopy extents: {canopy_extents}")
        self.log(f"Moran's I values: {morans_i_values}")
        self.log(f"Replicates per combination: {n_replicates}")
        self.log(f"Tolerances: Moran's I ±{morans_tolerance}, Canopy ±{canopy_tolerance:.1%}")
        
        if filter_unsuccessful:
            self.log("FILTERING: Only successful landscapes will be kept for analysis")
        else:
            self.log("NO FILTERING: All landscapes will be kept regardless of success")
        
        total_combinations = len(aoi_values) * len(canopy_extents) * len(morans_i_values) * n_replicates
        self.log(f"Total landscapes to generate: {total_combinations}")
        
        self.results = []
        combination_count = 0
        filtered_count = 0
        
        for aoi, canopy_extent, target_morans_i in product(aoi_values, canopy_extents, morans_i_values):
            for replicate in range(n_replicates):
                combination_count += 1
                start_time = time.time()
                
                self.log(f"Combination {combination_count}/{total_combinations}")
                self.log(f"AOI: {aoi}m², Canopy: {canopy_extent:.1%}, Moran's I: {target_morans_i:.2f}, Rep: {replicate+1}")
                
                nrows, ncols = self.calculate_dimensions(aoi)
                
                result = self.generate_landscape_with_target_morans(
                    nrows, ncols, canopy_extent, target_morans_i, algorithm,
                    morans_tolerance, canopy_tolerance
                )
                
                if filter_unsuccessful and not result['success']:
                    filtered_count += 1
                    self.log(f"FILTERED: Moran's I error={result['difference']:.4f}, Canopy error={result.get('canopy_difference', 0):.4f}")
                    continue
                
                x_coords, y_coords = self.generate_sample_points(aoi)
                sampled_values = self.sample_landscape(result['landscape'], x_coords, y_coords)
                true_canopy_proportion = np.mean(result['landscape'])
                sampling_stats = self.calculate_sampling_statistics(sampled_values, true_canopy_proportion)
                
                result_record = {
                    'combination_id': combination_count,
                    'replicate': replicate + 1,
                    'aoi_m2': aoi,
                    'nrows': nrows,
                    'ncols': ncols,
                    'actual_area_m2': nrows * ncols * self.cell_size**2,
                    'morans_i_target': target_morans_i,
                    'morans_i_achieved': result['achieved_morans_i'],
                    'morans_i_difference': result['difference'],
                    'canopy_extent_target': canopy_extent,
                    'canopy_extent_actual': result.get('actual_canopy', np.mean(result['landscape'])),
                    'canopy_difference': result.get('canopy_difference', abs(np.mean(result['landscape']) - canopy_extent)),
                    'optimal_parameter': result['optimal_parameter'],
                    'algorithm': result['algorithm'],
                    'success': result['success'],
                    'generation_time_seconds': time.time() - start_time,
                    'morans_tolerance_used': morans_tolerance,
                    'canopy_tolerance_used': canopy_tolerance,
                    # Sampling results
                    'n_sample_points': sampling_stats['n_sample_points'],
                    'n_canopy_hits': sampling_stats['n_canopy_hits'],
                    'estimated_canopy_proportion': sampling_stats['estimated_canopy_proportion'],
                    'sampling_bias': sampling_stats['bias'],
                    'sampling_absolute_error': sampling_stats['absolute_error'],
                    'sampling_relative_error': sampling_stats['relative_error'],
                    'ci_lower': sampling_stats['ci_lower'],
                    'ci_upper': sampling_stats['ci_upper'],
                    'ci_width': sampling_stats['ci_width'],
                    'ci_contains_true': sampling_stats['ci_contains_true'],
                    'standard_error': sampling_stats['standard_error'],
                    # Bootstrap results
                    'bootstrap_mean': sampling_stats['bootstrap_mean'],
                    'bootstrap_std': sampling_stats['bootstrap_std'],
                    'bootstrap_samples': sampling_stats['bootstrap_samples'],
                    'bootstrap_percentile_ci_lower': sampling_stats['bootstrap_percentile_ci_lower'],
                    'bootstrap_percentile_ci_upper': sampling_stats['bootstrap_percentile_ci_upper'],
                    'bootstrap_percentile_ci_width': sampling_stats['bootstrap_percentile_ci_width'],
                    'bootstrap_percentile_contains_true': sampling_stats['bootstrap_percentile_contains_true'],
                    'bootstrap_bca_ci_lower': sampling_stats['bootstrap_bca_ci_lower'],
                    'bootstrap_bca_ci_upper': sampling_stats['bootstrap_bca_ci_upper'],
                    'bootstrap_bca_ci_width': sampling_stats['bootstrap_bca_ci_width'],
                    'bootstrap_bca_contains_true': sampling_stats['bootstrap_bca_contains_true'],
                    'bias_correction': sampling_stats['bias_correction'],
                    'acceleration': sampling_stats['acceleration']
                }
                
                if save_landscapes:
                    result_record['landscape_array'] = result['landscape']
                    result_record['continuous_array'] = result['continuous_landscape']
                    result_record['sample_points_x'] = x_coords
                    result_record['sample_points_y'] = y_coords
                    result_record['sample_values'] = sampled_values
                
                self.results.append(result_record)
                
                # Progress update
                kept_results = len(self.results)
                if kept_results > 0:
                    success_rate = np.mean([r['success'] for r in self.results])
                    avg_morans_error = np.mean([r['morans_i_difference'] for r in self.results])
                    avg_canopy_error = np.mean([r['canopy_difference'] for r in self.results])
                    self.log(f"Success: {'✓' if result['success'] else '✗'} | "
                            f"Kept: {kept_results} | Filtered: {filtered_count} | "
                            f"Success rate: {success_rate:.1%} | "
                            f"Avg Moran's error: {avg_morans_error:.4f} | "
                            f"Avg canopy error: {avg_canopy_error:.4f}")
                else:
                    self.log(f"Success: {'✓' if result['success'] else '✗'} | "
                            f"Kept: 0 | Filtered: {filtered_count}")
        
        final_df = pd.DataFrame(self.results)
        
        self.log("="*60)
        self.log("FILTERING SUMMARY")
        self.log("="*60)
        self.log(f"Total combinations attempted: {combination_count}")
        self.log(f"Landscapes kept: {len(final_df)}")
        self.log(f"Landscapes filtered out: {filtered_count}")
        if combination_count > 0:
            self.log(f"Overall retention rate: {len(final_df)/combination_count:.1%}")
        
        return final_df
    
    def export_sample_points_csv(self, results_df, filename):
        """Export sample point results in the specified format"""
        
        if 'sample_points_x' not in results_df.columns:
            raise ValueError("Sample points data not found. Run parameter sweep with save_landscapes=True")
        
        self.log(f"Exporting sample points data to {filename}...")
        
        sample_points_data = []
        
        for aoi in results_df['aoi_m2'].unique():
            aoi_data = results_df[results_df['aoi_m2'] == aoi].iloc[0]
            x_coords = aoi_data['sample_points_x']
            y_coords = aoi_data['sample_points_y']
            
            aoi_sample_df = pd.DataFrame({
                'Sample_Point_ID': range(1, len(x_coords) + 1),
                'Sample_Point_X_Location': x_coords,
                'Sample_Point_Y_Location': y_coords
            })
            
            aoi_results = results_df[results_df['aoi_m2'] == aoi]
