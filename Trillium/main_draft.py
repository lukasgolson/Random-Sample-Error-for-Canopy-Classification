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
warnings.filterwarnings('ignore')

class NeutralLandscapeGenerator:
    """
    Generate neutral landscapes with specified AOI, canopy extent, and Moran's I values
    Apply consistent random sampling across all landscapes for comparison studies
    Includes comprehensive bootstrap analysis for confidence intervals
    """
    
    def __init__(self, cell_size=1, n_sample_points=100000, random_seed=42):
        """
        Initialize generator
        
        Parameters:
        -----------
        cell_size : float
            Size of each cell in meters (default=1m for easier calculations)
        n_sample_points : int
            Number of random sample points to generate (default=100,000)
        random_seed : int
            Random seed for reproducible sample point generation
        """
        self.cell_size = cell_size
        self.n_sample_points = n_sample_points
        self.random_seed = random_seed
        self.results = []
        self.sample_points_cache = {}  # Cache sample points by AOI size
        
    def generate_sample_points(self, aoi_m2):
        """
        Generate consistent random sample points for a given AOI
        
        Parameters:
        -----------
        aoi_m2 : float
            Area of interest in square meters
            
        Returns:
        --------
        tuple : (x_coords, y_coords) arrays of sample point coordinates
        """
        # Check if we already have sample points for this AOI
        if aoi_m2 in self.sample_points_cache:
            return self.sample_points_cache[aoi_m2]
        
        # Calculate landscape extent
        side_length = np.sqrt(aoi_m2)
        
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
        
        # Generate random coordinates within the AOI bounds
        x_coords = np.random.uniform(0, side_length, self.n_sample_points)
        y_coords = np.random.uniform(0, side_length, self.n_sample_points)
        
        # Cache the sample points
        self.sample_points_cache[aoi_m2] = (x_coords, y_coords)
        
        return x_coords, y_coords
    
    def sample_landscape(self, landscape, x_coords, y_coords):
        """
        Sample landscape values at given coordinates
        
        Parameters:
        -----------
        landscape : np.array
            2D binary landscape array (0s and 1s)
        x_coords, y_coords : np.array
            Sample point coordinates
            
        Returns:
        --------
        np.array : Binary array of canopy hits (1) or misses (0)
        """
        nrows, ncols = landscape.shape
        
        # Convert coordinates to grid indices
        # Assuming landscape spans from 0 to sqrt(AOI) in both directions
        side_length = nrows * self.cell_size
        
        col_indices = np.floor(x_coords / self.cell_size).astype(int)
        row_indices = np.floor(y_coords / self.cell_size).astype(int)
        
        # Ensure indices are within bounds
        col_indices = np.clip(col_indices, 0, ncols - 1)
        row_indices = np.clip(row_indices, 0, nrows - 1)
        
        # Sample landscape values
        sampled_values = landscape[row_indices, col_indices]
        
        return sampled_values
    
    def calculate_bootstrap_statistics(self, sampled_values, n_bootstrap=1000, confidence_level=0.95):
        """
        Calculate bootstrap statistics for sampling results
        
        Parameters:
        -----------
        sampled_values : np.array
            Binary array of sample results (0s and 1s)
        n_bootstrap : int
            Number of bootstrap samples (default=1000)
        confidence_level : float
            Confidence level for intervals (default=0.95)
            
        Returns:
        --------
        dict : Bootstrap statistics
        """
        n_samples = len(sampled_values)
        bootstrap_proportions = []
        
        # Set random seed for reproducible bootstrap
        np.random.seed(self.random_seed + 1000)  # Different seed from sample generation
        
        # Generate bootstrap samples
        for i in range(n_bootstrap):
            # Resample with replacement
            bootstrap_sample = np.random.choice(sampled_values, size=n_samples, replace=True)
            bootstrap_proportion = np.mean(bootstrap_sample)
            bootstrap_proportions.append(bootstrap_proportion)
        
        bootstrap_proportions = np.array(bootstrap_proportions)
        
        # Calculate statistics
        bootstrap_mean = np.mean(bootstrap_proportions)
        bootstrap_std = np.std(bootstrap_proportions)
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha/2) * 100
        upper_percentile = (1 - alpha/2) * 100
        
        ci_lower = np.percentile(bootstrap_proportions, lower_percentile)
        ci_upper = np.percentile(bootstrap_proportions, upper_percentile)
        
        # Calculate bias-corrected confidence interval (BCa method)
        original_proportion = np.mean(sampled_values)
        
        # Bias correction
        bias_correction = np.sum(bootstrap_proportions < original_proportion) / n_bootstrap
        bias_correction = 2 * bias_correction - 1
        
        # Acceleration (jackknife estimate)
        jackknife_proportions = []
        for i in range(n_samples):
            jackknife_sample = np.concatenate([sampled_values[:i], sampled_values[i+1:]])
            jackknife_proportions.append(np.mean(jackknife_sample))
        
        jackknife_proportions = np.array(jackknife_proportions)
        jackknife_mean = np.mean(jackknife_proportions)
        acceleration = np.sum((jackknife_mean - jackknife_proportions)**3) / (6 * (np.sum((jackknife_mean - jackknife_proportions)**2))**1.5)
        
        # BCa confidence intervals
        z_alpha_2 = norm.ppf(alpha/2)
        z_1_alpha_2 = norm.ppf(1 - alpha/2)
        
        bca_lower_z = bias_correction + (bias_correction + z_alpha_2) / (1 - acceleration * (bias_correction + z_alpha_2))
        bca_upper_z = bias_correction + (bias_correction + z_1_alpha_2) / (1 - acceleration * (bias_correction + z_1_alpha_2))
        
        bca_lower_percentile = norm.cdf(bca_lower_z) * 100
        bca_upper_percentile = norm.cdf(bca_upper_z) * 100
        
        # Ensure percentiles are within valid range
        bca_lower_percentile = np.clip(bca_lower_percentile, 0, 100)
        bca_upper_percentile = np.clip(bca_upper_percentile, 0, 100)
        
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
            'acceleration': acceleration,
            'bootstrap_proportions': bootstrap_proportions  # For additional analysis if needed
        }
    
    def calculate_sampling_statistics(self, sampled_values, true_canopy_proportion):
        """
        Calculate sampling statistics including bootstrap analysis
        
        Parameters:
        -----------
        sampled_values : np.array
            Binary array of sample results
        true_canopy_proportion : float
            True proportion of canopy in landscape
            
        Returns:
        --------
        dict : Comprehensive sampling statistics including bootstrap results
        """
        n_canopy_hits = np.sum(sampled_values)
        estimated_proportion = n_canopy_hits / len(sampled_values)
        bias = estimated_proportion - true_canopy_proportion
        absolute_error = abs(bias)
        
        # Calculate standard confidence interval (95%)
        std_error = np.sqrt(estimated_proportion * (1 - estimated_proportion) / len(sampled_values))
        ci_lower = estimated_proportion - 1.96 * std_error
        ci_upper = estimated_proportion + 1.96 * std_error
        ci_width = ci_upper - ci_lower
        
        # Calculate bootstrap statistics
        bootstrap_stats = self.calculate_bootstrap_statistics(sampled_values)
        
        # Check if true value is contained in different confidence intervals
        ci_contains_true = ci_lower <= true_canopy_proportion <= ci_upper
        bootstrap_percentile_contains_true = bootstrap_stats['percentile_ci_lower'] <= true_canopy_proportion <= bootstrap_stats['percentile_ci_upper']
        bootstrap_bca_contains_true = bootstrap_stats['bca_ci_lower'] <= true_canopy_proportion <= bootstrap_stats['bca_ci_upper']
        
        # Combine all statistics
        combined_stats = {
            'n_sample_points': len(sampled_values),
            'n_canopy_hits': n_canopy_hits,
            'estimated_canopy_proportion': estimated_proportion,
            'true_canopy_proportion': true_canopy_proportion,
            'bias': bias,
            'absolute_error': absolute_error,
            'relative_error': absolute_error / true_canopy_proportion if true_canopy_proportion > 0 else 0,
            'standard_error': std_error,
            # Standard confidence intervals
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_width': ci_width,
            'ci_contains_true': ci_contains_true,
            # Bootstrap statistics
            'bootstrap_mean': bootstrap_stats['bootstrap_mean'],
            'bootstrap_std': bootstrap_stats['bootstrap_std'],
            'bootstrap_samples': bootstrap_stats['bootstrap_samples'],
            # Bootstrap percentile confidence intervals
            'bootstrap_percentile_ci_lower': bootstrap_stats['percentile_ci_lower'],
            'bootstrap_percentile_ci_upper': bootstrap_stats['percentile_ci_upper'],
            'bootstrap_percentile_ci_width': bootstrap_stats['percentile_ci_width'],
            'bootstrap_percentile_contains_true': bootstrap_percentile_contains_true,
            # Bootstrap BCa confidence intervals  
            'bootstrap_bca_ci_lower': bootstrap_stats['bca_ci_lower'],
            'bootstrap_bca_ci_upper': bootstrap_stats['bca_ci_upper'],
            'bootstrap_bca_ci_width': bootstrap_stats['bca_ci_width'],
            'bootstrap_bca_contains_true': bootstrap_bca_contains_true,
            # Bootstrap correction parameters
            'bias_correction': bootstrap_stats['bias_correction'],
            'acceleration': bootstrap_stats['acceleration']
        }
        
        return combined_stats
        
    def calculate_dimensions(self, aoi_m2):
        """
        Calculate grid dimensions for given area of interest
        
        Parameters:
        -----------
        aoi_m2 : float
            Area of interest in square meters
            
        Returns:
        --------
        tuple : (nrows, ncols)
        """
        # Assume square landscape for simplicity
        side_length = np.sqrt(aoi_m2)
        cells_per_side = int(side_length / self.cell_size)
        return cells_per_side, cells_per_side
    
    def calculate_morans_i(self, binary_landscape):
        """
        Calculate Moran's I for a binary landscape
        
        Parameters:
        -----------
        binary_landscape : np.array
            2D binary array (0s and 1s)
            
        Returns:
        --------
        float : Moran's I value
        """
        nrows, ncols = binary_landscape.shape
        
        # Create spatial weights matrix (Queen's case - 8 neighbors)
        w = weights.lat2W(nrows, ncols, rook=False)
        
        # Calculate Moran's I
        moran = esda.Moran(binary_landscape.flatten(), w)
        return moran.I
    
    def select_optimal_algorithm(self, target_morans_i):
        """
        Select the best algorithm based on target Moran's I value
        
        Parameters:
        -----------
        target_morans_i : float
            Target Moran's I value
            
        Returns:
        --------
        str : Optimal algorithm name
        """
        if target_morans_i < 0.2:
            return 'random'  # For low/negative autocorrelation
        elif target_morans_i > 0.8:
            return 'randomClusterNN'  # For high clustering
        else:
            return 'mpd'  # For moderate autocorrelation
    
    def generate_landscape_with_target_morans(self, nrows, ncols, canopy_percent, 
                                            target_morans_i, algorithm=None, 
                                            tolerance=0.05, max_iterations=50):
        """
        Generate landscape with target Moran's I value using optimal algorithm selection
        
        Parameters:
        -----------
        nrows, ncols : int
            Landscape dimensions
        canopy_percent : float
            Proportion of landscape that should be canopy (0-1)
        target_morans_i : float
            Target Moran's I value (-1 to 1)
        algorithm : str, optional
            Algorithm to use ('mpd', 'randomClusterNN', 'random'). If None, auto-selects optimal algorithm
        tolerance : float
            Acceptable difference from target Moran's I
        max_iterations : int
            Maximum optimization iterations
            
        Returns:
        --------
        dict : Results including landscape, achieved Moran's I, and parameters
        """
        
        # Auto-select algorithm if not specified
        if algorithm is None:
            algorithm = self.select_optimal_algorithm(target_morans_i)
            print(f"  Auto-selected algorithm: {algorithm} for target Moran's I = {target_morans_i:.2f}")
        
        def objective_function(param):
            """Objective function to minimize difference from target Moran's I"""
            try:
                if algorithm == 'mpd':
                    # Midpoint displacement - param is h (roughness)
                    landscape = nlmpy.mpd(nrows, ncols, h=param)
                elif algorithm == 'randomClusterNN':
                    # Random cluster NN - param is p (proportion of cluster seeds)
                    landscape = nlmpy.randomClusterNN(nrows, ncols, p=param, n=4)
                elif algorithm == 'random':
                    # Pure random landscape - param controls smoothing level
                    landscape = nlmpy.random(nrows, ncols)
                    if param > 0.1:  # Apply minimal smoothing if needed
                        from scipy import ndimage
                        landscape = ndimage.gaussian_filter(landscape, sigma=param)
                else:
                    raise ValueError(f"Unknown algorithm: {algorithm}")
                
                # Convert to binary based on canopy percentage
                threshold = np.percentile(landscape, (1-canopy_percent)*100)
                binary_landscape = (landscape > threshold).astype(int)
                
                # Calculate Moran's I
                achieved_morans_i = self.calculate_morans_i(binary_landscape)
                
                # Return absolute difference from target
                return abs(achieved_morans_i - target_morans_i)
                
            except Exception as e:
                # Return large penalty if generation fails
                return 999.0
        
        # Set algorithm-specific parameter bounds and optimization strategy
        if algorithm == 'random':
            # For random landscapes, try direct generation first
            try:
                landscape = nlmpy.random(nrows, ncols)
                threshold = np.percentile(landscape, (1-canopy_percent)*100)
                binary_landscape = (landscape > threshold).astype(int)
                achieved_morans_i = self.calculate_morans_i(binary_landscape)
                
                # If close enough, use as-is
                if abs(achieved_morans_i - target_morans_i) <= tolerance:
                    return {
                        'landscape': binary_landscape,
                        'continuous_landscape': landscape,
                        'target_morans_i': target_morans_i,
                        'achieved_morans_i': achieved_morans_i,
                        'difference': abs(achieved_morans_i - target_morans_i),
                        'optimal_parameter': 0.0,
                        'algorithm': algorithm,
                        'success': True
                    }
                
                # If not close enough, try optimization with smoothing
                bounds = (0.0, 2.0)  # Smoothing parameter
                
            except Exception:
                bounds = (0.0, 2.0)
        elif algorithm == 'mpd':
            # Adjust bounds based on target Moran's I
            if target_morans_i < 0.3:
                bounds = (0.7, 0.99)  # High roughness for low autocorrelation
            elif target_morans_i > 0.7:
                bounds = (0.01, 0.3)  # Low roughness for high autocorrelation
            else:
                bounds = (0.01, 0.99)  # Full range for moderate autocorrelation
        else:  # randomClusterNN
            # Adjust bounds based on target Moran's I
            if target_morans_i > 0.7:
                bounds = (0.3, 0.8)  # High clustering
            else:
                bounds = (0.01, 0.5)  # Moderate clustering
        
        # Optimize parameter to achieve target Moran's I
        try:
            result = minimize_scalar(objective_function, bounds=bounds, 
                                   method='bounded', options={'maxiter': max_iterations})
            
            optimal_param = result.x
            
            # Generate final landscape with optimal parameter
            if algorithm == 'mpd':
                final_continuous = nlmpy.mpd(nrows, ncols, h=optimal_param)
            elif algorithm == 'randomClusterNN':
                final_continuous = nlmpy.randomClusterNN(nrows, ncols, p=optimal_param, n=4)
            elif algorithm == 'random':
                final_continuous = nlmpy.random(nrows, ncols)
                if optimal_param > 0.1:
                    from scipy import ndimage
                    final_continuous = ndimage.gaussian_filter(final_continuous, sigma=optimal_param)
            
            # Convert to binary
            threshold = np.percentile(final_continuous, (1-canopy_percent)*100)
            final_binary = (final_continuous > threshold).astype(int)
            
            # Calculate final Moran's I
            achieved_morans_i = self.calculate_morans_i(final_binary)
            
            return {
                'landscape': final_binary,
                'continuous_landscape': final_continuous,
                'target_morans_i': target_morans_i,
                'achieved_morans_i': achieved_morans_i,
                'difference': abs(achieved_morans_i - target_morans_i),
                'optimal_parameter': optimal_param,
                'algorithm': algorithm,
                'success': abs(achieved_morans_i - target_morans_i) <= tolerance
            }
            
        except Exception as e:
            print(f"Error in optimization: {e}")
            # Multi-algorithm fallback strategy
            fallback_algorithms = ['random', 'mpd', 'randomClusterNN']
            fallback_algorithms = [alg for alg in fallback_algorithms if alg != algorithm]  # Remove current algorithm
            
            for fallback_alg in fallback_algorithms:
                try:
                    print(f"  Trying fallback algorithm: {fallback_alg}")
                    fallback_result = self.generate_landscape_with_target_morans(
                        nrows, ncols, canopy_percent, target_morans_i, 
                        algorithm=fallback_alg, tolerance=tolerance*2, max_iterations=max_iterations//2
                    )
                    fallback_result['algorithm'] = f"{algorithm}_fallback_{fallback_alg}"
                    return fallback_result
                except Exception:
                    continue
            
            # Final fallback to pure random if all else fails
            print("  Using final random fallback")
            fallback = nlmpy.random(nrows, ncols)
            threshold = np.percentile(fallback, (1-canopy_percent)*100)
            fallback_binary = (fallback > threshold).astype(int)
            fallback_morans = self.calculate_morans_i(fallback_binary)
            
            return {
                'landscape': fallback_binary,
                'continuous_landscape': fallback,
                'target_morans_i': target_morans_i,
                'achieved_morans_i': fallback_morans,
                'difference': abs(fallback_morans - target_morans_i),
                'optimal_parameter': None,
                'algorithm': 'final_random_fallback',
                'success': False
            }
    
    def run_parameter_sweep(self, aoi_values, canopy_extents, morans_i_values, 
                          algorithm=None, n_replicates=1, save_landscapes=False):
        """
        Run complete parameter sweep across all combinations
        
        Parameters:
        -----------
        aoi_values : list
            Areas of interest in square meters [200, 600, 4000]
        canopy_extents : list
            Canopy coverage proportions [0.2, 0.4, 0.6, 0.8]
        morans_i_values : list
            Target Moran's I values [-0.5, 0, 0.5]
        algorithm : str, optional
            Algorithm to use ('mpd', 'randomClusterNN', 'random'). If None, auto-selects optimal algorithm for each target
        n_replicates : int
            Number of replicates per parameter combination
        save_landscapes : bool
            Whether to store actual landscape arrays (memory intensive)
            
        Returns:
        --------
        pd.DataFrame : Results summary
        """
        
        
        if algorithm is None:
            print(f"Starting parameter sweep with AUTO-SELECTED algorithms...")
            print(f"Algorithm selection strategy:")
            print(f"  Moran's I < 0.2: 'random' algorithm")
            print(f"  Moran's I 0.2-0.8: 'mpd' algorithm") 
            print(f"  Moran's I > 0.8: 'randomClusterNN' algorithm")
        else:
            print(f"Starting parameter sweep with {algorithm} algorithm...")
        print(f"AOI values: {aoi_values}")
        print(f"Canopy extents: {canopy_extents}")
        print(f"Moran's I values: {morans_i_values}")
        print(f"Replicates per combination: {n_replicates}")
        
        total_combinations = len(aoi_values) * len(canopy_extents) * len(morans_i_values) * n_replicates
        print(f"Total landscapes to generate: {total_combinations}")
        
        self.results = []
        combination_count = 0
        
        for aoi, canopy_extent, target_morans_i in product(aoi_values, canopy_extents, morans_i_values):
            for replicate in range(n_replicates):
                combination_count += 1
                start_time = time.time()
                
                print(f"\nCombination {combination_count}/{total_combinations}")
                print(f"AOI: {aoi}m², Canopy: {canopy_extent:.1%}, Moran's I: {target_morans_i:.2f}, Rep: {replicate+1}")
                
                # Calculate landscape dimensions
                nrows, ncols = self.calculate_dimensions(aoi)
                
                # Generate landscape
                result = self.generate_landscape_with_target_morans(
                    nrows, ncols, canopy_extent, target_morans_i, algorithm
                )
                
                # Generate sample points for this AOI (cached for consistency)
                x_coords, y_coords = self.generate_sample_points(aoi)
                
                # Sample the landscape
                sampled_values = self.sample_landscape(result['landscape'], x_coords, y_coords)
                
                # Calculate sampling statistics
                true_canopy_proportion = np.mean(result['landscape'])
                sampling_stats = self.calculate_sampling_statistics(sampled_values, true_canopy_proportion)
                
                # Store results
                result_record = {
                    'combination_id': combination_count,
                    'replicate': replicate + 1,
                    'aoi_m2': aoi,
                    'nrows': nrows,
                    'ncols': ncols,
                    'actual_area_m2': nrows * ncols * self.cell_size**2,
                    'canopy_extent_target': canopy_extent,
                    'canopy_extent_actual': np.mean(result['landscape']),
                    'morans_i_target': target_morans_i,
                    'morans_i_achieved': result['achieved_morans_i'],
                    'morans_i_difference': result['difference'],
                    'optimal_parameter': result['optimal_parameter'],
                    'algorithm': result['algorithm'],
                    'success': result['success'],
                    'generation_time_seconds': time.time() - start_time,
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
                success_rate = np.mean([r['success'] for r in self.results])
                avg_time = np.mean([r['generation_time_seconds'] for r in self.results])
                avg_bias = np.mean([abs(r['sampling_bias']) for r in self.results])
                print(f"Success: {'✓' if result['success'] else '✗'} | "
                      f"Success rate: {success_rate:.1%} | "
                      f"Avg time: {avg_time:.2f}s | "
                      f"Sampling bias: {sampling_stats['bias']:+.4f}")
        
        return pd.DataFrame(self.results)
    
    def plot_results_summary(self, results_df):
        """
        Create comprehensive summary plots including bootstrap analysis
        
        Parameters:
        -----------
        results_df : pd.DataFrame
            Results from run_parameter_sweep
        """
        fig, axes = plt.subplots(4, 2, figsize=(15, 24))
        fig.suptitle('Neutral Landscape Generation & Sampling Results Summary', fontsize=16)
        
        # 1. Success rate by parameters
        success_by_params = results_df.groupby(['aoi_m2', 'canopy_extent_target', 'morans_i_target'])['success'].mean().reset_index()
        
        ax1 = axes[0, 0]
        scatter = ax1.scatter(success_by_params['morans_i_target'], 
                             success_by_params['canopy_extent_target'], 
                             s=success_by_params['aoi_m2']/20, 
                             c=success_by_params['success'], 
                             cmap='RdYlGn', vmin=0, vmax=1, alpha=0.7)
        ax1.set_xlabel("Target Moran's I")
        ax1.set_ylabel('Canopy Extent')
        ax1.set_title('Success Rate by Parameters\n(Bubble size = AOI)')
        plt.colorbar(scatter, ax=ax1, label='Success Rate')
        
        # 2. Moran's I accuracy
        ax2 = axes[0, 1]
        ax2.scatter(results_df['morans_i_target'], results_df['morans_i_achieved'], 
                   alpha=0.6, c=results_df['aoi_m2'], cmap='viridis')
        ax2.plot([-1, 1], [-1, 1], 'r--', alpha=0.8, label='Perfect match')
        ax2.set_xlabel("Target Moran's I")
        ax2.set_ylabel("Achieved Moran's I")
        ax2.set_title("Moran's I Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Sampling bias vs Moran's I
        ax3 = axes[1, 0]
        scatter3 = ax3.scatter(results_df['morans_i_achieved'], results_df['sampling_bias'], 
                              c=results_df['canopy_extent_actual'], cmap='Greens', alpha=0.7)
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.8, label='No bias')
        ax3.set_xlabel("Achieved Moran's I")
        ax3.set_ylabel('Sampling Bias')
        ax3.set_title('Sampling Bias vs Spatial Autocorrelation')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter3, ax=ax3, label='Canopy Extent')
        
        # 4. Bootstrap vs Standard CI comparison
        ax4 = axes[1, 1]
        # Compare CI coverage rates
        standard_coverage = results_df['ci_contains_true'].mean()
        bootstrap_percentile_coverage = results_df['bootstrap_percentile_contains_true'].mean()
        bootstrap_bca_coverage = results_df['bootstrap_bca_contains_true'].mean()
        
        coverage_methods = ['Standard\n(Normal)', 'Bootstrap\n(Percentile)', 'Bootstrap\n(BCa)']
        coverage_rates = [standard_coverage, bootstrap_percentile_coverage, bootstrap_bca_coverage]
        colors = ['lightblue', 'orange', 'green']
        
        bars = ax4.bar(coverage_methods, coverage_rates, color=colors, alpha=0.7)
        ax4.axhline(y=0.95, color='red', linestyle='--', alpha=0.8, label='Expected 95%')
        ax4.set_ylabel('Coverage Rate')
        ax4.set_title('Confidence Interval Coverage Comparison')
        ax4.legend()
        ax4.set_ylim(0.8, 1.0)
        
        # Add percentage labels on bars
        for bar, rate in zip(bars, coverage_rates):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{rate:.1%}', ha='center', va='bottom')
        
        # 5. Bootstrap confidence interval widths
        ax5 = axes[2, 0]
        ci_widths_data = pd.DataFrame({
            'Standard CI': results_df['ci_width'],
            'Bootstrap Percentile': results_df['bootstrap_percentile_ci_width'],
            'Bootstrap BCa': results_df['bootstrap_bca_ci_width']
        })
        ci_widths_data.boxplot(ax=ax5)
        ax5.set_ylabel('Confidence Interval Width')
        ax5.set_title('CI Width Comparison Across Methods')
        ax5.grid(True, alpha=0.3)
        plt.suptitle('')  # Remove automatic boxplot title
        
        # 6. Bootstrap standard error vs analytical standard error
        ax6 = axes[2, 1]
        ax6.scatter(results_df['standard_error'], results_df['bootstrap_std'], 
                   alpha=0.7, c=results_df['canopy_extent_actual'], cmap='viridis')
        max_se = max(results_df['standard_error'].max(), results_df['bootstrap_std'].max())
        ax6.plot([0, max_se], [0, max_se], 'r--', alpha=0.8, label='Perfect agreement')
        ax6.set_xlabel('Analytical Standard Error')
        ax6.set_ylabel('Bootstrap Standard Error')
        ax6.set_title('Standard Error: Analytical vs Bootstrap')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Bias correction and acceleration parameters
        ax7 = axes[3, 0]
        scatter7 = ax7.scatter(results_df['bias_correction'], results_df['acceleration'], 
                              c=results_df['morans_i_achieved'], cmap='coolwarm', alpha=0.7)
        ax7.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax7.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax7.set_xlabel('Bias Correction')
        ax7.set_ylabel('Acceleration')
        ax7.set_title('Bootstrap BCa Parameters')
        plt.colorbar(scatter7, ax=ax7, label="Moran's I")
        ax7.grid(True, alpha=0.3)
        
        # 8. Coverage by landscape characteristics
        ax8 = axes[3, 1]
        coverage_by_morans = results_df.groupby('morans_i_target').agg({
            'ci_contains_true': 'mean',
            'bootstrap_percentile_contains_true': 'mean',
            'bootstrap_bca_contains_true': 'mean'
