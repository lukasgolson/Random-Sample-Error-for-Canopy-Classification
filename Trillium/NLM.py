import time
import warnings
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pysal.explore import esda
from pysal.lib import weights
from scipy.optimize import minimize_scalar
from scipy import ndimage
from nlmpy import nlmpy
#from multiprocessing import Pool
from tqdm import tqdm

warnings.filterwarnings('ignore')

# GLOBAL PARAMETERS
CELL_SIZE = 0.5 # 50 cm
N_SAMPLE_POINTS = 100000     # number of sample points
RANDOM_SEED = 42
TOLERANCE = 0.025 # acceptable Moran's I difference
MAX_ITERATIONS = 75 # max optimization iterations
SAVE_LANDSCAPES = False # whether to save full landscapes
USE_PARALLEL = False

# Parameter sweeps
AOI_VALUES = [200, 600] # AOI in m² - 200, 600, 4000
CANOPY_EXTENTS = np.round(np.arange(0, 1.01, 0.05), 2).tolist()  # canopy cover sweep
MORANS_I_VALUES = np.round(np.arange(-.5, 1.01, 0.05), 2).tolist()  # Moran's I sweep
N_REPLICATES = 200 # replicates per combo
ALGORITHM = None # None = auto-select

class NeutralLandscapeGenerator:
    """
    Generate neutral landscapes with specified AOI, canopy extent, and Moran's I values
    Apply consistent random sampling across all landscapes for comparison studies
    Uses three different algorithms based on target Moran's I values
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
        elif target_morans_i > 0.6:
            return 'randomClusterNN'  # For high clustering
        else:
            return 'mpd'  # For moderate autocorrelation

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
        col_indices = np.floor(x_coords / self.cell_size).astype(int)
        row_indices = np.floor(y_coords / self.cell_size).astype(int)

        # Ensure indices are within bounds
        col_indices = np.clip(col_indices, 0, ncols - 1)
        row_indices = np.clip(row_indices, 0, nrows - 1)

        # Sample landscape values
        sampled_values = landscape[row_indices, col_indices]

        return sampled_values

    def calculate_sampling_statistics(self, sampled_values, true_canopy_proportion):
        """
        Calculate sampling statistics

        Parameters:
        -----------
        sampled_values : np.array
            Binary array of sample results
        true_canopy_proportion : float
            True proportion of canopy in landscape

        Returns:
        --------
        dict : Sampling statistics
        """
        n_canopy_hits = np.sum(sampled_values)
        estimated_proportion = n_canopy_hits / len(sampled_values)
        bias = estimated_proportion - true_canopy_proportion
        absolute_error = abs(bias)

        # Calculate confidence interval (95%)
        std_error = np.sqrt(estimated_proportion * (1 - estimated_proportion) / len(sampled_values))
        ci_lower = estimated_proportion - 1.96 * std_error
        ci_upper = estimated_proportion + 1.96 * std_error
        ci_width = ci_upper - ci_lower

        return {
            'n_sample_points': len(sampled_values),
            'n_canopy_hits': n_canopy_hits,
            'estimated_canopy_proportion': estimated_proportion,
            'true_canopy_proportion': true_canopy_proportion,
            'bias': bias,
            'absolute_error': absolute_error,
            'relative_error': absolute_error / true_canopy_proportion if true_canopy_proportion > 0 else 0,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_width': ci_width,
            'ci_contains_true': ci_lower <= true_canopy_proportion <= ci_upper
        }

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

    def generate_landscape_with_target_morans(self, nrows, ncols, canopy_percent,
                                              target_morans_i, algorithm=None,
                                              tolerance=0.025, max_iterations=75):
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
            Algorithm to use ('mpd', 'randomClusterNN', 'random'). If None, auto-selects
        tolerance : float
            Acceptable difference from target Moran's I (default=0.025)
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
                        landscape = ndimage.gaussian_filter(landscape, sigma=param)
                else:
                    raise ValueError(f"Unknown algorithm: {algorithm}")

                # Convert to binary based on canopy percentage
                threshold = np.percentile(landscape, (1 - canopy_percent) * 100)
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
                threshold = np.percentile(landscape, (1 - canopy_percent) * 100)
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
                    final_continuous = ndimage.gaussian_filter(final_continuous, sigma=optimal_param)

            # Convert to binary
            threshold = np.percentile(final_continuous, (1 - canopy_percent) * 100)
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
            fallback_algorithms = [alg for alg in fallback_algorithms if alg != algorithm]

            for fallback_alg in fallback_algorithms:
                try:
                    print(f"  Trying fallback algorithm: {fallback_alg}")
                    fallback_result = self.generate_landscape_with_target_morans(
                        nrows, ncols, canopy_percent, target_morans_i,
                        algorithm=fallback_alg, tolerance=tolerance * 2, max_iterations=max_iterations // 2
                    )
                    fallback_result['algorithm'] = f"{algorithm}_fallback_{fallback_alg}"
                    return fallback_result
                except Exception:
                    continue

            # Final fallback to pure random
            print("  Using final random fallback")
            fallback = nlmpy.random(nrows, ncols)
            threshold = np.percentile(fallback, (1 - canopy_percent) * 100)
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

    def run_parameter_sweep(
            self,
            aoi_values=None,
            canopy_extents=None,
            morans_i_values=None,
            algorithm=None,
            n_replicates=None,
            save_landscapes=False,
            tolerance=None,
            max_iterations=None
    ):
        """
        Run complete parameter sweep across all combinations using globals if arguments are None.
        """

        # Use global defaults if parameters are None
        aoi_values = aoi_values if aoi_values is not None else AOI_VALUES
        canopy_extents = canopy_extents if canopy_extents is not None else CANOPY_EXTENTS
        morans_i_values = morans_i_values if morans_i_values is not None else MORANS_I_VALUES
        n_replicates = n_replicates if n_replicates is not None else N_REPLICATES
        tolerance = tolerance if tolerance is not None else TOLERANCE
        max_iterations = max_iterations if max_iterations is not None else MAX_ITERATIONS

        if algorithm is None:
            print("Starting parameter sweep with AUTO-SELECTED algorithms...")
            print("Algorithm selection strategy:")
            print("  Moran's I < 0.2: 'random' algorithm")
            print("  Moran's I 0.2-0.8: 'mpd' algorithm")
            print("  Moran's I > 0.8: 'randomClusterNN' algorithm")
        else:
            print(f"Starting parameter sweep with {algorithm} algorithm...")

        total_combinations = len(aoi_values) * len(canopy_extents) * len(morans_i_values) * n_replicates
        print(f"Total landscapes to generate: {total_combinations}")

        self.results = []
        combination_count = 0

        for aoi, canopy_extent, target_morans_i in product(aoi_values, canopy_extents, morans_i_values):
            for replicate in range(n_replicates):
                combination_count += 1
                start_time = time.time()

                nrows, ncols = self.calculate_dimensions(aoi)

                # Pass tolerance and max_iterations to landscape generator
                result = self.generate_landscape_with_target_morans(
                    nrows, ncols, canopy_extent, target_morans_i,
                    algorithm=algorithm,
                    tolerance=tolerance,
                    max_iterations=max_iterations
                )

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
                    'actual_area_m2': nrows * ncols * self.cell_size ** 2,
                    'canopy_extent_target': canopy_extent,
                    'canopy_extent_actual': np.mean(result['landscape']),
                    'morans_i_target': target_morans_i,
                    'morans_i_achieved': result['achieved_morans_i'],
                    'morans_i_difference': result['difference'],
                    'optimal_parameter': result['optimal_parameter'],
                    'algorithm': result['algorithm'],
                    'success': result['success'],
                    'generation_time_seconds': time.time() - start_time,
                    'n_sample_points': sampling_stats['n_sample_points'],
                    'n_canopy_hits': sampling_stats['n_canopy_hits'],
                    'estimated_canopy_proportion': sampling_stats['estimated_canopy_proportion'],
                    'sampling_bias': sampling_stats['bias'],
                    'sampling_absolute_error': sampling_stats['absolute_error'],
                    'sampling_relative_error': sampling_stats['relative_error'],
                    'ci_lower': sampling_stats['ci_lower'],
                    'ci_upper': sampling_stats['ci_upper'],
                    'ci_width': sampling_stats['ci_width'],
                    'ci_contains_true': sampling_stats['ci_contains_true']
                }

                if save_landscapes:
                    result_record['landscape_array'] = result['landscape']
                    result_record['continuous_array'] = result['continuous_landscape']
                    result_record['sample_points_x'] = x_coords
                    result_record['sample_points_y'] = y_coords
                    result_record['sample_values'] = sampled_values

                self.results.append(result_record)

        return pd.DataFrame(self.results)

    def plot_results_summary(self, results_df):
        """
        Create summary plots of the parameter sweep results including algorithm comparison

        Parameters:
        -----------
        results_df : pd.DataFrame
            Results from run_parameter_sweep
        """
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        fig.suptitle('Multi-Algorithm Neutral Landscape Generation Results', fontsize=16)

        # 1. Success rate by parameters and algorithm
        success_by_params = results_df.groupby(['aoi_m2', 'canopy_extent_target', 'morans_i_target'])[
            'success'].mean().reset_index()

        ax1 = axes[0, 0]
        scatter = ax1.scatter(success_by_params['morans_i_target'],
                              success_by_params['canopy_extent_target'],
                              s=success_by_params['aoi_m2'] / 20,
                              c=success_by_params['success'],
                              cmap='RdYlGn', vmin=0, vmax=1, alpha=0.7)
        ax1.set_xlabel("Target Moran's I")
        ax1.set_ylabel('Canopy Extent')
        ax1.set_title('Success Rate by Parameters\n(Bubble size = AOI)')
        plt.colorbar(scatter, ax=ax1, label='Success Rate')

        # 2. Algorithm selection visualization
        ax2 = axes[0, 1]
        # Extract primary algorithm (before any fallback suffixes)
        results_df['primary_algorithm'] = results_df['algorithm'].str.split('_').str[0]
        algorithm_counts = results_df.groupby(['morans_i_target', 'primary_algorithm']).size().unstack(fill_value=0)

        algorithm_counts.plot(kind='bar', stacked=True, ax=ax2,
                              color=['lightcoral', 'lightblue', 'lightgreen'])
        ax2.set_xlabel("Target Moran's I")
        ax2.set_ylabel('Number of Landscapes')
        ax2.set_title('Algorithm Selection by Target Moran\'s I')
        ax2.legend(title='Algorithm')
        ax2.tick_params(axis='x', rotation=45)

        # 3. Moran's I accuracy by algorithm
        ax3 = axes[1, 0]
        colors = {'random': 'red', 'mpd': 'blue', 'randomClusterNN': 'green'}
        for alg in results_df['primary_algorithm'].unique():
            alg_data = results_df[results_df['primary_algorithm'] == alg]
            ax3.scatter(alg_data['morans_i_target'], alg_data['morans_i_achieved'],
                        alpha=0.6, label=alg, color=colors.get(alg, 'gray'))
        ax3.plot([-1, 1], [-1, 1], 'k--', alpha=0.8, label='Perfect match')
        ax3.set_xlabel("Target Moran's I")
        ax3.set_ylabel("Achieved Moran's I")
        ax3.set_title("Moran's I Accuracy by Algorithm")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Sampling bias vs Moran's I
        ax4 = axes[1, 1]
        scatter4 = ax4.scatter(results_df['morans_i_achieved'], results_df['sampling_bias'],
                               c=results_df['canopy_extent_actual'], cmap='Greens', alpha=0.7)
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.8, label='No bias')
        ax4.set_xlabel("Achieved Moran's I")
        ax4.set_ylabel('Sampling Bias')
        ax4.set_title('Sampling Bias vs Spatial Autocorrelation')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter4, ax=ax4, label='Canopy Extent')

        # 5. Generation time by algorithm and AOI
        ax5 = axes[2, 0]
        time_by_alg = results_df.groupby(['primary_algorithm', 'aoi_m2'])['generation_time_seconds'].mean().unstack()
        time_by_alg.plot(kind='bar', ax=ax5)
        ax5.set_xlabel('Algorithm')
        ax5.set_ylabel('Average Generation Time (seconds)')
        ax5.set_title('Generation Time by Algorithm and AOI')
        ax5.legend(title='AOI (m²)')
        ax5.tick_params(axis='x', rotation=45)

        # 6. Success rate by algorithm
        ax6 = axes[2, 1]
        success_by_alg = results_df.groupby('primary_algorithm')['success'].mean()
        bars = ax6.bar(success_by_alg.index, success_by_alg.values,
                       color=[colors.get(alg, 'gray') for alg in success_by_alg.index])
        ax6.set_xlabel('Algorithm')
        ax6.set_ylabel('Success Rate')
        ax6.set_title('Success Rate by Algorithm')
        ax6.set_ylim(0, 1)

        # Add percentage labels on bars
        for bar, rate in zip(bars, success_by_alg.values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{rate:.1%}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig("landscape.png", dpi=300, bbox_inches='tight')
        plt.close()  # free memory

        # Print comprehensive summary statistics
        print("\n" + "=" * 80)
        print("MULTI-ALGORITHM PARAMETER SWEEP SUMMARY")
        print("=" * 80)
        print(f"Total landscapes generated: {len(results_df)}")
        print(f"Overall success rate: {results_df['success'].mean():.1%}")
        print(f"Average Moran's I error: {results_df['morans_i_difference'].mean():.4f}")
        print(f"Average generation time: {results_df['generation_time_seconds'].mean():.2f} seconds")

        print(f"\nALGORITHM PERFORMANCE:")
        for alg in sorted(results_df['primary_algorithm'].unique()):
            alg_data = results_df[results_df['primary_algorithm'] == alg]
            success_rate = alg_data['success'].mean()
            avg_error = alg_data['morans_i_difference'].mean()
            count = len(alg_data)
            print(f"  {alg}: {success_rate:.1%} success, {avg_error:.4f} avg error, {count} landscapes")

        print(f"\nSAMPLING PERFORMANCE:")
        print(f"Average sampling bias: {results_df['sampling_bias'].mean():+.6f}")
        print(f"Average absolute error: {results_df['sampling_absolute_error'].mean():.6f}")
        print(f"Average relative error: {results_df['sampling_relative_error'].mean():.4%}")
        print(f"CI coverage rate: {results_df['ci_contains_true'].mean():.1%}")

    def export_sample_points_csv(self, results_df, filename='sample_points_results.csv'):
        """
        Export sample point results in the specified format

        Parameters:
        -----------
        results_df : pd.DataFrame
            Results from run_parameter_sweep (must have save_landscapes=True)
        filename : str
            Output CSV filename

        Returns:
        --------
        pd.DataFrame : Sample points results in wide format
        """

        if 'sample_points_x' not in results_df.columns:
            raise ValueError("Sample points data not found. Run parameter sweep with save_landscapes=True")

        print(f"Exporting sample points data to {filename}...")

        # Get sample points from first result (they're the same for each AOI)
        sample_points_data = []

        # Process each AOI separately since sample points are consistent within AOI
        for aoi in results_df['aoi_m2'].unique():
            aoi_data = results_df[results_df['aoi_m2'] == aoi].iloc[0]
            x_coords = aoi_data['sample_points_x']
            y_coords = aoi_data['sample_points_y']

            # Create base dataframe for this AOI
            aoi_sample_df = pd.DataFrame({
                'Sample_Point_ID': range(1, len(x_coords) + 1),
                'Sample_Point_X_Location': x_coords,
                'Sample_Point_Y_Location': y_coords
            })

            # Add results for each landscape as columns
            aoi_results = results_df[results_df['aoi_m2'] == aoi]

            for idx, row in aoi_results.iterrows():
                # Create column name: {AOI}_{Target_Extent}_{Target_Moran}_{True_Extent}_{True_Moran}
                col_name = f"{int(row['aoi_m2'])}_{row['canopy_extent_target']:.2f}_{row['morans_i_target']:.2f}_{row['canopy_extent_actual']:.3f}_{row['morans_i_achieved']:.3f}"

                # Add sample values for this landscape
                aoi_sample_df[col_name] = row['sample_values']

            sample_points_data.append(aoi_sample_df)

        # Handle multiple AOIs
        if len(sample_points_data) == 1:
            final_df = sample_points_data[0]
        else:
            # Create separate files for each AOI
            for i, (aoi, aoi_df) in enumerate(zip(sorted(results_df['aoi_m2'].unique()), sample_points_data)):
                aoi_filename = filename.replace('.csv', f'_AOI_{int(aoi)}m2.csv')
                aoi_df.to_csv(aoi_filename, index=False)
                print(f"  Exported AOI {int(aoi)} m² data to {aoi_filename}")

            # Use first AOI as base for main file
            final_df = sample_points_data[0]

        # Export main file
        final_df.to_csv(filename, index=False)
        print(f"Sample points data exported successfully!")
        print(f"  File: {filename}")
        print(f"  Sample points: {len(final_df):,}")
        print(
            f"  Landscape columns: {len([col for col in final_df.columns if col not in ['Sample_Point_ID', 'Sample_Point_X_Location', 'Sample_Point_Y_Location']])}")

        return final_df

    def create_sample_points_summary(self, results_df):
        """
        Create a summary of landscape characteristics for reference

        Parameters:
        -----------
        results_df : pd.DataFrame
            Results from run_parameter_sweep

        Returns:
        --------
        pd.DataFrame : Summary of landscape characteristics
        """

        summary_data = []

        for idx, row in results_df.iterrows():
            # Create column name matching the sample points export
            col_name = f"{int(row['aoi_m2'])}_{row['canopy_extent_target']:.2f}_{row['morans_i_target']:.2f}_{row['canopy_extent_actual']:.3f}_{row['morans_i_achieved']:.3f}"

            summary_data.append({
                'Column_Name': col_name,
                'AOI_m2': int(row['aoi_m2']),
                'Landscape_Dimensions': f"{row['nrows']}x{row['ncols']}",
                'Target_Canopy_Extent': row['canopy_extent_target'],
                'Actual_Canopy_Extent': row['canopy_extent_actual'],
                'Target_Morans_I': row['morans_i_target'],
                'Actual_Morans_I': row['morans_i_achieved'],
                'Morans_I_Error': row['morans_i_difference'],
                'Algorithm_Used': row['algorithm'],
                'Optimal_Parameter': row['optimal_parameter'],
                'Success': row['success'],
                'Replicate': row['replicate'],
                'Combination_ID': row['combination_id'],
                'Sample_Points_Used': row['n_sample_points'],
                'Canopy_Hits': row['n_canopy_hits'],
                'Estimated_Proportion': row['estimated_canopy_proportion'],
                'Sampling_Bias': row['sampling_bias'],
                'Generation_Time_Sec': row['generation_time_seconds']
            })

        summary_df = pd.DataFrame(summary_data)
        return summary_df

if __name__ == "__main__":
    from itertools import product
    from multiprocessing import Pool

    # Initialize generator with global parameters
    generator = NeutralLandscapeGenerator(
        cell_size=CELL_SIZE,
        n_sample_points=N_SAMPLE_POINTS,
        random_seed=RANDOM_SEED
    )

    # Function to generate a single combination + replicate
    def generate_combination(args):
        aoi, canopy_extent, target_morans_i, replicate, gen = args
        start_time = time.time()
        nrows, ncols = gen.calculate_dimensions(aoi)

        result = gen.generate_landscape_with_target_morans(
            nrows, ncols, canopy_extent, target_morans_i,
            algorithm=ALGORITHM,
            tolerance=TOLERANCE,
            max_iterations=MAX_ITERATIONS
        )

        x_coords, y_coords = gen.generate_sample_points(aoi)
        sampled_values = gen.sample_landscape(result['landscape'], x_coords, y_coords)
        true_canopy_proportion = np.mean(result['landscape'])
        sampling_stats = gen.calculate_sampling_statistics(sampled_values, true_canopy_proportion)

        record = {
            'replicate': replicate + 1,
            'aoi_m2': aoi,
            'nrows': nrows,
            'ncols': ncols,
            'actual_area_m2': nrows * ncols * gen.cell_size ** 2,
            'canopy_extent_target': canopy_extent,
            'canopy_extent_actual': np.mean(result['landscape']),
            'morans_i_target': target_morans_i,
            'morans_i_achieved': result['achieved_morans_i'],
            'morans_i_difference': result['difference'],
            'optimal_parameter': result['optimal_parameter'],
            'algorithm': result['algorithm'],
            'success': result['success'],
            'generation_time_seconds': time.time() - start_time,
            'n_sample_points': sampling_stats['n_sample_points'],
            'n_canopy_hits': sampling_stats['n_canopy_hits'],
            'estimated_canopy_proportion': sampling_stats['estimated_canopy_proportion'],
            'sampling_bias': sampling_stats['bias'],
            'sampling_absolute_error': sampling_stats['absolute_error'],
            'sampling_relative_error': sampling_stats['relative_error'],
            'ci_lower': sampling_stats['ci_lower'],
            'ci_upper': sampling_stats['ci_upper'],
            'ci_width': sampling_stats['ci_width'],
            'ci_contains_true': sampling_stats['ci_contains_true']
        }

        if SAVE_LANDSCAPES:
            record['landscape_array'] = result['landscape']
            record['continuous_array'] = result['continuous_landscape']
            record['sample_points_x'] = x_coords
            record['sample_points_y'] = y_coords
            record['sample_values'] = sampled_values

        return record

    # Prepare all combinations
    args_list = [
        (aoi, canopy_extent, target_morans_i, rep, generator)
        for aoi, canopy_extent, target_morans_i in product(AOI_VALUES, CANOPY_EXTENTS, MORANS_I_VALUES)
        for rep in range(N_REPLICATES)
    ]

    print(f"Running parallel parameter sweep with {len(args_list)} total tasks using 192 CPUs...")

    # Run in parallel
    results = []
    if USE_PARALLEL:
        from multiprocessing import cpu_count

        n_cpus = min(192, cpu_count())  # don’t exceed available cores
        print(f"Running parallel parameter sweep with {len(args_list)} tasks using {n_cpus} CPUs...")
        with Pool(processes=n_cpus) as pool:
            for r in tqdm(pool.imap(generate_combination, args_list), total=len(args_list),
                          desc="Generating landscapes"):
                results.append(r)
    else:
        print(f"Running serial parameter sweep with {len(args_list)} tasks...")
        for r in tqdm(map(generate_combination, args_list), total=len(args_list), desc="Generating landscapes"):
            results.append(r)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Keep only best replicate per combination
    best_results = (
        results_df
        .sort_values("morans_i_difference")
        .groupby(["aoi_m2", "canopy_extent_target", "morans_i_target"], group_keys=False)
        .apply(lambda g: g.head(1))
        .reset_index(drop=True)
    )

    # Export sample points results
    print("\nExporting sample points data...")
    sample_points_df = generator.export_sample_points_csv(
        best_results,
        filename='sample_points_canopy_analysis.csv'
    )

    # Create and export landscape summary
    print("\nCreating landscape summary...")
    summary_df = generator.create_sample_points_summary(best_results)
    summary_df.to_csv('landscape_characteristics_summary.csv', index=False)
    print("Landscape summary exported to 'landscape_characteristics_summary.csv'")

    # Generate summary plots
    print("\nGenerating summary plots...")
    generator.plot_results_summary(results_df)
