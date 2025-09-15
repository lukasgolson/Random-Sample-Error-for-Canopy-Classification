import numpy as np
import pandas as pd
import nlmpy
from pysal.lib import weights
from pysal.explore import esda
from scipy.optimize import minimize_scalar
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
                                            target_morans_i, algorithm='mpd', 
                                            tolerance=0.05, max_iterations=50):
        """
        Generate landscape with target Moran's I value
        
        Parameters:
        -----------
        nrows, ncols : int
            Landscape dimensions
        canopy_percent : float
            Proportion of landscape that should be canopy (0-1)
        target_morans_i : float
            Target Moran's I value (-1 to 1)
        algorithm : str
            'mpd' or 'randomClusterNN'
        tolerance : float
            Acceptable difference from target Moran's I
        max_iterations : int
            Maximum optimization iterations
            
        Returns:
        --------
        dict : Results including landscape, achieved Moran's I, and parameters
        """
        
        def objective_function(param):
            """Objective function to minimize difference from target Moran's I"""
            try:
                if algorithm == 'mpd':
                    # Midpoint displacement - param is h (roughness)
                    landscape = nlmpy.mpd(nrows, ncols, h=param)
                elif algorithm == 'randomClusterNN':
                    # Random cluster NN - param is p (proportion of cluster seeds)
                    landscape = nlmpy.randomClusterNN(nrows, ncols, p=param, n=4)
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
        
        # Optimize parameter to achieve target Moran's I
        try:
            if algorithm == 'mpd':
                # h parameter bounds for midpoint displacement
                bounds = (0.01, 0.99)
            else:
                # p parameter bounds for random cluster
                bounds = (0.01, 0.8)
                
            result = minimize_scalar(objective_function, bounds=bounds, 
                                   method='bounded', options={'maxiter': max_iterations})
            
            optimal_param = result.x
            
            # Generate final landscape with optimal parameter
            if algorithm == 'mpd':
                final_continuous = nlmpy.mpd(nrows, ncols, h=optimal_param)
            else:
                final_continuous = nlmpy.randomClusterNN(nrows, ncols, p=optimal_param, n=4)
            
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
            # Return fallback landscape if optimization fails
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
                'algorithm': 'random_fallback',
                'success': False
            }
    
    def run_parameter_sweep(self, aoi_values, canopy_extents, morans_i_values, 
                          algorithm='mpd', n_replicates=1, save_landscapes=False):
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
        algorithm : str
            Algorithm to use ('mpd' or 'randomClusterNN')
        n_replicates : int
            Number of replicates per parameter combination
        save_landscapes : bool
            Whether to store actual landscape arrays (memory intensive)
            
        Returns:
        --------
        pd.DataFrame : Results summary
        """
        
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
                    'ci_contains_true': sampling_stats['ci_contains_true']
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
        Create summary plots of the parameter sweep results including sampling analysis
        
        Parameters:
        -----------
        results_df : pd.DataFrame
            Results from run_parameter_sweep
        """
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
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
        
        # 4. Confidence interval coverage
        ax4 = axes[1, 1]
        ci_coverage = results_df.groupby(['aoi_m2', 'morans_i_target'])['ci_contains_true'].mean().reset_index()
        for aoi in sorted(ci_coverage['aoi_m2'].unique()):
            data = ci_coverage[ci_coverage['aoi_m2'] == aoi]
            ax4.plot(data['morans_i_target'], data['ci_contains_true'], 
                    marker='o', label=f'{aoi} m²', linewidth=2)
        ax4.axhline(y=0.95, color='red', linestyle='--', alpha=0.8, label='Expected 95%')
        ax4.set_xlabel("Moran's I")
        ax4.set_ylabel('CI Coverage Rate')
        ax4.set_title('95% Confidence Interval Coverage')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0.8, 1.0)
        
        # 5. Sampling error vs AOI
        ax5 = axes[2, 0]
        results_df.boxplot(column='sampling_absolute_error', by='aoi_m2', ax=ax5)
        ax5.set_xlabel('Area of Interest (m²)')
        ax5.set_ylabel('Absolute Sampling Error')
        ax5.set_title('Sampling Error by AOI Size')
        plt.suptitle('')  # Remove automatic boxplot title
        
        # 6. Relative error by canopy extent
        ax6 = axes[2, 1]
        results_df.boxplot(column='sampling_relative_error', by='canopy_extent_target', ax=ax6)
        ax6.set_xlabel('Target Canopy Extent')
        ax6.set_ylabel('Relative Sampling Error')
        ax6.set_title('Relative Error by Canopy Coverage')
        plt.suptitle('')  # Remove automatic boxplot title
        
        plt.tight_layout()
        plt.show()
        
        # Print comprehensive summary statistics
        print("\n" + "="*70)
        print("PARAMETER SWEEP & SAMPLING SUMMARY")
        print("="*70)
        print(f"Total landscapes generated: {len(results_df)}")
        print(f"Overall success rate: {results_df['success'].mean():.1%}")
        print(f"Average Moran's I error: {results_df['morans_i_difference'].mean():.4f}")
        print(f"Average canopy extent error: {abs(results_df['canopy_extent_target'] - results_df['canopy_extent_actual']).mean():.4f}")
        print(f"Average generation time: {results_df['generation_time_seconds'].mean():.2f} seconds")
        
        print(f"\nSAMPLING PERFORMANCE:")
        print(f"Average sampling bias: {results_df['sampling_bias'].mean():+.6f}")
        print(f"Average absolute error: {results_df['sampling_absolute_error'].mean():.6f}")
        print(f"Average relative error: {results_df['sampling_relative_error'].mean():.4%}")
        print(f"CI coverage rate: {results_df['ci_contains_true'].mean():.1%}")
        
        print(f"\nSuccess rate by AOI:")
        for aoi in sorted(results_df['aoi_m2'].unique()):
            success_rate = results_df[results_df['aoi_m2'] == aoi]['success'].mean()
            avg_bias = results_df[results_df['aoi_m2'] == aoi]['sampling_bias'].mean()
            print(f"  {aoi:,} m²: {success_rate:.1%} success, {avg_bias:+.6f} avg bias")

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
        
        # For multiple AOIs, we need to handle them separately or combine
        # Since sample points differ by AOI, let's create separate files or sections
        if len(sample_points_data) == 1:
            # Single AOI case
            final_df = sample_points_data[0]
        else:
            # Multiple AOIs - create separate sheets or files
            for i, (aoi, aoi_df) in enumerate(zip(sorted(results_df['aoi_m2'].unique()), sample_points_data)):
                aoi_filename = filename.replace('.csv', f'_AOI_{int(aoi)}m2.csv')
                aoi_df.to_csv(aoi_filename, index=False)
                print(f"  Exported AOI {int(aoi)} m² data to {aoi_filename}")
            
            # Also create a combined summary file
            final_df = sample_points_data[0]  # Use first AOI as base
        
        # Export main file
        final_df.to_csv(filename, index=False)
        print(f"Sample points data exported successfully!")
        print(f"  File: {filename}")
        print(f"  Sample points: {len(final_df):,}")
        print(f"  Landscape columns: {len([col for col in final_df.columns if col not in ['Sample_Point_ID', 'Sample_Point_X_Location', 'Sample_Point_Y_Location']])}")
        
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
                'Algorithm': row['algorithm'],
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


# Example usage and parameter sweep
if __name__ == "__main__":
    # Initialize generator with 100,000 sample points
    generator = NeutralLandscapeGenerator(
        cell_size=1,  # 1m resolution
        n_sample_points=100000,  # 100,000 random sample points
        random_seed=42  # For reproducible results
    )
    
    # Define parameter ranges for your project
    aoi_values = [200, 600, 4000]  # Area of interest in m²
    canopy_extents = [0.2, 0.4, 0.6, 0.8]  # 20%, 40%, 60%, 80% canopy coverage
    morans_i_values = [-0.5, -0.2, 0.0, 0.2, 0.5, 0.8]  # Range of spatial autocorrelation
    
    # Run parameter sweep with landscape and sample point data saved
    print("Running parameter sweep...")
    results_df = generator.run_parameter_sweep(
        aoi_values=aoi_values,
        canopy_extents=canopy_extents,
        morans_i_values=morans_i_values,
        algorithm='mpd',  # Try 'randomClusterNN' as alternative
        n_replicates=2,  # Generate 2 landscapes per parameter combination
        save_landscapes=True  # REQUIRED for sample points export
    )
    
    # Export sample points results in specified format
    print("\nExporting sample points data...")
    sample_points_df = generator.export_sample_points_csv(
        results_df, 
        filename='sample_points_canopy_analysis.csv'
    )
    
    # Create and export landscape summary
    print("\nCreating landscape summary...")
    summary_df = generator.create_sample_points_summary(results_df)
    summary_df.to_csv('landscape_characteristics_summary.csv', index=False)
    print("Landscape summary exported to 'landscape_characteristics_summary.csv'")
    
    # Create summary plots
    print("\nGenerating summary plots...")
    generator.plot_results_summary(results_df)
    
    # Save detailed results
    results_df_export = results_df.
