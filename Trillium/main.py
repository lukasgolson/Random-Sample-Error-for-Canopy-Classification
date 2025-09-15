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
    """
    
    def __init__(self, cell_size=1):
        """
        Initialize generator
        
        Parameters:
        -----------
        cell_size : float
            Size of each cell in meters (default=1m for easier calculations)
        """
        self.cell_size = cell_size
        self.results = []
        
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
                    'generation_time_seconds': time.time() - start_time
                }
                
                if save_landscapes:
                    result_record['landscape_array'] = result['landscape']
                    result_record['continuous_array'] = result['continuous_landscape']
                
                self.results.append(result_record)
                
                # Progress update
                success_rate = np.mean([r['success'] for r in self.results])
                avg_time = np.mean([r['generation_time_seconds'] for r in self.results])
                print(f"Success: {'✓' if result['success'] else '✗'} | "
                      f"Success rate: {success_rate:.1%} | "
                      f"Avg time: {avg_time:.2f}s")
        
        return pd.DataFrame(self.results)
    
    def plot_results_summary(self, results_df):
        """
        Create summary plots of the parameter sweep results
        
        Parameters:
        -----------
        results_df : pd.DataFrame
            Results from run_parameter_sweep
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Neutral Landscape Generation Results Summary', fontsize=16)
        
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
        
        # 3. Canopy extent accuracy
        ax3 = axes[1, 0]
        ax3.scatter(results_df['canopy_extent_target'], results_df['canopy_extent_actual'], 
                   alpha=0.6, c=results_df['morans_i_target'], cmap='coolwarm')
        ax3.plot([0, 1], [0, 1], 'r--', alpha=0.8, label='Perfect match')
        ax3.set_xlabel('Target Canopy Extent')
        ax3.set_ylabel('Actual Canopy Extent')
        ax3.set_title('Canopy Extent Accuracy')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Generation time distribution
        ax4 = axes[1, 1]
        results_df.boxplot(column='generation_time_seconds', by='aoi_m2', ax=ax4)
        ax4.set_xlabel('Area of Interest (m²)')
        ax4.set_ylabel('Generation Time (seconds)')
        ax4.set_title('Generation Time by AOI')
        plt.suptitle('')  # Remove automatic boxplot title
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print("\n" + "="*60)
        print("PARAMETER SWEEP SUMMARY")
        print("="*60)
        print(f"Total landscapes generated: {len(results_df)}")
        print(f"Overall success rate: {results_df['success'].mean():.1%}")
        print(f"Average Moran's I error: {results_df['morans_i_difference'].mean():.4f}")
        print(f"Average canopy extent error: {abs(results_df['canopy_extent_target'] - results_df['canopy_extent_actual']).mean():.4f}")
        print(f"Average generation time: {results_df['generation_time_seconds'].mean():.2f} seconds")
        
        print(f"\nSuccess rate by AOI:")
        for aoi in sorted(results_df['aoi_m2'].unique()):
            success_rate = results_df[results_df['aoi_m2'] == aoi]['success'].mean()
            print(f"  {aoi:,} m²: {success_rate:.1%}")


# Example usage and parameter sweep
if __name__ == "__main__":
    # Initialize generator
    generator = NeutralLandscapeGenerator(cell_size=1)  # 1m resolution
    
    # Define parameter ranges for your project
    aoi_values = [200, 600, 4000]  # Area of interest in m²
    canopy_extents = [0.2, 0.4, 0.6, 0.8]  # 20%, 40%, 60%, 80% canopy coverage
    morans_i_values = [-0.5, -0.2, 0.0, 0.2, 0.5, 0.8]  # Range of spatial autocorrelation
    
    # Run parameter sweep
    results_df = generator.run_parameter_sweep(
        aoi_values=aoi_values,
        canopy_extents=canopy_extents,
        morans_i_values=morans_i_values,
        algorithm='mpd',  # Try 'randomClusterNN' as alternative
        n_replicates=3,  # Generate 3 landscapes per parameter combination
        save_landscapes=False  # Set to True if you want to save actual arrays
    )
    
    # Create summary plots
    generator.plot_results_summary(results_df)
    
    # Save results to CSV
    results_df.to_csv('neutral_landscape_parameter_sweep.csv', index=False)
    print(f"\nResults saved to 'neutral_landscape_parameter_sweep.csv'")
    
    # Example: Access a specific landscape
    # If save_landscapes=True, you can access individual landscapes:
    # landscape_array = results_df.iloc[0]['landscape_array']
    # plt.imshow(landscape_array, cmap='Greens')
    # plt.title(f"Example Landscape (AOI={results_df.iloc[0]['aoi_m2']}m², Canopy={results_df.iloc[0]['canopy_extent_target']:.1%})")
    # plt.show()
