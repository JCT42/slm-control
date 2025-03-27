"""
Advanced test script for pattern generation algorithms.
This script provides comprehensive testing and visualization of pattern generation algorithms.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import sys
import os
from matplotlib.colors import LogNorm
import argparse
import importlib.util

# Import the pattern generator from the main file
spec = importlib.util.spec_from_file_location("pattern_gen_2_0", "pattern_gen_2.0.py")
pattern_gen_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pattern_gen_module)
PatternGenerator = pattern_gen_module.PatternGenerator

class PatternTester:
    def __init__(self, size=128, test_name="default_test"):
        """Initialize pattern tester with specified size and test name"""
        self.size = size
        self.test_name = test_name
        self.results_dir = "pattern_test_results"
        
        # Create results directory if it doesn't exist
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            
        # Create test directory
        self.test_dir = os.path.join(self.results_dir, test_name)
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
    
    def create_test_patterns(self):
        """Create a variety of test patterns for comprehensive testing"""
        patterns = {}
        
        # Pattern 1: Single spot in center
        spot = np.zeros((self.size, self.size))
        center = self.size // 2
        radius = self.size // 20
        y, x = np.ogrid[-center:self.size-center, -center:self.size-center]
        mask = x*x + y*y <= radius*radius
        spot[mask] = 1.0
        patterns["single_spot"] = spot
        
        # Pattern 2: Multiple spots
        spots = np.zeros((self.size, self.size))
        positions = [
            (self.size//4, self.size//4),
            (self.size//4, 3*self.size//4),
            (3*self.size//4, self.size//4),
            (3*self.size//4, 3*self.size//4)
        ]
        for x, y in positions:
            y_grid, x_grid = np.ogrid[0:self.size, 0:self.size]
            mask = ((x_grid - x)**2 + (y_grid - y)**2) <= radius**2
            spots[mask] = 1.0
        patterns["multiple_spots"] = spots
        
        # Pattern 3: Line pattern
        line = np.zeros((self.size, self.size))
        line_width = self.size // 40
        line[:, center-line_width:center+line_width] = 1.0
        patterns["line"] = line
        
        # Pattern 4: Cross pattern
        cross = np.zeros((self.size, self.size))
        cross[center-line_width:center+line_width, :] = 1.0
        cross[:, center-line_width:center+line_width] = 1.0
        patterns["cross"] = cross
        
        # Pattern 5: Checkerboard
        checkerboard = np.zeros((self.size, self.size))
        check_size = self.size // 8
        for i in range(8):
            for j in range(8):
                if (i + j) % 2 == 0:
                    checkerboard[i*check_size:(i+1)*check_size, 
                                j*check_size:(j+1)*check_size] = 1.0
        patterns["checkerboard"] = checkerboard
        
        # Pattern 6: Gaussian spot
        gaussian = np.zeros((self.size, self.size))
        x = np.linspace(-3, 3, self.size)
        y = np.linspace(-3, 3, self.size)
        x_grid, y_grid = np.meshgrid(x, y)
        gaussian = np.exp(-(x_grid**2 + y_grid**2))
        patterns["gaussian"] = gaussian
        
        return patterns
    
    def create_signal_masks(self, patterns):
        """Create signal region masks for MRAF testing"""
        masks = {}
        
        for name, pattern in patterns.items():
            # Create a slightly larger mask than the pattern
            mask = np.zeros_like(pattern)
            mask[pattern > 0.1] = 1
            
            # Dilate the mask a bit
            from scipy import ndimage
            mask = ndimage.binary_dilation(mask, iterations=3).astype(float)
            
            masks[name] = mask
        
        return masks
    
    def calculate_metrics(self, target, reconstruction):
        """Calculate various quality metrics for pattern evaluation"""
        metrics = {}
        
        # Normalize patterns for fair comparison
        target_norm = target / np.max(target)
        recon_norm = reconstruction / np.max(reconstruction)
        
        # 1. Correlation coefficient
        target_flat = target_norm.flatten() - np.mean(target_norm)
        recon_flat = recon_norm.flatten() - np.mean(recon_norm)
        
        numerator = np.sum(target_flat * recon_flat)
        denominator = np.sqrt(np.sum(target_flat**2) * np.sum(recon_flat**2))
        correlation = numerator / denominator if denominator != 0 else 0
        metrics["correlation"] = correlation
        
        # 2. Mean Square Error (MSE)
        mse = np.mean((target_norm - recon_norm)**2)
        metrics["mse"] = mse
        
        # 3. Normalized Mean Square Error (NMSE)
        nmse = mse / np.mean(target_norm**2) if np.mean(target_norm**2) > 0 else float('inf')
        metrics["nmse"] = nmse
        
        # 4. Peak Signal-to-Noise Ratio (PSNR)
        if mse > 0:
            psnr = 10 * np.log10(1.0 / mse)  # Assuming max value is 1.0
        else:
            psnr = float('inf')
        metrics["psnr"] = psnr
        
        # 5. Signal region efficiency (for patterns with clear signal regions)
        signal_mask = target_norm > 0.1
        if np.sum(signal_mask) > 0:
            signal_power = np.sum(recon_norm[signal_mask])
            total_power = np.sum(recon_norm)
            efficiency = signal_power / total_power if total_power > 0 else 0
            metrics["efficiency"] = efficiency
        else:
            metrics["efficiency"] = 0
        
        return metrics
    
    def run_algorithm_test(self, pattern_name, target, algorithm="gs", max_iterations=50, 
                          signal_mask=None, mixing_parameter=0.4, tolerance=1e-4):
        """Run a single algorithm test on a specific pattern"""
        print(f"\n=== Testing {algorithm.upper()} on {pattern_name} pattern ===")
        
        # Create pattern generator
        pattern_gen = PatternGenerator(
            target_intensity=target.copy(),
            signal_region_mask=signal_mask,
            mixing_parameter=mixing_parameter
        )
        
        # Initialize random phase (same for both algorithms)
        np.random.seed(42)  # For reproducibility
        random_phase = np.exp(1j * 2 * np.pi * np.random.rand(self.size, self.size))
        
        # Apply initial field with target amplitude
        initial_field = np.sqrt(pattern_gen.target_intensity) * random_phase
        
        # Time the algorithm
        start_time = time.time()
        
        # Run optimization
        optimized_field, error_history, stop_reason = pattern_gen.optimize(
            initial_field=initial_field,
            algorithm=algorithm,
            max_iterations=max_iterations,
            tolerance=tolerance
        )
        
        # Calculate time taken
        elapsed_time = time.time() - start_time
        
        # Calculate SLM phase pattern
        slm_field = pattern_gen.propagate(optimized_field)
        slm_phase = np.angle(slm_field)
        
        # Calculate reconstruction
        recon_field = pattern_gen.inverse_propagate(np.exp(1j * slm_phase))
        reconstruction = np.abs(recon_field)**2
        
        # Calculate quality metrics
        metrics = self.calculate_metrics(target, reconstruction)
        
        # Save results
        results = {
            "pattern_name": pattern_name,
            "algorithm": algorithm,
            "max_iterations": max_iterations,
            "tolerance": tolerance,
            "mixing_parameter": mixing_parameter if algorithm == "mraf" else None,
            "stop_reason": stop_reason,
            "iterations_used": len(error_history),
            "elapsed_time": elapsed_time,
            "error_history": error_history,
            "metrics": metrics,
            "target": target,
            "reconstruction": reconstruction,
            "slm_phase": slm_phase
        }
        
        # Print summary
        print(f"Algorithm: {algorithm.upper()}")
        print(f"Pattern: {pattern_name}")
        print(f"Stop reason: {stop_reason}")
        print(f"Iterations: {len(error_history)}")
        print(f"Time: {elapsed_time:.3f} seconds")
        print(f"Correlation: {metrics['correlation']:.4f}")
        print(f"NMSE: {metrics['nmse']:.4e}")
        print(f"Efficiency: {metrics['efficiency']:.4f}")
        
        return results
    
    def plot_results(self, results, save_path=None):
        """Plot and save the results of an algorithm test"""
        plt.figure(figsize=(15, 10))
        
        # Get data from results
        pattern_name = results["pattern_name"]
        algorithm = results["algorithm"]
        target = results["target"]
        reconstruction = results["reconstruction"]
        slm_phase = results["slm_phase"]
        error_history = results["error_history"]
        metrics = results["metrics"]
        
        # Target
        plt.subplot(2, 3, 1)
        plt.imshow(target, cmap='viridis')
        plt.title('Target Pattern')
        plt.colorbar()
        
        # Reconstruction
        plt.subplot(2, 3, 2)
        plt.imshow(reconstruction, cmap='viridis')
        plt.title(f'Reconstruction\nCorrelation: {metrics["correlation"]:.4f}')
        plt.colorbar()
        
        # Phase pattern
        plt.subplot(2, 3, 3)
        plt.imshow(slm_phase, cmap='twilight')
        plt.title('SLM Phase Pattern')
        plt.colorbar()
        
        # Error history (linear scale)
        plt.subplot(2, 3, 4)
        plt.plot(error_history)
        plt.title('Error History (linear scale)')
        plt.xlabel('Iteration')
        plt.ylabel('NMSE')
        plt.grid(True)
        
        # Error history (log scale)
        plt.subplot(2, 3, 5)
        plt.semilogy(error_history)
        plt.title('Error History (log scale)')
        plt.xlabel('Iteration')
        plt.ylabel('NMSE (log scale)')
        plt.grid(True)
        
        # Difference map
        plt.subplot(2, 3, 6)
        diff = np.abs(reconstruction / np.max(reconstruction) - target / np.max(target))
        plt.imshow(diff, cmap='hot')
        plt.title(f'Difference Map\nNMSE: {metrics["nmse"]:.4e}')
        plt.colorbar()
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path)
            print(f"Results saved to {save_path}")
        
        plt.close()
    
    def run_comprehensive_test(self, algorithms=None, max_iterations=50, tolerance=1e-4):
        """Run comprehensive tests on all patterns with specified algorithms"""
        if algorithms is None:
            algorithms = ["gs", "mraf"]
        
        # Create test patterns
        patterns = self.create_test_patterns()
        
        # Create signal masks for MRAF
        masks = self.create_signal_masks(patterns)
        
        # Store all results
        all_results = []
        
        # Test each pattern with each algorithm
        for pattern_name, pattern in patterns.items():
            for algorithm in algorithms:
                # For MRAF, use signal mask
                signal_mask = masks[pattern_name] if algorithm == "mraf" else None
                
                # Run test
                results = self.run_algorithm_test(
                    pattern_name=pattern_name,
                    target=pattern,
                    algorithm=algorithm,
                    max_iterations=max_iterations,
                    signal_mask=signal_mask,
                    tolerance=tolerance
                )
                
                # Plot and save results
                save_path = os.path.join(self.test_dir, f"{pattern_name}_{algorithm}.png")
                self.plot_results(results, save_path)
                
                all_results.append(results)
        
        # Create summary report
        self.create_summary_report(all_results)
        
        return all_results
    
    def create_summary_report(self, all_results):
        """Create a summary report of all test results"""
        report_path = os.path.join(self.test_dir, "summary_report.txt")
        
        with open(report_path, "w") as f:
            f.write("=== PATTERN GENERATION ALGORITHM TEST SUMMARY ===\n\n")
            f.write(f"Test name: {self.test_name}\n")
            f.write(f"Pattern size: {self.size}x{self.size}\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Table header
            f.write(f"{'Pattern':<15} {'Algorithm':<6} {'Iterations':<10} {'Time (s)':<10} "
                   f"{'Correlation':<12} {'NMSE':<12} {'Efficiency':<10}\n")
            f.write("-" * 80 + "\n")
            
            # Table rows
            for result in all_results:
                pattern = result["pattern_name"]
                algorithm = result["algorithm"].upper()
                iterations = result["iterations_used"]
                time_taken = f"{result['elapsed_time']:.3f}"
                correlation = f"{result['metrics']['correlation']:.4f}"
                nmse = f"{result['metrics']['nmse']:.4e}"
                efficiency = f"{result['metrics']['efficiency']:.4f}"
                
                f.write(f"{pattern:<15} {algorithm:<6} {iterations:<10} {time_taken:<10} "
                       f"{correlation:<12} {nmse:<12} {efficiency:<10}\n")
            
            f.write("\n\n=== ALGORITHM COMPARISON ===\n\n")
            
            # Compare algorithms by pattern
            patterns = set(result["pattern_name"] for result in all_results)
            algorithms = set(result["algorithm"] for result in all_results)
            
            for pattern in patterns:
                f.write(f"\nPattern: {pattern}\n")
                f.write("-" * 40 + "\n")
                
                pattern_results = [r for r in all_results if r["pattern_name"] == pattern]
                
                for metric in ["correlation", "nmse", "efficiency"]:
                    f.write(f"{metric.capitalize()}:\n")
                    
                    for algorithm in algorithms:
                        alg_results = [r for r in pattern_results if r["algorithm"] == algorithm]
                        if alg_results:
                            value = alg_results[0]["metrics"][metric]
                            f.write(f"  {algorithm.upper()}: {value:.6f}\n")
                
                f.write("\n")
        
        print(f"\nSummary report saved to {report_path}")
        
        # Create comparison plots
        self.create_comparison_plots(all_results)
    
    def create_comparison_plots(self, all_results):
        """Create plots comparing algorithms across different patterns"""
        # Group results by pattern and algorithm
        patterns = set(result["pattern_name"] for result in all_results)
        algorithms = list(set(result["algorithm"] for result in all_results))
        
        # Prepare data for plotting
        correlation_data = {alg: [] for alg in algorithms}
        nmse_data = {alg: [] for alg in algorithms}
        efficiency_data = {alg: [] for alg in algorithms}
        time_data = {alg: [] for alg in algorithms}
        iterations_data = {alg: [] for alg in algorithms}
        
        pattern_names = []
        
        for pattern in patterns:
            pattern_names.append(pattern)
            
            for algorithm in algorithms:
                # Find result for this pattern and algorithm
                result = next((r for r in all_results 
                              if r["pattern_name"] == pattern and r["algorithm"] == algorithm), None)
                
                if result:
                    correlation_data[algorithm].append(result["metrics"]["correlation"])
                    nmse_data[algorithm].append(result["metrics"]["nmse"])
                    efficiency_data[algorithm].append(result["metrics"]["efficiency"])
                    time_data[algorithm].append(result["elapsed_time"])
                    iterations_data[algorithm].append(result["iterations_used"])
                else:
                    correlation_data[algorithm].append(0)
                    nmse_data[algorithm].append(0)
                    efficiency_data[algorithm].append(0)
                    time_data[algorithm].append(0)
                    iterations_data[algorithm].append(0)
        
        # Plot correlation comparison
        plt.figure(figsize=(15, 10))
        
        # Correlation
        plt.subplot(2, 2, 1)
        x = np.arange(len(pattern_names))
        width = 0.35
        
        for i, algorithm in enumerate(algorithms):
            plt.bar(x + i*width, correlation_data[algorithm], width, label=algorithm.upper())
        
        plt.xlabel('Pattern')
        plt.ylabel('Correlation')
        plt.title('Correlation by Pattern and Algorithm')
        plt.xticks(x + width/2, pattern_names, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # NMSE (log scale)
        plt.subplot(2, 2, 2)
        
        for i, algorithm in enumerate(algorithms):
            plt.semilogy(x + i*width, nmse_data[algorithm], 'o-', label=algorithm.upper())
        
        plt.xlabel('Pattern')
        plt.ylabel('NMSE (log scale)')
        plt.title('NMSE by Pattern and Algorithm')
        plt.xticks(x + width/2, pattern_names, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Efficiency
        plt.subplot(2, 2, 3)
        
        for i, algorithm in enumerate(algorithms):
            plt.bar(x + i*width, efficiency_data[algorithm], width, label=algorithm.upper())
        
        plt.xlabel('Pattern')
        plt.ylabel('Efficiency')
        plt.title('Efficiency by Pattern and Algorithm')
        plt.xticks(x + width/2, pattern_names, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Time
        plt.subplot(2, 2, 4)
        
        for i, algorithm in enumerate(algorithms):
            plt.bar(x + i*width, time_data[algorithm], width, label=algorithm.upper())
        
        plt.xlabel('Pattern')
        plt.ylabel('Time (seconds)')
        plt.title('Computation Time by Pattern and Algorithm')
        plt.xticks(x + width/2, pattern_names, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        comparison_path = os.path.join(self.test_dir, "algorithm_comparison.png")
        plt.savefig(comparison_path)
        plt.close()
        
        print(f"Algorithm comparison plot saved to {comparison_path}")

def main():
    """Main function to run the pattern tests"""
    parser = argparse.ArgumentParser(description='Test pattern generation algorithms')
    parser.add_argument('--size', type=int, default=128, help='Size of test patterns')
    parser.add_argument('--name', type=str, default=f"test_{time.strftime('%Y%m%d_%H%M%S')}", 
                        help='Name for this test run')
    parser.add_argument('--iterations', type=int, default=50, help='Maximum iterations')
    parser.add_argument('--tolerance', type=float, default=1e-4, help='Convergence tolerance')
    parser.add_argument('--algorithms', type=str, nargs='+', default=['gs', 'mraf'], 
                        help='Algorithms to test')
    
    args = parser.parse_args()
    
    print(f"=== Starting Pattern Generation Algorithm Test ===")
    print(f"Test name: {args.name}")
    print(f"Pattern size: {args.size}x{args.size}")
    print(f"Algorithms: {', '.join(args.algorithms)}")
    print(f"Max iterations: {args.iterations}")
    print(f"Tolerance: {args.tolerance}")
    
    # Create tester and run tests
    tester = PatternTester(size=args.size, test_name=args.name)
    results = tester.run_comprehensive_test(
        algorithms=args.algorithms,
        max_iterations=args.iterations,
        tolerance=args.tolerance
    )
    
    print(f"\n=== Test Complete ===")
    print(f"Results saved to: {tester.test_dir}")

if __name__ == "__main__":
    main()
