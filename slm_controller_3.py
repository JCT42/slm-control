"""
Advanced Pattern Generator for Sony LCX016AL-6 SLM - Raspberry Pi Version
Generates patterns for amplitude, phase, or combined modulation using advanced algorithms.

SLM Specifications:
- Resolution: 800 x 600 pixels
- Pixel Pitch: 32 μm
- Refresh Rate: 60 Hz
- Contrast Ratio: 200:1
- Default Wavelength: 650 nm (red laser)

Advanced Features:
- Multiple modulation modes (Amplitude, Phase, Combined)
- Advanced pattern generation algorithms
- Amplitude and phase coupling compensation
- Multi-scale optimization
- Phase quantization compensation
- Enhanced input beam handling
- Adaptive signal region selection
- Enhanced phase shift visualization
"""

import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import threading
import time
import os
import subprocess
from picamera2 import Picamera2
from libcamera import controls
import pygame
from tqdm import tqdm
import scipy.special
import traceback

class PhaseOptimizer:
    def __init__(self, target_intensity, signal_region_mask=None, mixing_parameter=0.4):
        """
        Initialize pattern generator with target intensity and optional MRAF parameters.
        
        Args:
            target_intensity (np.ndarray): Target intensity pattern (2D array)
            signal_region_mask (np.ndarray): Binary mask defining signal region for MRAF (2D array)
            mixing_parameter (float): Mixing parameter for MRAF algorithm (0 < m < 1)
        """
        self.target_intensity = target_intensity
        # Normalize target intensity by maximum value instead of sum
        # This provides more meaningful error values and better convergence
        self.target_intensity = self.target_intensity / np.max(self.target_intensity)
        
        # If no signal region mask is provided, use the entire region
        if signal_region_mask is None:
            self.signal_region_mask = np.ones_like(target_intensity)
        else:
            self.signal_region_mask = signal_region_mask
            
        self.mixing_parameter = mixing_parameter
        
        # Create a smooth window function to reduce artifacts
        h, w = target_intensity.shape
        y, x = np.ogrid[-h//2:h//2, -w//2:w//2]
        
        # Create a super-Gaussian window (smoother than regular Gaussian)
        # Using a higher order (10) for even sharper edge roll-off but still smooth transition
        self.window = np.exp(-0.5 * ((x/(w*0.45))**10 + (y/(h*0.45))**10))
        
        # Calculate radial distance from center for filtering
        r = np.sqrt(x**2 + y**2)
        
        # Create a more aggressive DC filter with smoother transition
        # This removes the central bright spot more effectively
        dc_radius = min(w,h)*0.005  # Smaller radius for more targeted DC suppression
        self.dc_filter = 1.0 - np.exp(-0.5 * (r/dc_radius)**2)
        
        # Create a high-frequency filter to suppress noise that causes star patterns
        # Using a Butterworth-like filter for sharper cutoff with minimal ringing
        hf_radius = min(w,h)*0.4
        butterworth_order = 4  # Higher order for steeper roll-off
        self.hf_filter = 1.0 / (1.0 + (r/hf_radius)**(2*butterworth_order))
        
        # Create cross-shaped notch filter to specifically target star-shaped artifacts
        # These typically appear along the x and y axes in Fourier space
        notch_width = min(w,h)*0.01
        x_axis_mask = np.exp(-0.5 * (y/notch_width)**2)
        y_axis_mask = np.exp(-0.5 * (x/notch_width)**2)
        self.cross_filter = 1.0 - np.maximum(x_axis_mask, y_axis_mask) * 0.9  # 90% attenuation along axes
    
    def propagate(self, field):
        """Propagate field from image plane to SLM plane"""
        # Apply window function to reduce edge artifacts
        windowed_field = field * self.window
        
        # Use proper FFT shifting for optical propagation
        fft_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(windowed_field)))
        
        # Apply DC, high-frequency, and cross filters in Fourier space
        filtered_fft = fft_field * self.dc_filter * self.hf_filter * self.cross_filter
        
        return filtered_fft
    
    def inverse_propagate(self, field):
        """Propagate field from SLM plane to image plane"""
        # Use proper FFT shifting for optical propagation
        result = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(field)))
        
        # Apply window on the way back to reduce ringing artifacts
        # Using a gentler window for the reconstruction
        return result * np.sqrt(self.window)
    
    def gs_iteration(self, field):
        """Single iteration of Gerchberg-Saxton algorithm"""
        # Propagate to SLM plane
        slm_field = self.propagate(field)
        # Apply phase-only constraint
        slm_field = np.exp(1j * np.angle(slm_field))
        # Propagate to image plane
        image_field = self.inverse_propagate(slm_field)
        # Apply amplitude constraint
        return np.sqrt(self.target_intensity) * np.exp(1j * np.angle(image_field))
    
    def mraf_iteration(self, field):
        """Single iteration of Mixed-Region Amplitude Freedom algorithm"""
        # Propagate to SLM plane
        slm_field = self.propagate(field)
        # Apply phase-only constraint
        slm_field = np.exp(1j * np.angle(slm_field))
        # Propagate to image plane
        image_field = self.inverse_propagate(slm_field)
        
        # Apply MRAF mixing in signal and noise regions
        m = self.mixing_parameter
        sr_mask = self.signal_region_mask
        nr_mask = 1 - sr_mask
        
        mixed_field = np.zeros_like(image_field, dtype=complex)
        # Signal region: maintain target amplitude
        mixed_field[sr_mask == 1] = np.sqrt(self.target_intensity[sr_mask == 1]) * np.exp(1j * np.angle(image_field[sr_mask == 1]))
        # Noise region: allow amplitude freedom
        mixed_field[nr_mask == 1] = ((1-m) * image_field[nr_mask == 1] + m * np.sqrt(self.target_intensity[nr_mask == 1]) * np.exp(1j * np.angle(image_field[nr_mask == 1])))
        
        return mixed_field
    
    def optimize(self, initial_field, algorithm='gs', max_iterations=100, tolerance=1e-4):
        """
        Optimize the phase pattern using specified algorithm.
        
        Args:
            initial_field (np.ndarray): Initial complex field
            algorithm (str): 'gs' for Gerchberg-Saxton or 'mraf' for Mixed-Region Amplitude Freedom
            max_iterations (int): Maximum number of iterations
            tolerance (float): Convergence tolerance
            
        Returns:
            tuple: (optimized_field, error_history, stop_reason)
        """
        field = initial_field.copy()
        error_history = []
        prev_error = float('inf')
        stop_reason = "Maximum iterations reached"
        
        # Calculate initial error for reference
        initial_error = self.calculate_error(field, algorithm)
        print(f"Initial error: {initial_error:.3e}")
        
        # Run optimization loop
        for i in range(max_iterations):
            # Store field before iteration for comparison
            prev_field = field.copy()
            
            # Apply iteration based on selected algorithm
            if algorithm.lower() == 'gs':
                field = self.gs_iteration(field)
            elif algorithm.lower() == 'mraf':
                field = self.mraf_iteration(field)
            else:
                raise ValueError("Algorithm must be 'gs' or 'mraf'")
                
            # Calculate error for monitoring
            current_error = self.calculate_error(field, algorithm)
            error_history.append(current_error)
            
            # Calculate field change for convergence check
            field_change = self.calculate_field_change(field, prev_field)
                
            # Print current metrics for debugging
            if i % 10 == 0:  # Print every 10 iterations to reduce console output
                print(f"Iteration {i}, Error: {current_error:.3e}, Field Change: {field_change:.3e}")
            
            # Check convergence based on field change (more reliable than error delta)
            if field_change < tolerance and i > 5:  # Require at least 5 iterations
                stop_reason = f"Convergence reached at iteration {i+1}: Field change ({field_change:.3e}) < tolerance ({tolerance:.3e})"
                print(stop_reason)
                break
                
            # Check for NaN or Inf in error
            if np.isnan(current_error) or np.isinf(current_error):
                stop_reason = f"Algorithm stopped at iteration {i+1}: Error value is {current_error}"
                print(stop_reason)
                break
                
            prev_error = current_error
        
        # Calculate final error for comparison
        final_error = self.calculate_error(field, algorithm)
        improvement = initial_error / final_error if final_error > 0 else float('inf')
        print(f"Final error: {final_error:.3e}, Improvement: {improvement:.2f}x")
        
        # Return the results
        return field, error_history, stop_reason
    
    def calculate_error(self, field, algorithm):
        """
        Calculate Normalized Mean Square Error (NMSE) between reconstructed and target intensity.
        Using mean-value normalization instead of max-value normalization for better convergence tracking.
        """
        recon_intensity = np.abs(field)**2
        
        if algorithm.lower() == 'gs':
            # For GS, calculate error over entire field
            # Calculate Mean Square Error (MSE)
            mse = np.mean((recon_intensity - self.target_intensity)**2)
            # Normalize by dividing by the mean of target intensity to get NMSE
            norm_error = mse / np.mean(self.target_intensity**2)
        else:
            # For MRAF, calculate error only in signal region
            sr_mask = self.signal_region_mask
            if np.sum(sr_mask) > 0:  # Ensure signal region is not empty
                # Calculate MSE in signal region
                mse = np.mean((recon_intensity[sr_mask == 1] - self.target_intensity[sr_mask == 1])**2)
                # Normalize by mean squared target intensity in signal region (NMSE)
                norm_error = mse / np.mean(self.target_intensity[sr_mask == 1]**2)
            else:
                norm_error = 0.0
                
        return norm_error
    
    def calculate_field_change(self, field, prev_field):
        """
        Calculate the mean change in field intensity between iterations.
        This is a more reliable convergence metric than error delta.
        """
        return np.mean(np.abs(np.abs(field)**2 - np.abs(prev_field)**2))

class AdvancedPatternGenerator:
    def __init__(self):
        """Initialize the advanced pattern generator with extended features"""
        # Initialize pygame
        pygame.init()
        
        # Sony LCX016AL-6 specifications
        self.width = 800
        self.height = 600
        self.pixel_size = 32e-6
        self.active_area = (26.6e-3, 20.0e-3)
        self.refresh_rate = 60
        self.contrast_ratio = 200
        
        # Default wavelength
        self.wavelength = 650e-9
        
        # Modulation parameters
        self.modulation_mode = "Phase"  # "Phase", "Amplitude", or "Combined"
        self.amplitude_coupling = 0.1
        self.phase_coupling = 0.1
        
        # Phase shift parameters for zero-order suppression
        self.phase_shift_x = 0.0  # Phase shift in x-direction (cycles per image)
        self.phase_shift_y = 0.0  # Phase shift in y-direction (cycles per image)
        
        # Initialize camera state
        self.camera_active = False
        self.camera_paused = False
        self.last_frame = None  # Store the last frame for pause functionality
        
        # Simulation parameters
        self.padding_factor = 2
        self.padded_width = self.width * self.padding_factor
        self.padded_height = self.height * self.padding_factor
        
        # Calculate important parameters
        self.k = 2 * np.pi / self.wavelength
        self.dx = self.pixel_size
        self.df_x = 1 / (self.padded_width * self.dx)
        self.df_y = 1 / (self.padded_height * self.dx)
        
        # Create coordinate grids
        self.x = np.linspace(-self.padded_width//2, self.padded_width//2-1, self.padded_width) * self.dx
        self.y = np.linspace(-self.padded_height//2, self.padded_height//2-1, self.padded_height) * self.dx
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Initialize input beam parameters
        self.input_beam_type = "Gaussian"  # Default beam type
        self.custom_input_beam = None
        
        # Initialize signal region mask for MRAF
        self.signal_region_mask = None
        
        # New parameters for multi-scale optimization
        self.use_multiscale = True
        self.scale_levels = 3  # Number of scale levels for multi-scale optimization
        
        # Phase quantization parameters
        self.phase_levels = 256  # 8-bit SLM
        self.use_phase_compensation = True
        
        # Initialize GUI
        self.setup_gui()

    def setup_gui(self):
        """Create the main GUI window and controls"""
        self.root = tk.Tk()
        self.root.title("Advanced SLM Pattern Generator 3.0")
        
        # Add ESC key binding to exit application
        self.root.bind('<Escape>', lambda e: self.quit_application())
        
        # Create main frames
        self.control_frame = ttk.Frame(self.root, padding="10")
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # Create scrollable frame for controls
        self.canvas = tk.Canvas(self.control_frame)
        self.scrollbar = ttk.Scrollbar(self.control_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Bind mousewheel to scroll
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Create parameter controls
        self.create_controls()
        
        # Create phase shift controls with enhanced visualization
        self.create_phase_shift_controls()
        
        # Create preview area
        self.preview_frame = ttk.Frame(self.root, padding="10")
        self.preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create preview with matplotlib
        self.create_preview()
        
        # Create camera preview
        self.create_camera_preview()
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Set up window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.quit_application)
        
        # Initialize camera if available
        self.initialize_camera()
        
        # Initialize target and pattern
        self.target = None
        self.pattern = None
        
        # Set up keyboard shortcuts
        self.root.bind("<Control-g>", lambda e: self.generate_pattern())
        self.root.bind("<Control-s>", lambda e: self.save_pattern())
        self.root.bind("<Control-l>", lambda e: self.load_image())
        self.root.bind("<Control-d>", lambda e: self.send_to_slm())
        
    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
    def create_controls(self):
        """Create parameter control widgets"""
        # Add buttons frame
        buttons_frame = ttk.Frame(self.scrollable_frame)
        buttons_frame.pack(fill=tk.X, pady=5)
        
        # Load target image button
        self.load_button = ttk.Button(buttons_frame, text="Load Target Image", command=self.load_image)
        self.load_button.pack(side=tk.LEFT, padx=5)
        
        # Load pattern button
        self.load_pattern_button = ttk.Button(buttons_frame, text="Load Pattern", command=self.load_pattern)
        self.load_pattern_button.pack(side=tk.LEFT, padx=5)
        
        # Send to SLM button
        self.send_to_slm_button = ttk.Button(buttons_frame, text="Send to SLM", command=self.send_to_slm)
        self.send_to_slm_button.pack(side=tk.LEFT, padx=5)
        
        # Save pattern button
        self.save_button = ttk.Button(buttons_frame, text="Save Pattern", command=self.save_pattern)
        self.save_button.pack(side=tk.LEFT, padx=5)
        
        # Create notebook for tabbed controls
        param_notebook = ttk.Notebook(self.scrollable_frame)
        param_notebook.pack(fill=tk.X, padx=5, pady=5)
        
        # Mode selection tab
        mode_frame = ttk.Frame(param_notebook)
        param_notebook.add(mode_frame, text="Mode")
        
        # Modulation mode
        ttk.Label(mode_frame, text="Mode:").grid(row=0, column=0, padx=5, pady=5)
        self.mode_var = tk.StringVar(value="Phase")
        mode_menu = ttk.OptionMenu(mode_frame, self.mode_var, "Phase", "Phase", "Amplitude", "Combined")
        mode_menu.grid(row=0, column=1, padx=5, pady=5)
        
        # Input beam selection
        ttk.Label(mode_frame, text="Input Beam:").grid(row=0, column=2, padx=5, pady=5)
        self.beam_type_var = tk.StringVar(value="Gaussian")
        beam_types = ["Gaussian", "Super Gaussian", "Top Hat", "Bessel", "LG01", "Custom"]
        beam_menu = ttk.OptionMenu(mode_frame, self.beam_type_var, "Gaussian", *beam_types, 
                                 command=self._on_beam_type_change)
        beam_menu.grid(row=0, column=3, padx=5, pady=5)
        
        # Amplitude coupling
        ttk.Label(mode_frame, text="Amplitude Coupling:").grid(row=0, column=4, padx=5, pady=5)
        self.amp_coupling_var = tk.StringVar(value="0.1")
        ttk.Entry(mode_frame, textvariable=self.amp_coupling_var, width=10).grid(row=0, column=5, padx=5, pady=5)
        
        # Phase coupling
        ttk.Label(mode_frame, text="Phase Coupling:").grid(row=0, column=6, padx=5, pady=5)
        self.phase_coupling_var = tk.StringVar(value="0.1")
        ttk.Entry(mode_frame, textvariable=self.phase_coupling_var, width=10).grid(row=0, column=7, padx=5, pady=5)
        
        # Algorithm parameters tab
        algo_frame = ttk.Frame(param_notebook)
        param_notebook.add(algo_frame, text="Algorithm")
        
        # Algorithm selection
        ttk.Label(algo_frame, text="Algorithm:").grid(row=0, column=0, padx=5, pady=5)
        self.algorithm_var = tk.StringVar(value="gs")
        algorithm_menu = ttk.OptionMenu(algo_frame, self.algorithm_var, "gs", "gs", "mraf")
        algorithm_menu.grid(row=0, column=1, padx=5, pady=5)
        
        # Number of iterations
        ttk.Label(algo_frame, text="Iterations:").grid(row=0, column=2, padx=5, pady=5)
        self.iterations_var = tk.StringVar(value="10")
        ttk.Entry(algo_frame, textvariable=self.iterations_var, width=10).grid(row=0, column=3, padx=5, pady=5)
        
        # MRAF parameters frame (initially hidden)
        self.mraf_frame = ttk.Frame(algo_frame)
        self.mraf_frame.grid(row=1, column=0, columnspan=8, padx=5, pady=5)
        
        # MRAF mixing parameter
        ttk.Label(self.mraf_frame, text="Mixing Parameter:").grid(row=0, column=0, padx=5, pady=5)
        self.mixing_parameter_var = tk.StringVar(value="0.5")
        ttk.Entry(self.mraf_frame, textvariable=self.mixing_parameter_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        # Signal region ratio
        ttk.Label(self.mraf_frame, text="Signal Region Ratio:").grid(row=0, column=2, padx=5, pady=5)
        self.signal_region_ratio_var = tk.StringVar(value="0.3")
        ttk.Entry(self.mraf_frame, textvariable=self.signal_region_ratio_var, width=10).grid(row=0, column=3, padx=5, pady=5)
        
        # Error parameters frame
        error_frame = ttk.LabelFrame(algo_frame, text="Error Parameters")
        error_frame.grid(row=3, column=0, columnspan=8, padx=5, pady=5, sticky="ew")
        
        # Tolerance
        ttk.Label(error_frame, text="Tolerance:").grid(row=0, column=0, padx=5, pady=5)
        self.tolerance_var = tk.StringVar(value="1e-35")
        ttk.Entry(error_frame, textvariable=self.tolerance_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        # Show error plot checkbox
        self.show_error_plot_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(error_frame, text="Show Error Plot", variable=self.show_error_plot_var).grid(row=0, column=4, padx=5, pady=5)
        
        # Add event handler for algorithm selection
        self.algorithm_var.trace_add("write", self._on_algorithm_change)
        
        # Beam width factor
        ttk.Label(algo_frame, text="Beam Width:").grid(row=0, column=4, padx=5, pady=5)
        self.beam_width_var = tk.StringVar(value="1.0")
        ttk.Entry(algo_frame, textvariable=self.beam_width_var, width=10).grid(row=0, column=5, padx=5, pady=5)
        
        # Phase range
        ttk.Label(algo_frame, text="Phase Range (π):").grid(row=0, column=6, padx=5, pady=5)
        self.phase_range_var = tk.StringVar(value="2.0")
        ttk.Entry(algo_frame, textvariable=self.phase_range_var, width=10).grid(row=0, column=7, padx=5, pady=5)
        
        # Optical parameters tab
        optical_frame = ttk.Frame(param_notebook)
        param_notebook.add(optical_frame, text="Optical")
        
        # Wavelength
        ttk.Label(optical_frame, text="Wavelength (nm):").grid(row=0, column=0, padx=5, pady=5)
        self.wavelength_var = tk.StringVar(value="650")
        wavelength_entry = ttk.Entry(optical_frame, textvariable=self.wavelength_var, width=10)
        wavelength_entry.grid(row=0, column=1, padx=5, pady=5)
        wavelength_entry.bind('<Return>', lambda e: self.update_wavelength())
        
        # Gamma correction
        ttk.Label(optical_frame, text="Gamma:").grid(row=0, column=2, padx=5, pady=5)
        self.gamma_var = tk.StringVar(value="0.7")
        ttk.Entry(optical_frame, textvariable=self.gamma_var, width=10).grid(row=0, column=3, padx=5, pady=5)
        
        # Advanced optimization tab - NEW
        advanced_frame = ttk.Frame(param_notebook)
        param_notebook.add(advanced_frame, text="Advanced")
        
        # Multi-scale optimization
        ttk.Label(advanced_frame, text="Multi-scale:").grid(row=0, column=0, padx=5, pady=5)
        self.use_multiscale_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(advanced_frame, variable=self.use_multiscale_var).grid(row=0, column=1, padx=5, pady=5)
        
        # Scale levels
        ttk.Label(advanced_frame, text="Scale Levels:").grid(row=0, column=2, padx=5, pady=5)
        self.scale_levels_var = tk.StringVar(value="3")
        ttk.Entry(advanced_frame, textvariable=self.scale_levels_var, width=10).grid(row=0, column=3, padx=5, pady=5)
        
        # Phase quantization
        ttk.Label(advanced_frame, text="Phase Levels:").grid(row=1, column=0, padx=5, pady=5)
        self.phase_levels_var = tk.StringVar(value="256")
        ttk.Entry(advanced_frame, textvariable=self.phase_levels_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        # Phase compensation
        ttk.Label(advanced_frame, text="Phase Compensation:").grid(row=1, column=2, padx=5, pady=5)
        self.use_phase_compensation_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(advanced_frame, variable=self.use_phase_compensation_var).grid(row=1, column=3, padx=5, pady=5)
        
        # Adaptive signal region
        ttk.Label(advanced_frame, text="Adaptive Signal Region:").grid(row=2, column=0, padx=5, pady=5)
        self.use_adaptive_signal_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(advanced_frame, variable=self.use_adaptive_signal_var).grid(row=2, column=1, padx=5, pady=5)
        
        # Generate button
        ttk.Button(self.control_frame, text="Generate Pattern", command=self.generate_pattern).pack(pady=10)

    def create_phase_shift_controls(self):
        """Create controls for adjusting phase shift to avoid zero-order diffraction with visualization"""
        phase_shift_frame = ttk.LabelFrame(self.scrollable_frame, text="Zero-Order Diffraction Control", padding="10")
        phase_shift_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # X-direction phase shift
        ttk.Label(phase_shift_frame, text="X-Shift (cycles):").grid(row=0, column=0, padx=5, pady=5)
        self.phase_shift_x_var = tk.StringVar(value="0.0")
        phase_shift_x_entry = ttk.Entry(phase_shift_frame, textvariable=self.phase_shift_x_var, width=8)
        phase_shift_x_entry.grid(row=0, column=1, padx=5, pady=5)
        
        # X-direction phase shift slider
        x_slider = ttk.Scale(phase_shift_frame, from_=-5.0, to=5.0, orient=tk.HORIZONTAL, length=200,
                           command=lambda v: self.phase_shift_x_var.set(f"{float(v):.1f}"))
        x_slider.set(0.0)
        x_slider.grid(row=0, column=2, padx=5, pady=5)
        
        # Y-direction phase shift
        ttk.Label(phase_shift_frame, text="Y-Shift (cycles):").grid(row=1, column=0, padx=5, pady=5)
        self.phase_shift_y_var = tk.StringVar(value="0.0")
        phase_shift_y_entry = ttk.Entry(phase_shift_frame, textvariable=self.phase_shift_y_var, width=8)
        phase_shift_y_entry.grid(row=1, column=1, padx=5, pady=5)
        
        # Y-direction phase shift slider
        y_slider = ttk.Scale(phase_shift_frame, from_=-5.0, to=5.0, orient=tk.HORIZONTAL, length=200,
                           command=lambda v: self.phase_shift_y_var.set(f"{float(v):.1f}"))
        y_slider.set(0.0)
        y_slider.grid(row=1, column=2, padx=5, pady=5)
        
        # Apply button
        apply_button = ttk.Button(phase_shift_frame, text="Apply Shift", 
                                command=lambda: self.apply_phase_shift())
        apply_button.grid(row=0, column=3, rowspan=2, padx=10, pady=5)
        
        # Create a small figure for phase shift visualization
        self.shift_fig = plt.figure(figsize=(3, 3))
        self.shift_ax = self.shift_fig.add_subplot(111)
        self.shift_ax.set_title('Diffraction Orders')
        self.shift_ax.set_xlim(-2, 2)
        self.shift_ax.set_ylim(-2, 2)
        self.shift_ax.set_xticks([-2, -1, 0, 1, 2])
        self.shift_ax.set_yticks([-2, -1, 0, 1, 2])
        self.shift_ax.grid(True)
        
        # Add markers for diffraction orders
        self.zero_order = self.shift_ax.plot(0, 0, 'ro', markersize=10, label='Zero Order')[0]
        self.target_order = self.shift_ax.plot(0, 0, 'go', markersize=8, label='Target')[0]
        self.shift_ax.legend(loc='upper right', fontsize='small')
        
        # Create canvas for the phase shift visualization
        self.shift_canvas = FigureCanvasTkAgg(self.shift_fig, master=phase_shift_frame)
        self.shift_canvas.draw()
        self.shift_canvas.get_tk_widget().grid(row=0, column=4, rowspan=3, padx=10, pady=5)
        
        # Update visualization when values change
        self.phase_shift_x_var.trace_add("write", self._update_shift_visualization)
        self.phase_shift_y_var.trace_add("write", self._update_shift_visualization)
        
        # Help text
        help_text = "Shift your image away from the zero-order (undiffracted) light by adding a linear phase ramp.\n"
        help_text += "Positive values shift right/down, negative values shift left/up.\n"
        help_text += "Values between 1.0 and 3.0 work well for shifting to the first diffraction order."
        help_label = ttk.Label(phase_shift_frame, text=help_text, wraplength=500)
        help_label.grid(row=2, column=0, columnspan=4, padx=5, pady=5)
        
    def _update_shift_visualization(self, *args):
        """Update the phase shift visualization"""
        try:
            # Get current shift values
            x_shift = float(self.phase_shift_x_var.get())
            y_shift = float(self.phase_shift_y_var.get())
            
            # Update target order position
            self.target_order.set_data(x_shift, y_shift)
            
            # Update canvas
            self.shift_canvas.draw()
        except ValueError:
            # Ignore invalid values
            pass

    def create_preview(self):
        """Create preview area with matplotlib"""
        # Create figure with subplots
        self.fig = plt.figure(figsize=(12, 8))
        
        # Pattern preview
        self.pattern_ax = self.fig.add_subplot(221)
        self.pattern_ax.set_title('SLM Pattern')
        self.pattern_im = self.pattern_ax.imshow(np.zeros((self.height, self.width)), cmap='gray')
        self.pattern_ax.set_axis_off()
        
        # Target preview
        self.target_ax = self.fig.add_subplot(222)
        self.target_ax.set_title('Target Image')
        self.target_im = self.target_ax.imshow(np.zeros((self.height, self.width)), cmap='gray')
        self.target_ax.set_axis_off()
        
        # Reconstruction preview
        self.recon_ax = self.fig.add_subplot(223)
        self.recon_ax.set_title('Simulated Reconstruction')
        self.recon_im = self.recon_ax.imshow(np.zeros((self.height, self.width)), cmap='viridis')
        self.recon_ax.set_axis_off()
        
        # Error plot
        self.error_ax = self.fig.add_subplot(224)
        self.error_ax.set_title('Error vs. Iteration')
        self.error_ax.set_xlabel('Iteration')
        self.error_ax.set_ylabel('NMSE')
        self.error_ax.set_yscale('log')
        self.error_line, = self.error_ax.plot([], [], 'b-')
        
        # Add colorbar for reconstruction
        self.recon_cbar = self.fig.colorbar(self.recon_im, ax=self.recon_ax, fraction=0.046, pad=0.04)
        self.recon_cbar.set_label('Intensity (log scale)')
        
        # Adjust layout
        self.fig.tight_layout()
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.preview_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Add toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.preview_frame)
        self.toolbar.update()
        
    def create_camera_preview(self):
        """Create camera preview area"""
        # Create camera control frame
        camera_control_frame = ttk.LabelFrame(self.scrollable_frame, text="Camera Control")
        camera_control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Camera control buttons
        ttk.Button(camera_control_frame, text="Start Camera", command=self.start_camera).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(camera_control_frame, text="Stop Camera", command=self.stop_camera).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(camera_control_frame, text="Pause/Resume", command=self.toggle_pause_camera).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(camera_control_frame, text="Capture Frame", command=self.capture_frame).grid(row=0, column=3, padx=5, pady=5)
        ttk.Button(camera_control_frame, text="Save Image", command=self.save_camera_image).grid(row=0, column=4, padx=5, pady=5)
        ttk.Button(camera_control_frame, text="Send to SLM", command=self.send_to_slm).grid(row=0, column=5, padx=5, pady=5)
        
        # Exposure control
        ttk.Label(camera_control_frame, text="Exposure:").grid(row=1, column=0, padx=5, pady=5)
        self.exposure_var = tk.StringVar(value="1000")
        exposure_entry = ttk.Entry(camera_control_frame, textvariable=self.exposure_var, width=10)
        exposure_entry.grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(camera_control_frame, text="Set", command=self.set_camera_exposure).grid(row=1, column=2, padx=5, pady=5)
        
        # Create camera figure
        self.camera_fig = plt.figure(figsize=(6, 4))
        self.camera_ax = self.camera_fig.add_subplot(111)
        self.camera_ax.set_title('Camera View')
        self.camera_im = self.camera_ax.imshow(np.zeros((480, 640, 3)))
        self.camera_ax.set_axis_off()
        
        # Create camera canvas in a separate window
        self.camera_window = tk.Toplevel(self.root)
        self.camera_window.title("Camera Preview")
        self.camera_window.protocol("WM_DELETE_WINDOW", lambda: self.camera_window.withdraw())
        self.camera_window.withdraw()  # Hide initially
        
        # Add ESC key binding to close camera window
        self.camera_window.bind('<Escape>', lambda e: self.camera_window.withdraw())
        
        self.camera_canvas = FigureCanvasTkAgg(self.camera_fig, master=self.camera_window)
        self.camera_canvas.draw()
        self.camera_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Add toolbar
        self.camera_toolbar = NavigationToolbar2Tk(self.camera_canvas, self.camera_window)
        self.camera_toolbar.update()
        
    def initialize_camera(self):
        """Initialize the camera if available"""
        try:
            self.camera = Picamera2()
            
            # Configure camera with more comprehensive settings
            preview_config = self.camera.create_preview_configuration(
                main={"size": (1920, 1080), "format": "RGB888"},
                lores={"size": (640, 360), "format": "YUV420"},
                display="lores"
            )
            
            still_config = self.camera.create_still_configuration(
                main={"size": (1920, 1080), "format": "RGB888"},
                lores={"size": (640, 360), "format": "YUV420"}
            )
            
            self.camera.configure(preview_config)
            
            # Set initial camera controls
            exposure_time = int(self.exposure_var.get())
            self.camera.set_controls({
                "ExposureTime": exposure_time,
                "AwbEnable": True,
                "AeEnable": False  # Manual exposure
            })
            
            self.camera.start()
            self.camera_active = True
            self.status_var.set(f"Camera initialized with exposure {exposure_time}μs")
            
            # Start camera update thread
            self.camera_thread = threading.Thread(target=self._camera_update_thread)
            self.camera_thread.daemon = True
            self.camera_thread.start()
            
            # Show camera window
            self.camera_window.deiconify()
            
        except Exception as e:
            self.status_var.set(f"Camera initialization failed: {str(e)}")
            self.camera_active = False

    def generate_pattern(self):
        """Generate pattern based on current settings"""
        try:
            # Check if we have a target image
            if self.target is None:
                self.status_var.set("No target image loaded")
                return
            
            # Get parameters
            iterations = int(self.iterations_var.get())
            algorithm = self.algorithm_var.get()
            tolerance = float(self.tolerance_var.get())
            
            # Generate phase-only pattern
            self.status_var.set("Generating pattern...")
            
            # Start a thread for pattern generation
            threading.Thread(target=self._generate_pattern_thread, 
                           args=(iterations, algorithm, tolerance)).start()
            
        except Exception as e:
            self.status_var.set(f"Error generating pattern: {str(e)}")
            traceback.print_exc()
    
    def _generate_pattern_thread(self, iterations, algorithm, tolerance):
        """Thread function for pattern generation"""
        try:
            # Generate input beam
            self.generate_input_beam()
            
            # Generate phase pattern
            self.generate_phase_pattern(iterations, algorithm, tolerance)
            
            # Apply phase shift if needed
            if float(self.phase_shift_x_var.get()) != 0.0 or float(self.phase_shift_y_var.get()) != 0.0:
                self.apply_phase_shift()
            
            # Update preview
            self.root.after(0, self.update_preview)
            self.root.after(0, lambda: self.status_var.set("Pattern generation complete"))
            
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Error in pattern generation: {str(e)}"))
            traceback.print_exc()
    
    def generate_input_beam(self):
        """Generate input beam profile based on selected type with enhanced handling"""
        # Get beam width factor
        beam_width = float(self.beam_width_var.get())
        
        # Create normalized coordinates
        x_norm = self.X / (self.padded_width * self.dx / 2)
        y_norm = self.Y / (self.padded_height * self.dx / 2)
        r = np.sqrt(x_norm**2 + y_norm**2)
        
        # Generate beam profile based on selected type
        if self.beam_type_var.get() == "Gaussian":
            # Gaussian beam
            self.input_beam = np.exp(-(r**2) / (2 * beam_width**2))
            
        elif self.beam_type_var.get() == "Super Gaussian":
            # Super Gaussian beam (higher order for flatter top)
            order = 4  # Higher order for flatter top
            self.input_beam = np.exp(-(r**2)**order / (2 * beam_width**2))
            
        elif self.beam_type_var.get() == "Top Hat":
            # Top hat beam
            self.input_beam = np.zeros_like(r)
            self.input_beam[r <= beam_width] = 1.0
            
            # Apply smooth edge
            edge_width = 0.1
            edge_region = (r > beam_width) & (r <= beam_width + edge_width)
            self.input_beam[edge_region] = 0.5 * (1 + np.cos(np.pi * (r[edge_region] - beam_width) / edge_width))
            
        elif self.beam_type_var.get() == "Bessel":
            # Bessel beam (approximation using Bessel function of first kind)
            k_r = 5.0 * beam_width  # Radial wave number
            self.input_beam = scipy.special.j0(k_r * r)**2
            
        elif self.beam_type_var.get() == "LG01":
            # Laguerre-Gaussian (LG01) mode
            self.input_beam = (r**2 / beam_width**2) * np.exp(-(r**2) / (2 * beam_width**2))
            
        elif self.beam_type_var.get() == "Custom" and self.custom_input_beam is not None:
            # Use custom beam profile
            self.input_beam = self.custom_input_beam
            
        else:
            # Default to uniform illumination
            self.input_beam = np.ones_like(r)
        
        # Normalize input beam
        self.input_beam = self.input_beam / np.max(self.input_beam)
        
        # Apply amplitude modulation if in combined mode
        if self.mode_var.get() == "Combined":
            # Apply amplitude coupling
            amplitude_coupling = float(self.amp_coupling_var.get())
            self.input_beam = self.input_beam * (1.0 - amplitude_coupling * (1.0 - self.input_beam))
    
    def generate_phase_pattern(self, iterations, algorithm='gs', tolerance=1e-6):
        """Generate phase-only pattern using selected algorithm with multi-scale optimization"""
        # Get parameters
        use_multiscale = self.use_multiscale_var.get()
        scale_levels = int(self.scale_levels_var.get()) if use_multiscale else 1
        use_phase_compensation = self.use_phase_compensation_var.get()
        phase_levels = int(self.phase_levels_var.get())
        use_adaptive_signal = self.use_adaptive_signal_var.get()
        
        # Prepare target image
        # Ensure target is padded to match SLM dimensions
        if self.target.shape != (self.padded_height, self.padded_width):
            # Center the target in a padded array
            padded_target = np.zeros((self.padded_height, self.padded_width))
            y_offset = (self.padded_height - self.target.shape[0]) // 2
            x_offset = (self.padded_width - self.target.shape[1]) // 2
            padded_target[y_offset:y_offset+self.target.shape[0], 
                         x_offset:x_offset+self.target.shape[1]] = self.target
            self.target = padded_target
        
        # Normalize target intensity
        self.target_intensity = self.target / np.max(self.target)
        
        # Initialize error history
        self.error_history = []
        
        # Initialize phase with random values
        initial_phase = 2 * np.pi * np.random.rand(self.padded_height, self.padded_width) - np.pi
        
        # Multi-scale optimization
        if use_multiscale and scale_levels > 1:
            # Start with lowest resolution
            current_phase = self._multiscale_optimization(initial_phase, algorithm, iterations, tolerance, scale_levels)
        else:
            # Single scale optimization
            current_phase = self._run_optimization(initial_phase, algorithm, iterations, tolerance)
        
        # Apply phase quantization if enabled
        if use_phase_compensation:
            current_phase = self._apply_phase_quantization(current_phase, phase_levels)
        
        # Store the final phase pattern
        self.phase_pattern = current_phase
        
        # Convert phase to grayscale for display
        self.pattern = self.phase_to_grayscale(current_phase)
        
        # Calculate the simulated reconstruction
        self.calculate_reconstruction()
    
    def _multiscale_optimization(self, initial_phase, algorithm, iterations, tolerance, scale_levels):
        """Perform multi-scale optimization to avoid local minima"""
        current_phase = initial_phase
        
        # Calculate scale factors for each level
        scale_factors = [2**i for i in range(scale_levels-1, -1, -1)]
        
        for level, scale in enumerate(scale_factors):
            self.status_var.set(f"Multi-scale optimization: level {level+1}/{scale_levels}")
            
            # Downsample target and phase
            if scale > 1:
                h_scaled = self.padded_height // scale
                w_scaled = self.padded_width // scale
                
                # Downsample target
                target_scaled = cv2.resize(self.target_intensity, (w_scaled, h_scaled), 
                                         interpolation=cv2.INTER_AREA)
                
                # Downsample phase
                phase_scaled = cv2.resize(current_phase, (w_scaled, h_scaled), 
                                        interpolation=cv2.INTER_LINEAR)
                
                # Store original target temporarily
                original_target = self.target_intensity
                self.target_intensity = target_scaled
                
                # Run optimization at this scale
                iterations_at_scale = max(5, iterations // (2 * (scale_levels - level)))
                phase_scaled = self._run_optimization(phase_scaled, algorithm, iterations_at_scale, tolerance)
                
                # Upsample result to full resolution
                current_phase = cv2.resize(phase_scaled, (self.padded_width, self.padded_height), 
                                         interpolation=cv2.INTER_LINEAR)
                
                # Restore original target
                self.target_intensity = original_target
            else:
                # Final full-resolution optimization
                iterations_at_scale = iterations
                current_phase = self._run_optimization(current_phase, algorithm, iterations_at_scale, tolerance)
        
        return current_phase
    
    def _run_optimization(self, initial_phase, algorithm, iterations, tolerance):
        """Run the selected optimization algorithm"""
        # Initialize variables
        current_phase = initial_phase.copy()
        current_error = float('inf')
        
        # Create signal region mask for MRAF algorithm
        if algorithm == 'mraf':
            if self.use_adaptive_signal_var.get():
                self.signal_region_mask = self._create_adaptive_signal_region()
            else:
                # Use fixed signal region based on ratio
                signal_ratio = float(self.signal_region_ratio_var.get())
                self.signal_region_mask = self._create_signal_region(signal_ratio)
        
        # Get mixing parameter for MRAF
        if algorithm == 'mraf':
            mixing_parameter = float(self.mixing_parameter_var.get())
        
        # Main optimization loop
        for i in range(iterations):
            # Create complex field with phase only
            slm_field = np.exp(1j * current_phase)
            
            # Apply input beam profile
            if self.input_beam is not None:
                slm_field = slm_field * np.sqrt(self.input_beam)
            
            # Perform FFT to get far-field
            shifted_field = np.fft.ifftshift(slm_field)
            fft_field = np.fft.fft2(shifted_field)
            far_field = np.fft.fftshift(fft_field)
            
            # Calculate current intensity
            current_intensity = np.abs(far_field)**2
            
            # Calculate error (NMSE)
            mse = np.mean((current_intensity - self.target_intensity)**2)
            current_error = mse / np.mean(self.target_intensity**2)
            self.error_history.append(current_error)
            
            # Check for convergence
            if current_error < tolerance:
                break
            
            # Apply algorithm-specific modifications
            if algorithm == 'gs':
                # Gerchberg-Saxton algorithm
                # Replace amplitude in far field while keeping phase
                far_field_phase = np.angle(far_field)
                modified_far_field = np.sqrt(self.target_intensity) * np.exp(1j * far_field_phase)
                
            elif algorithm == 'mraf':
                # Mixed-Region Amplitude Freedom algorithm
                far_field_phase = np.angle(far_field)
                
                # Apply mixing in signal region, keep phase only in noise region
                modified_far_field = far_field.copy()
                
                # Signal region: mix target amplitude with current amplitude
                modified_far_field[self.signal_region_mask] = (
                    (1 - mixing_parameter) * np.sqrt(self.target_intensity[self.signal_region_mask]) + 
                    mixing_parameter * np.abs(far_field[self.signal_region_mask])
                ) * np.exp(1j * far_field_phase[self.signal_region_mask])
                
                # Noise region: keep complex amplitude
                # (already set in the copy operation)
            
            # Inverse FFT to get back to SLM plane
            shifted_modified = np.fft.ifftshift(modified_far_field)
            ifft_field = np.fft.ifft2(shifted_modified)
            slm_plane_field = np.fft.fftshift(ifft_field)
            
            # Extract phase for next iteration
            new_phase = np.angle(slm_plane_field)
            
            # Apply input beam constraint if available
            if self.input_beam is not None:
                # Only update phase where input beam is significant
                beam_mask = self.input_beam > 0.01
                current_phase[beam_mask] = new_phase[beam_mask]
            else:
                current_phase = new_phase
        
        # Return the optimized phase pattern
        return current_phase
    
    def _create_signal_region(self, ratio):
        """Create signal region mask based on fixed ratio"""
        # Create empty mask
        mask = np.zeros_like(self.target_intensity, dtype=bool)
        
        # Find non-zero target pixels
        target_nonzero = self.target_intensity > 0.01 * np.max(self.target_intensity)
        
        # Calculate area to cover
        total_pixels = np.sum(target_nonzero)
        signal_pixels = int(ratio * self.padded_width * self.padded_height)
        
        # If target has fewer non-zero pixels than signal region, use target
        if total_pixels <= signal_pixels:
            mask[target_nonzero] = True
        else:
            # Sort pixels by intensity
            y_indices, x_indices = np.where(target_nonzero)
            intensities = self.target_intensity[y_indices, x_indices]
            sorted_indices = np.argsort(intensities)[::-1]  # Sort in descending order
            
            # Take top pixels to fill signal region
            for i in range(min(signal_pixels, len(sorted_indices))):
                idx = sorted_indices[i]
                mask[y_indices[idx], x_indices[idx]] = True
        
        return mask
    
    def _create_adaptive_signal_region(self):
        """Create adaptive signal region based on target characteristics"""
        # Start with threshold-based approach
        threshold = 0.1  # Initial threshold
        mask = self.target_intensity > threshold * np.max(self.target_intensity)
        
        # Calculate the current signal region ratio
        current_ratio = np.sum(mask) / (self.padded_width * self.padded_height)
        
        # Target ratio based on image complexity
        # More complex images need larger signal regions
        nonzero_ratio = np.sum(self.target_intensity > 0.01) / (self.padded_width * self.padded_height)
        entropy = -np.sum(self.target_intensity * np.log2(self.target_intensity + 1e-10))
        normalized_entropy = entropy / np.log2(self.padded_width * self.padded_height)
        
        # Adaptive target ratio based on entropy and non-zero ratio
        target_ratio = 0.2 + 0.3 * normalized_entropy + 0.2 * nonzero_ratio
        target_ratio = min(0.7, max(0.1, target_ratio))  # Clamp between 0.1 and 0.7
        
        # Adjust threshold to achieve target ratio
        max_iterations = 20
        for _ in range(max_iterations):
            if abs(current_ratio - target_ratio) < 0.01:
                break
                
            if current_ratio > target_ratio:
                threshold *= 1.2  # Increase threshold to reduce signal region
            else:
                threshold *= 0.8  # Decrease threshold to increase signal region
                
            mask = self.target_intensity > threshold * np.max(self.target_intensity)
            current_ratio = np.sum(mask) / (self.padded_width * self.padded_height)
        
        # Apply morphological operations to clean up the mask
        mask = mask.astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask.astype(bool)
    
    def _apply_phase_quantization(self, phase, levels):
        """Apply phase quantization with error diffusion for better results"""
        # Calculate the phase step
        phase_step = 2 * np.pi / levels
        
        # Normalize phase to [0, 2π)
        phase_norm = (phase + np.pi) % (2 * np.pi)
        
        # Initialize quantized phase and error
        quantized_phase = np.zeros_like(phase_norm)
        error = np.zeros_like(phase_norm)
        
        # Apply error diffusion (Floyd-Steinberg dithering)
        height, width = phase_norm.shape
        for y in range(height):
            for x in range(width):
                # Get current pixel with accumulated error
                old_phase = phase_norm[y, x] + error[y, x]
                
                # Quantize
                new_phase = phase_step * round(old_phase / phase_step)
                quantized_phase[y, x] = new_phase
                
                # Calculate error
                quant_error = old_phase - new_phase
                
                # Distribute error to neighboring pixels
                if x + 1 < width:
                    error[y, x + 1] += quant_error * 7 / 16
                if y + 1 < height:
                    if x > 0:
                        error[y + 1, x - 1] += quant_error * 3 / 16
                    error[y + 1, x] += quant_error * 5 / 16
                    if x + 1 < width:
                        error[y + 1, x + 1] += quant_error * 1 / 16
        
        # Shift back to [-π, π)
        return (quantized_phase - np.pi) % (2 * np.pi) - np.pi
    
    def apply_phase_shift(self):
        """Apply the phase shift and regenerate the pattern"""
        if not hasattr(self, 'phase_pattern'):
            self.status_var.set("No pattern to shift")
            return
        
        try:
            # Get shift values
            shift_x = float(self.phase_shift_x_var.get())
            shift_y = float(self.phase_shift_y_var.get())
            
            # Create normalized coordinates
            x_norm = np.linspace(-1, 1, self.padded_width)
            y_norm = np.linspace(-1, 1, self.padded_height)
            X_norm, Y_norm = np.meshgrid(x_norm, y_norm)
            
            # Calculate phase ramp (linear phase shift)
            phase_ramp = 2 * np.pi * (shift_x * X_norm + shift_y * Y_norm)
            
            # Apply phase shift to the existing phase pattern
            shifted_phase = self.phase_pattern + phase_ramp
            
            # Wrap phase to [-π, π)
            shifted_phase = (shifted_phase + np.pi) % (2 * np.pi) - np.pi
            
            # Update pattern
            self.phase_pattern = shifted_phase
            self.pattern = self.phase_to_grayscale(shifted_phase)
            
            # Recalculate reconstruction
            self.calculate_reconstruction()
            
            # Update preview
            self.update_preview()
            
            self.status_var.set(f"Applied phase shift: X={shift_x}, Y={shift_y}")
            
        except Exception as e:
            self.status_var.set(f"Error applying phase shift: {str(e)}")
            traceback.print_exc()
    
    def calculate_reconstruction(self):
        """Calculate the simulated reconstruction using FFT-based diffraction"""
        if not hasattr(self, 'phase_pattern'):
            return
        
        # Create complex field with phase only
        slm_field = np.exp(1j * self.phase_pattern)
        
        # Apply input beam profile if available
        if hasattr(self, 'input_beam') and self.input_beam is not None:
            slm_field = slm_field * np.sqrt(self.input_beam)
        
        # Perform FFT-based diffraction simulation
        shifted_field = np.fft.ifftshift(slm_field)
        fft_field = np.fft.fft2(shifted_field)
        image_field = np.fft.fftshift(fft_field)
        
        # Calculate intensity
        self.reconstruction = np.abs(image_field)**2
        
        # Apply logarithmic scaling for better visualization
        # Add small constant to avoid log(0)
        self.log_reconstruction = np.log10(self.reconstruction + 1e-10)
    
    def phase_to_grayscale(self, phase):
        """Convert phase [-π, π) to grayscale [0, 255]"""
        # Map [-π, π) to [0, 255]
        return np.round(((phase + np.pi) / (2 * np.pi)) * 255).astype(np.uint8)
    
    def grayscale_to_phase(self, grayscale):
        """Convert grayscale [0, 255] to phase [-π, π)"""
        # Map [0, 255] to [-π, π)
        return (grayscale / 255.0) * 2 * np.pi - np.pi

    def update_preview(self):
        """Update the preview displays"""
        try:
            # Update pattern preview
            if hasattr(self, 'pattern') and self.pattern is not None:
                self.pattern_im.set_data(self.pattern)
                self.pattern_ax.set_title('SLM Pattern (Phase)')
            
            # Update target preview
            if hasattr(self, 'target') and self.target is not None:
                self.target_im.set_data(self.target)
                self.target_ax.set_title('Target Image')
            
            # Update reconstruction preview
            if hasattr(self, 'log_reconstruction') and self.log_reconstruction is not None:
                self.recon_im.set_data(self.log_reconstruction)
                self.recon_ax.set_title('Simulated Reconstruction (log scale)')
                
                # Update colorbar range for better visualization
                vmin = np.min(self.log_reconstruction)
                vmax = np.max(self.log_reconstruction)
                self.recon_im.set_clim(vmin, vmax)
            
            # Update error plot
            if hasattr(self, 'error_history') and self.error_history:
                self.error_line.set_data(range(len(self.error_history)), self.error_history)
                self.error_ax.relim()
                self.error_ax.autoscale_view()
                self.error_ax.set_title(f'Final Error: {self.error_history[-1]:.2e}')
            
            # Redraw canvas
            self.canvas.draw()
            
        except Exception as e:
            self.status_var.set(f"Error updating preview: {str(e)}")
            traceback.print_exc()
    
    def save_pattern(self):
        """Save generated pattern to file"""
        try:
            if not hasattr(self, 'pattern') or self.pattern is None:
                self.status_var.set("No pattern to save")
                return
            
            # Generate default filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            default_name = f"pattern_{timestamp}.png"
            
            # Use zenity file dialog
            cmd = ['zenity', '--file-selection', '--save',
                   '--filename=' + default_name,
                   '--file-filter=Images | *.png *.jpg *.jpeg *.bmp *.tif *.tiff',
                   '--title=Save Pattern',
                   '--confirm-overwrite']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.status_var.set("Pattern save cancelled")
                return
                
            save_path = result.stdout.strip()
            if not save_path:
                return
                
            # Add extension if not present
            if not save_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                save_path += '.png'
            
            # Save the pattern
            cv2.imwrite(save_path, self.pattern)
            self.status_var.set(f"Pattern saved to: {os.path.basename(save_path)}")
            
        except Exception as e:
            self.status_var.set(f"Error saving pattern: {str(e)}")
            traceback.print_exc()
    
    def load_image(self):
        """Load target image from file"""
        try:
            # Use zenity file dialog
            cmd = ['zenity', '--file-selection',
                   '--file-filter=Images | *.png *.jpg *.jpeg *.bmp *.tif *.tiff',
                   '--title=Select Target Image']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.status_var.set("Image loading cancelled")
                return
                
            filename = result.stdout.strip()
            if not filename:
                return
            
            # Load image
            image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                self.status_var.set("Error loading image")
                return
            
            # Resize to SLM dimensions if needed
            if image.shape[0] != self.height or image.shape[1] != self.width:
                image = cv2.resize(image, (self.width, self.height))
            
            # Store target and normalize
            self.target = image.astype(float) / 255.0
            
            # Update preview
            self.update_preview()
            
            self.status_var.set(f"Loaded image: {os.path.basename(filename)}")
            
        except Exception as e:
            self.status_var.set(f"Error loading image: {str(e)}")
            traceback.print_exc()
    
    def load_pattern(self):
        """Load pattern from file"""
        try:
            # Use zenity file dialog
            cmd = ['zenity', '--file-selection', '--file-filter=Images | *.png *.jpg *.jpeg *.bmp *.tif *.tiff',
                   '--title=Select Pattern']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.status_var.set("Pattern loading cancelled")
                return
                
            file_path = result.stdout.strip()
            if not file_path:
                return
                
            # Load the pattern
            pattern = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if pattern is None:
                raise ValueError("Could not load pattern")
                
            # Resize if necessary
            if pattern.shape != (self.height, self.width):
                pattern = cv2.resize(pattern, (self.width, self.height))
            
            # Store pattern
            self.pattern = pattern
            # Convert grayscale to phase [-π to π)
            self.slm_phase = (pattern.astype(float) / 255.0 * 2 * np.pi - np.pi)
            
            # Update preview
            self.update_preview()
            self.status_var.set(f"Pattern loaded from: {os.path.basename(file_path)}")
            
        except Exception as e:
            self.status_var.set(f"Error loading pattern: {str(e)}")
            traceback.print_exc()
    
    def send_to_slm(self):
        """Send pattern to SLM via HDMI"""
        if not hasattr(self, 'pattern') or self.pattern is None:
            self.status_var.set("No pattern to display. Generate or load a pattern first.")
            return
        
        try:
            # Create a thread for SLM display
            self.slm_thread = threading.Thread(target=self._display_slm_pattern)
            self.slm_thread.daemon = True  # Thread will be terminated when main program exits
            self.slm_thread.start()
            self.status_var.set("Pattern sent to SLM. Press ESC in SLM window to close.")
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            print(f"Detailed error: {str(e)}")
            # Try to recover display
            pygame.display.quit()
            pygame.display.init()
    
    def _display_slm_pattern(self):
        """Internal method to handle SLM display in a separate thread"""
        try:
            # Force reinitialize pygame display
            if pygame.display.get_init():
                pygame.display.quit()
            pygame.display.init()
            
            # Set SDL environment variables for display control
            os.environ['SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS'] = '0'
            
            # Get display info
            print(f"Number of displays: {pygame.display.get_num_displays()}")
            for i in range(pygame.display.get_num_displays()):
                info = pygame.display.get_desktop_sizes()[i]
                print(f"Display {i}: {info}")
            
            # Create window on second display
            slm_window = pygame.display.set_mode(
                (self.width, self.height),
                pygame.NOFRAME,
                display=1
            )
            
            # Create and show pattern
            pattern_surface = pygame.Surface((self.width, self.height), depth=8)
            pattern_surface.set_palette([(i, i, i) for i in range(256)])
            pygame.surfarray.pixels2d(pattern_surface)[:] = self.pattern.T
            
            # Clear window and display pattern
            slm_window.fill((0, 0, 0))
            slm_window.blit(pattern_surface, (0, 0))
            pygame.display.flip()
            
            print("Pattern displayed. Press ESC to close.")
            
            # Event loop in separate thread
            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        running = False
                        break
                    elif event.type == pygame.QUIT:
                        running = False
                        break
                time.sleep(0.01)  # Small sleep to prevent high CPU usage
            
            # Cleanup
            pygame.display.quit()
            pygame.display.init()
            
        except Exception as e:
            print(f"Error in SLM display thread: {str(e)}")
            # Try to recover display
            pygame.display.quit()
            pygame.display.init()
    
    def start_camera(self):
        """Start camera preview"""
        if not self.camera_active:
            try:
                self.initialize_camera()
            except Exception as e:
                self.status_var.set(f"Error starting camera: {str(e)}")
                return
        
        # Show camera window
        self.camera_window.deiconify()
        
        # Start update thread if not already running
        if not hasattr(self, 'camera_thread') or not self.camera_thread.is_alive():
            self.camera_paused = False
            self.camera_thread = threading.Thread(target=self._camera_update_thread)
            self.camera_thread.daemon = True
            self.camera_thread.start()
    
    def stop_camera(self):
        """Stop camera preview"""
        self.camera_paused = True
        self.camera_window.withdraw()
    
    def toggle_pause_camera(self):
        """Toggle camera pause state"""
        if not self.camera_active:
            self.status_var.set("Camera is not active")
            return
            
        self.camera_paused = not self.camera_paused
        if self.camera_paused:
            self.status_var.set("Camera paused")
        else:
            self.status_var.set("Camera resumed")
    
    def capture_frame(self):
        """Capture current camera frame as target"""
        if not self.camera_active or self.last_frame is None:
            self.status_var.set("Camera not active or no frame available")
            return
        
        # Convert to grayscale if needed
        if len(self.last_frame.shape) == 3:
            frame = cv2.cvtColor(self.last_frame, cv2.COLOR_RGB2GRAY)
        else:
            frame = self.last_frame.copy()
        
        # Resize to SLM dimensions
        frame = cv2.resize(frame, (self.width, self.height))
        
        # Store as target
        self.target = frame
        
        # Update preview
        self.update_preview()
        
        self.status_var.set("Captured frame as target")
    
    def set_camera_exposure(self):
        """Set camera exposure time"""
        try:
            exposure = int(self.exposure_var.get())
            if self.camera_active:
                self.camera.set_controls({"ExposureTime": exposure})
                self.status_var.set(f"Camera exposure set to {exposure}")
        except Exception as e:
            self.status_var.set(f"Error setting exposure: {str(e)}")
    
    def _camera_update_thread(self):
        """Thread function for updating camera preview"""
        try:
            while self.camera_active and hasattr(self, 'camera'):
                if not self.camera_paused:
                    # Capture frame
                    frame = self.camera.capture_array()
                    
                    # Store frame
                    self.last_frame = frame
                    
                    # Update preview
                    self.camera_im.set_data(frame)
                    self.camera_canvas.draw_idle()
                
                # Sleep to limit frame rate
                time.sleep(0.033)  # ~30 FPS
                
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Camera error: {str(e)}"))
    
    def update_wavelength(self):
        """Update wavelength parameter"""
        try:
            wavelength_nm = float(self.wavelength_var.get())
            self.wavelength = wavelength_nm * 1e-9
            self.k = 2 * np.pi / self.wavelength
            self.status_var.set(f"Wavelength updated to {wavelength_nm} nm")
        except ValueError:
            self.status_var.set("Invalid wavelength value")
    
    def _on_algorithm_change(self, *args):
        """Handle algorithm selection change"""
        if self.algorithm_var.get() == "mraf":
            self.mraf_frame.grid()
        else:
            self.mraf_frame.grid_remove()
    
    def _on_beam_type_change(self, *args):
        """Handle beam type selection change"""
        if self.beam_type_var.get() == "Custom":
            # Open file dialog to load custom beam profile
            file_path = filedialog.askopenfilename(
                title="Select Custom Beam Profile",
                filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"), ("All files", "*.*")]
            )
            
            if file_path:
                try:
                    # Load beam profile
                    beam = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    
                    # Resize to padded dimensions
                    beam = cv2.resize(beam, (self.padded_width, self.padded_height))
                    
                    # Normalize
                    self.custom_input_beam = beam / np.max(beam)
                    
                    self.status_var.set(f"Loaded custom beam profile: {os.path.basename(file_path)}")
                except Exception as e:
                    self.status_var.set(f"Error loading beam profile: {str(e)}")
                    self.beam_type_var.set("Gaussian")  # Revert to default
    
    def quit_application(self):
        """Clean up and quit"""
        # Stop camera if active
        if self.camera_active and hasattr(self, 'camera'):
            try:
                self.camera.stop()
            except:
                pass
                
        # Close SLM window if open
        if hasattr(self, 'slm_window') and self.slm_window is not None:
            try:
                self.slm_window.close()
            except:
                pass
                
        # Close camera window if open
        if hasattr(self, 'camera_window') and self.camera_window is not None:
            try:
                self.camera_window.withdraw()
            except:
                pass
        
        # Close pygame
        pygame.quit()
                
        # Destroy the root window and exit
        self.root.destroy()
        
    def save_camera_image(self):
        """Save the current camera frame to a file"""
        if self.last_frame is None:
            self.status_var.set("No camera frame to save")
            return
            
        try:
            # Generate default filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            default_name = f"camera_{timestamp}.png"
            
            # Use zenity file dialog
            cmd = ['zenity', '--file-selection', '--save',
                   '--filename=' + default_name,
                   '--file-filter=Images | *.png *.jpg *.jpeg',
                   '--title=Save Camera Image',
                   '--confirm-overwrite']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.status_var.set("Save cancelled")
                return
                
            file_path = result.stdout.strip()
            if not file_path:
                return
                
            # Add extension if not present
            if not file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path += '.png'
            
            # Save the image
            cv2.imwrite(file_path, self.last_frame)
            self.status_var.set(f"Camera image saved to: {file_path}")
            
        except Exception as e:
            self.status_var.set(f"Error saving camera image: {str(e)}")
            traceback.print_exc()
    
# Main entry point
if __name__ == "__main__":
    app = AdvancedPatternGenerator()
    app.root.mainloop()
