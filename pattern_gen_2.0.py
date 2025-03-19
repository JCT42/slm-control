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
- Integrated camera control for direct feedback

"""

import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backends.backend_agg import FigureCanvasAgg
import threading
import time
import os
import subprocess
import pygame
from tqdm import tqdm
import scipy.special
import traceback
import datetime
from PIL import Image, ImageTk
from matplotlib.figure import Figure
from typing import Dict, List, Tuple, Optional, Union, Any
import json

# Check if PiCamera2 is available
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("PiCamera2 not available. Install with: pip install picamera2")

class AdvancedPatternGenerator:
    def __init__(self):
        """Initialize the advanced pattern generator with extended features"""
        # Initialize pygame
        pygame.init()
        
        # SLM dimensions
        self.width = 800
        self.height = 600
        self.pixel_pitch = 32e-6  # 32 μm
        
        # Default wavelength
        self.wavelength = 650e-9
        
        # Modulation parameters
        self.modulation_mode = "Phase"  # "Phase", "Amplitude", or "Combined"
        
        # Initialize pattern arrays
        self.pattern = np.zeros((self.height, self.width), dtype=np.uint8)
        self.slm_phase = np.zeros((self.height, self.width), dtype=float)
        self.target_intensity = np.zeros((self.height, self.width), dtype=float)
        
        # Create the GUI
        self.create_gui()
        
        # Initialize camera controller
        self.camera_controller = None
        self.camera_gui = None
        
    def create_gui(self):
        """Create the GUI for the pattern generator"""
        # Create main window
        self.root = tk.Tk()
        self.root.title("Advanced Pattern Generator for Sony LCX016AL-6 SLM")
        self.root.geometry("1200x800")
        
        # Main frame
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a notebook for tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create tabs
        self.pattern_frame = ttk.Frame(self.notebook)
        self.settings_frame = ttk.Frame(self.notebook)
        self.camera_frame = ttk.Frame(self.notebook)
        
        self.notebook.add(self.pattern_frame, text="Pattern Generator")
        self.notebook.add(self.settings_frame, text="Settings")
        self.notebook.add(self.camera_frame, text="Camera")
        
        # Create status bar
        status_frame = ttk.Frame(self.main_frame)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=5)
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(fill=tk.X)
        
        # Create pattern generator UI
        self.create_pattern_ui()
        
        # Create camera controller UI
        self.create_camera_ui()
        
        # Bind ESC key to quit
        self.root.bind('<Escape>', lambda e: self.quit_application())
        # Also bind using protocol for window close button
        self.root.protocol("WM_DELETE_WINDOW", self.quit_application)
        
    def create_pattern_ui(self):
        """Create pattern generator UI"""
        # Create canvas for scrolling
        self.canvas = tk.Canvas(self.pattern_frame)
        scrollbar = ttk.Scrollbar(self.pattern_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        # Configure scrolling
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw", width=self.canvas.winfo_width())
        
        # Configure canvas scrolling
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        
        # Configure canvas resize
        self.canvas.bind('<Configure>', self._on_canvas_configure)
        
        # Create frames for different sections
        self.control_frame = ttk.LabelFrame(self.scrollable_frame, text="Controls", padding="10")
        self.control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.preview_frame = ttk.LabelFrame(self.scrollable_frame, text="Pattern Preview", padding="10")
        self.preview_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Add status bar
        self.status_bar = ttk.Label(self.scrollable_frame, textvariable=self.status_var)
        self.status_bar.pack(fill=tk.X, padx=5, pady=5)
        
        # Create controls
        self.create_controls()
        
        # Create preview area
        self.create_preview()
        
        # Add phase shift controls
        self.create_phase_shift_controls()
        
    def _on_canvas_configure(self, event):
        """Handle canvas resize"""
        width = event.width
        self.canvas.itemconfig(1, width=width)
        
    def create_controls(self):
        """Create parameter control widgets"""
        # Add buttons frame
        buttons_frame = ttk.Frame(self.control_frame)
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
        
        # Create a notebook for tabbed parameter groups
        param_notebook = ttk.Notebook(self.control_frame)
        param_notebook.pack(fill=tk.X, pady=5)
        
        # Modulation mode tab
        mode_frame = ttk.Frame(param_notebook)
        param_notebook.add(mode_frame, text="Modulation")
        
        # Modulation mode selection
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
        
        # Generate button
        ttk.Button(self.control_frame, text="Generate Pattern", command=self.generate_pattern).pack(pady=10)
        
    def create_phase_shift_controls(self):
        """Create controls for adjusting phase shift to avoid zero-order diffraction"""
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
        
        # Help text
        help_text = "Shift your image away from the zero-order (undiffracted) light by adding a linear phase ramp.\n"
        help_text += "Positive values shift right/down, negative values shift left/up."
        help_label = ttk.Label(phase_shift_frame, text=help_text, wraplength=500)
        help_label.grid(row=2, column=0, columnspan=4, padx=5, pady=5)
    
    def apply_phase_shift(self):
        """Apply the phase shift and regenerate the pattern"""
        try:
            # Update phase shift parameters
            self.phase_shift_x = float(self.phase_shift_x_var.get())
            self.phase_shift_y = float(self.phase_shift_y_var.get())
            
            # If we have a pattern already, apply the phase shift directly
            if hasattr(self, 'pattern') and hasattr(self, 'slm_phase'):
                # Create coordinate grids for the SLM plane
                y, x = np.indices((self.height, self.width))
                
                # Normalize coordinates to [-0.5, 0.5] range
                x_norm = (x - self.width // 2) / self.width
                y_norm = (y - self.height // 2) / self.height
                
                # Calculate linear phase ramp
                phase_ramp = 2 * np.pi * (self.phase_shift_x * x_norm + self.phase_shift_y * y_norm)
                
                # Apply phase ramp to existing SLM phase
                self.slm_phase = np.mod(self.slm_phase + phase_ramp, 2 * np.pi) - np.pi
                
                # Convert to pattern (8-bit grayscale)
                gamma = float(self.gamma_var.get())
                normalized_phase = (self.slm_phase + np.pi) / (2 * np.pi)
                self.pattern = (normalized_phase ** gamma * 255).astype(np.uint8)
                
                # Update reconstruction preview
                # Create full-sized phase with the shift
                padded_phase = np.zeros((self.padded_height, self.padded_width))
                start_y = (self.padded_height - self.height) // 2
                end_y = start_y + self.height
                start_x = (self.padded_width - self.width) // 2
                end_x = start_x + self.width
                padded_phase[start_y:end_y, start_x:end_x] = self.slm_phase
                
                # Calculate reconstruction with shift
                image_field = self.pattern_generator.inverse_propagate(np.exp(1j * padded_phase))
                self.reconstruction = np.abs(image_field)**2
                
                # Normalize reconstruction for display
                if np.max(self.reconstruction) > 0:
                    self.reconstruction = self.reconstruction / np.max(self.reconstruction)
                
                # Update preview
                self.update_preview()
                
                self.status_var.set(f"Phase shift applied: X={self.phase_shift_x}, Y={self.phase_shift_y}")
            else:
                self.status_var.set("Generate a pattern first before applying phase shift")
        except ValueError as e:
            self.status_var.set(f"Invalid phase shift values: {str(e)}")
        except Exception as e:
            self.status_var.set(f"Error applying phase shift: {str(e)}")
            print(f"Detailed error: {str(e)}")
    
    def create_preview(self):
        """Create preview area with matplotlib plots"""
        # Create figure and subplots with 2x2 grid
        self.fig = plt.figure(figsize=(16, 10))
        gs = self.fig.add_gridspec(2, 2)
        
        # Create subplots
        self.ax1 = self.fig.add_subplot(gs[0, 0])  # Target
        self.ax2 = self.fig.add_subplot(gs[0, 1])  # Pattern
        self.ax3 = self.fig.add_subplot(gs[1, 0])  # Reconstruction
        self.ax4 = self.fig.add_subplot(gs[1, 1])  # Error plot
        
        # Create canvas and toolbar
        self.preview_canvas = FigureCanvasTkAgg(self.fig, master=self.preview_frame)
        toolbar = NavigationToolbar2Tk(self.preview_canvas, self.preview_frame)
        toolbar.update()
        
        # Add a safe save button to the preview frame
        safe_save_button = ttk.Button(
            self.preview_frame, 
            text="Safe Save Plot", 
            command=lambda: self.safe_plot_save(self.fig, "pattern_preview")
        )
        safe_save_button.pack(side=tk.BOTTOM, pady=5)
        
        # Initialize empty plots
        self.ax1.set_title('Target Image')
        self.ax1.set_xticks([])
        self.ax1.set_yticks([])
        
        self.ax2.set_title('Generated Pattern')
        self.ax2.set_xticks([])
        self.ax2.set_yticks([])
        
        self.ax3.set_title('Simulated Reconstruction')
        self.ax3.set_xticks([])
        self.ax3.set_yticks([])
        
        self.ax4.set_title('Optimization Error')
        self.ax4.set_xlabel('Iteration')
        self.ax4.set_ylabel('Error')
        self.ax4.grid(True)
        
        # Pack canvas
        self.preview_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initial draw
        self.preview_canvas.draw()
        
    def update_preview(self):
        """Update the preview plots with current patterns and reconstructions"""
        try:
            # Clear axes
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()
            self.ax4.clear()
            
            # Plot target image
            if hasattr(self, 'target'):
                self.ax1.imshow(self.target, cmap='gray')
                self.ax1.set_title('Target')
                self.ax1.set_xticks([])
                self.ax1.set_yticks([])
            
            # Plot current pattern
            if hasattr(self, 'pattern'):
                # Use gray colormap for phase patterns
                if self.modulation_mode == "Phase":
                    self.ax2.imshow(self.pattern, cmap='gray')
                else:
                    self.ax2.imshow(self.pattern, cmap='gray')
                self.ax2.set_title('SLM Pattern')
                self.ax2.set_xticks([])
                self.ax2.set_yticks([])
            
            # Plot simulated reconstruction
            if hasattr(self, 'reconstruction'):
                # Extract central region of reconstruction to match target size
                start_y = (self.padded_height - self.height) // 2
                end_y = start_y + self.height
                start_x = (self.padded_width - self.width) // 2
                end_x = start_x + self.width
                central_recon = self.reconstruction[start_y:end_y, start_x:end_x]
                
                # Display the central region of the reconstruction
                self.ax3.imshow(central_recon, cmap='hot')  # Use hot colormap for intensity
                self.ax3.set_title('Simulated Reconstruction')
                self.ax3.set_xticks([])
                self.ax3.set_yticks([])
            
            # Plot error history if available and enabled
            if hasattr(self, 'error_history') and self.show_error_plot_var.get():
                iterations = range(0, len(self.error_history))
                self.ax4.plot(iterations, self.error_history, 'b-', marker='o')
                self.ax4.set_title('Optimization Error')
                self.ax4.set_xlabel('Iteration')
                self.ax4.set_ylabel('Error')
                self.ax4.grid(True)
                
            # Update canvas
            self.preview_canvas.draw()
            
        except Exception as e:
            self.status_var.set(f"Error updating preview: {str(e)}")
            print(f"Detailed error: {str(e)}")
    
    def generate_pattern(self):
        """Generate pattern based on current settings"""
        try:
            # Get parameters from UI
            iterations = int(self.iterations_var.get())
            gamma = float(self.gamma_var.get())
            algorithm = self.algorithm_var.get()
            tolerance = float(self.tolerance_var.get())
            
            # Check if target image is loaded
            if self.target is None:
                self.status_var.set("Please load a target image first")
                return
                
            # Create padded target array
            self.padded_target = np.zeros((self.padded_height, self.padded_width))
            start_y = (self.padded_height - self.height) // 2
            end_y = start_y + self.height
            start_x = (self.padded_width - self.width) // 2
            end_x = start_x + self.width
            self.padded_target[start_y:end_y, start_x:end_x] = self.target
            
            # Generate pattern based on mode
            if self.modulation_mode == "Phase":
                # Generate phase pattern
                optimized_field, slm_phase, stop_reason = self.generate_phase_pattern(iterations, algorithm, tolerance)
                
            elif self.modulation_mode == "Amplitude":
                # Generate amplitude pattern
                optimized_field, slm_phase, stop_reason = self.generate_amplitude_pattern(iterations, algorithm, tolerance)
                
            else:  # Combined mode
                # Generate combined pattern
                optimized_field, slm_phase, stop_reason = self.generate_combined_pattern(iterations, algorithm, tolerance)
            
            if optimized_field is None:
                return
            
            # Extract central region
            start_y = (self.padded_height - self.height) // 2
            end_y = start_y + self.height
            start_x = (self.padded_width - self.width) // 2
            end_x = start_x + self.width
            central_phase = slm_phase[start_y:end_y, start_x:end_x]
            
            # Generate pattern based on mode
            if self.modulation_mode == "Phase":
                # Extract phase and normalize to [0, 1]
                normalized_phase = (central_phase + np.pi) / (2 * np.pi)
                self.pattern = (normalized_phase ** gamma * 255).astype(np.uint8)
                
            elif self.modulation_mode == "Amplitude":
                # Extract amplitude and normalize
                amplitude = np.abs(optimized_field)
                normalized_amplitude = amplitude / np.max(amplitude)
                self.pattern = (normalized_amplitude ** gamma * 255).astype(np.uint8)
                
            else:  # Combined mode
                # Extract both amplitude and phase
                amplitude = np.abs(optimized_field)
                phase = np.angle(optimized_field)
                
                # Normalize both components
                normalized_amplitude = amplitude / np.max(amplitude)
                normalized_phase = (phase + np.pi) / (2 * np.pi)
                
                # Apply coupling parameters and combine
                amp_component = normalized_amplitude ** self.amplitude_coupling
                phase_component = normalized_phase ** self.phase_coupling
                
                # Weighted sum of components
                combined = (amp_component + phase_component) / 2
                
                # Apply gamma correction and scale to 8-bit
                self.pattern = (combined ** gamma * 255).astype(np.uint8)
            
            # Update the preview with the full field for proper reconstruction
            self.update_preview()
            
            self.status_var.set(f"Pattern generated using {algorithm.upper()} algorithm. Stopped due to: {stop_reason}")
            
        except Exception as e:
            self.status_var.set(f"Error generating pattern: {str(e)}")
            print(f"Detailed error: {str(e)}")
            traceback.print_exc()
    
    def load_pattern(self):
        """Load a pattern from file"""
        try:
            # Use zenity file dialog
            cmd = ['zenity', '--file-selection',
                   '--file-filter=Images | *.png *.jpg *.jpeg *.bmp *.tif *.tiff',
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
            if pattern.shape != (600, 800):
                pattern = cv2.resize(pattern, (800, 600))
            
            self.pattern = pattern
            self.slm_phase = (pattern.astype(float) / 255.0 * 2 * np.pi - np.pi)
            
            # Update preview plots
            self.ax2.clear()
            self.ax2.imshow(pattern, cmap='gray')
            self.ax2.set_title('Loaded Pattern')
            self.ax2.set_xticks([])
            self.ax2.set_yticks([])
            
            self.preview_canvas.draw()
            self.status_var.set(f"Pattern loaded from: {file_path}")
            
        except Exception as e:
            self.status_var.set(f"Error loading pattern: {str(e)}")
    
    def send_to_slm(self):
        """Send pattern to SLM via HDMI-A-2"""
        if not hasattr(self, 'pattern'):
            self.status_var.set("No pattern to display. Generate or load a pattern first.")
            return
        
        try:
            # Set SDL environment variables for display control before initializing pygame
            os.environ['SDL_VIDEO_WINDOW_POS'] = '1280,0'  # Position at main monitor width
            os.environ['SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS'] = '0'
            
            # Create a thread for SLM display
            self.slm_thread = threading.Thread(target=self._display_slm_pattern)
            self.slm_thread.daemon = True  # Thread will be terminated when main program exits
            self.slm_thread.start()
            self.status_var.set("Pattern sent to HDMI-A-2. Press ESC in SLM window to close.")
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            print(f"Detailed error: {str(e)}")
            # Try to recover display
            try:
                pygame.display.quit()
                pygame.display.init()
            except:
                pass

    def _display_slm_pattern(self):
        """Internal method to handle SLM display in a separate thread"""
        slm_window = None
        try:
            # Force reinitialize pygame display system
            pygame.display.quit()
            pygame.init()  # Full initialization to ensure all subsystems are ready
            
            # Get display info
            num_displays = pygame.display.get_num_displays()
            print(f"Number of displays: {num_displays}")
            
            if num_displays < 2:
                print("Warning: Only one display detected. Using current display.")
                display_index = 0
            else:
                display_index = 1  # Use second display
                
            for i in range(num_displays):
                info = pygame.display.get_desktop_sizes()[i]
                print(f"Display {i}: {info}")
            
            # Get the size of the target display
            target_display_size = pygame.display.get_desktop_sizes()[display_index]
            print(f"Using display {display_index} with size {target_display_size}")
            
            # Create window on target display with proper size
            try:
                slm_window = pygame.display.set_mode(
                    target_display_size,
                    pygame.NOFRAME,
                    display=display_index
                )
            except pygame.error:
                # Fallback if display parameter fails
                slm_window = pygame.display.set_mode(
                    target_display_size,
                    pygame.NOFRAME
                )
                # Try to move window to correct position
                pygame.display.set_caption("SLM Pattern")
            
            # Create and show pattern - resize to match display if needed
            pattern_height, pattern_width = self.pattern.shape
            if (pattern_width, pattern_height) != target_display_size:
                print(f"Resizing pattern from {pattern_width}x{pattern_height} to {target_display_size[0]}x{target_display_size[1]}")
                resized_pattern = cv2.resize(self.pattern, target_display_size)
            else:
                resized_pattern = self.pattern
                
            # Create surface with proper depth for grayscale
            pattern_surface = pygame.Surface(target_display_size, depth=8)
            pattern_surface.set_palette([(i, i, i) for i in range(256)])
            
            # Update surface with pattern data
            pygame_array = pygame.surfarray.pixels2d(pattern_surface)
            pygame_array[:] = resized_pattern.T
            del pygame_array  # Release the surface lock
            
            # Clear window and display pattern
            slm_window.fill((0, 0, 0))
            slm_window.blit(pattern_surface, (0, 0))
            pygame.display.flip()
            
            print("Pattern displayed. Press ESC to close.")
            
            # Event loop in separate thread - with improved error handling
            running = True
            while running:
                try:
                    # Check if pygame is still initialized
                    if not pygame.display.get_init():
                        print("Pygame display no longer initialized, exiting event loop")
                        break
                        
                    # Process events safely
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                            running = False
                            break
                        elif event.type == pygame.QUIT:
                            running = False
                            break
                except pygame.error as e:
                    print(f"Pygame error in event loop: {e}")
                    break
                except Exception as e:
                    print(f"Unexpected error in event loop: {e}")
                    break
                    
                # Small sleep to prevent high CPU usage
                time.sleep(0.01)
            
        except Exception as e:
            print(f"Error in SLM display thread: {str(e)}")
            traceback.print_exc()  # Print full traceback for debugging
        finally:
            # Ensure cleanup happens even if there was an error
            try:
                # Only quit if we're still initialized
                if pygame.display.get_init():
                    pygame.display.quit()
                pygame.display.init()
            except Exception as cleanup_error:
                print(f"Error during pygame cleanup: {cleanup_error}")
    
    def quit_application(self):
        """Clean up and quit the application"""
        print("Quitting application...")
        try:
            # Clean up pygame if initialized
            if pygame.display.get_init():
                pygame.display.quit()
            
            # Stop camera if running
            if self.camera_controller is not None and self.camera_controller.is_running:
                self.camera_controller.stop()
            
            # Destroy and quit Tkinter
            self.root.destroy()
            
        except Exception as e:
            print(f"Error in main loop: {str(e)}")
        finally:
            print("Application closed")

    def load_image(self):
        """Load and display target image"""
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
            
            # Read image using cv2
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError("Failed to load image")
            
            # Resize if needed
            if img.shape != (600, 800):
                img = cv2.resize(img, (800, 600))
            
            # Store original image and normalize
            self.target = img.astype(float) / 255.0
            
            # Update preview plots
            self.ax1.clear()
            self.ax1.imshow(self.target, cmap='gray')
            self.ax1.set_title('Target Pattern')
            self.ax1.set_xticks([])
            self.ax1.set_yticks([])
            
            self.ax2.clear()
            self.ax2.set_title('Generated Pattern')
            self.ax2.set_xticks([])
            self.ax2.set_yticks([])
            
            self.ax3.clear()
            self.ax3.set_title('Simulated Reconstruction')
            self.ax3.set_xticks([])
            self.ax3.set_yticks([])
            
            # Update the canvas
            self.preview_canvas.draw()
            
            self.status_var.set(f"Loaded image: {os.path.basename(filename)}")
            
        except Exception as e:
            self.status_var.set(f"Error loading image: {str(e)}")
            print(f"Detailed error: {str(e)}")

    def create_camera_ui(self):
        """Create the camera controller UI"""
        # Initialize camera controller if not already done
        if self.camera_controller is None:
            # Initialize with the full resolution of 1456x1088 to preserve all intensity values
            # Camera uses RGB888 format as specified in camera_controller.py
            self.camera_controller = CameraController(resolution=(1456, 1088))
        
        # Initialize camera GUI
        self.camera_gui = CameraGUI(self.camera_frame, self.camera_controller)

    def save_pattern(self):
        """Save the generated pattern"""
        if not hasattr(self, 'pattern'):
            self.status_var.set("No pattern to save")
            return
            
        try:
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
            
            save_path = result.stdout.strip()
            if not save_path:
                return
            
            # Add extension if not present
            if not save_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                save_path += '.png'
            
            # Save the pattern
            cv2.imwrite(save_path, self.pattern)
            self.status_var.set(f"Pattern saved to: {save_path}")
            
        except Exception as e:
            self.status_var.set(f"Error saving pattern: {str(e)}")

    def safe_plot_save(self, figure, default_filename=None):
        """Safely save a plot by temporarily pausing the camera to avoid OpenCV threading issues"""
        try:
            # Create a file dialog to get save location
            filetypes = [('PNG', '*.png'), ('PDF', '*.pdf'), ('SVG', '*.svg'), ('JPEG', '*.jpg')]
            filename = filedialog.asksaveasfilename(
                title='Save Figure',
                defaultextension=".png",
                filetypes=filetypes,
                initialfile=default_filename or "figure"
            )
            
            if filename:
                figure.savefig(filename)
                self.status_var.set(f"Figure saved to {filename}")
        except Exception as e:
            self.status_var.set(f"Error saving figure: {str(e)}")

    def generate_input_beam(self):
        """Generate input beam profile based on selected type"""
        try:
            beam_type = self.beam_type_var.get()
            beam_width = float(self.beam_width_var.get())
            
            if beam_type == "Custom" and self.custom_input_beam is not None:
                return self.custom_input_beam
            
            # Calculate basic parameters
            sigma_x = self.active_area[0] / (2.355 * beam_width)  # FWHM = 2.355 * sigma
            sigma_y = self.active_area[1] / (2.355 * beam_width)
            r = np.sqrt(self.X**2 + self.Y**2)
            
            if beam_type == "Gaussian":
                beam = np.exp(-self.X**2 / (2 * sigma_x**2) - self.Y**2 / (2 * sigma_y**2))
            
            elif beam_type == "Super Gaussian":
                n = 4  # Super Gaussian order
                beam = np.exp(-(self.X**2 / (2 * sigma_x**2) + self.Y**2 / (2 * sigma_y**2))**n)
            
            elif beam_type == "Top Hat":
                radius = min(self.active_area) / 2
                beam = np.where(r <= radius, 1.0, 0.0)
            
            elif beam_type == "Bessel":
                k_r = 5.0 / sigma_x  # Radial wave number
                beam = np.abs(scipy.special.j0(k_r * r))
            
            elif beam_type == "LG01":
                # Laguerre-Gaussian LG01 mode
                r_squared = self.X**2 / sigma_x**2 + self.Y**2 / sigma_y**2
                phi = np.arctan2(self.Y, self.X)
                beam = r_squared * np.exp(-r_squared/2) * np.exp(1j * phi)
                beam = np.abs(beam)
            
            else:  # Default to Gaussian if something goes wrong
                beam = np.exp(-self.X**2 / (2 * sigma_x**2) - self.Y**2 / (2 * sigma_y**2))
            
            return beam / np.max(beam)
            
        except Exception as e:
            self.status_var.set(f"Error generating input beam: {str(e)}")
            return np.ones((self.padded_height, self.padded_width))

    def _on_beam_type_change(self, *args):
        """Handle input beam type change"""
        if self.beam_type_var.get() == "Custom":
            self.load_custom_beam()
        else:
            # Clear any existing custom beam
            self.custom_input_beam = None
            # Update status
            self.status_var.set(f"Input beam type changed to: {self.beam_type_var.get()}")

    def load_custom_beam(self):
        """Load a custom input beam profile from file"""
        try:
            # Use zenity file dialog
            cmd = ['zenity', '--file-selection',
                   '--file-filter=Images | *.png *.jpg *.jpeg *.bmp *.tif *.tiff',
                   '--title=Select Custom Input Beam Profile']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.status_var.set("Custom beam loading cancelled")
                self.beam_type_var.set("Gaussian")  # Reset to Gaussian
                return
                
            filename = result.stdout.strip()
            if not filename:
                self.beam_type_var.set("Gaussian")  # Reset to Gaussian
                return
            
            # Read image using cv2
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError("Failed to load image")
            
            # Resize to padded dimensions
            img = cv2.resize(img, (self.padded_width, self.padded_height))
            
            # Normalize
            self.custom_input_beam = img.astype(float) / 255.0
            self.status_var.set(f"Custom input beam loaded from: {os.path.basename(filename)}")
            
        except Exception as e:
            self.status_var.set(f"Error loading custom beam: {str(e)}")
            self.beam_type_var.set("Gaussian")  # Reset to Gaussian
            self.custom_input_beam = None

    def update_wavelength(self):
        """Update wavelength and related parameters"""
        try:
            wavelength_nm = float(self.wavelength_var.get())
            self.wavelength = wavelength_nm * 1e-9
            self.k = 2 * np.pi / self.wavelength
            self.status_var.set(f"Wavelength updated to {wavelength_nm} nm")
        except ValueError:
            self.status_var.set("Invalid wavelength value")

    def run(self):
        """Start the GUI application"""
        try:
            self.root.mainloop()
        except Exception as e:
            print(f"Error in main loop: {str(e)}")
        finally:
            self.quit_application()

    def generate_phase_pattern(self, iterations, algorithm='gs', tolerance=1e-6):
        """Generate phase-only pattern using selected algorithm"""
        try:
            # Initialize random phase
            random_phase = np.exp(1j * 2 * np.pi * np.random.rand(self.padded_height, self.padded_width))
            field = random_phase
            
            # Get input beam
            input_beam = self.generate_input_beam()
            
            # Apply input beam to initial field
            initial_field = input_beam * np.exp(1j * np.angle(field))
            
            # Create signal region mask for MRAF if needed
            if algorithm.lower() == 'mraf':
                # Create signal region mask based on the target
                if self.signal_region_mask is None or self.signal_region_mask.shape != self.padded_target.shape:
                    signal_ratio = float(self.signal_region_ratio_var.get())
                    # Create a binary mask where target intensity > threshold
                    threshold = signal_ratio * np.max(self.padded_target)
                    self.signal_region_mask = (self.padded_target > threshold).astype(float)
                    print(f"Created signal region mask with ratio {signal_ratio}, covering {np.mean(self.signal_region_mask)*100:.1f}% of the image")
            else:
                # For GS algorithm, no mask needed
                self.signal_region_mask = None
            
            # Create pattern generator with target intensity
            self.pattern_generator = PatternGenerator(
                target_intensity=self.padded_target,
                signal_region_mask=self.signal_region_mask,
                mixing_parameter=float(self.mixing_parameter_var.get())
            )
            
            # Run optimization with selected algorithm
            optimized_field, self.error_history, stop_reason = self.pattern_generator.optimize(
                initial_field=initial_field,
                algorithm=algorithm,
                max_iterations=iterations,
                tolerance=float(self.tolerance_var.get())
            )
            
            # Get SLM phase pattern
            slm_field = self.pattern_generator.propagate(optimized_field)
            slm_phase = np.angle(slm_field)
            
            # Extract central region
            start_y = (self.padded_height - self.height) // 2
            end_y = start_y + self.height
            start_x = (self.padded_width - self.width) // 2
            end_x = start_x + self.width
            central_phase = slm_phase[start_y:end_y, start_x:end_x]
            
            # Store the original SLM phase for later modification
            self.slm_phase = central_phase.copy()
            
            # Apply phase shift to avoid zero-order diffraction
            if self.phase_shift_x != 0 or self.phase_shift_y != 0:
                # Create coordinate grids for the SLM plane
                y, x = np.indices((self.height, self.width))
                
                # Normalize coordinates to [-0.5, 0.5] range
                x_norm = (x - self.width // 2) / self.width
                y_norm = (y - self.height // 2) / self.height
                
                # Calculate linear phase ramp
                phase_ramp = 2 * np.pi * (self.phase_shift_x * x_norm + self.phase_shift_y * y_norm)
                
                # Add phase ramp to SLM phase
                self.slm_phase = np.mod(self.slm_phase + phase_ramp, 2 * np.pi) - np.pi
            
            # Extract phase and normalize to [0, 1]
            gamma = float(self.gamma_var.get())
            normalized_phase = (self.slm_phase + np.pi) / (2 * np.pi)
            self.pattern = (normalized_phase ** gamma * 255).astype(np.uint8)
            
            # Calculate and store reconstruction for preview
            try:
                # Create full-sized phase with the shift
                padded_phase = np.zeros((self.padded_height, self.padded_width))
                start_y = (self.padded_height - self.height) // 2
                end_y = start_y + self.height
                start_x = (self.padded_width - self.width) // 2
                end_x = start_x + self.width
                padded_phase[start_y:end_y, start_x:end_x] = self.slm_phase
                
                # Create complex field with phase only (amplitude = 1)
                # Apply uniform amplitude across the SLM area
                amplitude = np.zeros((self.padded_height, self.padded_width))
                amplitude[start_y:end_y, start_x:end_x] = 1.0
                
                # Create the complex field with uniform amplitude and the calculated phase
                slm_field = amplitude * np.exp(1j * padded_phase)
                
                # Simulate propagation to far field (exactly like in pattern_generator_windows.py)
                far_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(slm_field)))
                self.reconstruction = np.abs(far_field)**2
                
                # Normalize reconstruction for display
                if np.max(self.reconstruction) > 0:
                    self.reconstruction = self.reconstruction / np.max(self.reconstruction)
                
                # Apply logarithmic scaling for better visualization of dynamic range
                # This helps see details that might be lost in the high intensity regions
                self.reconstruction = np.log1p(self.reconstruction * 10) / np.log1p(10)
                
            except Exception as e:
                print(f"Warning: Error calculating reconstruction: {str(e)}")
                traceback.print_exc()
                # Create a fallback reconstruction
                self.reconstruction = np.ones((self.padded_height, self.padded_width)) * 0.1
                self.reconstruction[start_y:end_y, start_x:end_x] = self.target
            
            self.status_var.set(f"Pattern generated using {algorithm.upper()} algorithm. Stopped due to: {stop_reason}")
            
            # Update the preview to show all plots
            self.update_preview()
            
            # Return the results
            return optimized_field, slm_phase, stop_reason
            
        except Exception as e:
            self.status_var.set(f"Error generating pattern: {str(e)}")
            print(f"Detailed error: {str(e)}")
            traceback.print_exc()
            return None, None, "Error in pattern generation"

    def generate_amplitude_pattern(self, iterations, algorithm='gs', tolerance=1e-6):
        """Generate amplitude-only pattern"""
        try:
            # Initialize random phase
            random_phase = np.exp(1j * 2 * np.pi * np.random.rand(self.padded_height, self.padded_width))
            field = random_phase
            
            # Get input beam
            input_beam = self.generate_input_beam()
            
            # Apply input beam to initial field
            initial_field = input_beam * np.exp(1j * np.angle(field))
            
            # Create signal region mask for MRAF if needed
            if algorithm.lower() == 'mraf':
                # Create signal region mask based on the target
                if self.signal_region_mask is None or self.signal_region_mask.shape != self.padded_target.shape:
                    signal_ratio = float(self.signal_region_ratio_var.get())
                    # Create a binary mask where target intensity > threshold
                    threshold = signal_ratio * np.max(self.padded_target)
                    self.signal_region_mask = (self.padded_target > threshold).astype(float)
                    print(f"Created signal region mask with ratio {signal_ratio}, covering {np.mean(self.signal_region_mask)*100:.1f}% of the image")
            else:
                # For GS algorithm, no mask needed
                self.signal_region_mask = None
            
            # Create pattern generator with target intensity
            self.pattern_generator = PatternGenerator(
                target_intensity=self.padded_target,
                signal_region_mask=self.signal_region_mask,
                mixing_parameter=float(self.mixing_parameter_var.get())
            )
            
            # Run optimization with selected algorithm
            optimized_field, self.error_history, stop_reason = self.pattern_generator.optimize(
                initial_field=initial_field,
                algorithm=algorithm,
                max_iterations=iterations,
                tolerance=float(self.tolerance_var.get())
            )
            
            # Get SLM phase pattern
            slm_field = self.pattern_generator.propagate(optimized_field)
            slm_phase = np.angle(slm_field)
            
            # Extract central region
            start_y = (self.padded_height - self.height) // 2
            end_y = start_y + self.height
            start_x = (self.padded_width - self.width) // 2
            end_x = start_x + self.width
            central_phase = slm_phase[start_y:end_y, start_x:end_x]
            
            # Store the original SLM phase for later modification
            self.slm_phase = central_phase.copy()
            
            # Apply phase shift to avoid zero-order diffraction
            if self.phase_shift_x != 0 or self.phase_shift_y != 0:
                # Create coordinate grids for the SLM plane
                y, x = np.indices((self.height, self.width))
                
                # Normalize coordinates to [-0.5, 0.5] range
                x_norm = (x - self.width // 2) / self.width
                y_norm = (y - self.height // 2) / self.height
                
                # Calculate linear phase ramp
                phase_ramp = 2 * np.pi * (self.phase_shift_x * x_norm + self.phase_shift_y * y_norm)
                
                # Add phase ramp to SLM phase
                self.slm_phase = np.mod(self.slm_phase + phase_ramp, 2 * np.pi) - np.pi
            
            # Extract amplitude and normalize
            gamma = float(self.gamma_var.get())
            normalized_phase = (self.slm_phase + np.pi) / (2 * np.pi)
            self.pattern = (normalized_phase ** gamma * 255).astype(np.uint8)
            
            # Calculate and store reconstruction for preview
            try:
                # Create full-sized phase with the shift
                padded_phase = np.zeros((self.padded_height, self.padded_width))
                start_y = (self.padded_height - self.height) // 2
                end_y = start_y + self.height
                start_x = (self.padded_width - self.width) // 2
                end_x = start_x + self.width
                padded_phase[start_y:end_y, start_x:end_x] = self.slm_phase
                
                # Create complex field with phase only (amplitude = 1)
                # Apply uniform amplitude across the SLM area
                amplitude = np.zeros((self.padded_height, self.padded_width))
                amplitude[start_y:end_y, start_x:end_x] = 1.0
                
                # Create the complex field with uniform amplitude and the calculated phase
                slm_field = amplitude * np.exp(1j * padded_phase)
                
                # Simulate propagation to far field (exactly like in pattern_generator_windows.py)
                far_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(slm_field)))
                self.reconstruction = np.abs(far_field)**2
                
                # Normalize reconstruction for display
                if np.max(self.reconstruction) > 0:
                    self.reconstruction = self.reconstruction / np.max(self.reconstruction)
                
                # Apply logarithmic scaling for better visualization of dynamic range
                # This helps see details that might be lost in the high intensity regions
                self.reconstruction = np.log1p(self.reconstruction * 10) / np.log1p(10)
                
            except Exception as e:
                print(f"Warning: Error calculating reconstruction: {str(e)}")
                traceback.print_exc()
                # Create a fallback reconstruction
                self.reconstruction = np.ones((self.padded_height, self.padded_width)) * 0.1
                self.reconstruction[start_y:end_y, start_x:end_x] = self.target
            
            self.status_var.set(f"Pattern generated using {algorithm.upper()} algorithm. Stopped due to: {stop_reason}")
            
            # Update the preview to show all plots
            self.update_preview()
            
            # Return the results
            return optimized_field, slm_phase, stop_reason
            
        except Exception as e:
            self.status_var.set(f"Error generating pattern: {str(e)}")
            print(f"Detailed error: {str(e)}")
            traceback.print_exc()
            return None, None, "Error in pattern generation"

    def generate_combined_pattern(self, iterations, algorithm='gs', tolerance=1e-6):
        """Generate combined amplitude and phase pattern"""
        try:
            # Initialize random phase
            random_phase = np.exp(1j * 2 * np.pi * np.random.rand(self.padded_height, self.padded_width))
            field = random_phase
            
            # Get input beam
            input_beam = self.generate_input_beam()
            
            # Apply input beam to initial field
            initial_field = input_beam * np.exp(1j * np.angle(field))
            
            # Create signal region mask for MRAF if needed
            if algorithm.lower() == 'mraf':
                # Create signal region mask based on the target
                if self.signal_region_mask is None or self.signal_region_mask.shape != self.padded_target.shape:
                    signal_ratio = float(self.signal_region_ratio_var.get())
                    # Create a binary mask where target intensity > threshold
                    threshold = signal_ratio * np.max(self.padded_target)
                    self.signal_region_mask = (self.padded_target > threshold).astype(float)
                    print(f"Created signal region mask with ratio {signal_ratio}, covering {np.mean(self.signal_region_mask)*100:.1f}% of the image")
            else:
                # For GS algorithm, no mask needed
                self.signal_region_mask = None
            
            # Create pattern generator with target intensity
            self.pattern_generator = PatternGenerator(
                target_intensity=self.padded_target,
                signal_region_mask=self.signal_region_mask,
                mixing_parameter=float(self.mixing_parameter_var.get())
            )
            
            # Run optimization with selected algorithm
            optimized_field, self.error_history, stop_reason = self.pattern_generator.optimize(
                initial_field=initial_field,
                algorithm=algorithm,
                max_iterations=iterations,
                tolerance=float(self.tolerance_var.get())
            )
            
            # Get SLM phase pattern
            slm_field = self.pattern_generator.propagate(optimized_field)
            slm_phase = np.angle(slm_field)
            
            # Extract central region
            start_y = (self.padded_height - self.height) // 2
            end_y = start_y + self.height
            start_x = (self.padded_width - self.width) // 2
            end_x = start_x + self.width
            central_phase = slm_phase[start_y:end_y, start_x:end_x]
            
            # Store the original SLM phase for later modification
            self.slm_phase = central_phase.copy()
            
            # Apply phase shift to avoid zero-order diffraction
            if self.phase_shift_x != 0 or self.phase_shift_y != 0:
                # Create coordinate grids for the SLM plane
                y, x = np.indices((self.height, self.width))
                
                # Normalize coordinates to [-0.5, 0.5] range
                x_norm = (x - self.width // 2) / self.width
                y_norm = (y - self.height // 2) / self.height
                
                # Calculate linear phase ramp
                phase_ramp = 2 * np.pi * (self.phase_shift_x * x_norm + self.phase_shift_y * y_norm)
                
                # Add phase ramp to SLM phase
                self.slm_phase = np.mod(self.slm_phase + phase_ramp, 2 * np.pi) - np.pi
            
            # Extract both amplitude and phase
            gamma = float(self.gamma_var.get())
            normalized_phase = (self.slm_phase + np.pi) / (2 * np.pi)
            self.pattern = (normalized_phase ** gamma * 255).astype(np.uint8)
            
            # Calculate and store reconstruction for preview
            try:
                # Create full-sized phase with the shift
                padded_phase = np.zeros((self.padded_height, self.padded_width))
                start_y = (self.padded_height - self.height) // 2
                end_y = start_y + self.height
                start_x = (self.padded_width - self.width) // 2
                end_x = start_x + self.width
                padded_phase[start_y:end_y, start_x:end_x] = self.slm_phase
                
                # Create complex field with phase only (amplitude = 1)
                # Apply uniform amplitude across the SLM area
                amplitude = np.zeros((self.padded_height, self.padded_width))
                amplitude[start_y:end_y, start_x:end_x] = 1.0
                
                # Create the complex field with uniform amplitude and the calculated phase
                slm_field = amplitude * np.exp(1j * padded_phase)
                
                # Simulate propagation to far field (exactly like in pattern_generator_windows.py)
                far_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(slm_field)))
                self.reconstruction = np.abs(far_field)**2
                
                # Normalize reconstruction for display
                if np.max(self.reconstruction) > 0:
                    self.reconstruction = self.reconstruction / np.max(self.reconstruction)
                
                # Apply logarithmic scaling for better visualization of dynamic range
                # This helps see details that might be lost in the high intensity regions
                self.reconstruction = np.log1p(self.reconstruction * 10) / np.log1p(10)
                
            except Exception as e:
                print(f"Warning: Error calculating reconstruction: {str(e)}")
                traceback.print_exc()
                # Create a fallback reconstruction
                self.reconstruction = np.ones((self.padded_height, self.padded_width)) * 0.1
                self.reconstruction[start_y:end_y, start_x:end_x] = self.target
            
            self.status_var.set(f"Pattern generated using {algorithm.upper()} algorithm. Stopped due to: {stop_reason}")
            
            # Update the preview to show all plots
            self.update_preview()
            
            # Return the results
            return optimized_field, slm_phase, stop_reason
            
        except Exception as e:
            self.status_var.set(f"Error generating pattern: {str(e)}")
            print(f"Detailed error: {str(e)}")
            traceback.print_exc()
            return None, None, "Error in pattern generation"

    def _on_algorithm_change(self, *args):
        """Handle algorithm selection change"""
        if self.algorithm_var.get() == "mraf":
            self.mraf_frame.grid(row=1, column=0, columnspan=8, padx=5, pady=5)
        else:
            self.mraf_frame.grid_remove()

class PatternGenerator:
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
    
    def optimize(self, initial_field, algorithm='gs', max_iterations=10, tolerance=1e-4):
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
        for i in tqdm(range(max_iterations), desc=f"Running {algorithm.upper()} optimization"):
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
            print(f"Iteration {i}, Error: {current_error:.3e}, Field Change: {field_change:.3e}, Tolerance: {tolerance:.3e}")
            
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
        
        # If we reached max iterations, note that
        if i == max_iterations - 1:
            print(stop_reason)
        
        # Calculate final error for comparison
        final_error = self.calculate_error(field, algorithm)
        improvement = initial_error / final_error if final_error > 0 else float('inf')
        print(f"Final error: {final_error:.3e}, Improvement: {improvement:.2f}x")
        
        # Return the results
        return field, error_history, stop_reason
    
    def calculate_error(self, field, algorithm):
        """
        Calculate Normalized Mean Square Error (NMSE) between reconstructed and target intensity.
        This provides a more meaningful error metric than absolute difference.
        """
        recon_intensity = np.abs(field)**2
        
        if algorithm.lower() == 'gs':
            # For GS, calculate error over entire field
            mse = np.mean((recon_intensity - self.target_intensity)**2)
            # Normalize by mean squared target intensity (NMSE)
            norm_error = mse / np.mean(self.target_intensity**2)
        else:
            # For MRAF, calculate error only in signal region
            sr_mask = self.signal_region_mask
            if np.sum(sr_mask) > 0:  # Ensure signal region is not empty
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

class CameraController:
    """
    Camera controller for IMX296 monochrome camera that outputs in RGB3 format.
    Preserves intensity values for scientific analysis.
    """
    
    def __init__(self, 
                 resolution: Tuple[int, int] = (1456, 1088),
                 device: str = "/dev/video0"):
        """
        Initialize the camera controller.
        
        Args:
            resolution: Camera resolution (width, height)
            device: Camera device path
        """
        self.resolution = resolution
        self.device = device
        self.camera = None
        self.is_running = False
        self.thread = None
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_histogram = None
        
        # Camera settings
        self.settings = {
            'exposure': 10.0,  # ms
            'gain': 1.0,       # analog gain
            'brightness': 0,   # -255 to 255
            'contrast': 1.0,   # 0.0 to 2.0
        }
        
        # Initialize camera
        self._initialize_camera()
    
    def _initialize_camera(self) -> bool:
        """Initialize the camera based on available hardware"""
        if PICAMERA2_AVAILABLE:
            return self._initialize_picamera2()
        else:
            return self._initialize_opencv()
    
    def _initialize_opencv(self) -> bool:
        """Initialize OpenCV camera"""
        try:
            self.camera = cv2.VideoCapture(self.device)
            
            # Set resolution
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            
            # Check if camera opened successfully
            if not self.camera.isOpened():
                print("Error: Could not open camera.")
                return False
                
            # Apply initial settings
            self.apply_settings()
            
            print("OpenCV camera initialized successfully")
            return True
            
        except Exception as e:
            print(f"Error initializing OpenCV camera: {str(e)}")
            return False
    
    def _initialize_picamera2(self) -> bool:
        """Initialize PiCamera2"""
        try:
            self.camera = Picamera2()
            
            # Configure camera for still capture
            config = self.camera.create_still_configuration(
                main={"size": self.resolution, "format": "RGB888"},
                lores={"size": (640, 480), "format": "YUV420"},
                display="lores"
            )
            self.camera.configure(config)
            
            # Apply initial settings
            self.apply_settings()
            
            print("PiCamera2 initialized successfully")
            return True
            
        except Exception as e:
            print(f"Error initializing PiCamera2: {str(e)}")
            return False
    
    def start(self) -> bool:
        """Start the camera capture thread"""
        if self.is_running:
            print("Camera is already running")
            return True
            
        if self.camera is None:
            print("Camera not initialized")
            return False
            
        # Start the camera
        if PICAMERA2_AVAILABLE and isinstance(self.camera, Picamera2):
            self.camera.start()
        
        # Start the capture thread
        self.is_running = True
        self.thread = threading.Thread(target=self._capture_thread, daemon=True)
        self.thread.start()
        
        print("Camera started")
        return True
    
    def stop(self) -> None:
        """Stop the camera capture thread"""
        self.is_running = False
        
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None
            
        # Stop the camera
        if PICAMERA2_AVAILABLE and isinstance(self.camera, Picamera2):
            self.camera.stop()
        elif self.camera is not None and not PICAMERA2_AVAILABLE:
            self.camera.release()
            
        self.camera = None
        print("Camera stopped")
    
    def _capture_thread(self) -> None:
        """Thread function for continuous frame capture"""
        while self.is_running:
            frame = self._capture_frame()
            
            if frame is not None:
                with self.lock:
                    self.latest_frame = frame
                    self.latest_histogram = self.generate_histogram(frame)
                    
            time.sleep(0.03)  # ~30 FPS
    
    def _capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame"""
        try:
            if PICAMERA2_AVAILABLE and isinstance(self.camera, Picamera2):
                # Capture using PiCamera2
                frame = self.camera.capture_array()
                
                # PiCamera2 returns RGB888 format, take only one channel for grayscale
                # We use the green channel as it typically has the best sensitivity
                frame = frame[:, :, 1]
                
            else:
                # Capture using OpenCV
                ret, frame = self.camera.read()
                if not ret:
                    print("Failed to capture frame")
                    return None
                
                # Convert to grayscale if needed
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            return frame
            
        except Exception as e:
            print(f"Error capturing frame: {str(e)}")
            return None
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the latest captured frame"""
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()
    
    def get_latest_histogram(self) -> Optional[np.ndarray]:
        """Get the latest histogram image"""
        with self.lock:
            if self.latest_histogram is None:
                return None
            return self.latest_histogram.copy()
    
    def generate_histogram(self, frame: np.ndarray) -> np.ndarray:
        """Generate a histogram image from the frame"""
        try:
            # Create a figure for the histogram with a smaller size
            fig = Figure(figsize=(4.5, 3.5), dpi=100)
            ax = fig.add_subplot(111)
            
            # Add more padding around the plot for axis labels
            fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.9)
            
            # Calculate histogram - ensure we use the full 8-bit range (0-255)
            hist = cv2.calcHist([frame], [0], None, [256], [0, 256])
            
            # Normalize the histogram
            total_pixels = frame.shape[0] * frame.shape[1]
            normalized_hist = hist / total_pixels
            
            # Plot the normalized histogram
            ax.plot(normalized_hist, color='blue')
            ax.set_xlim([0, 256])
            ax.set_title('Normalized Intensity Histogram (8-bit)')
            ax.set_xlabel('Intensity Value (0-255)')
            ax.set_ylabel('Normalized Pixel Count')
            
            # Add vertical lines at min and max values
            min_val = np.min(frame)
            max_val = np.max(frame)
            mean_val = np.mean(frame)
            
            # Add min/max/mean lines
            ax.axvline(x=min_val, color='r', linestyle='--', alpha=0.7, label=f'Min: {min_val:.1f}')
            ax.axvline(x=max_val, color='g', linestyle='--', alpha=0.7, label=f'Max: {max_val:.1f}')
            ax.axvline(x=mean_val, color='y', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.1f}')
            
            # Add legend
            ax.legend(loc='upper right', fontsize='small')
            
            ax.grid(True, alpha=0.3)
            
            # Render the figure to a numpy array
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            buf = canvas.buffer_rgba()
            hist_image = np.asarray(buf)
            plt.close(fig)
            
            return hist_image
            
        except Exception as e:
            print(f"Error generating histogram: {str(e)}")
            # Return a blank image on error
            return np.zeros((350, 450, 4), dtype=np.uint8)
    
    def get_intensity_stats(self, frame: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Get intensity statistics from the frame"""
        if frame is None:
            frame = self.get_latest_frame()
            
        if frame is None:
            return {'min': 0, 'max': 0, 'mean': 0, 'std': 0}
            
        return {
            'min': float(np.min(frame)),
            'max': float(np.max(frame)),
            'mean': float(np.mean(frame)),
            'std': float(np.std(frame))
        }
    
    def apply_settings(self) -> None:
        """Apply camera settings"""
        try:
            if PICAMERA2_AVAILABLE and isinstance(self.camera, Picamera2):
                # Apply PiCamera2 settings
                # Convert exposure from ms to μs
                self.camera.set_controls({
                    "ExposureTime": int(self.settings['exposure'] * 1000),
                    "AnalogueGain": float(self.settings['gain']),
                })
                
            else:
                # Apply OpenCV settings
                self.camera.set(cv2.CAP_PROP_EXPOSURE, self.settings['exposure'])
                self.camera.set(cv2.CAP_PROP_GAIN, self.settings['gain'])
                self.camera.set(cv2.CAP_PROP_BRIGHTNESS, self.settings['brightness'])
                self.camera.set(cv2.CAP_PROP_CONTRAST, self.settings['contrast'])
                
        except Exception as e:
            print(f"Error applying camera settings: {str(e)}")
    
    def save_frame(self, filename: str, frame: Optional[np.ndarray] = None) -> bool:
        """Save the frame to a file"""
        if frame is None:
            frame = self.get_latest_frame()
            
        if frame is None:
            print("No frame available to save")
            return False
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
            
            # Save the image
            cv2.imwrite(filename, frame)
            
            print(f"Frame saved to {filename}")
            return True
            
        except Exception as e:
            print(f"Error saving frame: {str(e)}")
            return False
    
    def reset_settings(self) -> None:
        """Reset camera settings to defaults"""
        self.settings = {
            'exposure': 10.0,  # ms
            'gain': 1.0,       # analog gain
            'brightness': 0,   # -255 to 255
            'contrast': 1.0,   # 0.0 to 2.0
        }
        
        self.apply_settings()
        print("Camera settings reset to defaults")

class CameraGUI:
    """GUI for camera control and display"""
    
    def __init__(self, root: tk.Tk, camera: Optional[CameraController] = None):
        """Initialize the camera control GUI"""
        self.root = root
        
        # Create camera controller if not provided
        if camera is None:
            self.camera = CameraController()
        else:
            self.camera = camera
        
        # Preview dimensions
        self.preview_width = 640
        self.preview_height = 480
        
        # Pause state
        self.is_paused = False
        
        # Create the GUI
        self._create_widgets()
        
        # Start the camera
        if not self.camera.is_running:
            self.camera.start()
        
        # Start the update loop
        self._update_preview()
    
    def _create_widgets(self):
        """Create the GUI widgets"""
        # Main notebook
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Preview tab
        preview_tab = ttk.Frame(notebook)
        notebook.add(preview_tab, text="Preview")
        
        # Create preview area
        self._create_preview_area(preview_tab)
        
        # Status bar
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=2)
        
        self.status_var = tk.StringVar(value="Camera ready")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, anchor=tk.W)
        status_label.pack(fill=tk.X, padx=5)
    
    def _create_preview_area(self, parent):
        """Create the camera preview area"""
        # Preview frame
        preview_frame = ttk.Frame(parent)
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Split into preview and histogram/settings
        preview_pane = ttk.Frame(preview_frame)
        preview_pane.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        right_pane = ttk.Frame(preview_frame)
        right_pane.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)
        
        # Preview canvas
        self.preview_canvas = tk.Canvas(preview_pane, 
                                       width=self.preview_width, 
                                       height=self.preview_height,
                                       bg="black")
        self.preview_canvas.pack(pady=5)
        
        # Add intensity info label
        self.intensity_info = ttk.Label(preview_pane, 
                                      text="Intensity (8-bit) - Max: 0, Mean: 0",
                                      font=("Arial", 10))
        self.intensity_info.pack(pady=5)
        
        # Control buttons for preview pane
        control_frame = ttk.Frame(preview_pane)
        control_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(control_frame, text="Capture", command=self._on_capture).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Save", command=self._on_save).pack(side=tk.LEFT, padx=5)
        
        # Pause/Resume toggle button
        self.pause_text = tk.StringVar(value="Pause Camera")
        self.pause_button = ttk.Button(control_frame, textvariable=self.pause_text, command=self._on_toggle_pause)
        self.pause_button.pack(side=tk.LEFT, padx=5)
        
        # Histogram canvas - reduced size
        histogram_frame = ttk.LabelFrame(right_pane, text="Histogram")
        histogram_frame.pack(pady=5, fill=tk.X)
        
        self.histogram_canvas = tk.Canvas(histogram_frame,
                                        width=450,
                                        height=350,
                                        bg="white")
        self.histogram_canvas.pack(pady=5)
        
        # Settings frame - now under histogram in the right pane
        settings_frame = ttk.LabelFrame(right_pane, text="Camera Settings")
        settings_frame.pack(fill=tk.X, pady=10, padx=5)
        
        # Create numerical entry fields instead of sliders
        # Exposure control
        exposure_frame = ttk.Frame(settings_frame)
        exposure_frame.pack(fill=tk.X, pady=2)
        ttk.Label(exposure_frame, text="Exposure (ms):").pack(side=tk.LEFT, padx=5)
        self.exposure_var = tk.StringVar(value=str(self.camera.settings['exposure']))
        exposure_entry = ttk.Entry(exposure_frame, textvariable=self.exposure_var, width=8)
        exposure_entry.pack(side=tk.RIGHT, padx=5)
        
        # Gain control
        gain_frame = ttk.Frame(settings_frame)
        gain_frame.pack(fill=tk.X, pady=2)
        ttk.Label(gain_frame, text="Gain:").pack(side=tk.LEFT, padx=5)
        self.gain_var = tk.StringVar(value=str(self.camera.settings['gain']))
        gain_entry = ttk.Entry(gain_frame, textvariable=self.gain_var, width=8)
        gain_entry.pack(side=tk.RIGHT, padx=5)
        
        # Brightness control
        brightness_frame = ttk.Frame(settings_frame)
        brightness_frame.pack(fill=tk.X, pady=2)
        ttk.Label(brightness_frame, text="Brightness:").pack(side=tk.LEFT, padx=5)
        self.brightness_var = tk.StringVar(value=str(self.camera.settings['brightness']))
        brightness_entry = ttk.Entry(brightness_frame, textvariable=self.brightness_var, width=8)
        brightness_entry.pack(side=tk.RIGHT, padx=5)
        
        # Contrast control
        contrast_frame = ttk.Frame(settings_frame)
        contrast_frame.pack(fill=tk.X, pady=2)
        ttk.Label(contrast_frame, text="Contrast:").pack(side=tk.LEFT, padx=5)
        self.contrast_var = tk.StringVar(value=str(self.camera.settings['contrast']))
        contrast_entry = ttk.Entry(contrast_frame, textvariable=self.contrast_var, width=8)
        contrast_entry.pack(side=tk.RIGHT, padx=5)
        
        # Settings buttons frame
        settings_buttons_frame = ttk.Frame(settings_frame)
        settings_buttons_frame.pack(fill=tk.X, pady=5)
        
        # Apply settings button
        apply_button = ttk.Button(settings_buttons_frame, text="Apply Settings", command=self._on_apply_settings)
        apply_button.pack(side=tk.LEFT, padx=5)
        
        # Reset button
        reset_button = ttk.Button(settings_buttons_frame, text="Reset Settings", command=self._on_reset)
        reset_button.pack(side=tk.RIGHT, padx=5)
    
    def _update_setting_from_entry(self, setting, var):
        """Update a camera setting from an entry field"""
        try:
            value = float(var.get())
            self.camera.settings[setting] = value
            return True
        except ValueError:
            self.status_var.set(f"Invalid value for {setting}")
            return False
    
    def _on_capture(self):
        """Handle capture button click"""
        try:
            # Get the current frame
            frame = self.camera.get_latest_frame()
            
            if frame is None:
                self.status_var.set("No frame available to capture")
                return
                
            # Create default capture directory
            default_dir = "/home/surena/slm-control/Captures"
            
            # Create a default filename with timestamp
            default_filename = f"image_{time.strftime('%Y%m%d_%H%M%S')}.png"
            default_path = os.path.join(default_dir, default_filename)
            
            # Make sure the default directory exists
            os.makedirs(default_dir, exist_ok=True)
            
            # Check if zenity is available
            zenity_available = False
            try:
                # Check if zenity is installed
                result = subprocess.run(["which", "zenity"], capture_output=True, text=True)
                zenity_available = result.returncode == 0
            except Exception:
                zenity_available = False
                
            # Use zenity if available
            if zenity_available:
                try:
                    # Run zenity file selection dialog
                    cmd = [
                        "zenity", "--file-selection",
                        "--save",
                        "--filename", default_path,
                        "--title", "Save Captured Image",
                        "--file-filter", "Images | *.png *.jpg *.jpeg *.tif *.tiff"
                    ]
                    
                    # Execute zenity command
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    # Check if user canceled
                    if result.returncode != 0:
                        self.status_var.set("Capture canceled")
                        return
                        
                    # Get the selected filename
                    filename = result.stdout.strip()
                    
                    # If empty, user canceled
                    if not filename:
                        self.status_var.set("Capture canceled")
                        return
                        
                    # Save the frame
                    if self.camera.save_frame(filename, frame):
                        self.status_var.set(f"Image captured and saved to {filename}")
                    else:
                        self.status_var.set("Failed to save captured image")
                        
                except Exception as zenity_error:
                    print(f"Zenity error: {str(zenity_error)}")
                    self.status_var.set(f"Zenity error: {str(zenity_error)}")
                    
                    # Fallback to default path
                    if self.camera.save_frame(default_path, frame):
                        self.status_var.set(f"Image captured and saved to {default_path}")
                    else:
                        self.status_var.set("Failed to save captured image")
            else:
                # Zenity not available, use default path
                print("Zenity not available, saving to default path")
                if self.camera.save_frame(default_path, frame):
                    self.status_var.set(f"Image captured and saved to {default_path}")
                else:
                    self.status_var.set("Failed to save captured image")
                
        except Exception as e:
            self.status_var.set(f"Capture error: {str(e)}")
            print(f"Capture error: {str(e)}")
            traceback.print_exc()
    
    def _on_save(self):
        """Handle save button click"""
        try:
            # Get the current frame
            frame = self.camera.get_latest_frame()
            
            if frame is None:
                self.status_var.set("No frame available to save")
                return
                
            # Create default save directory
            default_dir = "/home/surena/slm-control/Captures"
            
            # Create a default filename with timestamp
            default_filename = f"image_{time.strftime('%Y%m%d_%H%M%S')}.png"
            default_path = os.path.join(default_dir, default_filename)
            
            # Make sure the default directory exists
            os.makedirs(default_dir, exist_ok=True)
            
            # Check if zenity is available
            zenity_available = False
            try:
                # Check if zenity is installed
                result = subprocess.run(["which", "zenity"], capture_output=True, text=True)
                zenity_available = result.returncode == 0
            except Exception:
                zenity_available = False
                
            # Use zenity if available
            if zenity_available:
                try:
                    # Run zenity file selection dialog
                    cmd = [
                        "zenity", "--file-selection",
                        "--save",
                        "--filename", default_path,
                        "--title", "Save Image",
                        "--file-filter", "Images | *.png *.jpg *.jpeg *.tif *.tiff"
                    ]
                    
                    # Execute zenity command
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    # Check if user canceled
                    if result.returncode != 0:
                        self.status_var.set("Save canceled")
                        return
                        
                    # Get the selected filename
                    filename = result.stdout.strip()
                    
                    # If empty, user canceled
                    if not filename:
                        self.status_var.set("Save canceled")
                        return
                        
                    # Save the frame
                    if self.camera.save_frame(filename, frame):
                        self.status_var.set(f"Image saved to {filename}")
                    else:
                        self.status_var.set("Failed to save image")
                        
                except Exception as zenity_error:
                    print(f"Zenity error: {str(zenity_error)}")
                    self.status_var.set(f"Zenity error: {str(zenity_error)}")
                    
                    # Fallback to default path
                    if self.camera.save_frame(default_path, frame):
                        self.status_var.set(f"Image saved to {default_path}")
                    else:
                        self.status_var.set("Failed to save image")
            else:
                # Zenity not available, use default path
                print("Zenity not available, saving to default path")
                if self.camera.save_frame(default_path, frame):
                    self.status_var.set(f"Image saved to {default_path}")
                else:
                    self.status_var.set("Failed to save image")
                
        except Exception as e:
            self.status_var.set(f"Save error: {str(e)}")
            print(f"Save error: {str(e)}")
            traceback.print_exc()
    
    def _on_reset(self):
        """Handle reset button click"""
        self.camera.reset_settings()
        
        # Update the UI
        self.exposure_var.set(str(self.camera.settings['exposure']))
        self.gain_var.set(str(self.camera.settings['gain']))
        self.brightness_var.set(str(self.camera.settings['brightness']))
        self.contrast_var.set(str(self.camera.settings['contrast']))
        
        self.status_var.set("Camera settings reset to defaults")
    
    def _on_toggle_pause(self):
        """Handle pause/resume button click"""
        self.is_paused = not self.is_paused
        
        if self.is_paused:
            self.pause_text.set("Resume Camera")
            self.status_var.set("Camera paused")
        else:
            self.pause_text.set("Pause Camera")
            self.status_var.set("Camera resumed")
    
    def _update_preview(self):
        """Update the preview display"""
        if not self.is_paused:
            try:
                # Get the latest frame
                frame = self.camera.get_latest_frame()
                
                if frame is not None:
                    # Resize for display
                    display_frame = cv2.resize(frame, (self.preview_width, self.preview_height))
                    
                    # Convert to PIL format
                    pil_img = Image.fromarray(display_frame)
                    
                    # Convert to PhotoImage
                    img_tk = ImageTk.PhotoImage(image=pil_img)
                    
                    # Update canvas
                    self.preview_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
                    self.preview_canvas.image = img_tk  # Keep a reference
                    
                    # Update intensity info
                    stats = self.camera.get_intensity_stats(frame)
                    self.intensity_info.config(
                        text=f"Intensity (8-bit) - Min: {stats['min']:.1f}, Max: {stats['max']:.1f}, Mean: {stats['mean']:.1f}"
                    )
                    
                    # Update histogram
                    hist_img = self.camera.get_latest_histogram()
                    if hist_img is not None:
                        # Convert RGBA to RGB
                        hist_rgb = cv2.cvtColor(hist_img, cv2.COLOR_RGBA2RGB)
                        
                        # Resize to fit canvas
                        hist_rgb = cv2.resize(hist_rgb, (450, 350))
                        
                        # Convert to PIL and then to PhotoImage
                        hist_pil = Image.fromarray(hist_rgb)
                        hist_tk = ImageTk.PhotoImage(image=hist_pil)
                        
                        # Update canvas
                        self.histogram_canvas.create_image(0, 0, anchor=tk.NW, image=hist_tk)
                        self.histogram_canvas.image = hist_tk  # Keep a reference
                
            except Exception as e:
                print(f"Error updating preview: {str(e)}")
                traceback.print_exc()
                
        # Schedule the next update
        self.root.after(33, self._update_preview)  # ~30 FPS
    
    def _on_apply_settings(self):
        """Handle apply settings button click"""
        # Update settings from entry fields
        if (self._update_setting_from_entry('exposure', self.exposure_var) and
            self._update_setting_from_entry('gain', self.gain_var) and
            self._update_setting_from_entry('brightness', self.brightness_var) and
            self._update_setting_from_entry('contrast', self.contrast_var)):
            
            # Apply settings
            self.camera.apply_settings()
            self.status_var.set("Camera settings applied")
        else:
            self.status_var.set("Failed to apply settings - check values")

if __name__ == "__main__":
    app = AdvancedPatternGenerator()
    app.run()
