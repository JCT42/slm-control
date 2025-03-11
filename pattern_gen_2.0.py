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
- Real-time pattern optimization
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
        
        # Initialize GUI
        self.setup_gui()

    def setup_gui(self):
        """Create the main GUI window and controls"""
        self.root = tk.Tk()
        self.root.title("SLM Pattern Generator")
        self.root.geometry("1600x900")  # Increased window size
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas for scrolling
        self.canvas = tk.Canvas(self.main_frame)
        scrollbar = ttk.Scrollbar(self.main_frame, orient="vertical", command=self.canvas.yview)
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
        
        self.camera_frame = ttk.LabelFrame(self.scrollable_frame, text="Camera Preview", padding="10")
        self.camera_frame.pack(fill=tk.BOTH, padx=5, pady=5)
        
        # Add status bar
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(self.scrollable_frame, textvariable=self.status_var)
        self.status_bar.pack(fill=tk.X, padx=5, pady=5)
        
        # Create controls
        self.create_controls()
        
        # Create preview area
        self.create_preview()
        
        # Initialize camera
        self.initialize_camera()
        
        # Bind ESC key to quit
        self.root.bind('<Escape>', lambda e: self.quit_application())
        
        # Bind mouse wheel to scrolling
        self.root.bind("<MouseWheel>", self._on_mousewheel)

    def _on_canvas_configure(self, event):
        """Handle canvas resize"""
        width = event.width
        self.canvas.itemconfig(1, width=width)

    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

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
        self.iterations_var = tk.StringVar(value="100")
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
        self.tolerance_var = tk.StringVar(value="1e-24")
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
        
        self.ax4.set_title('Optimization Error (NMSE)')
        self.ax4.set_xlabel('Iteration')
        self.ax4.set_ylabel('Normalized Mean Square Error')
        self.ax4.grid(True)
        
        # Pack canvas
        self.preview_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initial draw
        self.preview_canvas.draw()
        
    def create_camera_preview(self):
        """Create camera preview area"""
        if not self.camera_active:
            return
            
        try:
            # Create camera preview figure
            self.camera_fig, self.camera_ax = plt.subplots(figsize=(16, 6))  # Match preview size
            self.camera_canvas = FigureCanvasTkAgg(self.camera_fig, master=self.camera_frame)
            
            # Initialize camera preview with black image
            self.camera_image = self.camera_ax.imshow(np.zeros((1080, 1920)), cmap='gray', vmin=0, vmax=255)
            self.camera_ax.set_title('Camera Feed')
            self.camera_ax.set_xticks([])
            self.camera_ax.set_yticks([])
            
            # Create frame for camera controls
            controls_frame = ttk.Frame(self.camera_frame)
            controls_frame.pack(side=tk.RIGHT, padx=10, pady=5, fill=tk.Y)
            
            # Create sections for different control groups
            buttons_frame = ttk.LabelFrame(controls_frame, text="Camera Controls", padding=5)
            buttons_frame.pack(fill=tk.X, pady=5)
            
            # Pause/Resume button
            self.pause_camera_button = ttk.Button(buttons_frame, text="Pause Camera", command=self.pause_camera)
            self.pause_camera_button.pack(fill=tk.X, pady=2)
            
            # Capture button
            self.capture_button = ttk.Button(buttons_frame, text="Capture Image", command=self.capture_camera_image)
            self.capture_button.pack(fill=tk.X, pady=2)
            
            # Save button
            self.save_camera_button = ttk.Button(buttons_frame, text="Save Image", 
                                               command=lambda: self.save_camera_image())
            self.save_camera_button.pack(fill=tk.X, pady=2)
            
            # Exposure settings frame
            exposure_frame = ttk.LabelFrame(controls_frame, text="Exposure Settings", padding=5)
            exposure_frame.pack(fill=tk.X, pady=5)
            
            # Exposure control
            ttk.Label(exposure_frame, text="Exposure Time (ms):").pack(fill=tk.X)
            exposure_control_frame = ttk.Frame(exposure_frame)
            exposure_control_frame.pack(fill=tk.X)
            
            self.exposure_var = tk.StringVar(value="10")
            exposure_entry = ttk.Entry(exposure_control_frame, textvariable=self.exposure_var, width=8)
            exposure_entry.pack(side=tk.LEFT, padx=2)
            
            ttk.Button(exposure_control_frame, text="Set", 
                      command=lambda: self.set_exposure(float(self.exposure_var.get()))).pack(side=tk.LEFT, padx=2)
            
            # Gain settings frame
            gain_frame = ttk.LabelFrame(controls_frame, text="Gain Settings", padding=5)
            gain_frame.pack(fill=tk.X, pady=5)
            
            # Gain control
            ttk.Label(gain_frame, text="Analog Gain:").pack(fill=tk.X)
            gain_control_frame = ttk.Frame(gain_frame)
            gain_control_frame.pack(fill=tk.X)
            
            self.gain_var = tk.StringVar(value="1.0")
            gain_entry = ttk.Entry(gain_control_frame, textvariable=self.gain_var, width=8)
            gain_entry.pack(side=tk.LEFT, padx=2)
            
            ttk.Button(gain_control_frame, text="Set",
                      command=lambda: self.set_gain(float(self.gain_var.get()))).pack(side=tk.LEFT, padx=2)
            
            # Pack the camera canvas
            self.camera_canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
        except Exception as e:
            self.status_var.set(f"Error creating camera preview: {str(e)}")
            print(f"Detailed error: {str(e)}")

    def initialize_camera(self):
        """Initialize Raspberry Pi Camera"""
        try:
            # Initialize PiCamera2
            self.picam = Picamera2()
            
            # Configure camera
            preview_config = self.picam.create_preview_configuration(
                main={"size": (1920, 1080), "format": "RGB888"},
                lores={"size": (640, 360), "format": "YUV420"},
                display="lores"
            )
            
            still_config = self.picam.create_still_configuration(
                main={"size": (1920, 1080), "format": "RGB888"},
                lores={"size": (640, 360), "format": "YUV420"}
            )
            
            self.picam.configure(preview_config)
            
            # Set camera controls
            self.picam.set_controls({
                "FrameDurationLimits": (33333, 33333),  # 30fps
                "ExposureTime": 10000,  # 10ms exposure
                "AnalogueGain": 1.0
            })
            
            # Start the camera
            self.picam.start()
            time.sleep(2)  # Wait for camera to warm up
            
            # Test if camera is working
            test_frame = self.picam.capture_array()
            if test_frame is not None:
                print("Successfully connected to Raspberry Pi Camera")
                print(f"Frame shape: {test_frame.shape}")
                print(f"Frame type: {test_frame.dtype}")
                print(f"Frame range: min={test_frame.min()}, max={test_frame.max()}")
                self.camera_active = True
                
                # Create camera frame
                self.create_camera_preview()
                
                # Start camera thread
                self.camera_thread = threading.Thread(target=self.update_camera_preview, daemon=True)
                self.camera_thread.start()
                
                self.status_var.set("Raspberry Pi Camera initialized")
            else:
                self.status_var.set("Could not capture frame from Pi Camera")
                
        except Exception as e:
            self.status_var.set(f"Pi Camera initialization error: {str(e)}")
            print(f"Detailed camera error: {str(e)}")
            self.camera_active = False

    def update_camera_preview(self):
        """Update camera preview in a separate thread"""
        while self.camera_active:
            try:
                if not self.camera_paused:
                    # Capture new frame from Pi Camera
                    frame = self.picam.capture_array()
                    
                    if frame is not None:
                        # Convert to grayscale if needed
                        if len(frame.shape) == 3:
                            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        else:
                            frame_gray = frame
                        
                        # Store as last frame (at native resolution)
                        self.last_frame = frame_gray
                        
                        # Update matplotlib image (display at native resolution)
                        self.camera_image.set_array(frame_gray)
                        self.camera_canvas.draw_idle()
                else:
                    # If paused and we have a last frame, keep displaying it
                    if self.last_frame is not None:
                        self.camera_image.set_array(self.last_frame)
                        self.camera_canvas.draw_idle()
                    
            except Exception as e:
                print(f"Camera preview error: {str(e)}")
                time.sleep(0.1)
            
            time.sleep(0.033)  # ~30 FPS

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
                self.ax4.set_title('Optimization Error (NMSE)')
                self.ax4.set_xlabel('Iteration')
                self.ax4.set_ylabel('Normalized Mean Square Error')
                self.ax4.grid(True)
                
                # Use log scale if the error values span multiple orders of magnitude
                if len(self.error_history) > 1:
                    max_error = max(self.error_history)
                    min_error = min(self.error_history)
                    if max_error / min_error > 100:  # More than 2 orders of magnitude
                        self.ax4.set_yscale('log')
            else:
                self.ax4.set_title('Optimization Error (NMSE)')
                self.ax4.set_xlabel('Iteration')
                self.ax4.set_ylabel('Normalized Mean Square Error')
            
            # Update canvas
            self.preview_canvas.draw()
            
        except Exception as e:
            self.status_var.set(f"Error updating preview: {str(e)}")
            print(f"Detailed error: {str(e)}")

    def generate_pattern(self):
        """Generate pattern based on selected modulation mode"""
        if not hasattr(self, 'target'):
            self.status_var.set("Please load a target image first")
            return
            
        try:
            # Get parameters from GUI
            self.modulation_mode = self.mode_var.get()
            self.amplitude_coupling = float(self.amp_coupling_var.get())
            self.phase_coupling = float(self.phase_coupling_var.get())
            iterations = int(self.iterations_var.get())
            gamma = float(self.gamma_var.get())
            phase_range = float(self.phase_range_var.get()) * np.pi
            algorithm = self.algorithm_var.get()
            tolerance = float(self.tolerance_var.get())
            
            # Create signal region mask for MRAF if needed
            if algorithm == "mraf":
                try:
                    mixing_parameter = float(self.mixing_parameter_var.get())
                    signal_region_ratio = float(self.signal_region_ratio_var.get())
                    
                    # Create circular signal region mask
                    y, x = np.ogrid[:self.padded_height, :self.padded_width]
                    center_y, center_x = self.padded_height // 2, self.padded_width // 2
                    radius = min(center_x, center_y) * signal_region_ratio
                    signal_mask = ((x - center_x)**2 + (y - center_y)**2 <= radius**2).astype(float)
                    
                    # Initialize PatternGenerator with MRAF parameters
                    self.pattern_generator = PatternGenerator(
                        target_intensity=self.padded_target,
                        signal_region_mask=signal_mask,
                        mixing_parameter=mixing_parameter
                    )
                except ValueError as e:
                    self.status_var.set(f"Invalid MRAF parameters: {str(e)}")
                    return
            else:
                # Initialize PatternGenerator without MRAF parameters
                self.pattern_generator = PatternGenerator(
                    target_intensity=self.padded_target
                )
            
            # Generate pattern based on mode
            if self.modulation_mode == "Phase":
                field, slm_phase, stop_reason = self.generate_phase_pattern(iterations, algorithm, tolerance)
            elif self.modulation_mode == "Amplitude":
                field, slm_phase, stop_reason = self.generate_amplitude_pattern(iterations, algorithm, tolerance)
            else:  # Combined mode
                field, slm_phase, stop_reason = self.generate_combined_pattern(iterations, algorithm, tolerance)
            
            if field is None:
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
                amplitude = np.abs(field)
                normalized_amplitude = amplitude / np.max(amplitude)
                self.pattern = (normalized_amplitude ** gamma * 255).astype(np.uint8)
                
            else:  # Combined mode
                # Extract both amplitude and phase
                amplitude = np.abs(field)
                phase = np.angle(field)
                
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
            
            self.status_var.set(f"Pattern generated using {self.modulation_mode} mode. Stopped due to: {stop_reason}")
            
        except Exception as e:
            self.status_var.set(f"Error generating pattern: {str(e)}")
            print(f"Detailed error: {str(e)}")

    def load_pattern(self):
        """Load a pattern from file"""
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
            if pattern.shape != (600, 800):
                pattern = cv2.resize(pattern, (800, 600))
            
            self.pattern = pattern
            self.slm_phase = (pattern.astype(float) / 255.0 * 2 * np.pi - np.pi)
            
            # Update preview
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
            # Create a thread for SLM display
            self.slm_thread = threading.Thread(target=self._display_slm_pattern)
            self.slm_thread.daemon = True  # Thread will be terminated when main program exits
            self.slm_thread.start()
            self.status_var.set("Pattern sent to HDMI-A-2. Press ESC in SLM window to close.")
            
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
            os.environ['SDL_VIDEO_WINDOW_POS'] = '1280,0'  # Position at main monitor width
            os.environ['SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS'] = '0'
            
            # Get display info
            print(f"Number of displays: {pygame.display.get_num_displays()}")
            for i in range(pygame.display.get_num_displays()):
                info = pygame.display.get_desktop_sizes()[i]
                print(f"Display {i}: {info}")
            
            # Create window on second display
            slm_window = pygame.display.set_mode(
                (800, 600),
                pygame.NOFRAME,
                display=1
            )
            
            # Create and show pattern
            pattern_surface = pygame.Surface((800, 600), depth=8)
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

    def quit_application(self):
        """Clean up and quit the application"""
        if self.camera_active:
            self.camera_active = False
            self.picam.stop()
            self.picam.close()
        cv2.destroyAllWindows()
        self.root.quit()
        
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
            
            # Create padded target for Gerchberg-Saxton
            self.padded_target = np.zeros((self.padded_height, self.padded_width))
            start_y = (self.padded_height - self.height) // 2
            end_y = start_y + self.height
            start_x = (self.padded_width - self.width) // 2
            end_x = start_x + self.width
            self.padded_target[start_y:end_y, start_x:end_x] = self.target
            
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

    def save_pattern(self):
        """Save the generated pattern"""
        if not hasattr(self, 'pattern'):
            self.status_var.set("No pattern to save")
            return
            
        try:
            # Generate default filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            default_name = f"pattern_{timestamp}.png"
            
            # Use zenity file save dialog
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
                
            # Ensure .png extension
            if not save_path.lower().endswith('.png'):
                save_path += '.png'
            
            # Save the pattern
            cv2.imwrite(save_path, self.pattern)
            self.status_var.set(f"Pattern saved to: {save_path}")
            
        except Exception as e:
            self.status_var.set(f"Error saving pattern: {str(e)}")
        
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
            
            # Run optimization with selected algorithm
            optimized_field, self.error_history, stop_reason = self.pattern_generator.optimize(
                initial_field=initial_field,
                algorithm=algorithm,
                max_iterations=iterations,
                tolerance=tolerance
            )
            
            # Get SLM phase pattern
            slm_field = self.pattern_generator.propagate(optimized_field)
            slm_phase = np.angle(slm_field)
            
            # Calculate and store reconstruction for preview
            image_field = self.pattern_generator.inverse_propagate(np.exp(1j * slm_phase))
            self.reconstruction = np.abs(image_field)**2
            
            # Normalize reconstruction for display
            self.reconstruction = self.reconstruction / np.max(self.reconstruction)
            
            self.status_var.set(f"Phase pattern generated using {algorithm.upper()} algorithm")
            return optimized_field, slm_phase, stop_reason
            
        except Exception as e:
            self.status_var.set(f"Error in pattern generation: {str(e)}")
            print(f"Detailed error: {str(e)}")
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
            
            # Run optimization with selected algorithm
            optimized_field, self.error_history, stop_reason = self.pattern_generator.optimize(
                initial_field=initial_field,
                algorithm=algorithm,
                max_iterations=iterations,
                tolerance=tolerance
            )
            
            # Get SLM phase pattern
            slm_field = self.pattern_generator.propagate(optimized_field)
            slm_phase = np.angle(slm_field)
            
            # Calculate and store reconstruction for preview
            image_field = self.pattern_generator.inverse_propagate(np.exp(1j * slm_phase))
            self.reconstruction = np.abs(image_field)**2
            
            # Normalize reconstruction for display
            self.reconstruction = self.reconstruction / np.max(self.reconstruction)
            
            self.status_var.set(f"Amplitude pattern generated using {algorithm.upper()} algorithm")
            return optimized_field, slm_phase, stop_reason
            
        except Exception as e:
            self.status_var.set(f"Error in amplitude pattern generation: {str(e)}")
            print(f"Detailed error: {str(e)}")
            return None, None, "Error in amplitude pattern generation"

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
            
            # Run optimization with selected algorithm
            optimized_field, self.error_history, stop_reason = self.pattern_generator.optimize(
                initial_field=initial_field,
                algorithm=algorithm,
                max_iterations=iterations,
                tolerance=tolerance
            )
            
            # Get SLM phase pattern
            slm_field = self.pattern_generator.propagate(optimized_field)
            slm_phase = np.angle(slm_field)
            
            # Calculate and store reconstruction for preview
            image_field = self.pattern_generator.inverse_propagate(np.exp(1j * slm_phase))
            self.reconstruction = np.abs(image_field)**2
            
            # Normalize reconstruction for display
            self.reconstruction = self.reconstruction / np.max(self.reconstruction)
            
            self.status_var.set(f"Combined pattern generated using {algorithm.upper()} algorithm")
            return optimized_field, slm_phase, stop_reason
            
        except Exception as e:
            self.status_var.set(f"Error in combined pattern generation: {str(e)}")
            print(f"Detailed error: {str(e)}")
            return None, None, "Error in combined pattern generation"

    def pause_camera(self):
        """Pause or resume the camera feed"""
        if not self.camera_active:
            return
            
        try:
            self.camera_paused = not self.camera_paused
            button_text = "Resume Camera" if self.camera_paused else "Pause Camera"
            self.pause_camera_button.configure(text=button_text)
            
            if self.camera_paused:
                self.status_var.set("Camera feed paused")
            else:
                self.status_var.set("Camera feed resumed")
                
        except Exception as e:
            self.status_var.set(f"Error toggling camera pause: {str(e)}")
            print(f"Detailed error: {str(e)}")

    def set_exposure(self, exposure_ms):
        """Set camera exposure time in milliseconds"""
        try:
            exposure_us = int(exposure_ms * 1000)  # Convert to microseconds
            self.picam.set_controls({"ExposureTime": exposure_us})
            self.status_var.set(f"Exposure set to {exposure_ms}ms")
        except Exception as e:
            self.status_var.set(f"Error setting exposure: {str(e)}")

    def set_gain(self, gain):
        """Set camera analog gain"""
        try:
            self.picam.set_controls({"AnalogueGain": gain})
            self.status_var.set(f"Gain set to {gain}")
        except Exception as e:
            self.status_var.set(f"Error setting gain: {str(e)}")

    def capture_camera_image(self):
        """Capture current camera frame as target image"""
        try:
            if self.camera_active:
                frame = self.picam.capture_array()
                if frame is not None:
                    # Convert to grayscale
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Resize to match SLM resolution (keep camera preview at native resolution)
                    # Only resize when using the image for pattern generation
                    self.target = cv2.resize(gray, (800, 600))
                    
                    # Update preview
                    self.update_preview()
                    self.status_var.set("Image captured from camera")
                else:
                    self.status_var.set("Failed to capture image from camera")
        except Exception as e:
            self.status_var.set(f"Error capturing image: {str(e)}")

    def save_camera_image(self):
        """Save the current camera frame to a file"""
        if self.last_frame is None:
            self.status_var.set("No camera frame to save")
            return
            
        try:
            # Use zenity file dialog
            cmd = ['zenity', '--file-selection', '--save',
                   '--file-filter=Images | *.png *.jpg *.jpeg',
                   '--title=Save Camera Image']
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

    def _on_algorithm_change(self, *args):
        """Handle algorithm selection change"""
        if self.algorithm_var.get() == "mraf":
            self.mraf_frame.pack(fill=tk.X, padx=5, pady=5)
        else:
            self.mraf_frame.pack_forget()

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
        # Normalize target intensity to preserve energy
        self.target_intensity = self.target_intensity / np.sum(self.target_intensity)
        
        # If no signal region mask is provided, use the entire region
        if signal_region_mask is None:
            self.signal_region_mask = np.ones_like(target_intensity)
        else:
            self.signal_region_mask = signal_region_mask
            
        self.mixing_parameter = mixing_parameter
    
    def propagate(self, field):
        """Propagate field from image plane to SLM plane"""
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
    
    def inverse_propagate(self, field):
        """Propagate field from SLM plane to image plane"""
        return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(field)))
    
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
    
    def optimize(self, initial_field, algorithm='gs', max_iterations=100, tolerance=1e-6):
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
        
        # Run optimization loop
        for i in tqdm(range(max_iterations), desc=f"Running {algorithm.upper()} optimization"):
            # Apply iteration based on selected algorithm
            if algorithm.lower() == 'gs':
                field = self.gs_iteration(field)
            elif algorithm.lower() == 'mraf':
                field = self.mraf_iteration(field)
            else:
                raise ValueError("Algorithm must be 'gs' or 'mraf'")
                
            # Calculate normalized error for convergence check
            if algorithm.lower() == 'gs':
                # For GS, calculate normalized error over entire field (NMSE)
                current_intensity = np.abs(field)**2
                # Normalized Mean Square Error
                current_error = np.sum((current_intensity - self.target_intensity)**2) / np.sum(self.target_intensity**2)
            else:
                # For MRAF, calculate normalized error only in signal region (NMSE)
                sr_mask = self.signal_region_mask
                current_intensity = np.abs(field)**2
                target_sr = self.target_intensity[sr_mask == 1]
                current_sr = current_intensity[sr_mask == 1]
                # Normalized Mean Square Error in signal region
                current_error = np.sum((current_sr - target_sr)**2) / np.sum(target_sr**2)
            
            # Record error at every iteration
            error_history.append(current_error)
                
            # Print current error for debugging with scientific notation for very small values
            print(f"Iteration {i}, NMSE: {current_error:.3e}, Delta: {abs(current_error - prev_error):.3e}, Tolerance: {tolerance:.3e}")
            
            # Check convergence at every iteration
            if abs(current_error - prev_error) < tolerance:
                stop_reason = f"Convergence reached at iteration {i+1}: Error delta ({abs(current_error - prev_error):.3e}) < tolerance ({tolerance:.3e})"
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
        
        return field, error_history, stop_reason

if __name__ == "__main__":
    app = AdvancedPatternGenerator()
    app.run()
