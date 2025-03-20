"""
Image Intensity Distribution Visualizer

This script loads an image and visualizes its intensity distribution in various ways:
1. Original image
2. Intensity histogram
3. 2D intensity distribution
4. 3D surface plot of intensity

Compatible with the SLM phase mapping used in pattern_gen_2.0.py:
- Phase range: [-π to π]
- Grayscale mapping: 0 (black) → -π, 128 (gray) → 0, 255 (white) → π

Supports both 8-bit (0-255) and 10-bit (0-1023) image data.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import ttk, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import os
from scipy import ndimage
from matplotlib.widgets import Slider

import time
import traceback

class IntensityVisualizer:
    def __init__(self):
        """Initialize the intensity visualizer application"""
        self.image = None
        self.intensity = None
        self.phase = None
        self.bit_depth = 8  # Default bit depth
        self.max_value = 255  # Default max value for 8-bit
        
        # Create the main window
        self.root = tk.Tk()
        self.root.title("Image Intensity Distribution Visualizer")
        self.root.geometry("1200x800")
        
        # Create the main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create the control panel
        self.create_control_panel()
        
        # Create the visualization panel
        self.create_visualization_panel()
        
        # Create the status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def create_control_panel(self):
        """Create the control panel with buttons and options"""
        control_frame = ttk.LabelFrame(self.main_frame, text="Controls")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Load image button
        load_btn = ttk.Button(control_frame, text="Load Image", command=self.load_image)
        load_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # Capture from camera button
        capture_btn = ttk.Button(control_frame, text="Capture from Camera", command=self.capture_from_camera)
        capture_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # Bit depth frame
        bit_frame = ttk.LabelFrame(control_frame, text="Bit Depth")
        bit_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Bit depth options
        self.bit_depth_var = tk.IntVar(value=8)
        
        bit_options = [
            ("8-bit (0-255)", 8),
            ("10-bit (0-1023)", 10),
            ("12-bit (0-4095)", 12),
            ("16-bit (0-65535)", 16)
        ]
        
        for text, value in bit_options:
            ttk.Radiobutton(bit_frame, text=text, value=value, 
                           variable=self.bit_depth_var, command=self.update_bit_depth).pack(
                           anchor=tk.W, padx=5, pady=2)
        
        # Visualization type frame
        viz_frame = ttk.LabelFrame(control_frame, text="Visualization Type")
        viz_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Visualization type options
        self.viz_type = tk.StringVar(value="all")
        
        viz_options = [
            ("All Visualizations", "all"),
            ("Original Image", "original"),
            ("Intensity Histogram", "histogram"),
            ("2D Intensity Map", "intensity_2d"),
            ("3D Intensity Surface", "intensity_3d")
        ]
        
        for text, value in viz_options:
            ttk.Radiobutton(viz_frame, text=text, value=value, 
                           variable=self.viz_type, command=self.update_visualization).pack(
                           anchor=tk.W, padx=5, pady=2)
        
        # Color map selection
        cmap_frame = ttk.LabelFrame(control_frame, text="Color Map")
        cmap_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.colormap = tk.StringVar(value="viridis")
        cmaps = ["viridis", "plasma", "inferno", "magma", "jet", "hot", "cool", "rainbow", "gray"]
        
        cmap_combo = ttk.Combobox(cmap_frame, textvariable=self.colormap, values=cmaps)
        cmap_combo.pack(fill=tk.X, padx=5, pady=5)
        cmap_combo.bind("<<ComboboxSelected>>", lambda e: self.update_visualization())
        
        # Normalization options
        norm_frame = ttk.LabelFrame(control_frame, text="Normalization")
        norm_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.normalize = tk.BooleanVar(value=True)
        ttk.Checkbutton(norm_frame, text="Normalize Intensity", 
                       variable=self.normalize, command=self.update_visualization).pack(
                       anchor=tk.W, padx=5, pady=2)
        
        # Log scale option
        self.log_scale = tk.BooleanVar(value=False)
        ttk.Checkbutton(norm_frame, text="Logarithmic Scale", 
                       variable=self.log_scale, command=self.update_visualization).pack(
                       anchor=tk.W, padx=5, pady=2)
        
        # Add smoothing options
        smooth_frame = ttk.LabelFrame(control_frame, text="Histogram Smoothing")
        smooth_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Smoothing enabled checkbox
        self.smooth_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(smooth_frame, text="Enable Smoothing", 
                       variable=self.smooth_enabled, command=self.update_visualization).pack(
                       anchor=tk.W, padx=5, pady=2)
        
        # Smoothing amount slider
        slider_frame = ttk.Frame(smooth_frame)
        slider_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(slider_frame, text="Smoothing Amount:").pack(side=tk.LEFT)
        
        self.smooth_amount = tk.IntVar(value=3)
        smooth_slider = ttk.Scale(slider_frame, from_=1, to=20, 
                                 variable=self.smooth_amount, 
                                 orient=tk.HORIZONTAL,
                                 command=lambda e: self.update_visualization())
        smooth_slider.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
        
        # Add Gaussian fit options
        gauss_frame = ttk.LabelFrame(control_frame, text="Gaussian Fit")
        gauss_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Gaussian fit enabled checkbox
        self.gauss_fit_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(gauss_frame, text="Show Gaussian Fit", 
                       variable=self.gauss_fit_enabled, command=self.update_visualization).pack(
                       anchor=tk.W, padx=5, pady=2)
        
        # Contour levels for 2D plot
        contour_frame = ttk.Frame(gauss_frame)
        contour_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(contour_frame, text="Contour Levels:").pack(side=tk.LEFT)
        
        self.contour_levels = tk.IntVar(value=5)
        contour_slider = ttk.Scale(contour_frame, from_=3, to=15, 
                                  variable=self.contour_levels, 
                                  orient=tk.HORIZONTAL,
                                  command=lambda e: self.update_visualization())
        contour_slider.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
        
        # Add 3D plane controls
        planes_frame = ttk.LabelFrame(control_frame, text="3D Cross-Section Planes")
        planes_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Enable/disable planes
        planes_enable_frame = ttk.Frame(planes_frame)
        planes_enable_frame.pack(fill=tk.X, padx=5, pady=2)
        
        # XY plane (constant Z)
        xy_enable_frame = ttk.Frame(planes_frame)
        xy_enable_frame.pack(fill=tk.X, padx=5, pady=2)
        
        self.xy_plane_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(xy_enable_frame, text="XY Plane", 
                       variable=self.xy_plane_enabled, command=self.update_visualization).pack(
                       side=tk.LEFT, padx=5, pady=2)
        
        # XY plane position (Z value)
        self.xy_plane_pos = tk.DoubleVar(value=0.5)
        xy_slider = ttk.Scale(xy_enable_frame, from_=0.0, to=1.0, 
                             variable=self.xy_plane_pos, 
                             orient=tk.HORIZONTAL,
                             command=lambda e: self.update_visualization())
        xy_slider.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
        
        # YZ plane (constant X)
        yz_enable_frame = ttk.Frame(planes_frame)
        yz_enable_frame.pack(fill=tk.X, padx=5, pady=2)
        
        self.yz_plane_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(yz_enable_frame, text="YZ Plane", 
                       variable=self.yz_plane_enabled, command=self.update_visualization).pack(
                       side=tk.LEFT, padx=5, pady=2)
        
        # YZ plane position (X value)
        self.yz_plane_pos = tk.DoubleVar(value=0.5)
        yz_slider = ttk.Scale(yz_enable_frame, from_=0.0, to=1.0, 
                             variable=self.yz_plane_pos, 
                             orient=tk.HORIZONTAL,
                             command=lambda e: self.update_visualization())
        yz_slider.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
        
        # XZ plane (constant Y)
        xz_enable_frame = ttk.Frame(planes_frame)
        xz_enable_frame.pack(fill=tk.X, padx=5, pady=2)
        
        self.xz_plane_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(xz_enable_frame, text="XZ Plane", 
                       variable=self.xz_plane_enabled, command=self.update_visualization).pack(
                       side=tk.LEFT, padx=5, pady=2)
        
        # XZ plane position (Y value)
        self.xz_plane_pos = tk.DoubleVar(value=0.5)
        xz_slider = ttk.Scale(xz_enable_frame, from_=0.0, to=1.0, 
                             variable=self.xz_plane_pos, 
                             orient=tk.HORIZONTAL,
                             command=lambda e: self.update_visualization())
        xz_slider.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
        
        # Save visualization button
        save_btn = ttk.Button(control_frame, text="Save Visualization", command=self.save_visualization)
        save_btn.pack(fill=tk.X, padx=5, pady=5)
    
    def create_visualization_panel(self):
        """Create the visualization panel with matplotlib figures"""
        self.viz_frame = ttk.LabelFrame(self.main_frame, text="Visualization")
        self.viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create figure and canvas
        self.fig = plt.Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.viz_frame)
        self.toolbar.update()
        
        # Initialize with empty plots
        self.initialize_plots()
    
    def initialize_plots(self):
        """Initialize the plots with empty data"""
        self.fig.clear()
        
        # Show message to load an image
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, "Please load an image or capture from camera", 
                ha='center', va='center', fontsize=16)
        ax.axis('off')
        
        self.canvas.draw()
    
    def update_bit_depth(self):
        """Update the bit depth and recalculate intensity values"""
        self.bit_depth = self.bit_depth_var.get()
        self.max_value = 2**self.bit_depth - 1
        
        if self.image is not None:
            # Recalculate intensity and phase based on new bit depth
            self.calculate_intensity_and_phase()
            self.update_visualization()
    
    def calculate_intensity_and_phase(self):
        """Calculate intensity and phase based on current bit depth"""
        # Calculate intensity (normalized to 0-1)
        self.intensity = self.image.astype(float) / self.max_value
        
        # Calculate phase using SLM mapping (0 → -π, 128 → 0, 255 → π)
        # Adjust for different bit depths by scaling to the range [-π to π]
        self.phase = (self.intensity * 2 - 1) * np.pi
    
    def load_image(self):
        """Load an image file and prepare it for visualization"""
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("Raw files", "*.raw"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            # Load the image
            self.status_var.set(f"Loading image: {os.path.basename(file_path)}")
            self.root.update()
            
            # Check if it's a raw file
            if file_path.lower().endswith('.raw'):
                # Ask for dimensions for raw file
                raw_dialog = tk.Toplevel(self.root)
                raw_dialog.title("Raw Image Parameters")
                raw_dialog.geometry("300x200")
                raw_dialog.transient(self.root)
                raw_dialog.grab_set()
                
                ttk.Label(raw_dialog, text="Width:").grid(row=0, column=0, padx=5, pady=5)
                width_var = tk.IntVar(value=800)
                ttk.Entry(raw_dialog, textvariable=width_var).grid(row=0, column=1, padx=5, pady=5)
                
                ttk.Label(raw_dialog, text="Height:").grid(row=1, column=0, padx=5, pady=5)
                height_var = tk.IntVar(value=600)
                ttk.Entry(raw_dialog, textvariable=height_var).grid(row=1, column=1, padx=5, pady=5)
                
                ttk.Label(raw_dialog, text="Bit Depth:").grid(row=2, column=0, padx=5, pady=5)
                raw_bit_depth = tk.IntVar(value=self.bit_depth)
                ttk.Combobox(raw_dialog, textvariable=raw_bit_depth, 
                            values=[8, 10, 12, 16]).grid(row=2, column=1, padx=5, pady=5)
                
                result = [False]
                
                def on_ok():
                    result[0] = True
                    raw_dialog.destroy()
                
                ttk.Button(raw_dialog, text="OK", command=on_ok).grid(row=3, column=0, columnspan=2, pady=10)
                
                self.root.wait_window(raw_dialog)
                
                if not result[0]:
                    return
                
                # Update bit depth
                self.bit_depth = raw_bit_depth.get()
                self.bit_depth_var.set(self.bit_depth)
                self.max_value = 2**self.bit_depth - 1
                
                # Read raw file
                with open(file_path, 'rb') as f:
                    data = f.read()
                
                width = width_var.get()
                height = height_var.get()
                
                if self.bit_depth <= 8:
                    # 8-bit data
                    self.image = np.frombuffer(data, dtype=np.uint8).reshape(height, width)
                elif self.bit_depth <= 16:
                    # 10, 12, or 16-bit data
                    self.image = np.frombuffer(data, dtype=np.uint16).reshape(height, width)
                else:
                    raise ValueError(f"Unsupported bit depth: {self.bit_depth}")
                
                # Create RGB version for display
                self.image_rgb = cv2.cvtColor(
                    (self.image * (255.0 / self.max_value)).astype(np.uint8), 
                    cv2.COLOR_GRAY2RGB
                )
            else:
                # Read image using OpenCV
                image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                
                if image is None:
                    raise ValueError("Failed to load image")
                
                # Determine bit depth from image
                if image.dtype == np.uint8:
                    self.bit_depth = 8
                    self.max_value = 255
                elif image.dtype == np.uint16:
                    # Check actual bit depth based on max value
                    max_val = np.max(image)
                    if max_val <= 1023:
                        self.bit_depth = 10
                        self.max_value = 1023
                    elif max_val <= 4095:
                        self.bit_depth = 12
                        self.max_value = 4095
                    else:
                        self.bit_depth = 16
                        self.max_value = 65535
                
                # Update bit depth radio button
                self.bit_depth_var.set(self.bit_depth)
                
                # Convert to RGB for display if it's color
                if len(image.shape) == 3:
                    self.image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # Convert to grayscale for intensity analysis
                    self.image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    self.image = image
                    # Create RGB version for display
                    self.image_rgb = cv2.cvtColor(
                        (self.image * (255.0 / self.max_value)).astype(np.uint8), 
                        cv2.COLOR_GRAY2RGB
                    )
            
            # Calculate intensity and phase
            self.calculate_intensity_and_phase()
            
            self.status_var.set(
                f"Loaded image: {os.path.basename(file_path)} - "
                f"Shape: {self.image.shape} - "
                f"Bit Depth: {self.bit_depth}-bit (0-{self.max_value})"
            )
            
            # Update visualization
            self.update_visualization()
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            print(f"Error loading image: {str(e)}")
            traceback.print_exc()
    
    def capture_from_camera(self):
        """Capture an image from the Raspberry Pi camera"""
        try:
            self.status_var.set("Initializing camera...")
            self.root.update()
            
            # Initialize camera
            picam2 = Picamera2()
            
            # Configure camera for 10-bit capture
            config = picam2.create_still_configuration(
                main={"size": (1920, 1080), "format": "RGB888"}
            )
            
            # For 10-bit raw capture
            if self.bit_depth == 10:
                config = picam2.create_still_configuration(
                    main={"size": (1920, 1080), "format": "SRGGB10_CSI2P"},
                    raw={"size": picam2.sensor_resolution, "format": "SRGGB10"}
                )
            
            picam2.configure(config)
            picam2.start()
            
            # Allow camera to adjust
            self.status_var.set("Adjusting camera settings...")
            self.root.update()
            time.sleep(2)
            
            # Capture image
            self.status_var.set("Capturing image...")
            self.root.update()
            
            if self.bit_depth == 10:
                # Capture 10-bit raw image
                buffer = picam2.capture_array("raw")
                # Convert raw Bayer data to grayscale
                # This is a simplified conversion - a real implementation would do proper demosaicing
                self.image = buffer[0::2, 0::2]  # Just take one color channel for simplicity
                self.max_value = 1023  # 10-bit max value
            else:
                # Capture standard image
                buffer = picam2.capture_array("main")
                # Convert to grayscale
                self.image = cv2.cvtColor(buffer, cv2.COLOR_RGB2GRAY)
                self.max_value = 255  # 8-bit max value
                self.bit_depth = 8
            
            # Update bit depth radio button
            self.bit_depth_var.set(self.bit_depth)
            
            # Create RGB version for display
            self.image_rgb = cv2.cvtColor(
                (self.image * (255.0 / self.max_value)).astype(np.uint8), 
                cv2.COLOR_GRAY2RGB
            )
            
            # Calculate intensity and phase
            self.calculate_intensity_and_phase()
            
            # Stop camera
            picam2.stop()
            
            self.status_var.set(
                f"Captured image - Shape: {self.image.shape} - "
                f"Bit Depth: {self.bit_depth}-bit (0-{self.max_value})"
            )
            
            # Update visualization
            self.update_visualization()
            
        except Exception as e:
            self.status_var.set(f"Camera error: {str(e)}")
            print(f"Error capturing from camera: {str(e)}")
            traceback.print_exc()
    
    def update_visualization(self):
        """Update the visualization based on current settings"""
        if self.image is None:
            return
            
        # Clear the figure
        self.fig.clear()
        
        # Calculate intensity and phase
        self.calculate_intensity_and_phase()
        
        # Get raw pixel data for visualization
        raw_intensity = self.image.copy().astype(float)
        
        # Get normalized intensity data for certain visualizations
        if self.normalize.get():
            intensity = self.intensity.copy()
        else:
            # Use actual pixel values (not normalized)
            intensity = raw_intensity / self.max_value
        
        # Apply log scale if selected
        if self.log_scale.get() and np.max(intensity) > 0:
            # Add small value to avoid log(0)
            intensity = np.log1p(intensity)
            # Renormalize after log if needed
            if self.normalize.get():
                intensity = intensity / np.max(intensity)
        
        # Get the selected colormap
        cmap = self.colormap.get()
        
        # Apply smoothing to the intensity data if enabled
        if self.smooth_enabled.get():
            # Get smoothing amount from slider
            sigma = self.smooth_amount.get() / 5.0  # Scale down for finer control
            
            # Create a copy of the data for smoothing
            smoothed_intensity = ndimage.gaussian_filter(intensity, sigma=sigma)
            raw_smoothed_intensity = ndimage.gaussian_filter(raw_intensity, sigma=sigma)
        else:
            # Use original data
            smoothed_intensity = intensity
            raw_smoothed_intensity = raw_intensity
        
        # Fit Gaussian model if enabled
        if self.gauss_fit_enabled.get():
            # Fit 2D Gaussian to the intensity data
            self.gaussian_params = self.fit_2d_gaussian(smoothed_intensity)
            # Generate the Gaussian model
            self.gaussian_model = self.generate_gaussian_model(self.gaussian_params, smoothed_intensity.shape)
            # Scale the model to match raw intensity range if needed
            if not self.normalize.get():
                self.raw_gaussian_model = self.gaussian_model * self.max_value
            else:
                self.raw_gaussian_model = self.gaussian_model * self.max_value
        else:
            self.gaussian_params = None
            self.gaussian_model = None
            self.raw_gaussian_model = None
        
        # Get the selected visualization type
        viz_type = self.viz_type.get()
        
        if viz_type == "original":
            self.create_original_visualization()
        elif viz_type == "histogram":
            self.create_histogram_visualization(smoothed_intensity, raw_smoothed_intensity)
        elif viz_type == "intensity_2d":
            self.create_intensity_2d_visualization(smoothed_intensity, raw_smoothed_intensity, cmap)
        elif viz_type == "intensity_3d":
            self.create_intensity_3d_visualization(smoothed_intensity, raw_smoothed_intensity, cmap)
        else:
            # Default to showing all visualizations
            self.create_all_visualizations(smoothed_intensity, raw_smoothed_intensity, cmap)
        
        # Update the canvas
        self.fig.tight_layout()
        self.canvas.draw()
    
    def create_all_visualizations(self, intensity, raw_intensity, cmap):
        """Create all visualizations in a grid"""
        # Create a 2x2 grid
        gs = self.fig.add_gridspec(2, 2)
        
        # Original image
        ax1 = self.fig.add_subplot(gs[0, 0])
        ax1.imshow(self.image_rgb)
        ax1.set_title(f"Original Image ({self.bit_depth}-bit)")
        ax1.axis('off')
        
        # Intensity histogram
        ax2 = self.fig.add_subplot(gs[0, 1])
        
        # Determine which data to use for histogram
        if self.log_scale.get():
            hist_data = intensity.flatten()
            bins = 256
            title_prefix = "Log"
            x_limits = (0, 1)
        else:
            # Use actual pixel values for histogram
            hist_data = raw_intensity.flatten()
            bins = min(256, self.max_value)
            title_prefix = ""
            x_limits = (0, self.max_value)
        
        # Calculate histogram
        hist, bin_edges = np.histogram(hist_data, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Apply smoothing if enabled
        if self.smooth_enabled.get():
            # Get smoothing amount from slider
            sigma = self.smooth_amount.get() / 5.0  # Scale down for finer control
            
            # Apply Gaussian smoothing
            from scipy import ndimage
            hist_smooth = ndimage.gaussian_filter1d(hist, sigma=sigma)
            
            # Plot smoothed histogram as a line
            ax2.plot(bin_centers, hist_smooth, color='blue', linewidth=2, 
                    label=f'Smoothed (σ={sigma:.1f})')
            
            # Add original histogram as semi-transparent bars for reference
            ax2.bar(bin_centers, hist, width=bin_centers[1]-bin_centers[0], 
                   color='blue', alpha=0.3, label='Original')
            
            # Update title to indicate smoothing
            title_prefix = f"{title_prefix} Smoothed"
        else:
            # Plot regular histogram
            ax2.hist(hist_data, bins=bins, color='blue', alpha=0.7)
        
        # Set title and axis labels
        ax2.set_title(f"{title_prefix} Intensity Histogram ({self.bit_depth}-bit)")
        ax2.set_xlim(x_limits)
        
        # Set y-axis to logarithmic scale
        ax2.set_yscale('log')
        ax2.set_xlabel("Pixel Value")
        ax2.set_ylabel("Frequency (log scale)")
        ax2.grid(True, alpha=0.3)
        
        # Add vertical lines for key values in histogram
        if not self.log_scale.get():
            # Add line for mean value
            mean_val = np.mean(raw_intensity)
            ax2.axvline(mean_val, color='r', linestyle='--', 
                      label=f'Mean: {mean_val:.1f}')
            
            # Add line for median value
            median_val = np.median(raw_intensity)
            ax2.axvline(median_val, color='g', linestyle=':', 
                      label=f'Median: {median_val:.1f}')
            
            # Add line for max value
            max_val = np.max(raw_intensity)
            ax2.axvline(max_val, color='purple', linestyle='-.',
                      label=f'Max: {max_val}')
            
            ax2.legend(fontsize='small')
        
        # 2D intensity map
        ax3 = self.fig.add_subplot(gs[1, 0])
        
        # Use actual pixel values if not normalized
        if not self.normalize.get() and not self.log_scale.get():
            im = ax3.imshow(raw_intensity, cmap=cmap, vmin=0, vmax=self.max_value)
            ax3.set_title(f"2D Intensity Map (Raw Values)")
        else:
            im = ax3.imshow(intensity, cmap=cmap)
            ax3.set_title(f"2D Intensity Map ({self.bit_depth}-bit)")
            
        cbar = self.fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        
        if not self.normalize.get() and not self.log_scale.get():
            # Show actual pixel values in colorbar
            cbar.set_label(f"Pixel Value (0-{self.max_value})")
        else:
            # Show normalized values
            cbar.set_label("Normalized Intensity")
            
        ax3.axis('off')
        
        # 3D intensity surface plot
        ax4 = self.fig.add_subplot(gs[1, 1], projection='3d')
        
        # Create coordinate grid
        y, x = np.mgrid[0:raw_intensity.shape[0], 0:raw_intensity.shape[1]]
        
        # Downsample for performance if image is large
        if max(raw_intensity.shape) > 100:
            step = max(1, int(max(raw_intensity.shape) / 100))
            x = x[::step, ::step]
            y = y[::step, ::step]
            
            # Use actual pixel values if not normalized
            if not self.normalize.get() and not self.log_scale.get():
                intensity_3d = raw_intensity[::step, ::step]
            else:
                intensity_3d = intensity[::step, ::step]
        else:
            # Use actual pixel values if not normalized
            if not self.normalize.get() and not self.log_scale.get():
                intensity_3d = raw_intensity
            else:
                intensity_3d = intensity
        
        # Create the surface plot
        surf = ax4.plot_surface(x, y, intensity_3d, cmap=cmap, linewidth=0, antialiased=True, alpha=0.9)
        
        if not self.normalize.get() and not self.log_scale.get():
            ax4.set_title(f"3D Intensity Surface (Raw Values)")
        else:
            ax4.set_title(f"3D Intensity Surface ({self.bit_depth}-bit)")
            
        # Hide axis labels for cleaner look in the grid
        ax4.set_xlabel("")
        ax4.set_ylabel("")
        ax4.set_zlabel("")
        
        # Set z-axis limits based on data
        if not self.normalize.get() and not self.log_scale.get():
            ax4.set_zlim(0, self.max_value)
    
    def create_original_visualization(self):
        """Create visualization of the original image"""
        ax = self.fig.add_subplot(111)
        ax.imshow(self.image_rgb)
        ax.set_title(f"Original Image ({self.bit_depth}-bit)")
        ax.axis('off')
    
    def create_histogram_visualization(self, intensity, raw_intensity):
        """Create histogram visualization of intensity values"""
        ax = self.fig.add_subplot(111)
        
        # Determine which data to use
        if self.log_scale.get():
            hist_data = intensity.flatten()
            bins = 256
            title_prefix = "Log"
            x_limits = (0, 1)
        else:
            # Use actual pixel values for histogram
            hist_data = raw_intensity.flatten()
            bins = min(256, self.max_value)
            title_prefix = ""
            x_limits = (0, self.max_value)
        
        # Calculate histogram
        hist, bin_edges = np.histogram(hist_data, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Apply smoothing if enabled
        if self.smooth_enabled.get():
            # Get smoothing amount from slider
            sigma = self.smooth_amount.get() / 5.0  # Scale down for finer control
            
            # Apply Gaussian smoothing
            from scipy import ndimage
            hist_smooth = ndimage.gaussian_filter1d(hist, sigma=sigma)
            
            # Plot smoothed histogram as a line
            ax.plot(bin_centers, hist_smooth, color='blue', linewidth=2, 
                   label=f'Smoothed (σ={sigma:.1f})')
            
            # Add original histogram as semi-transparent bars for reference
            ax.bar(bin_centers, hist, width=bin_centers[1]-bin_centers[0], 
                  color='blue', alpha=0.3, label='Original')
            
            # Update title to indicate smoothing
            title_prefix = f"{title_prefix} Smoothed"
        else:
            # Plot regular histogram
            ax.hist(hist_data, bins=bins, color='blue', alpha=0.7)
        
        # Set title and axis labels
        ax.set_title(f"{title_prefix} Intensity Histogram ({self.bit_depth}-bit)")
        ax.set_xlim(x_limits)
        
        # Set y-axis to logarithmic scale
        ax.set_yscale('log')
        ax.set_xlabel("Pixel Value")
        ax.set_ylabel("Frequency (log scale)")
        ax.grid(True, alpha=0.3)
        
        # Add vertical lines for key values
        if not self.log_scale.get():
            # Add line for mean value
            mean_val = np.mean(raw_intensity)
            ax.axvline(mean_val, color='r', linestyle='--', 
                      label=f'Mean: {mean_val:.1f}')
            
            # Add line for median value
            median_val = np.median(raw_intensity)
            ax.axvline(median_val, color='g', linestyle=':', 
                      label=f'Median: {median_val:.1f}')
            
            # Add line for max value
            max_val = np.max(raw_intensity)
            ax.axvline(max_val, color='purple', linestyle='-.',
                      label=f'Max: {max_val}')
        
        # Add legend
        ax.legend(fontsize='small')
    
    def create_intensity_2d_visualization(self, intensity, raw_intensity, cmap):
        """Create 2D visualization of intensity distribution"""
        ax = self.fig.add_subplot(111)
        
        # Use actual pixel values if not normalized
        if not self.normalize.get() and not self.log_scale.get():
            im = ax.imshow(raw_intensity, cmap=cmap, vmin=0, vmax=self.max_value)
            ax.set_title(f"2D Intensity Map (Raw Values)")
        else:
            im = ax.imshow(intensity, cmap=cmap)
            ax.set_title(f"2D Intensity Map ({self.bit_depth}-bit)")
            
        cbar = self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        if not self.normalize.get() and not self.log_scale.get():
            # Show actual pixel values in colorbar
            cbar.set_label(f"Pixel Value (0-{self.max_value})")
        else:
            # Show normalized values
            cbar.set_label("Normalized Intensity")
            
        ax.axis('off')
        
        # Add Gaussian fit contours if enabled
        if self.gauss_fit_enabled.get() and self.gaussian_model is not None:
            # Get number of contour levels from slider
            levels = self.contour_levels.get()
            
            # Create contour levels
            if not self.normalize.get() and not self.log_scale.get():
                # Use raw intensity values for contours
                contour_data = self.raw_gaussian_model
                vmin, vmax = np.min(contour_data), np.max(contour_data)
                contour_levels = np.linspace(vmin + (vmax-vmin)*0.1, vmax, levels)
            else:
                # Use normalized values for contours
                contour_data = self.gaussian_model
                vmin, vmax = np.min(contour_data), np.max(contour_data)
                contour_levels = np.linspace(vmin + (vmax-vmin)*0.1, vmax, levels)
            
            # Plot contours
            contours = ax.contour(contour_data, levels=contour_levels, colors='r', linewidths=1.5, alpha=0.8)
            
            # Add contour labels
            ax.clabel(contours, inline=True, fontsize=8, fmt='%.2f')
            
            # Add text with Gaussian parameters
            if self.gaussian_params is not None:
                amplitude, x0, y0, sigma_x, sigma_y, theta, offset = self.gaussian_params
                
                # Convert to degrees for display
                theta_deg = np.degrees(theta)
                
                # Create text for parameters
                param_text = (
                    f"Gaussian Fit Parameters:\n"
                    f"Center: ({x0:.1f}, {y0:.1f})\n"
                    f"σx: {sigma_x:.2f}, σy: {sigma_y:.2f}\n"
                    f"θ: {theta_deg:.1f}°"
                )
                
                # Add text box with parameters
                ax.text(0.02, 0.98, param_text, transform=ax.transAxes,
                       fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Add axis labels
        ax.set_xlabel("X Pixel")
        ax.set_ylabel("Y Pixel")
    
    def create_intensity_3d_visualization(self, intensity, raw_intensity, cmap):
        """Create 3D surface plot of intensity distribution"""
        ax = self.fig.add_subplot(111, projection='3d')
        
        # Get dimensions
        height, width = intensity.shape
        
        # Create coordinate grid
        x = np.arange(0, width, 1)
        y = np.arange(0, height, 1)
        x, y = np.meshgrid(x, y)
        
        # Use actual pixel values if not normalized
        if not self.normalize.get() and not self.log_scale.get():
            intensity_3d = raw_intensity
            z_max = self.max_value
        else:
            intensity_3d = intensity
            z_max = 1.0
        
        # Create the surface plot
        surf = ax.plot_surface(x, y, intensity_3d, cmap=cmap, linewidth=0, antialiased=True, alpha=0.9)
        
        # Add cross-section planes if enabled
        # XY plane (constant Z)
        if self.xy_plane_enabled.get():
            # Calculate Z position based on slider value
            z_pos = self.xy_plane_pos.get() * z_max
            
            # Create XY plane
            xx, yy = np.meshgrid(np.arange(0, width), np.arange(0, height))
            zz = np.ones_like(xx) * z_pos
            
            # Plot the plane
            ax.plot_surface(xx, yy, zz, color='r', alpha=0.4)
            
            # Add contour plot on the plane
            xy_plane_data = intensity_3d.copy()
            levels = np.linspace(np.min(xy_plane_data), np.max(xy_plane_data), 10)
            cset = ax.contourf(xx, yy, xy_plane_data, zdir='z', offset=z_pos, 
                              levels=levels, cmap=cmap, alpha=0.6)
        
        # YZ plane (constant X)
        if self.yz_plane_enabled.get():
            # Calculate X position based on slider value
            x_pos = int(self.yz_plane_pos.get() * (width - 1))
            
            # Create YZ plane
            yy, zz = np.meshgrid(np.arange(0, height), np.linspace(0, z_max, 50))
            xx = np.ones_like(yy) * x_pos
            
            # Get intensity values along the YZ plane
            yz_intensity = intensity_3d[:, x_pos]
            
            # Plot the plane with intensity values
            yz_plane = np.zeros((50, height))
            for i in range(height):
                yz_plane[:, i] = np.linspace(0, yz_intensity[i], 50)
            
            ax.plot_surface(xx, yy, zz, facecolors=plt.cm.get_cmap(cmap)(yz_plane), alpha=0.8)
        
        # XZ plane (constant Y)
        if self.xz_plane_enabled.get():
            # Calculate Y position based on slider value
            y_pos = int(self.xz_plane_pos.get() * (height - 1))
            
            # Create XZ plane
            xx, zz = np.meshgrid(np.arange(0, width), np.linspace(0, z_max, 50))
            yy = np.ones_like(xx) * y_pos
            
            # Get intensity values along the XZ plane
            xz_intensity = intensity_3d[y_pos, :]
            
            # Plot the plane with intensity values
            xz_plane = np.zeros((50, width))
            for i in range(width):
                xz_plane[:, i] = np.linspace(0, xz_intensity[i], 50)
            
            ax.plot_surface(xx, yy, zz, facecolors=plt.cm.get_cmap(cmap)(xz_plane), alpha=0.8)
        
        # Add Gaussian fit surface if enabled
        if self.gauss_fit_enabled.get() and self.gaussian_model is not None:
            # Use appropriate model based on normalization
            if not self.normalize.get() and not self.log_scale.get():
                gaussian_surface = self.raw_gaussian_model
            else:
                gaussian_surface = self.gaussian_model
            
            # Plot the Gaussian fit as a wireframe
            wireframe = ax.plot_wireframe(x, y, gaussian_surface, color='r', linewidth=1, alpha=0.7)
            
            # Add text with Gaussian parameters
            if self.gaussian_params is not None:
                amplitude, x0, y0, sigma_x, sigma_y, theta, offset = self.gaussian_params
                
                # Convert to degrees for display
                theta_deg = np.degrees(theta)
                
                # Create text for parameters
                param_text = (
                    f"Gaussian Fit Parameters:\n"
                    f"Center: ({x0:.1f}, {y0:.1f})\n"
                    f"σx: {sigma_x:.2f}, σy: {sigma_y:.2f}\n"
                    f"θ: {theta_deg:.1f}°"
                )
                
                # Add text to the figure
                ax.text2D(0.02, 0.98, param_text, transform=ax.transAxes,
                         fontsize=9, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        if not self.normalize.get() and not self.log_scale.get():
            ax.set_title(f"3D Intensity Surface (Raw Values)")
        else:
            ax.set_title(f"3D Intensity Surface ({self.bit_depth}-bit)")
            
        # Add axis labels
        ax.set_xlabel("X Pixel")
        ax.set_ylabel("Y Pixel")
        ax.set_zlabel("Intensity")
        
        # Set z-axis limits based on data
        if not self.normalize.get() and not self.log_scale.get():
            ax.set_zlim(0, self.max_value)
    
    def create_xy_plane_visualization(self, intensity, raw_intensity, cmap):
        """Create XY plane visualization"""
        ax = self.fig.add_subplot(111)
        
        # Get dimensions
        height, width = intensity.shape
        
        # Calculate Z position based on slider value
        z_pos_percentage = self.xy_plane_pos.get()
        z_pos = z_pos_percentage * self.max_value
        
        # Create XY plane data
        xy_plane_data = raw_intensity.copy()
        
        # Use actual pixel values if not normalized
        if not self.normalize.get() and not self.log_scale.get():
            im = ax.imshow(xy_plane_data, cmap=cmap, vmin=0, vmax=self.max_value)
            ax.set_title(f"XY Plane (Z={z_pos:.1f}, {z_pos_percentage*100:.0f}%)")
        else:
            im = ax.imshow(intensity, cmap=cmap)
            ax.set_title(f"XY Plane (Z={z_pos_percentage:.2f}, {z_pos_percentage*100:.0f}%)")
            
        cbar = self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        if not self.normalize.get() and not self.log_scale.get():
            # Show actual pixel values in colorbar
            cbar.set_label(f"Pixel Value (0-{self.max_value})")
        else:
            # Show normalized values
            cbar.set_label("Normalized Intensity")
        
        # Add axis labels
        ax.set_xlabel("X Pixel")
        ax.set_ylabel("Y Pixel")
        
        # Add a slider directly on the visualization for adjusting Z position
        slider_ax = self.fig.add_axes([0.25, 0.02, 0.5, 0.03])
        z_slider = Slider(
            ax=slider_ax,
            label='Z Position',
            valmin=0,
            valmax=1,
            valinit=self.xy_plane_pos.get(),
            color='red'
        )
        
        def update_z(val):
            self.xy_plane_pos.set(val)
            self.update_visualization()
            
        z_slider.on_changed(update_z)
    
    def create_yz_plane_visualization(self, intensity, raw_intensity, cmap):
        """Create YZ plane visualization"""
        ax = self.fig.add_subplot(111)
        
        # Get dimensions
        height, width = intensity.shape
        
        # Calculate X position based on slider value
        x_pos_percentage = self.yz_plane_pos.get()
        x_pos = int(x_pos_percentage * (width - 1))
        
        # Create YZ plane data
        yz_slice = raw_intensity[:, x_pos]
        
        # Create a 2D representation of the YZ plane
        y_coords = np.arange(0, height)
        z_coords = np.linspace(0, self.max_value, 100)
        Y, Z = np.meshgrid(y_coords, z_coords)
        
        # Create intensity values for each point in the YZ plane
        YZ_intensity = np.zeros((100, height))
        for i in range(height):
            YZ_intensity[:, i] = np.linspace(0, yz_slice[i], 100)
        
        # Use actual pixel values if not normalized
        if not self.normalize.get() and not self.log_scale.get():
            im = ax.imshow(YZ_intensity, cmap=cmap, vmin=0, vmax=self.max_value, 
                          aspect='auto', origin='lower', extent=[0, height, 0, self.max_value])
            ax.set_title(f"YZ Plane (X={x_pos}, {x_pos_percentage*100:.0f}%)")
        else:
            normalized_YZ = YZ_intensity / self.max_value if self.max_value > 0 else YZ_intensity
            im = ax.imshow(normalized_YZ, cmap=cmap, aspect='auto', origin='lower', 
                          extent=[0, height, 0, 1.0])
            ax.set_title(f"YZ Plane (X={x_pos}, {x_pos_percentage*100:.0f}%)")
            
        cbar = self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        if not self.normalize.get() and not self.log_scale.get():
            # Show actual pixel values in colorbar
            cbar.set_label(f"Pixel Value (0-{self.max_value})")
        else:
            # Show normalized values
            cbar.set_label("Normalized Intensity")
        
        # Add axis labels
        ax.set_xlabel("Y Pixel")
        ax.set_ylabel("Z (Intensity)")
        
        # Add a slider directly on the visualization for adjusting X position
        slider_ax = self.fig.add_axes([0.25, 0.02, 0.5, 0.03])
        x_slider = Slider(
            ax=slider_ax,
            label='X Position',
            valmin=0,
            valmax=1,
            valinit=self.yz_plane_pos.get(),
            color='red'
        )
        
        def update_x(val):
            self.yz_plane_pos.set(val)
            self.update_visualization()
            
        x_slider.on_changed(update_x)
    
    def create_xz_plane_visualization(self, intensity, raw_intensity, cmap):
        """Create XZ plane visualization"""
        ax = self.fig.add_subplot(111)
        
        # Get dimensions
        height, width = intensity.shape
        
        # Calculate Y position based on slider value
        y_pos_percentage = self.xz_plane_pos.get()
        y_pos = int(y_pos_percentage * (height - 1))
        
        # Create XZ plane data
        xz_slice = raw_intensity[y_pos, :]
        
        # Create a 2D representation of the XZ plane
        x_coords = np.arange(0, width)
        z_coords = np.linspace(0, self.max_value, 100)
        X, Z = np.meshgrid(x_coords, z_coords)
        
        # Create intensity values for each point in the XZ plane
        XZ_intensity = np.zeros((100, width))
        for i in range(width):
            XZ_intensity[:, i] = np.linspace(0, xz_slice[i], 100)
        
        # Use actual pixel values if not normalized
        if not self.normalize.get() and not self.log_scale.get():
            im = ax.imshow(XZ_intensity, cmap=cmap, vmin=0, vmax=self.max_value, 
                          aspect='auto', origin='lower', extent=[0, width, 0, self.max_value])
            ax.set_title(f"XZ Plane (Y={y_pos}, {y_pos_percentage*100:.0f}%)")
        else:
            normalized_XZ = XZ_intensity / self.max_value if self.max_value > 0 else XZ_intensity
            im = ax.imshow(normalized_XZ, cmap=cmap, aspect='auto', origin='lower', 
                          extent=[0, width, 0, 1.0])
            ax.set_title(f"XZ Plane (Y={y_pos}, {y_pos_percentage*100:.0f}%)")
            
        cbar = self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        if not self.normalize.get() and not self.log_scale.get():
            # Show actual pixel values in colorbar
            cbar.set_label(f"Pixel Value (0-{self.max_value})")
        else:
            # Show normalized values
            cbar.set_label("Normalized Intensity")
        
        # Add axis labels
        ax.set_xlabel("X Pixel")
        ax.set_ylabel("Z (Intensity)")
        
        # Add a slider directly on the visualization for adjusting Y position
        slider_ax = self.fig.add_axes([0.25, 0.02, 0.5, 0.03])
        y_slider = Slider(
            ax=slider_ax,
            label='Y Position',
            valmin=0,
            valmax=1,
            valinit=self.xz_plane_pos.get(),
            color='red'
        )
        
        def update_y(val):
            self.xz_plane_pos.set(val)
            self.update_visualization()
            
        y_slider.on_changed(update_y)
    
    def create_phase_visualization(self, cmap):
        """Create visualization of the phase distribution"""
        ax = self.fig.add_subplot(111)
        im = ax.imshow(self.phase, cmap=cmap, vmin=-np.pi, vmax=np.pi)
        ax.set_title(f"Phase Distribution [-π to π] ({self.bit_depth}-bit)")
        cbar = self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        cbar.set_ticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
        ax.axis('off')
    
    def save_visualization(self):
        """Save the current visualization to a file"""
        if self.image is None:
            self.status_var.set("Error: No image loaded")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Visualization",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
            self.status_var.set(f"Visualization saved to: {os.path.basename(file_path)}")
        except Exception as e:
            self.status_var.set(f"Error saving file: {str(e)}")
    
    def fit_2d_gaussian(self, intensity):
        """Fit a 2D Gaussian to the intensity data
        
        Returns:
            tuple: (amplitude, x0, y0, sigma_x, sigma_y, theta, offset)
                amplitude: height of the gaussian
                x0, y0: center coordinates
                sigma_x, sigma_y: standard deviations in x and y directions
                theta: rotation angle
                offset: background offset
        """
        from scipy.optimize import curve_fit
        
        # Create coordinate grids
        height, width = intensity.shape
        y, x = np.mgrid[:height, :width]
        
        # Flatten the arrays
        x_flat = x.flatten()
        y_flat = y.flatten()
        z_flat = intensity.flatten()
        
        # Define 2D Gaussian function
        def gaussian_2d(coords, amplitude, x0, y0, sigma_x, sigma_y, theta, offset):
            x, y = coords
            x0 = float(x0)
            y0 = float(y0)
            
            # Rotation
            a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
            b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
            c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
            
            # Gaussian function
            exponent = a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) + c*((y-y0)**2)
            return offset + amplitude * np.exp(-exponent)
        
        # Initial guess
        # Find the maximum intensity point as initial center
        max_idx = np.argmax(z_flat)
        x0_guess = x_flat[max_idx]
        y0_guess = y_flat[max_idx]
        
        # Estimate amplitude and offset
        amplitude_guess = np.max(z_flat) - np.min(z_flat)
        offset_guess = np.min(z_flat)
        
        # Estimate standard deviations
        sigma_guess = min(width, height) / 8.0  # Initial guess for sigma
        
        # Initial parameter guess
        initial_guess = [amplitude_guess, x0_guess, y0_guess, sigma_guess, sigma_guess, 0, offset_guess]
        
        try:
            # Fit the 2D Gaussian
            popt, _ = curve_fit(lambda coords, *params: gaussian_2d((x_flat, y_flat), *params),
                               (x_flat, y_flat), z_flat, p0=initial_guess,
                               bounds=([0, 0, 0, 0, 0, -np.pi/4, 0],
                                      [np.inf, width, height, width/2, height/2, np.pi/4, np.inf]))
            
            return popt
        except (RuntimeError, ValueError) as e:
            print(f"Gaussian fitting error: {e}")
            # Return a default fit based on initial guess if fitting fails
            return initial_guess
    
    def generate_gaussian_model(self, params, shape):
        """Generate a 2D Gaussian model based on fit parameters
        
        Args:
            params: (amplitude, x0, y0, sigma_x, sigma_y, theta, offset)
            shape: (height, width) of the output array
            
        Returns:
            2D numpy array with the Gaussian model
        """
        amplitude, x0, y0, sigma_x, sigma_y, theta, offset = params
        height, width = shape
        
        # Create coordinate grids
        y, x = np.mgrid[:height, :width]
        
        # Rotation
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        
        # Gaussian function
        exponent = a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) + c*((y-y0)**2)
        gaussian = offset + amplitude * np.exp(-exponent)
        
        return gaussian
    
    def run(self):
        """Run the application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = IntensityVisualizer()
    app.run()
