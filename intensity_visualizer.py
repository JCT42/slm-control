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
import os

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
            ("3D Intensity Surface", "intensity_3d"),
            ("Phase Distribution", "phase")
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
        
        # Calculate phase using SLM mapping (0 -> -π, 128 -> 0, 255 -> π)
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
        
        # Get current visualization type
        viz_type = self.viz_type.get()
        
        # Get raw pixel data for visualization
        raw_intensity = self.image.copy()
        
        # Get normalized intensity data for certain visualizations
        if self.normalize.get():
            intensity = self.intensity.copy()
        else:
            # Use actual pixel values (not normalized)
            intensity = raw_intensity.astype(float)
        
        # Apply log scale if selected
        if self.log_scale.get() and np.max(intensity) > 0:
            # Add small value to avoid log(0)
            intensity = np.log1p(intensity)
            # Renormalize after log if needed
            if self.normalize.get():
                intensity = intensity / np.max(intensity)
        
        # Get colormap
        cmap = self.colormap.get()
        
        # Create appropriate visualizations based on type
        if viz_type == "all":
            self.create_all_visualizations(intensity, raw_intensity, cmap)
        elif viz_type == "original":
            self.create_original_visualization()
        elif viz_type == "histogram":
            self.create_histogram_visualization(intensity, raw_intensity)
        elif viz_type == "intensity_2d":
            self.create_intensity_2d_visualization(intensity, raw_intensity, cmap)
        elif viz_type == "intensity_3d":
            self.create_intensity_3d_visualization(intensity, raw_intensity, cmap)
        elif viz_type == "phase":
            self.create_phase_visualization(cmap)
        
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
        if self.log_scale.get():
            ax2.hist(intensity.flatten(), bins=256, color='blue', alpha=0.7)
            ax2.set_title(f"Log Intensity Histogram ({self.bit_depth}-bit)")
        else:
            # Use actual pixel values for histogram
            ax2.hist(raw_intensity.flatten(), bins=min(256, self.max_value), color='blue', alpha=0.7)
            ax2.set_title(f"Intensity Histogram ({self.bit_depth}-bit)")
            # Set x-axis limits based on bit depth
            ax2.set_xlim(0, self.max_value)
        ax2.set_xlabel("Pixel Value")
        ax2.set_ylabel("Frequency")
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
            ax3.set_title(f"Intensity Distribution (Raw Values)")
        else:
            im = ax3.imshow(intensity, cmap=cmap)
            ax3.set_title(f"Intensity Distribution ({self.bit_depth}-bit)")
            
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
        surf = ax4.plot_surface(x, y, intensity_3d, cmap=cmap, linewidth=0, antialiased=True)
        
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
        
        if self.log_scale.get():
            ax.hist(intensity.flatten(), bins=256, color='blue', alpha=0.7)
            ax.set_title(f"Log Intensity Histogram ({self.bit_depth}-bit)")
        else:
            # Use actual pixel values for histogram
            ax.hist(raw_intensity.flatten(), bins=min(256, self.max_value), color='blue', alpha=0.7)
            ax.set_title(f"Intensity Histogram ({self.bit_depth}-bit)")
            # Set x-axis limits based on bit depth
            ax.set_xlim(0, self.max_value)
            
        ax.set_xlabel("Pixel Value")
        ax.set_ylabel("Frequency")
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
            
            ax.legend()
    
    def create_intensity_2d_visualization(self, intensity, raw_intensity, cmap):
        """Create 2D visualization of intensity distribution"""
        ax = self.fig.add_subplot(111)
        
        # Use actual pixel values if not normalized
        if not self.normalize.get() and not self.log_scale.get():
            im = ax.imshow(raw_intensity, cmap=cmap, vmin=0, vmax=self.max_value)
            ax.set_title(f"Intensity Distribution (Raw Values)")
        else:
            im = ax.imshow(intensity, cmap=cmap)
            ax.set_title(f"Intensity Distribution ({self.bit_depth}-bit)")
        
        # Add colorbar with actual pixel value range
        cbar = self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        if not self.normalize.get() and not self.log_scale.get():
            # Show actual pixel values in colorbar
            cbar.set_label(f"Pixel Value (0-{self.max_value})")
        else:
            # Show normalized values
            cbar.set_label("Normalized Intensity")
            
        ax.axis('off')
    
    def create_intensity_3d_visualization(self, intensity, raw_intensity, cmap):
        """Create 3D surface plot of intensity distribution"""
        ax = self.fig.add_subplot(111, projection='3d')
        
        # Create coordinate grid
        y, x = np.mgrid[0:raw_intensity.shape[0], 0:raw_intensity.shape[1]]
        
        # Downsample for performance if image is large
        if max(raw_intensity.shape) > 200:
            step = max(1, int(max(raw_intensity.shape) / 200))
            x = x[::step, ::step]
            y = y[::step, ::step]
            
            # Use actual pixel values if not normalized
            if not self.normalize.get() and not self.log_scale.get():
                intensity_plot = raw_intensity[::step, ::step]
            else:
                intensity_plot = intensity[::step, ::step]
        else:
            # Use actual pixel values if not normalized
            if not self.normalize.get() and not self.log_scale.get():
                intensity_plot = raw_intensity
            else:
                intensity_plot = intensity
        
        # Create the surface plot
        surf = ax.plot_surface(x, y, intensity_plot, cmap=cmap, linewidth=0, antialiased=True)
        
        if not self.normalize.get() and not self.log_scale.get():
            ax.set_title(f"3D Intensity Surface (Raw Values)")
        else:
            ax.set_title(f"3D Intensity Surface ({self.bit_depth}-bit)")
            
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        
        if not self.normalize.get() and not self.log_scale.get():
            # Show actual pixel values on z-axis
            ax.set_zlabel(f"Pixel Value (0-{self.max_value})")
            ax.set_zlim(0, self.max_value)
        else:
            # Show normalized values
            ax.set_zlabel("Normalized Intensity")
            ax.set_zlim(0, 1.0)
        
        # Add colorbar
        cbar = self.fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        if not self.normalize.get() and not self.log_scale.get():
            # Show actual pixel values in colorbar
            cbar.set_label(f"Pixel Value (0-{self.max_value})")
        else:
            # Show normalized values
            cbar.set_label("Normalized Intensity")
    
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
    
    def run(self):
        """Run the application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = IntensityVisualizer()
    app.run()
