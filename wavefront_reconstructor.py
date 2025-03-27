#!/usr/bin/env python3
"""
Wavefront Reconstructor

A standalone application for reconstructing wavefronts from Shack-Hartmann spot pattern images.
Allows loading of reference and measurement images, setting lenslet array parameters,
and visualizing the reconstructed wavefront in 2D and 3D.

This application preserves the scientific integrity of intensity measurements
without converting them to phase values during the image processing steps.
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

class SpotDetector:
    """Detects spots in Shack-Hartmann wavefront sensor images."""
    
    def __init__(self, threshold=50, min_distance=10, block_size=11):
        """
        Initialize the spot detector.
        
        Args:
            threshold: Threshold for spot detection (0-255)
            min_distance: Minimum distance between spots in pixels
            block_size: Block size for adaptive thresholding
        """
        self.threshold = threshold
        self.min_distance = min_distance
        self.block_size = block_size
        self.reference_spots = None
        
    def detect_spots(self, image):
        """
        Detect spots in an image.
        
        Args:
            image: Input image (grayscale)
            
        Returns:
            List of spot coordinates (x, y)
        """
        # Convert to grayscale if needed
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
            cv2.THRESH_BINARY, self.block_size, -self.threshold
        )
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate centroids
        spots = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                spots.append((cx, cy))
        
        # Filter spots that are too close
        filtered_spots = []
        for spot in spots:
            if all(np.sqrt((spot[0] - s[0])**2 + (spot[1] - s[1])**2) > self.min_distance 
                  for s in filtered_spots):
                filtered_spots.append(spot)
        
        return filtered_spots

class WavefrontReconstructor:
    """Reconstructs wavefront from spot displacements."""
    
    def __init__(self, lenslet_pitch_mm=1.5, focal_length_mm=100, pixel_size_um=32):
        """
        Initialize the wavefront reconstructor.
        
        Args:
            lenslet_pitch_mm: Pitch of the lenslets in mm
            focal_length_mm: Focal length of the lenslets in mm
            pixel_size_um: Pixel size in microns
        """
        self.lenslet_pitch_mm = lenslet_pitch_mm
        self.focal_length_mm = focal_length_mm
        self.pixel_size_um = pixel_size_um
        
    def reconstruct_wavefront(self, reference_spots, measurement_spots):
        """
        Reconstruct wavefront from spot displacements.
        
        Args:
            reference_spots: List of reference spot coordinates (x, y)
            measurement_spots: List of measurement spot coordinates (x, y)
            
        Returns:
            Reconstructed wavefront as 2D numpy array
        """
        if len(reference_spots) != len(measurement_spots):
            # Match spots if counts don't match
            measurement_spots = self._match_spots(reference_spots, measurement_spots)
        
        # Calculate displacements
        displacements = []
        for ref_spot, meas_spot in zip(reference_spots, measurement_spots):
            dx = meas_spot[0] - ref_spot[0]
            dy = meas_spot[1] - ref_spot[1]
            displacements.append((dx, dy))
        
        # Convert displacements to slopes
        # Slope = displacement / focal_length
        focal_length_pixels = self.focal_length_mm * 1000 / self.pixel_size_um
        slopes_x = [dx / focal_length_pixels for dx, dy in displacements]
        slopes_y = [dy / focal_length_pixels for dx, dy in displacements]
        
        # Create grid for wavefront
        x_positions = [spot[0] for spot in reference_spots]
        y_positions = [spot[1] for spot in reference_spots]
        
        # Determine grid dimensions
        min_x, max_x = min(x_positions), max(x_positions)
        min_y, max_y = min(y_positions), max(y_positions)
        
        # Calculate number of lenslets in each direction
        lenslet_pitch_pixels = self.lenslet_pitch_mm * 1000 / self.pixel_size_um
        num_lenslets_x = int((max_x - min_x) / lenslet_pitch_pixels) + 1
        num_lenslets_y = int((max_y - min_y) / lenslet_pitch_pixels) + 1
        
        # Create regular grid for wavefront
        grid_x = np.linspace(min_x, max_x, num_lenslets_x)
        grid_y = np.linspace(min_y, max_y, num_lenslets_y)
        X, Y = np.meshgrid(grid_x, grid_y)
        
        # Integrate slopes to get wavefront
        # Simple integration: cumulative sum of slopes
        wavefront = np.zeros((num_lenslets_y, num_lenslets_x))
        
        # Map slopes to grid points
        slope_x_grid = np.zeros_like(wavefront)
        slope_y_grid = np.zeros_like(wavefront)
        
        for (x, y), sx, sy in zip(reference_spots, slopes_x, slopes_y):
            # Find nearest grid point
            i = np.argmin(np.abs(grid_y - y))
            j = np.argmin(np.abs(grid_x - x))
            if 0 <= i < num_lenslets_y and 0 <= j < num_lenslets_x:
                slope_x_grid[i, j] = sx
                slope_y_grid[i, j] = sy
        
        # Integrate x-slopes along x-direction
        for i in range(num_lenslets_y):
            wavefront[i, :] = np.cumsum(slope_x_grid[i, :]) * lenslet_pitch_pixels
        
        # Integrate y-slopes along y-direction and add to wavefront
        for j in range(num_lenslets_x):
            wavefront[:, j] += np.cumsum(slope_y_grid[:, j]) * lenslet_pitch_pixels
        
        # Remove piston (mean value)
        wavefront -= np.mean(wavefront)
        
        return wavefront, X, Y
    
    def _match_spots(self, reference_spots, measurement_spots):
        """Match measurement spots to reference spots based on proximity."""
        matched_spots = []
        for ref_spot in reference_spots:
            # Find closest measurement spot
            distances = [np.sqrt((ref_spot[0] - m[0])**2 + (ref_spot[1] - m[1])**2) 
                        for m in measurement_spots]
            if distances:
                closest_idx = np.argmin(distances)
                matched_spots.append(measurement_spots[closest_idx])
            else:
                # If no measurement spots, use reference spot (zero displacement)
                matched_spots.append(ref_spot)
        
        return matched_spots

class WavefrontReconstructorApp:
    """Standalone application for wavefront reconstruction from images."""
    
    def __init__(self, root):
        """
        Initialize the application.
        
        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("Wavefront Reconstructor")
        self.root.geometry("1200x800")
        
        # Initialize variables
        self.reference_image = None
        self.measurement_image = None
        self.reference_spots = None
        self.measurement_spots = None
        self.wavefront = None
        self.use_generated_reference = tk.BooleanVar(value=False)
        
        # Initialize components
        self.spot_detector = SpotDetector()
        self.wavefront_reconstructor = WavefrontReconstructor()
        
        # Create GUI
        self._create_gui()
    
    def _create_gui(self):
        """Create the GUI components."""
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create left frame for controls and images
        left_frame = ttk.Frame(main_frame, width=400)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5)
        
        # Create right frame for visualizations
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create scrollable frame for controls
        control_canvas = tk.Canvas(left_frame, width=380)
        control_scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=control_canvas.yview)
        control_scrollable_frame = ttk.Frame(control_canvas)
        
        control_scrollable_frame.bind(
            "<Configure>",
            lambda e: control_canvas.configure(scrollregion=control_canvas.bbox("all"))
        )
        
        control_canvas.create_window((0, 0), window=control_scrollable_frame, anchor="nw")
        control_canvas.configure(yscrollcommand=control_scrollbar.set)
        
        # Add mouse wheel scrolling
        def _on_mousewheel(event):
            control_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        control_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        control_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add controls to scrollable frame
        # File controls
        file_frame = ttk.LabelFrame(control_scrollable_frame, text="Image Files")
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(file_frame, text="Load Reference Image", 
                  command=self._load_reference_image).pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(file_frame, text="Load Measurement Image", 
                  command=self._load_measurement_image).pack(fill=tk.X, padx=5, pady=5)
        
        # Reference generation option
        ttk.Checkbutton(file_frame, text="Generate ideal reference grid (no reference image needed)", 
                       variable=self.use_generated_reference).pack(fill=tk.X, padx=5, pady=5)
        
        # Lenslet array parameters
        lenslet_frame = ttk.LabelFrame(control_scrollable_frame, text="Lenslet Array Parameters")
        lenslet_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(lenslet_frame, text="Lenslet Pitch (mm):").pack(anchor=tk.W, padx=5, pady=2)
        self.lenslet_pitch_var = tk.DoubleVar(value=1.5)
        ttk.Entry(lenslet_frame, textvariable=self.lenslet_pitch_var).pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(lenslet_frame, text="Focal Length (mm):").pack(anchor=tk.W, padx=5, pady=2)
        self.focal_length_var = tk.DoubleVar(value=100.0)
        ttk.Entry(lenslet_frame, textvariable=self.focal_length_var).pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(lenslet_frame, text="Pixel Size (µm):").pack(anchor=tk.W, padx=5, pady=2)
        self.pixel_size_var = tk.DoubleVar(value=32.0)
        ttk.Entry(lenslet_frame, textvariable=self.pixel_size_var).pack(fill=tk.X, padx=5, pady=2)
        
        # Spot detection parameters
        spot_frame = ttk.LabelFrame(control_scrollable_frame, text="Spot Detection Parameters")
        spot_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(spot_frame, text="Threshold:").pack(anchor=tk.W, padx=5, pady=2)
        self.threshold_var = tk.IntVar(value=50)
        ttk.Scale(spot_frame, from_=0, to=255, variable=self.threshold_var, 
                 orient=tk.HORIZONTAL).pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(spot_frame, text="Minimum Distance (pixels):").pack(anchor=tk.W, padx=5, pady=2)
        self.min_distance_var = tk.IntVar(value=10)
        ttk.Scale(spot_frame, from_=1, to=50, variable=self.min_distance_var, 
                 orient=tk.HORIZONTAL).pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(spot_frame, text="Block Size:").pack(anchor=tk.W, padx=5, pady=2)
        self.block_size_var = tk.IntVar(value=11)
        block_size_scale = ttk.Scale(spot_frame, from_=3, to=51, variable=self.block_size_var, 
                                    orient=tk.HORIZONTAL)
        block_size_scale.pack(fill=tk.X, padx=5, pady=2)
        
        # Ensure block size is odd
        def _update_block_size(event):
            value = self.block_size_var.get()
            if value % 2 == 0:
                self.block_size_var.set(value + 1)
        block_size_scale.bind("<ButtonRelease-1>", _update_block_size)
        
        # Process buttons
        process_frame = ttk.LabelFrame(control_scrollable_frame, text="Processing")
        process_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(process_frame, text="Detect Spots", 
                  command=self._detect_spots).pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(process_frame, text="Reconstruct Wavefront", 
                  command=self._reconstruct_wavefront).pack(fill=tk.X, padx=5, pady=5)
        
        # Image preview
        preview_frame = ttk.LabelFrame(control_scrollable_frame, text="Image Preview")
        preview_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.preview_label = ttk.Label(preview_frame)
        self.preview_label.pack(padx=5, pady=5)
        
        # Create visualization notebook
        viz_notebook = ttk.Notebook(right_frame)
        viz_notebook.pack(fill=tk.BOTH, expand=True)
        
        # 2D wavefront tab
        self.wavefront_2d_frame = ttk.Frame(viz_notebook)
        viz_notebook.add(self.wavefront_2d_frame, text="2D Wavefront")
        
        # 3D wavefront tab
        self.wavefront_3d_frame = ttk.Frame(viz_notebook)
        viz_notebook.add(self.wavefront_3d_frame, text="3D Wavefront")
        
        # Spots tab
        self.spots_frame = ttk.Frame(viz_notebook)
        viz_notebook.add(self.spots_frame, text="Detected Spots")
        
        # Create initial plots
        self._create_plots()
    
    def _create_plots(self):
        """Create the initial plots."""
        # 2D wavefront plot
        self.fig_2d = Figure(figsize=(6, 5), dpi=100)
        self.ax_2d = self.fig_2d.add_subplot(111)
        self.ax_2d.set_title("2D Wavefront")
        self.ax_2d.set_xlabel("X (pixels)")
        self.ax_2d.set_ylabel("Y (pixels)")
        
        self.canvas_2d = FigureCanvasTkAgg(self.fig_2d, master=self.wavefront_2d_frame)
        self.canvas_2d.draw()
        self.canvas_2d.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar_2d = NavigationToolbar2Tk(self.canvas_2d, self.wavefront_2d_frame)
        toolbar_2d.update()
        
        # 3D wavefront plot
        self.fig_3d = Figure(figsize=(6, 5), dpi=100)
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        self.ax_3d.set_title("3D Wavefront")
        self.ax_3d.set_xlabel("X (pixels)")
        self.ax_3d.set_ylabel("Y (pixels)")
        self.ax_3d.set_zlabel("Wavefront (waves)")
        
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, master=self.wavefront_3d_frame)
        self.canvas_3d.draw()
        self.canvas_3d.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar_3d = NavigationToolbar2Tk(self.canvas_3d, self.wavefront_3d_frame)
        toolbar_3d.update()
        
        # Spots plot
        self.fig_spots = Figure(figsize=(6, 5), dpi=100)
        self.ax_spots = self.fig_spots.add_subplot(111)
        self.ax_spots.set_title("Detected Spots")
        self.ax_spots.set_xlabel("X (pixels)")
        self.ax_spots.set_ylabel("Y (pixels)")
        
        self.canvas_spots = FigureCanvasTkAgg(self.fig_spots, master=self.spots_frame)
        self.canvas_spots.draw()
        self.canvas_spots.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar_spots = NavigationToolbar2Tk(self.canvas_spots, self.spots_frame)
        toolbar_spots.update()
    
    def _load_reference_image(self):
        """Load reference image from file."""
        file_path = filedialog.askopenfilename(
            title="Select Reference Image",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")]
        )
        
        if file_path:
            self.reference_image = cv2.imread(file_path)
            if self.reference_image is None:
                messagebox.showerror("Error", "Failed to load image")
                return
            
            # Display preview
            self._display_preview(self.reference_image, "Reference Image")
    
    def _load_measurement_image(self):
        """Load measurement image from file."""
        file_path = filedialog.askopenfilename(
            title="Select Measurement Image",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")]
        )
        
        if file_path:
            self.measurement_image = cv2.imread(file_path)
            if self.measurement_image is None:
                messagebox.showerror("Error", "Failed to load image")
                return
            
            # Display preview
            self._display_preview(self.measurement_image, "Measurement Image")
    
    def _display_preview(self, image, title):
        """Display image preview."""
        # Convert to RGB for display
        if image is not None:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize for preview
            height, width = rgb_image.shape[:2]
            max_size = 300
            if width > max_size or height > max_size:
                scale = max_size / max(width, height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                rgb_image = cv2.resize(rgb_image, (new_width, new_height))
            
            # Convert to PhotoImage
            pil_image = Image.fromarray(rgb_image)
            tk_image = ImageTk.PhotoImage(pil_image)
            
            # Update preview label
            self.preview_label.configure(image=tk_image)
            self.preview_label.image = tk_image  # Keep reference
    
    def _generate_ideal_reference_grid(self):
        """Generate an ideal reference grid based on lenslet parameters."""
        if self.measurement_image is None:
            messagebox.showerror("Error", "Measurement image not loaded")
            return None
        
        # Get image dimensions
        height, width = self.measurement_image.shape[:2]
        
        # Calculate lenslet pitch in pixels
        pixel_size_mm = self.pixel_size_var.get() / 1000  # Convert from µm to mm
        lenslet_pitch_pixels = self.lenslet_pitch_var.get() / pixel_size_mm
        
        # Calculate number of lenslets in each direction
        num_lenslets_x = int(width / lenslet_pitch_pixels) + 1
        num_lenslets_y = int(height / lenslet_pitch_pixels) + 1
        
        # Calculate the offset to center the pattern
        offset_x = (width - (num_lenslets_x - 1) * lenslet_pitch_pixels) / 2
        offset_y = (height - (num_lenslets_y - 1) * lenslet_pitch_pixels) / 2
        
        # Create reference spots at regular intervals
        reference_spots = []
        for i in range(num_lenslets_y):
            for j in range(num_lenslets_x):
                # Calculate spot position
                x = int(offset_x + j * lenslet_pitch_pixels)
                y = int(offset_y + i * lenslet_pitch_pixels)
                
                # Ensure spot is within image boundaries
                if 0 <= x < width and 0 <= y < height:
                    reference_spots.append((x, y))
        
        return reference_spots
    
    def _detect_spots(self):
        """Detect spots in reference and measurement images."""
        # Update spot detector parameters
        self.spot_detector.threshold = self.threshold_var.get()
        self.spot_detector.min_distance = self.min_distance_var.get()
        self.spot_detector.block_size = self.block_size_var.get()
        
        # Detect spots in reference image if available and not using generated reference
        if not self.use_generated_reference.get():
            if self.reference_image is None:
                messagebox.showerror("Error", "Reference image not loaded")
                return
            self.reference_spots = self.spot_detector.detect_spots(self.reference_image)
        
        # Detect spots in measurement image if available
        if self.measurement_image is not None:
            self.measurement_spots = self.spot_detector.detect_spots(self.measurement_image)
            
            # Generate ideal reference grid if needed
            if self.use_generated_reference.get():
                self.reference_spots = self._generate_ideal_reference_grid()
                if not self.reference_spots:
                    return
        else:
            messagebox.showerror("Error", "Measurement image not loaded")
            return
        
        # Plot detected spots
        self._plot_spots()
        
        messagebox.showinfo("Spot Detection", 
                           f"Detected {len(self.reference_spots)} reference spots " +
                           f"({'generated' if self.use_generated_reference.get() else 'from image'})\n" +
                           f"Detected {len(self.measurement_spots)} spots in measurement image")
    
    def _plot_spots(self):
        """Plot detected spots."""
        self.ax_spots.clear()
        self.ax_spots.set_title("Detected Spots")
        self.ax_spots.set_xlabel("X (pixels)")
        self.ax_spots.set_ylabel("Y (pixels)")
        
        # Plot reference spots
        if self.reference_spots:
            x = [spot[0] for spot in self.reference_spots]
            y = [spot[1] for spot in self.reference_spots]
            self.ax_spots.plot(x, y, 'bo', label="Reference")
        
        # Plot measurement spots
        if self.measurement_spots:
            x = [spot[0] for spot in self.measurement_spots]
            y = [spot[1] for spot in self.measurement_spots]
            self.ax_spots.plot(x, y, 'rx', label="Measurement")
        
        self.ax_spots.legend()
        self.ax_spots.grid(True)
        self.canvas_spots.draw()
    
    def _reconstruct_wavefront(self):
        """Reconstruct wavefront from spot displacements."""
        if not self.reference_spots:
            messagebox.showerror("Error", "No reference spots detected")
            return
        
        if not self.measurement_spots:
            messagebox.showerror("Error", "No measurement spots detected")
            return
        
        # Update wavefront reconstructor parameters
        self.wavefront_reconstructor.lenslet_pitch_mm = self.lenslet_pitch_var.get()
        self.wavefront_reconstructor.focal_length_mm = self.focal_length_var.get()
        self.wavefront_reconstructor.pixel_size_um = self.pixel_size_var.get()
        
        # Reconstruct wavefront
        self.wavefront, X, Y = self.wavefront_reconstructor.reconstruct_wavefront(
            self.reference_spots, self.measurement_spots
        )
        
        # Plot wavefront
        self._plot_wavefront(self.wavefront, X, Y)
        
        messagebox.showinfo("Wavefront Reconstruction", "Wavefront reconstructed successfully")
    
    def _plot_wavefront(self, wavefront, X, Y):
        """Plot wavefront in 2D and 3D."""
        # 2D plot
        self.ax_2d.clear()
        self.ax_2d.set_title("2D Wavefront")
        self.ax_2d.set_xlabel("X (pixels)")
        self.ax_2d.set_ylabel("Y (pixels)")
        
        contour = self.ax_2d.contourf(X, Y, wavefront, 50, cmap='viridis')
        self.fig_2d.colorbar(contour, ax=self.ax_2d, label="Wavefront (waves)")
        self.canvas_2d.draw()
        
        # 3D plot
        self.ax_3d.clear()
        self.ax_3d.set_title("3D Wavefront")
        self.ax_3d.set_xlabel("X (pixels)")
        self.ax_3d.set_ylabel("Y (pixels)")
        self.ax_3d.set_zlabel("Wavefront (waves)")
        
        surf = self.ax_3d.plot_surface(X, Y, wavefront, cmap='viridis', edgecolor='none')
        self.fig_3d.colorbar(surf, ax=self.ax_3d, label="Wavefront (waves)")
        self.canvas_3d.draw()

def main():
    """Main function."""
    root = tk.Tk()
    app = WavefrontReconstructorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
