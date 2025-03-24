#!/usr/bin/env python3
"""
Shack-Hartmann Wavefront Sensor Implementation for SLM

This module provides a Shack-Hartmann wavefront sensor implementation using:
1. An SLM to display a lenslet array pattern
2. A camera to capture the resulting spot pattern
3. Analysis tools to reconstruct the wavefront

Features:
- Generate lenslet array patterns for SLM display
- Process camera images to detect spot positions
- Reconstruct wavefront using zonal or modal methods
- Visualize wavefront and aberrations
- Integration with existing SLM and camera controllers
"""

import os
import time
import threading
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from scipy.interpolate import griddata, RectBivariateSpline
from scipy.ndimage import gaussian_filter
import json

# Try to import camera controller
try:
    from camera_controller import CameraController
except ImportError:
    print("Warning: camera_controller module not found. Camera functionality will be limited.")
    CameraController = None

class LensletArrayGenerator:
    """
    Generates phase patterns for lenslet arrays to be displayed on an SLM.
    """
    
    def __init__(self, slm_width=800, slm_height=600, pixel_pitch=32e-6, wavelength=650e-9):
        """
        Initialize the lenslet array generator.
        
        Args:
            slm_width: Width of the SLM in pixels
            slm_height: Height of the SLM in pixels
            pixel_pitch: SLM pixel pitch in meters
            wavelength: Wavelength of light in meters
        """
        self.slm_width = slm_width
        self.slm_height = slm_height
        self.pixel_pitch = pixel_pitch
        self.wavelength = wavelength
        
        # Physical dimensions of SLM
        self.slm_width_m = slm_width * pixel_pitch
        self.slm_height_m = slm_height * pixel_pitch
        
        # Wave number
        self.k = 2 * np.pi / wavelength
        
        # Create coordinate grids
        self.x = np.linspace(-self.slm_width_m/2, self.slm_width_m/2, slm_width)
        self.y = np.linspace(-self.slm_height_m/2, self.slm_height_m/2, slm_height)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
    def generate_lenslet_array(self, lenslet_pitch, focal_length):
        """
        Generate a phase pattern for a lenslet array.
        
        Args:
            lenslet_pitch: Distance between lenslet centers in meters
            focal_length: Focal length of each lenslet in meters
            
        Returns:
            Phase pattern for the lenslet array (2D numpy array)
        """
        # Number of lenslets in each dimension
        num_lenslets_x = int(self.slm_width_m / lenslet_pitch)
        num_lenslets_y = int(self.slm_height_m / lenslet_pitch)
        
        # Initialize phase pattern
        phase = np.zeros((self.slm_height, self.slm_width))
        
        # Generate lenslet array
        for i in range(num_lenslets_y):
            for j in range(num_lenslets_x):
                # Center of current lenslet
                x0 = -self.slm_width_m/2 + (j + 0.5) * lenslet_pitch
                y0 = -self.slm_height_m/2 + (i + 0.5) * lenslet_pitch
                
                # Distance from lenslet center
                r_squared = (self.X - x0)**2 + (self.Y - y0)**2
                
                # Phase of a lens: φ(r) = -k*r²/(2f)
                lens_phase = -self.k * r_squared / (2 * focal_length)
                
                # Apply lens phase within lenslet area
                mask = (np.abs(self.X - x0) < lenslet_pitch/2) & (np.abs(self.Y - y0) < lenslet_pitch/2)
                phase[mask] = lens_phase[mask]
        
        # Wrap phase to [-π, π] range as required by the SLM
        phase = np.mod(phase + np.pi, 2 * np.pi) - np.pi
        
        return phase
    
    def phase_to_grayscale(self, phase, gamma=1.0):
        """
        Convert phase values to grayscale values for SLM display.
        Maps the phase range [-π, π] to grayscale [0, 255].
        
        Args:
            phase: Phase pattern (2D numpy array)
            gamma: Gamma correction factor
            
        Returns:
            Grayscale pattern (2D numpy array, uint8)
        """
        # Map [-π, π] to [0, 1]
        normalized_phase = (phase + np.pi) / (2 * np.pi)
        
        # Apply gamma correction and scale to [0, 255]
        grayscale = (normalized_phase ** gamma * 255).astype(np.uint8)
        
        return grayscale
    
    def add_blazed_grating(self, phase, shift_x=1.0, shift_y=0.0):
        """
        Add a blazed grating to the phase pattern to shift the pattern
        away from the zero-order diffraction.
        
        Args:
            phase: Original phase pattern (2D numpy array)
            shift_x: Phase shift in x-direction (cycles per image)
            shift_y: Phase shift in y-direction (cycles per image)
            
        Returns:
            Modified phase pattern with blazed grating (2D numpy array)
        """
        # Normalized coordinates
        x_norm = self.X / self.slm_width_m
        y_norm = self.Y / self.slm_height_m
        
        # Calculate linear phase ramp (blazed grating)
        phase_ramp = 2 * np.pi * (shift_x * x_norm + shift_y * y_norm)
        
        # Add phase ramp to original phase
        shifted_phase = np.mod(phase + phase_ramp + np.pi, 2 * np.pi) - np.pi
        
        return shifted_phase

class SpotDetector:
    """
    Detects and analyzes spot patterns from Shack-Hartmann sensor images.
    """
    
    def __init__(self, num_lenslets_x, num_lenslets_y, reference_spots=None):
        """
        Initialize the spot detector.
        
        Args:
            num_lenslets_x: Number of lenslets in x-direction
            num_lenslets_y: Number of lenslets in y-direction
            reference_spots: Reference spot positions (2D numpy array, shape (n, 2))
        """
        self.num_lenslets_x = num_lenslets_x
        self.num_lenslets_y = num_lenslets_y
        self.reference_spots = reference_spots
        
    def detect_spots(self, image, threshold=50, min_distance=10):
        """
        Detect spots in an image using blob detection.
        
        Args:
            image: Input image (2D numpy array)
            threshold: Intensity threshold for spot detection
            min_distance: Minimum distance between spots
            
        Returns:
            List of spot positions (x, y)
        """
        # Ensure image is grayscale
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to create binary image
        _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
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
            if not any(np.sqrt((spot[0] - s[0])**2 + (spot[1] - s[1])**2) < min_distance 
                      for s in filtered_spots):
                filtered_spots.append(spot)
        
        return np.array(filtered_spots)
    
    def calculate_spot_shifts(self, spots):
        """
        Calculate spot shifts relative to reference positions.
        
        Args:
            spots: Detected spot positions (2D numpy array, shape (n, 2))
            
        Returns:
            Spot shifts (2D numpy array, shape (n, 2))
        """
        if self.reference_spots is None:
            # If no reference spots are provided, create a regular grid
            x_spacing = spots[:, 0].max() / (self.num_lenslets_x - 1)
            y_spacing = spots[:, 1].max() / (self.num_lenslets_y - 1)
            
            x_ref = np.linspace(0, spots[:, 0].max(), self.num_lenslets_x)
            y_ref = np.linspace(0, spots[:, 1].max(), self.num_lenslets_y)
            
            X_ref, Y_ref = np.meshgrid(x_ref, y_ref)
            self.reference_spots = np.column_stack((X_ref.flatten(), Y_ref.flatten()))
        
        # Match detected spots to reference spots
        shifts = np.zeros_like(self.reference_spots)
        for i, ref_spot in enumerate(self.reference_spots):
            # Find the closest detected spot
            distances = np.sqrt(np.sum((spots - ref_spot)**2, axis=1))
            closest_idx = np.argmin(distances)
            
            # Calculate shift
            shifts[i] = spots[closest_idx] - ref_spot
        
        return shifts

class WavefrontReconstructor:
    """
    Reconstructs wavefront from spot shifts using zonal or modal methods.
    """
    
    def __init__(self, lenslet_pitch, focal_length):
        """
        Initialize the wavefront reconstructor.
        
        Args:
            lenslet_pitch: Distance between lenslet centers in meters
            focal_length: Focal length of each lenslet in meters
        """
        self.lenslet_pitch = lenslet_pitch
        self.focal_length = focal_length
        
    def reconstruct_zonal(self, spot_shifts, num_lenslets_x, num_lenslets_y):
        """
        Reconstruct wavefront using zonal integration method.
        
        Args:
            spot_shifts: Spot shifts (2D numpy array, shape (n, 2))
            num_lenslets_x: Number of lenslets in x-direction
            num_lenslets_y: Number of lenslets in y-direction
            
        Returns:
            Reconstructed wavefront (2D numpy array)
        """
        # Reshape spot shifts to grid
        dx = spot_shifts[:, 0].reshape(num_lenslets_y, num_lenslets_x)
        dy = spot_shifts[:, 1].reshape(num_lenslets_y, num_lenslets_x)
        
        # Convert shifts to slopes
        slopes_x = dx * self.lenslet_pitch / self.focal_length
        slopes_y = dy * self.lenslet_pitch / self.focal_length
        
        # Create grid for wavefront
        wavefront = np.zeros((num_lenslets_y, num_lenslets_x))
        
        # Integrate slopes to get wavefront (simple cumulative sum method)
        # Note: This is a basic implementation. More sophisticated methods exist.
        for i in range(1, num_lenslets_y):
            wavefront[i, 0] = wavefront[i-1, 0] + slopes_y[i-1, 0] * self.lenslet_pitch
        
        for i in range(num_lenslets_y):
            for j in range(1, num_lenslets_x):
                wavefront[i, j] = wavefront[i, j-1] + slopes_x[i, j-1] * self.lenslet_pitch
        
        # Remove piston (mean value)
        wavefront -= np.mean(wavefront)
        
        return wavefront
    
    def reconstruct_modal(self, spot_shifts, num_lenslets_x, num_lenslets_y, num_modes=15):
        """
        Reconstruct wavefront using modal method with Zernike polynomials.
        
        Args:
            spot_shifts: Spot shifts (2D numpy array, shape (n, 2))
            num_lenslets_x: Number of lenslets in x-direction
            num_lenslets_y: Number of lenslets in y-direction
            num_modes: Number of Zernike modes to use
            
        Returns:
            Reconstructed wavefront (2D numpy array)
        """
        # This is a placeholder for a more complex implementation
        # A full implementation would use Zernike polynomials and least squares fitting
        
        # For now, fall back to zonal reconstruction
        return self.reconstruct_zonal(spot_shifts, num_lenslets_x, num_lenslets_y)

class ShackHartmannGUI:
    """
    GUI for Shack-Hartmann wavefront sensor.
    """
    
    def __init__(self, root=None):
        """
        Initialize the Shack-Hartmann GUI.
        
        Args:
            root: Tkinter root window (if None, create a new one)
        """
        if root is None:
            self.root = tk.Tk()
            self.root.title("Shack-Hartmann Wavefront Sensor")
            self.root.geometry("1280x800")
        else:
            self.root = root
            
        # Create camera controller if available
        if CameraController is not None:
            self.camera = CameraController()
        else:
            self.camera = None
            
        # Create lenslet array generator
        self.lenslet_generator = LensletArrayGenerator()
        
        # Default parameters
        self.num_lenslets_x = 10
        self.num_lenslets_y = 8
        self.lenslet_pitch = 2.5e-3  # 2.5 mm
        self.focal_length = 0.1  # 100 mm
        
        # Initialize variables
        self.lenslet_pattern = None
        self.detected_spots = None
        self.reference_spots = None
        self.wavefront = None
        
        # Create GUI
        self._create_widgets()
        
        # Start camera if available
        if self.camera is not None:
            self.camera.start()
            
    def _create_widgets(self):
        """Create the GUI widgets."""
        # Create main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create left panel for controls
        self.control_frame = ttk.Frame(self.main_frame, width=300)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Create right panel for display
        self.display_frame = ttk.Frame(self.main_frame)
        self.display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create controls
        self._create_controls()
        
        # Create display
        self._create_display()
        
    def _create_controls(self):
        """Create control widgets."""
        # Lenslet array parameters
        lenslet_frame = ttk.LabelFrame(self.control_frame, text="Lenslet Array Parameters")
        lenslet_frame.pack(fill=tk.X, pady=5)
        
        # Number of lenslets X
        ttk.Label(lenslet_frame, text="Number of lenslets X:").pack(anchor=tk.W)
        self.num_lenslets_x_var = tk.StringVar(value=str(self.num_lenslets_x))
        ttk.Entry(lenslet_frame, textvariable=self.num_lenslets_x_var).pack(fill=tk.X, pady=2)
        
        # Number of lenslets Y
        ttk.Label(lenslet_frame, text="Number of lenslets Y:").pack(anchor=tk.W)
        self.num_lenslets_y_var = tk.StringVar(value=str(self.num_lenslets_y))
        ttk.Entry(lenslet_frame, textvariable=self.num_lenslets_y_var).pack(fill=tk.X, pady=2)
        
        # Lenslet pitch (mm)
        ttk.Label(lenslet_frame, text="Lenslet pitch (mm):").pack(anchor=tk.W)
        self.lenslet_pitch_var = tk.StringVar(value=str(self.lenslet_pitch * 1000))
        ttk.Entry(lenslet_frame, textvariable=self.lenslet_pitch_var).pack(fill=tk.X, pady=2)
        
        # Focal length (mm)
        ttk.Label(lenslet_frame, text="Focal length (mm):").pack(anchor=tk.W)
        self.focal_length_var = tk.StringVar(value=str(self.focal_length * 1000))
        ttk.Entry(lenslet_frame, textvariable=self.focal_length_var).pack(fill=tk.X, pady=2)
        
        # Blazed grating parameters
        grating_frame = ttk.LabelFrame(self.control_frame, text="Blazed Grating Parameters")
        grating_frame.pack(fill=tk.X, pady=5)
        
        # Shift X
        ttk.Label(grating_frame, text="Shift X:").pack(anchor=tk.W)
        self.shift_x_var = tk.StringVar(value="1.0")
        ttk.Entry(grating_frame, textvariable=self.shift_x_var).pack(fill=tk.X, pady=2)
        
        # Shift Y
        ttk.Label(grating_frame, text="Shift Y:").pack(anchor=tk.W)
        self.shift_y_var = tk.StringVar(value="0.0")
        ttk.Entry(grating_frame, textvariable=self.shift_y_var).pack(fill=tk.X, pady=2)
        
        # Spot detection parameters
        detection_frame = ttk.LabelFrame(self.control_frame, text="Spot Detection Parameters")
        detection_frame.pack(fill=tk.X, pady=5)
        
        # Threshold
        ttk.Label(detection_frame, text="Threshold:").pack(anchor=tk.W)
        self.threshold_var = tk.StringVar(value="50")
        ttk.Entry(detection_frame, textvariable=self.threshold_var).pack(fill=tk.X, pady=2)
        
        # Min distance
        ttk.Label(detection_frame, text="Min distance:").pack(anchor=tk.W)
        self.min_distance_var = tk.StringVar(value="10")
        ttk.Entry(detection_frame, textvariable=self.min_distance_var).pack(fill=tk.X, pady=2)
        
        # Action buttons
        action_frame = ttk.Frame(self.control_frame)
        action_frame.pack(fill=tk.X, pady=10)
        
        # Generate lenslet pattern button
        ttk.Button(action_frame, text="Generate Lenslet Pattern", 
                  command=self._on_generate_pattern).pack(fill=tk.X, pady=2)
        
        # Display on SLM button
        ttk.Button(action_frame, text="Display on SLM", 
                  command=self._on_display_slm).pack(fill=tk.X, pady=2)
        
        # Capture reference button
        ttk.Button(action_frame, text="Capture Reference", 
                  command=self._on_capture_reference).pack(fill=tk.X, pady=2)
        
        # Reconstruct wavefront button
        ttk.Button(action_frame, text="Reconstruct Wavefront", 
                  command=self._on_reconstruct_wavefront).pack(fill=tk.X, pady=2)
        
        # Save wavefront button
        ttk.Button(action_frame, text="Save Wavefront", 
                  command=self._on_save_wavefront).pack(fill=tk.X, pady=2)
        
        # Status bar
        status_frame = ttk.Frame(self.control_frame)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=5)
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status_frame, textvariable=self.status_var).pack(anchor=tk.W)
        
    def _create_display(self):
        """Create display widgets."""
        # Create notebook for tabbed display
        self.notebook = ttk.Notebook(self.display_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.lenslet_tab = ttk.Frame(self.notebook)
        self.camera_tab = ttk.Frame(self.notebook)
        self.spots_tab = ttk.Frame(self.notebook)
        self.wavefront_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.lenslet_tab, text="Lenslet Pattern")
        self.notebook.add(self.camera_tab, text="Camera View")
        self.notebook.add(self.spots_tab, text="Spot Detection")
        self.notebook.add(self.wavefront_tab, text="Wavefront")
        
        # Create figures for each tab
        self._create_lenslet_figure()
        self._create_camera_figure()
        self._create_spots_figure()
        self._create_wavefront_figure()
        
    def _create_lenslet_figure(self):
        """Create figure for lenslet pattern display."""
        self.lenslet_fig = Figure(figsize=(6, 5), dpi=100)
        self.lenslet_canvas = FigureCanvasTkAgg(self.lenslet_fig, master=self.lenslet_tab)
        self.lenslet_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.lenslet_ax = self.lenslet_fig.add_subplot(111)
        self.lenslet_ax.set_title("Lenslet Array Pattern")
        self.lenslet_ax.set_xlabel("X (pixels)")
        self.lenslet_ax.set_ylabel("Y (pixels)")
        
        # Initial empty display
        self.lenslet_im = self.lenslet_ax.imshow(np.zeros((100, 100)), cmap='gray')
        self.lenslet_fig.colorbar(self.lenslet_im, ax=self.lenslet_ax, label="Phase (rad)")
        
        self.lenslet_canvas.draw()
        
    def _create_camera_figure(self):
        """Create figure for camera view display."""
        self.camera_fig = Figure(figsize=(6, 5), dpi=100)
        self.camera_canvas = FigureCanvasTkAgg(self.camera_fig, master=self.camera_tab)
        self.camera_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.camera_ax = self.camera_fig.add_subplot(111)
        self.camera_ax.set_title("Camera View")
        self.camera_ax.set_xlabel("X (pixels)")
        self.camera_ax.set_ylabel("Y (pixels)")
        
        # Initial empty display
        self.camera_im = self.camera_ax.imshow(np.zeros((100, 100)), cmap='gray')
        
        self.camera_canvas.draw()
        
        # Start camera update if available
        if self.camera is not None:
            self._update_camera_view()
        
    def _create_spots_figure(self):
        """Create figure for spot detection display."""
        self.spots_fig = Figure(figsize=(6, 5), dpi=100)
        self.spots_canvas = FigureCanvasTkAgg(self.spots_fig, master=self.spots_tab)
        self.spots_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.spots_ax = self.spots_fig.add_subplot(111)
        self.spots_ax.set_title("Spot Detection")
        self.spots_ax.set_xlabel("X (pixels)")
        self.spots_ax.set_ylabel("Y (pixels)")
        
        # Initial empty display
        self.spots_im = self.spots_ax.imshow(np.zeros((100, 100)), cmap='gray')
        
        self.spots_canvas.draw()
        
    def _create_wavefront_figure(self):
        """Create figure for wavefront display."""
        self.wavefront_fig = Figure(figsize=(6, 5), dpi=100)
        self.wavefront_canvas = FigureCanvasTkAgg(self.wavefront_fig, master=self.wavefront_tab)
        self.wavefront_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.wavefront_ax = self.wavefront_fig.add_subplot(111)
        self.wavefront_ax.set_title("Reconstructed Wavefront")
        self.wavefront_ax.set_xlabel("X (mm)")
        self.wavefront_ax.set_ylabel("Y (mm)")
        
        # Initial empty display
        self.wavefront_im = self.wavefront_ax.imshow(np.zeros((10, 10)), cmap='viridis')
        self.wavefront_fig.colorbar(self.wavefront_im, ax=self.wavefront_ax, label="Wavefront (waves)")
        
        self.wavefront_canvas.draw()
        
    def _update_camera_view(self):
        """Update camera view."""
        if self.camera is not None and self.camera.is_running:
            # Get latest frame
            frame = self.camera.get_latest_frame()
            
            if frame is not None:
                # Update camera view
                self.camera_ax.clear()
                self.camera_ax.set_title("Camera View")
                self.camera_ax.set_xlabel("X (pixels)")
                self.camera_ax.set_ylabel("Y (pixels)")
                self.camera_im = self.camera_ax.imshow(frame, cmap='gray')
                self.camera_canvas.draw()
        
        # Schedule next update
        self.root.after(100, self._update_camera_view)
        
    def _on_generate_pattern(self):
        """Generate lenslet array pattern."""
        try:
            # Get parameters from GUI
            self.num_lenslets_x = int(self.num_lenslets_x_var.get())
            self.num_lenslets_y = int(self.num_lenslets_y_var.get())
            self.lenslet_pitch = float(self.lenslet_pitch_var.get()) / 1000  # Convert mm to m
            self.focal_length = float(self.focal_length_var.get()) / 1000  # Convert mm to m
            
            # Generate lenslet array pattern
            self.lenslet_pattern = self.lenslet_generator.generate_lenslet_array(
                self.lenslet_pitch, self.focal_length)
            
            # Add blazed grating if specified
            shift_x = float(self.shift_x_var.get())
            shift_y = float(self.shift_y_var.get())
            
            if shift_x != 0 or shift_y != 0:
                self.lenslet_pattern = self.lenslet_generator.add_blazed_grating(
                    self.lenslet_pattern, shift_x, shift_y)
            
            # Convert to grayscale for display
            grayscale_pattern = self.lenslet_generator.phase_to_grayscale(self.lenslet_pattern)
            
            # Update display
            self.lenslet_ax.clear()
            self.lenslet_ax.set_title("Lenslet Array Pattern")
            self.lenslet_ax.set_xlabel("X (pixels)")
            self.lenslet_ax.set_ylabel("Y (pixels)")
            self.lenslet_im = self.lenslet_ax.imshow(grayscale_pattern, cmap='gray')
            self.lenslet_fig.colorbar(self.lenslet_im, ax=self.lenslet_ax, label="Grayscale")
            self.lenslet_canvas.draw()
            
            self.status_var.set("Lenslet pattern generated")
            
        except Exception as e:
            self.status_var.set(f"Error generating pattern: {str(e)}")
            messagebox.showerror("Error", f"Error generating pattern: {str(e)}")
            
    def _on_display_slm(self):
        """Display pattern on SLM."""
        if self.lenslet_pattern is None:
            self.status_var.set("Generate a pattern first")
            return
            
        try:
            # Convert to grayscale for SLM
            grayscale_pattern = self.lenslet_generator.phase_to_grayscale(self.lenslet_pattern)
            
            # Save pattern to file
            cv2.imwrite("lenslet_pattern.png", grayscale_pattern)
            
            # Display on SLM (this would need to be adapted to your SLM control method)
            # For now, just show a message
            self.status_var.set("Pattern saved to lenslet_pattern.png")
            messagebox.showinfo("SLM Display", 
                               "Pattern saved to lenslet_pattern.png\n"
                               "Please load this pattern into your SLM control software.")
            
        except Exception as e:
            self.status_var.set(f"Error displaying on SLM: {str(e)}")
            messagebox.showerror("Error", f"Error displaying on SLM: {str(e)}")
            
    def _on_capture_reference(self):
        """Capture reference spot pattern."""
        if self.camera is None:
            self.status_var.set("Camera not available")
            return
            
        try:
            # Capture frame
            frame = self.camera.capture_frame()
            
            if frame is None:
                self.status_var.set("Failed to capture frame")
                return
                
            # Save frame as reference
            cv2.imwrite("reference_spots.png", frame)
            self.reference_spots = self.spot_detector.detect_spots(frame)
            
            self.status_var.set("Reference spots captured")
            
        except Exception as e:
            self.status_var.set(f"Error capturing reference: {str(e)}")
            messagebox.showerror("Error", f"Error capturing reference: {str(e)}")   