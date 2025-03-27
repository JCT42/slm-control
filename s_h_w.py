#!/usr/bin/env python3
"""
Shack-Hartmann Wavefront Sensor Implementation

This module provides a streamlined implementation for wavefront reconstruction
using the Shack-Hartmann method. It processes an input image containing spots
from a lenslet array and reconstructs the wavefront.

Features:
- Process images to detect spot positions
- Calculate spot shifts from reference positions
- Reconstruct wavefront using zonal or modal methods
- Visualize wavefront and aberrations
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cv2
from scipy.interpolate import griddata, RectBivariateSpline
from scipy.ndimage import gaussian_filter
import argparse
import os
import sys
import subprocess
import platform
import tkinter as tk
from tkinter import filedialog


def select_file(title="Select Image", file_types=(("Image files", "*.jpg;*.jpeg;*.png;*.bmp"), ("All files", "*.*"))):
    """
    Open a file dialog to select a file, compatible with both Windows and Raspberry Pi.
    
    Args:
        title: Dialog title
        file_types: File types to filter
        
    Returns:
        Selected file path or None if canceled
    """
    # Check if running on Raspberry Pi or similar Linux
    if platform.system() == "Linux":
        try:
            # Try to use zenity for file selection on Raspberry Pi
            file_types_str = " ".join([f"--file-filter='{name} ({patterns})' {patterns}" for name, patterns in file_types])
            cmd = f"zenity --file-selection --title='{title}' {file_types_str}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                return result.stdout.strip()
            return None
        except Exception as e:
            print(f"Zenity error: {e}")
            # Fall back to tkinter if zenity fails
            pass
    
    # Use tkinter file dialog for Windows or as fallback
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    # Convert file_types to tkinter format
    tk_filetypes = []
    for desc, exts in file_types:
        # Split extensions if multiple are provided with semicolons
        ext_list = exts.split(';')
        for ext in ext_list:
            tk_filetypes.append((desc, ext))
    
    file_path = filedialog.askopenfilename(title=title, filetypes=tk_filetypes)
    root.destroy()
    
    return file_path if file_path else None


class ShackHartmannWavefrontSensor:
    """
    Processes Shack-Hartmann sensor images and reconstructs wavefronts.
    """
    
    def __init__(self, lenslet_pitch=0.5, focal_length=10.0):
        """
        Initialize the Shack-Hartmann wavefront sensor.
        
        Args:
            lenslet_pitch: Distance between lenslet centers in mm
            focal_length: Focal length of each lenslet in mm
        """
        self.lenslet_pitch = lenslet_pitch
        self.focal_length = focal_length
        self.reference_spots = None
        self.current_spots = None
        self.spot_shifts = None
        self.wavefront = None
        
    def detect_spots(self, image, threshold=50, min_area=5, max_area=500):
        """
        Detect spots in an image using blob detection.
        
        Args:
            image: Input image (2D numpy array)
            threshold: Intensity threshold for spot detection
            min_area: Minimum area of spots to detect
            max_area: Maximum area of spots to detect
            
        Returns:
            Array of spot positions (x, y)
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
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    spots.append((cx, cy))
        
        # Sort spots by y, then x for consistent ordering
        spots.sort(key=lambda p: (p[1], p[0]))
        
        return np.array(spots)
    
    def set_reference_spots(self, spots=None, image=None, threshold=50):
        """
        Set reference spot positions either directly or from an image.
        
        Args:
            spots: Array of reference spot positions (x, y)
            image: Reference image to detect spots from
            threshold: Threshold for spot detection if using image
        """
        if spots is not None:
            self.reference_spots = np.array(spots)
        elif image is not None:
            self.reference_spots = self.detect_spots(image, threshold)
        else:
            raise ValueError("Either spots or image must be provided")
            
        return self.reference_spots
    
    def calculate_spot_shifts(self, spots=None, image=None, threshold=50):
        """
        Calculate spot shifts from reference positions.
        
        Args:
            spots: Array of current spot positions (x, y)
            image: Current image to detect spots from
            threshold: Threshold for spot detection if using image
            
        Returns:
            Array of spot shifts (dx, dy)
        """
        if self.reference_spots is None:
            raise ValueError("Reference spots not set. Call set_reference_spots first.")
            
        if spots is not None:
            self.current_spots = np.array(spots)
        elif image is not None:
            self.current_spots = self.detect_spots(image, threshold)
        else:
            raise ValueError("Either spots or image must be provided")
            
        # Ensure same number of spots
        if len(self.current_spots) != len(self.reference_spots):
            raise ValueError(f"Number of current spots ({len(self.current_spots)}) does not match reference spots ({len(self.reference_spots)})")
            
        # Calculate shifts
        self.spot_shifts = self.current_spots - self.reference_spots
        
        return self.spot_shifts
    
    def reconstruct_wavefront_zonal(self, spot_shifts=None, smoothing=0.5):
        """
        Reconstruct wavefront using zonal integration method.
        
        Args:
            spot_shifts: Array of spot shifts (dx, dy)
            smoothing: Smoothing factor for the wavefront
            
        Returns:
            Reconstructed wavefront (2D numpy array)
        """
        if spot_shifts is not None:
            self.spot_shifts = spot_shifts
        elif self.spot_shifts is None:
            raise ValueError("Spot shifts not calculated. Call calculate_spot_shifts first.")
            
        # Extract spot positions and shifts
        ref_spots = self.reference_spots
        shifts = self.spot_shifts
        
        # Calculate slopes (proportional to wavefront derivatives)
        # Convert from pixels to physical units
        slopes_x = -shifts[:, 0] * self.lenslet_pitch / self.focal_length
        slopes_y = -shifts[:, 1] * self.lenslet_pitch / self.focal_length
        
        # Create grid for interpolation
        x_min, y_min = np.min(ref_spots, axis=0)
        x_max, y_max = np.max(ref_spots, axis=0)
        
        grid_size = max(x_max - x_min, y_max - y_min) // 2
        x_grid = np.linspace(x_min, x_max, grid_size)
        y_grid = np.linspace(y_min, y_max, grid_size)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Interpolate slopes onto regular grid
        points = ref_spots
        grid_slopes_x = griddata(points, slopes_x, (X, Y), method='cubic', fill_value=0)
        grid_slopes_y = griddata(points, slopes_y, (X, Y), method='cubic', fill_value=0)
        
        # Apply smoothing
        if smoothing > 0:
            grid_slopes_x = gaussian_filter(grid_slopes_x, sigma=smoothing)
            grid_slopes_y = gaussian_filter(grid_slopes_y, sigma=smoothing)
        
        # Integrate slopes to get wavefront
        # Using cumulative integration in both directions and averaging
        wavefront_x = np.zeros_like(X)
        wavefront_y = np.zeros_like(Y)
        
        # Integrate x-slopes along x-direction
        for i in range(grid_size):
            wavefront_x[i, :] = np.cumsum(grid_slopes_x[i, :]) * (x_max - x_min) / grid_size
            
        # Integrate y-slopes along y-direction
        for j in range(grid_size):
            wavefront_y[:, j] = np.cumsum(grid_slopes_y[:, j]) * (y_max - y_min) / grid_size
            
        # Average the two integrations
        self.wavefront = (wavefront_x + wavefront_y) / 2
        
        # Remove piston (mean value)
        self.wavefront -= np.mean(self.wavefront)
        
        return self.wavefront, X, Y
    
    def reconstruct_wavefront_modal(self, spot_shifts=None, num_modes=15):
        """
        Reconstruct wavefront using modal method with Zernike polynomials.
        
        Args:
            spot_shifts: Array of spot shifts (dx, dy)
            num_modes: Number of Zernike modes to use
            
        Returns:
            Reconstructed wavefront (2D numpy array)
        """
        # This is a simplified implementation that could be expanded
        # For now, we'll use the zonal method
        return self.reconstruct_wavefront_zonal(spot_shifts)
    
    def visualize_results(self, original_image=None, title="Wavefront Reconstruction"):
        """
        Visualize the results of wavefront reconstruction.
        
        Args:
            original_image: Original image (optional)
            title: Title for the plot
        """
        if self.wavefront is None:
            raise ValueError("Wavefront not reconstructed. Call reconstruct_wavefront first.")
            
        # Create figure
        fig = plt.figure(figsize=(15, 10))
        
        # Plot original image if provided
        if original_image is not None:
            ax1 = fig.add_subplot(2, 2, 1)
            ax1.imshow(original_image, cmap='gray')
            ax1.set_title("Original Image")
            ax1.axis('off')
            
            # Plot detected spots
            ax2 = fig.add_subplot(2, 2, 2)
            ax2.imshow(original_image, cmap='gray')
            if self.reference_spots is not None:
                ax2.plot(self.reference_spots[:, 0], self.reference_spots[:, 1], 'bo', markersize=3, label='Reference')
            if self.current_spots is not None:
                ax2.plot(self.current_spots[:, 0], self.current_spots[:, 1], 'ro', markersize=3, label='Current')
            ax2.set_title("Detected Spots")
            ax2.legend()
            ax2.axis('off')
            
            # Plot wavefront in 2D
            ax3 = fig.add_subplot(2, 2, 3)
            im = ax3.imshow(self.wavefront, cmap='jet')
            ax3.set_title("Wavefront")
            plt.colorbar(im, ax=ax3, label='Wavefront [wavelengths]')
            
            # Plot wavefront in 3D
            ax4 = fig.add_subplot(2, 2, 4, projection='3d')
            X, Y = np.meshgrid(np.arange(self.wavefront.shape[1]), np.arange(self.wavefront.shape[0]))
            surf = ax4.plot_surface(X, Y, self.wavefront, cmap='jet', linewidth=0, antialiased=True)
            ax4.set_title("3D Wavefront")
            plt.colorbar(surf, ax=ax4, shrink=0.5, aspect=5, label='Wavefront [wavelengths]')
        else:
            # Just plot wavefront in 2D and 3D
            ax1 = fig.add_subplot(1, 2, 1)
            im = ax1.imshow(self.wavefront, cmap='jet')
            ax1.set_title("Wavefront")
            plt.colorbar(im, ax=ax1, label='Wavefront [wavelengths]')
            
            ax2 = fig.add_subplot(1, 2, 2, projection='3d')
            X, Y = np.meshgrid(np.arange(self.wavefront.shape[1]), np.arange(self.wavefront.shape[0]))
            surf = ax2.plot_surface(X, Y, self.wavefront, cmap='jet', linewidth=0, antialiased=True)
            ax2.set_title("3D Wavefront")
            plt.colorbar(surf, ax=ax2, shrink=0.5, aspect=5, label='Wavefront [wavelengths]')
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig


def process_image(image_path=None, lenslet_pitch=0.5, focal_length=10.0, 
                  reference_image_path=None, threshold=50, smoothing=0.5):
    """
    Process an image and reconstruct the wavefront.
    
    Args:
        image_path: Path to the image to process (if None, open file dialog)
        lenslet_pitch: Distance between lenslet centers in mm
        focal_length: Focal length of each lenslet in mm
        reference_image_path: Path to reference image (if None, use ideal grid)
        threshold: Threshold for spot detection
        smoothing: Smoothing factor for wavefront reconstruction
        
    Returns:
        ShackHartmannWavefrontSensor object with results
    """
    # Initialize sensor
    sensor = ShackHartmannWavefrontSensor(lenslet_pitch, focal_length)
    
    # If no image path provided, open file dialog
    if image_path is None:
        image_path = select_file(title="Select Shack-Hartmann Image")
        if not image_path:
            print("No image selected. Exiting.")
            return None
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Set reference spots
    if reference_image_path is not None:
        reference_image = cv2.imread(reference_image_path)
        if reference_image is None:
            raise ValueError(f"Could not load reference image: {reference_image_path}")
        sensor.set_reference_spots(image=reference_image, threshold=threshold)
    else:
        # Ask if user wants to select a reference image
        if platform.system() == "Linux":
            try:
                cmd = "zenity --question --text='Do you want to select a reference image?' --title='Reference Image'"
                result = subprocess.run(cmd, shell=True)
                use_reference = (result.returncode == 0)
            except:
                use_reference = False
        else:
            root = tk.Tk()
            root.withdraw()
            use_reference = tk.messagebox.askyesno("Reference Image", "Do you want to select a reference image?")
            root.destroy()
        
        if use_reference:
            ref_path = select_file(title="Select Reference Image")
            if ref_path:
                reference_image = cv2.imread(ref_path)
                if reference_image is not None:
                    sensor.set_reference_spots(image=reference_image, threshold=threshold)
                else:
                    print("Could not load reference image. Using ideal grid.")
                    use_reference = False
            else:
                print("No reference image selected. Using ideal grid.")
                use_reference = False
        
        if not use_reference:
            # Create ideal grid as reference
            spots = sensor.detect_spots(image, threshold)
            if len(spots) == 0:
                raise ValueError("No spots detected in the image. Try adjusting the threshold.")
                
            num_spots = len(spots)
            
            # Estimate grid dimensions
            aspect_ratio = image.shape[1] / image.shape[0]
            num_y = int(np.sqrt(num_spots / aspect_ratio))
            num_x = int(num_spots / num_y)
            
            # Create ideal grid
            x = np.linspace(0, image.shape[1], num_x+2)[1:-1]
            y = np.linspace(0, image.shape[0], num_y+2)[1:-1]
            X, Y = np.meshgrid(x, y)
            ideal_spots = np.column_stack((X.flatten(), Y.flatten()))
            
            sensor.set_reference_spots(spots=ideal_spots)
    
    # Calculate spot shifts
    sensor.calculate_spot_shifts(image=image, threshold=threshold)
    
    # Reconstruct wavefront
    sensor.reconstruct_wavefront_zonal(smoothing=smoothing)
    
    return sensor, image


def main():
    """Main function to run the Shack-Hartmann wavefront sensor from command line."""
    parser = argparse.ArgumentParser(description='Shack-Hartmann Wavefront Sensor')
    parser.add_argument('--image', help='Path to the image to process (if not provided, will open file dialog)')
    parser.add_argument('--reference', help='Path to reference image')
    parser.add_argument('--lenslet-pitch', type=float, default=0.5, help='Distance between lenslet centers in mm')
    parser.add_argument('--focal-length', type=float, default=10.0, help='Focal length of each lenslet in mm')
    parser.add_argument('--threshold', type=int, default=50, help='Threshold for spot detection')
    parser.add_argument('--smoothing', type=float, default=0.5, help='Smoothing factor for wavefront reconstruction')
    parser.add_argument('--output', help='Path to save the output plot')
    
    args = parser.parse_args()
    
    try:
        # Process image
        sensor, image = process_image(
            image_path=args.image, 
            lenslet_pitch=args.lenslet_pitch,
            focal_length=args.focal_length,
            reference_image_path=args.reference,
            threshold=args.threshold,
            smoothing=args.smoothing
        )
        
        if sensor is None:
            return 1
        
        # Visualize results
        fig = sensor.visualize_results(image)
        
        if args.output:
            plt.savefig(args.output)
        
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    main()