#!/usr/bin/env python3
"""
Generate Lenslet Arrays for Shack-Hartmann Wavefront Sensor

This script generates lenslet array phase patterns with varying pitches and focal distances.
It also generates corresponding reference spot patterns for each lenslet array configuration.

The lenslet arrays are generated as phase patterns in the range [-π to π],
maintaining the scientific integrity of the phase representation for SLM patterns.
The reference spots are generated as intensity patterns, preserving the raw intensity values.
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter

class LensletArrayGenerator:
    """Generate lenslet array patterns for Shack-Hartmann wavefront sensing."""
    
    def __init__(self, width=800, height=600):
        """
        Initialize the lenslet array generator.
        
        Args:
            width: Width of the SLM in pixels
            height: Height of the SLM in pixels
        """
        self.width = width
        self.height = height
        
        # Create coordinate grids
        self.x = np.linspace(-width/2, width/2, width)
        self.y = np.linspace(-height/2, height/2, height)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Create output directories if they don't exist
        os.makedirs("lenslet_arrays", exist_ok=True)
        os.makedirs("reference_spots", exist_ok=True)
    
    def generate_lenslet_array(self, lenslet_pitch_mm, focal_length_mm):
        """
        Generate a lenslet array phase pattern.
        
        Args:
            lenslet_pitch_mm: Pitch of the lenslets in mm
            focal_length_mm: Focal length of the lenslets in mm
            
        Returns:
            2D numpy array containing the phase pattern in the range [-π, π]
        """
        # Convert mm to m
        lenslet_pitch = lenslet_pitch_mm / 1000
        focal_length = focal_length_mm / 1000
        
        # SLM parameters (typical values)
        pixel_pitch = 32e-6  # 32 microns per pixel
        
        # Calculate the lenslet pitch in pixels
        lenslet_pitch_pixels = lenslet_pitch / pixel_pitch
        
        # Calculate the phase pattern for a single lenslet
        k = 2 * np.pi / 650e-9  # Wave number for 650nm wavelength
        
        # Create modulo grid for repeating lenslets
        X_mod = np.mod(self.X + self.width/2, lenslet_pitch_pixels) - lenslet_pitch_pixels/2
        Y_mod = np.mod(self.Y + self.height/2, lenslet_pitch_pixels) - lenslet_pitch_pixels/2
        
        # Calculate squared distance from center of each lenslet
        R_squared = X_mod**2 + Y_mod**2
        
        # Calculate phase pattern for lenslet array
        # Quadratic phase for a lens: φ(r) = -k*r²/(2f)
        phase = -k * R_squared * (pixel_pitch**2) / (2 * focal_length)
        
        # Wrap phase to [-π, π]
        phase = np.mod(phase + np.pi, 2 * np.pi) - np.pi
        
        return phase
    
    def generate_reference_spots(self, lenslet_pitch_mm, focal_length_mm):
        """
        Generate a reference spot pattern for a given lenslet array configuration.
        
        Args:
            lenslet_pitch_mm: Pitch of the lenslets in mm
            focal_length_mm: Focal length of the lenslets in mm
            
        Returns:
            2D numpy array containing the reference spot pattern as intensity values
        """
        # SLM parameters
        pixel_pitch = 32e-6  # 32 microns per pixel
        
        # Calculate the lenslet pitch in pixels
        lenslet_pitch_pixels = lenslet_pitch_mm * 1000 / pixel_pitch
        
        # Create an empty image
        spots_image = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Calculate the number of lenslets in each direction
        num_lenslets_x = int(self.width / lenslet_pitch_pixels) + 2
        num_lenslets_y = int(self.height / lenslet_pitch_pixels) + 2
        
        # Calculate the offset to center the pattern
        offset_x = (self.width - (num_lenslets_x - 2) * lenslet_pitch_pixels) / 2
        offset_y = (self.height - (num_lenslets_y - 2) * lenslet_pitch_pixels) / 2
        
        # Create spots at regular intervals
        for i in range(num_lenslets_y):
            for j in range(num_lenslets_x):
                # Calculate spot position
                x = int(offset_x + j * lenslet_pitch_pixels)
                y = int(offset_y + i * lenslet_pitch_pixels)
                
                # Ensure spot is within image boundaries
                if 0 <= x < self.width and 0 <= y < self.height:
                    # Create a small Gaussian spot
                    spot_radius = 3  # Radius of spot in pixels
                    y_min = max(0, y - spot_radius)
                    y_max = min(self.height, y + spot_radius + 1)
                    x_min = max(0, x - spot_radius)
                    x_max = min(self.width, x + spot_radius + 1)
                    
                    for sy in range(y_min, y_max):
                        for sx in range(x_min, x_max):
                            # Calculate distance from spot center
                            dist = np.sqrt((sx - x)**2 + (sy - y)**2)
                            if dist <= spot_radius:
                                # Create a Gaussian intensity profile
                                intensity = int(255 * np.exp(-(dist/spot_radius)**2))
                                spots_image[sy, sx] = intensity
        
        # Apply Gaussian blur to make spots more realistic
        spots_image = gaussian_filter(spots_image, sigma=1.0)
        
        return spots_image
    
    def phase_to_grayscale(self, phase):
        """
        Convert phase pattern to grayscale image for display.
        
        Args:
            phase: Phase pattern in the range [-π, π]
            
        Returns:
            Grayscale image in the range [0, 255]
        """
        # Scale from [-π, π] to [0, 255]
        grayscale = ((phase + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
        return grayscale
    
    def save_lenslet_array(self, lenslet_pitch_mm, focal_length_mm, show_plot=False):
        """
        Generate and save a lenslet array pattern and its corresponding reference spots.
        
        Args:
            lenslet_pitch_mm: Pitch of the lenslets in mm
            focal_length_mm: Focal length of the lenslets in mm
            show_plot: Whether to display the patterns
        """
        # Generate the lenslet array
        phase = self.generate_lenslet_array(lenslet_pitch_mm, focal_length_mm)
        
        # Convert to grayscale for saving
        grayscale = self.phase_to_grayscale(phase)
        
        # Save the lenslet array pattern
        lenslet_filename = f"lenslet_arrays/lenslet_p{lenslet_pitch_mm:.1f}mm_f{focal_length_mm:.1f}mm.png"
        cv2.imwrite(lenslet_filename, grayscale)
        
        # Generate and save reference spots
        spots_image = self.generate_reference_spots(lenslet_pitch_mm, focal_length_mm)
        spots_filename = f"reference_spots/refspots_p{lenslet_pitch_mm:.1f}mm_f{focal_length_mm:.1f}mm.png"
        cv2.imwrite(spots_filename, spots_image)
        
        # Display the patterns if requested
        if show_plot:
            plt.figure(figsize=(15, 8))
            
            # Plot phase pattern
            plt.subplot(131)
            plt.imshow(phase, cmap='viridis', norm=Normalize(vmin=-np.pi, vmax=np.pi))
            plt.colorbar(label='Phase (rad)')
            plt.title(f'Lenslet Array Phase Pattern\nPitch: {lenslet_pitch_mm}mm, f: {focal_length_mm}mm')
            
            # Plot grayscale pattern
            plt.subplot(132)
            plt.imshow(grayscale, cmap='gray')
            plt.colorbar(label='Grayscale Value')
            plt.title('Grayscale Pattern for SLM')
            
            # Plot reference spots
            plt.subplot(133)
            plt.imshow(spots_image, cmap='gray')
            plt.colorbar(label='Intensity')
            plt.title('Reference Spots Pattern')
            
            plt.tight_layout()
            plt.savefig(f"lenslet_arrays/lenslet_p{lenslet_pitch_mm:.1f}mm_f{focal_length_mm:.1f}mm_plot.png")
            plt.show()
        
        print(f"Saved lenslet array and reference spots with pitch {lenslet_pitch_mm}mm and focal length {focal_length_mm}mm")

def main():
    """Generate lenslet arrays with varying focal lengths."""
    # Create lenslet array generator
    generator = LensletArrayGenerator(width=800, height=600)
    
    # Generate lenslet arrays with different pitches and focal lengths
    # Smaller pitch = more lenses in the array
    lenslet_pitches = [0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0]
    focal_lengths = [80, 90, 100, 110, 120]
    
    for lenslet_pitch_mm in lenslet_pitches:
        for focal_length_mm in focal_lengths:
            # Only show plots for a subset of combinations to avoid too many windows
            show_plot = (lenslet_pitch_mm == 0.3)
            generator.save_lenslet_array(lenslet_pitch_mm, focal_length_mm, show_plot=show_plot)
    
    print(f"Generated {len(lenslet_pitches) * len(focal_lengths)} lenslet arrays with corresponding reference spots")
    print("Lenslet arrays saved in the 'lenslet_arrays' directory")
    print("Reference spots saved in the 'reference_spots' directory")

if __name__ == "__main__":
    main()
