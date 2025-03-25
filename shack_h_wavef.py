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
import pygame
import traceback
from scipy.special import eval_jacobi
from PIL import Image, ImageTk

# Import camera controller
try:
    from camera_controller import CameraController
    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False
    print("Camera controller not available. Install with: pip install picamera2")

# Define Zernike polynomial functions since scipy.special.zernike_r is not available
def zernike_radial(n, m, rho):
    """
    Calculate the radial component of Zernike polynomial.
    
    Args:
        n: Radial degree
        m: Azimuthal degree (|m| <= n and n-|m| is even)
        rho: Radial distance (0 <= rho <= 1)
        
    Returns:
        Radial component of Zernike polynomial
    """
    if (n - m) % 2 != 0:
        return np.zeros_like(rho)
    
    k_max = (n - abs(m)) // 2
    R = np.zeros_like(rho)
    
    for k in range(k_max + 1):
        coef = (-1)**k * np.math.factorial(n - k)
        coef /= (np.math.factorial(k) * 
                np.math.factorial((n + abs(m))//2 - k) * 
                np.math.factorial((n - abs(m))//2 - k))
        R += coef * rho**(n - 2*k)
    
    return R

def zernike(m, n, rho, theta=None):
    """
    Calculate Zernike polynomial.
    
    Args:
        m: Azimuthal degree
        n: Radial degree (n >= |m|)
        rho: Radial distance (0 <= rho <= 1)
        theta: Azimuthal angle (0 <= theta <= 2*pi)
        
    Returns:
        Zernike polynomial
    """
    if theta is None:
        # Just return the radial part for m=0
        if m == 0:
            return zernike_radial(n, 0, rho)
        else:
            # For non-zero m, we need theta
            return zernike_radial(n, abs(m), rho)
    
    if m == 0:
        return zernike_radial(n, 0, rho)
    elif m > 0:
        return zernike_radial(n, m, rho) * np.cos(m * theta)
    else:  # m < 0
        return zernike_radial(n, abs(m), rho) * np.sin(abs(m) * theta)

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
        # Calculate number of lenslets needed to cover the entire SLM
        # Add extra lenslets to ensure full coverage including edges
        num_lenslets_x = int(np.ceil(self.slm_width_m / lenslet_pitch)) + 1
        num_lenslets_y = int(np.ceil(self.slm_height_m / lenslet_pitch)) + 1
        
        # Calculate offset to center the array
        offset_x = (num_lenslets_x * lenslet_pitch - self.slm_width_m) / 2
        offset_y = (num_lenslets_y * lenslet_pitch - self.slm_height_m) / 2
        
        # Initialize phase pattern
        phase = np.zeros((self.slm_height, self.slm_width))
        
        # Generate lenslet array
        for i in range(num_lenslets_y):
            for j in range(num_lenslets_x):
                # Center of current lenslet
                x0 = -self.slm_width_m/2 - offset_x + j * lenslet_pitch
                y0 = -self.slm_height_m/2 - offset_y + i * lenslet_pitch
                
                # Distance from lenslet center
                r_squared = (self.X - x0)**2 + (self.Y - y0)**2
                
                # Phase of a lens: φ(r) = -k*r²/(2f)
                lens_phase = -self.k * r_squared / (2 * focal_length)
                
                # Apply lens phase within lenslet area - use circular mask for better coverage
                mask = r_squared <= (lenslet_pitch/2)**2
                # Update phase where this lenslet provides the minimum r_squared (closest to center)
                # This ensures seamless coverage without gaps
                phase = np.where(mask & (r_squared < phase.size * 1e10), lens_phase, phase)
        
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
        # Ensure phase is within [-π, π]
        phase = np.clip(phase, -np.pi, np.pi)
        
        # Map [-π, π] to [0, 255] with 0 phase at 128 grayscale
        # -π → 0, 0 → 128, π → 255
        normalized_phase = (phase + np.pi) / (2 * np.pi)  # Map to [0, 1]
        grayscale = (normalized_phase * 255).astype(np.uint8)
        
        # Apply gamma correction if needed
        if gamma != 1.0:
            # Normalize to [0, 1], apply gamma, then scale back to [0, 255]
            normalized = grayscale / 255.0
            corrected = normalized ** gamma
            grayscale = (corrected * 255).astype(np.uint8)
        
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
    
    def __init__(self, lenslet_pitch=2.5e-3, focal_length=0.1):
        """
        Initialize the wavefront reconstructor.
        
        Args:
            lenslet_pitch: Distance between lenslet centers in meters
            focal_length: Focal length of each lenslet in meters
        """
        self.lenslet_pitch = lenslet_pitch
        self.focal_length = focal_length
        
    def reconstruct_zonal(self, spot_shifts, num_lenslets_x, num_lenslets_y, upsampling_factor=4):
        """
        Reconstruct wavefront using zonal integration method with improved resolution.
        
        Args:
            spot_shifts: Spot shifts (2D numpy array, shape (n, 2))
            num_lenslets_x: Number of lenslets in x-direction
            num_lenslets_y: Number of lenslets in y-direction
            upsampling_factor: Factor by which to increase the resolution (default: 4)
            
        Returns:
            Reconstructed wavefront (2D numpy array) with higher resolution
        """
        # Reshape spot shifts to grid
        dx = spot_shifts[:, 0].reshape(num_lenslets_y, num_lenslets_x)
        dy = spot_shifts[:, 1].reshape(num_lenslets_y, num_lenslets_x)
        
        # Convert shifts to slopes
        slopes_x = dx * self.lenslet_pitch / self.focal_length
        slopes_y = dy * self.lenslet_pitch / self.focal_length
        
        # Create high-resolution grid for wavefront
        high_res_y = num_lenslets_y * upsampling_factor
        high_res_x = num_lenslets_x * upsampling_factor
        
        # Create coordinate grids for original and upsampled data
        y_orig, x_orig = np.mgrid[0:num_lenslets_y, 0:num_lenslets_x]
        y_new, x_new = np.mgrid[0:num_lenslets_y:high_res_y*1j, 0:num_lenslets_x:high_res_x*1j]
        
        # Interpolate slopes to higher resolution grid
        slopes_x_interp = griddata((y_orig.flatten(), x_orig.flatten()), 
                                   slopes_x.flatten(), 
                                   (y_new, x_new), 
                                   method='cubic', 
                                   fill_value=0)
        
        slopes_y_interp = griddata((y_orig.flatten(), x_orig.flatten()), 
                                   slopes_y.flatten(), 
                                   (y_new, x_new), 
                                   method='cubic', 
                                   fill_value=0)
        
        # Calculate high-resolution lenslet pitch
        high_res_pitch = self.lenslet_pitch / upsampling_factor
        
        # Initialize high-resolution wavefront
        wavefront = np.zeros((high_res_y, high_res_x))
        
        # Integrate slopes using Southwell method (improved integration)
        # First integrate along y-direction
        for i in range(1, high_res_y):
            wavefront[i, 0] = wavefront[i-1, 0] + slopes_y_interp[i-1, 0] * high_res_pitch
        
        # Then integrate along x-direction for each row
        for i in range(high_res_y):
            for j in range(1, high_res_x):
                wavefront[i, j] = wavefront[i, j-1] + slopes_x_interp[i, j-1] * high_res_pitch
        
        # Apply smoothing to reduce noise and integration artifacts
        wavefront = gaussian_filter(wavefront, sigma=upsampling_factor/2)
        
        # Remove piston (mean value)
        wavefront -= np.mean(wavefront)
        
        return wavefront
    
    def reconstruct_modal(self, spot_shifts, num_lenslets_x, num_lenslets_y, num_modes=15, upsampling_factor=4):
        """
        Reconstruct wavefront using modal method with Zernike polynomials.
        
        Args:
            spot_shifts: Spot shifts (2D numpy array, shape (n, 2))
            num_lenslets_x: Number of lenslets in x-direction
            num_lenslets_y: Number of lenslets in y-direction
            num_modes: Number of Zernike modes to use
            upsampling_factor: Factor by which to increase the resolution (default: 4)
            
        Returns:
            Reconstructed wavefront (2D numpy array)
        """
        # Reshape spot shifts to grid
        dx = spot_shifts[:, 0].reshape(num_lenslets_y, num_lenslets_x)
        dy = spot_shifts[:, 1].reshape(num_lenslets_y, num_lenslets_x)
        
        # Convert shifts to slopes
        slopes_x = dx * self.lenslet_pitch / self.focal_length
        slopes_y = dy * self.lenslet_pitch / self.focal_length
        
        # Create normalized coordinates for Zernike polynomials
        y_grid, x_grid = np.mgrid[-1:1:num_lenslets_y*1j, -1:1:num_lenslets_x*1j]
        r_grid = np.sqrt(x_grid**2 + y_grid**2)
        theta = np.arctan2(y_grid, x_grid)
        
        # Mask out points outside the unit circle
        mask = r_grid <= 1.0
        
        # Flatten arrays for fitting
        x_flat = x_grid[mask]
        y_flat = y_grid[mask]
        slopes_x_flat = slopes_x[mask]
        slopes_y_flat = slopes_y[mask]
        
        # Calculate Zernike derivatives for fitting
        zernike_dx = []
        zernike_dy = []
        
        # Start from j=2 to skip piston term
        for j in range(2, num_modes + 2):
            n, m = self._get_zernike_indices(j)
            dzdr, dzdtheta = self._zernike_derivatives(n, m, r_grid[mask], theta[mask])
            
            # Convert to Cartesian derivatives
            dx = (dzdr * np.cos(theta[mask]) - dzdtheta * np.sin(theta[mask]) / r_grid[mask])
            dy = (dzdr * np.sin(theta[mask]) + dzdtheta * np.cos(theta[mask]) / r_grid[mask])
            
            zernike_dx.append(dx)
            zernike_dy.append(dy)
        
        # Stack derivatives for matrix operations
        zernike_dx = np.array(zernike_dx).T
        zernike_dy = np.array(zernike_dy).T
        
        # Combine x and y derivatives for fitting
        zernike_derivatives = np.vstack([zernike_dx, zernike_dy])
        slopes = np.concatenate([slopes_x_flat, slopes_y_flat])
        
        # Solve for Zernike coefficients using least squares
        coefficients, residuals, rank, s = np.linalg.lstsq(zernike_derivatives, slopes, rcond=None)
        
        # Create high-resolution grid for reconstruction
        high_res_y = num_lenslets_y * upsampling_factor
        high_res_x = num_lenslets_x * upsampling_factor
        y_high, x_high = np.mgrid[-1:1:high_res_y*1j, -1:1:high_res_x*1j]
        r_high = np.sqrt(x_high**2 + y_high**2)
        theta_high = np.arctan2(y_high, x_high)
        mask_high = r_high <= 1.0
        
        # Reconstruct wavefront using Zernike polynomials
        wavefront = np.zeros((high_res_y, high_res_x))
        for j, coef in enumerate(coefficients):
            n, m = self._get_zernike_indices(j + 2)  # +2 because we skipped piston
            wavefront[mask_high] += coef * self._zernike(n, m, r_high[mask_high], theta_high[mask_high])
        
        # Set points outside the unit circle to NaN or 0
        wavefront[~mask_high] = 0
        
        # Remove piston (mean value)
        wavefront -= np.mean(wavefront[mask_high])
        
        return wavefront
    
    def _get_zernike_indices(self, j):
        """
        Convert single Zernike index j to (n,m) indices using OSA/ANSI standard.
        """
        n = int(np.floor((-1 + np.sqrt(1 + 8*j)) / 2))
        m = 2*j - n*(n+2)
        if m > n:
            n, m = n+1, m-n-1
        return n, m
    
    def _zernike(self, n, m, r, theta):
        """
        Calculate Zernike polynomial Z_n^m(r, theta).
        """
        if m >= 0:
            return self._radial_zernike(n, m, r) * np.cos(m * theta)
        else:
            return self._radial_zernike(n, abs(m), r) * np.sin(abs(m) * theta)
    
    def _radial_zernike(self, n, m, r):
        """
        Calculate radial part of Zernike polynomial.
        """
        R = np.zeros_like(r)
        for k in range((n-m)//2 + 1):
            coef = (-1)**k * np.math.factorial(n-k) / (np.math.factorial(k) * 
                   np.math.factorial((n+m)//2 - k) * np.math.factorial((n-m)//2 - k))
            R += coef * r**(n-2*k)
        return R
    
    def _zernike_derivatives(self, n, m, r, theta):
        """
        Calculate derivatives of Zernike polynomial Z_n^m(r, theta).
        Returns (dZ/dr, dZ/dtheta).
        """
        if m >= 0:
            # For cos(m*theta) terms
            dZdr = self._radial_zernike_derivative(n, m, r) * np.cos(m * theta)
            dZdtheta = -m * self._radial_zernike(n, m, r) * np.sin(m * theta)
        else:
            # For sin(|m|*theta) terms
            m_abs = abs(m)
            dZdr = self._radial_zernike_derivative(n, m_abs, r) * np.sin(m_abs * theta)
            dZdtheta = m_abs * self._radial_zernike(n, m_abs, r) * np.cos(m_abs * theta)
        
        return dZdr, dZdtheta
    
    def _radial_zernike_derivative(self, n, m, r):
        """
        Calculate derivative of radial part of Zernike polynomial.
        """
        if n == m:
            return n * r**(n-1)
        
        dRdr = np.zeros_like(r)
        for k in range((n-m)//2 + 1):
            if n-2*k > 0:  # Skip terms where power becomes negative after differentiation
                coef = (-1)**k * np.math.factorial(n-k) * (n-2*k) / (np.math.factorial(k) * 
                       np.math.factorial((n+m)//2 - k) * np.math.factorial((n-m)//2 - k))
                dRdr += coef * r**(n-2*k-1)
        
        return dRdr

class WavefrontSimulator:
    """
    Simulates wavefront data for testing the Shack-Hartmann wavefront sensor.
    Generates spot patterns based on different wavefront aberrations.
    """
    
    def __init__(self, width=1456, height=1088, num_lenslets_x=10, num_lenslets_y=10):
        """
        Initialize the wavefront simulator.
        
        Args:
            width: Width of the simulated image in pixels
            height: Height of the simulated image in pixels
            num_lenslets_x: Number of lenslets in x-direction
            num_lenslets_y: Number of lenslets in y-direction
        """
        self.width = width
        self.height = height
        self.num_lenslets_x = num_lenslets_x
        self.num_lenslets_y = num_lenslets_y
        
    def generate_reference_spots(self):
        """
        Generate a reference spot pattern with no aberrations.
        
        Returns:
            Tuple of (image with spots, spot positions)
        """
        # Create a blank image
        image = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Calculate grid spacing
        x_spacing = (self.width - 100) / (self.num_lenslets_x - 1)
        y_spacing = (self.height - 100) / (self.num_lenslets_y - 1)
        
        # Create grid of spot positions
        x_positions = np.linspace(50, self.width - 50, self.num_lenslets_x)
        y_positions = np.linspace(50, self.height - 50, self.num_lenslets_y)
        
        xx, yy = np.meshgrid(x_positions, y_positions)
        spot_positions = np.column_stack((xx.flatten(), yy.flatten()))
        
        # Draw spots on the image
        for x, y in spot_positions:
            cv2.circle(image, (int(x), int(y)), 5, 255, -1)
        
        # Apply Gaussian blur to make spots look more realistic
        image = cv2.GaussianBlur(image, (15, 15), 3)
        
        return image, spot_positions
    
    def generate_aberrated_spots(self, aberration_type='defocus', magnitude=1.0, custom_wavefront=None):
        """
        Generate a spot pattern with specified aberration.
        
        Args:
            aberration_type: Type of aberration ('defocus', 'astigmatism', 'coma', 'spherical', 'random', 'custom')
            magnitude: Magnitude of the aberration
            custom_wavefront: Custom wavefront array (only used if aberration_type is 'custom')
            
        Returns:
            Tuple of (image with spots, spot positions, wavefront)
        """
        # Create a blank image
        image = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Calculate grid spacing
        x_spacing = (self.width - 100) / (self.num_lenslets_x - 1)
        y_spacing = (self.height - 100) / (self.num_lenslets_y - 1)
        
        # Create grid of reference spot positions
        x_positions = np.linspace(50, self.width - 50, self.num_lenslets_x)
        y_positions = np.linspace(50, self.height - 50, self.num_lenslets_y)
        
        xx, yy = np.meshgrid(x_positions, y_positions)
        reference_spots = np.column_stack((xx.flatten(), yy.flatten()))
        
        # Create normalized coordinate grid for wavefront calculation
        x_norm = np.linspace(-1, 1, self.num_lenslets_x)
        y_norm = np.linspace(-1, 1, self.num_lenslets_y)
        X_norm, Y_norm = np.meshgrid(x_norm, y_norm)
        r_norm = np.sqrt(X_norm**2 + Y_norm**2)
        theta = np.arctan2(Y_norm, X_norm)
        
        # Calculate wavefront based on aberration type
        if aberration_type == 'custom' and custom_wavefront is not None:
            # Resize custom wavefront to match lenslet array
            wavefront = cv2.resize(custom_wavefront, (self.num_lenslets_x, self.num_lenslets_y))
        else:
            wavefront = np.zeros((self.num_lenslets_y, self.num_lenslets_x))
            
            if aberration_type == 'defocus':
                # Defocus: Z(2,0) = 2r² - 1
                wavefront = magnitude * (2 * r_norm**2 - 1)
            elif aberration_type == 'astigmatism':
                # Astigmatism: Z(2,2) = r² * cos(2θ)
                wavefront = magnitude * r_norm**2 * np.cos(2 * theta)
            elif aberration_type == 'coma':
                # Coma: Z(3,1) = (3r³ - 2r) * cos(θ)
                wavefront = magnitude * (3 * r_norm**3 - 2 * r_norm) * np.cos(theta)
            elif aberration_type == 'spherical':
                # Spherical: Z(4,0) = 6r⁴ - 6r² + 1
                wavefront = magnitude * (6 * r_norm**4 - 6 * r_norm**2 + 1)
            elif aberration_type == 'random':
                # Random combination of Zernike polynomials
                np.random.seed(int(time.time()))
                for n in range(2, 5):
                    for m in range(-n, n+1, 2):
                        if m >= 0:
                            coef = magnitude * np.random.uniform(-1, 1)
                            Z = zernike(m, n, r_norm) * np.cos(m * theta)
                            wavefront += coef * Z
                        elif m < 0:
                            coef = magnitude * np.random.uniform(-1, 1)
                            Z = zernike(abs(m), n, r_norm) * np.sin(abs(m) * theta)
                            wavefront += coef * Z
        
        # Calculate wavefront gradient (slopes)
        grad_y, grad_x = np.gradient(wavefront)
        
        # Scale gradients to pixel shifts
        shift_scale = 20.0  # Pixels per unit gradient
        x_shifts = grad_x.flatten() * shift_scale
        y_shifts = grad_y.flatten() * shift_scale
        
        # Apply shifts to reference spots
        aberrated_spots = reference_spots.copy()
        aberrated_spots[:, 0] += x_shifts
        aberrated_spots[:, 1] += y_shifts
        
        # Draw spots on the image
        for x, y in aberrated_spots:
            cv2.circle(image, (int(x), int(y)), 5, 255, -1)
        
        # Apply Gaussian blur to make spots look more realistic
        image = cv2.GaussianBlur(image, (15, 15), 3)
        
        return image, aberrated_spots, wavefront
    
    def load_custom_wavefront(self, image_path):
        """
        Load a custom wavefront from an image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Normalized wavefront array
        """
        try:
            # Load image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
                
            # Normalize to [-1, 1] range
            wavefront = (image.astype(float) / 255.0) * 2.0 - 1.0
            
            return wavefront
        except Exception as e:
            print(f"Error loading custom wavefront: {str(e)}")
            return None

class ShackHartmannGUI:
    """
    GUI for Shack-Hartmann wavefront sensor.
    """
    
    def __init__(self, slm_width=800, slm_height=600, num_lenslets_x=10, num_lenslets_y=10, root=None):
        """
        Initialize the Shack-Hartmann GUI.
        
        Args:
            slm_width: Width of the SLM in pixels
            slm_height: Height of the SLM in pixels
            num_lenslets_x: Number of lenslets in x-direction
            num_lenslets_y: Number of lenslets in y-direction
            root: Tkinter root window (if None, a new one will be created)
        """
        # Store parameters
        self.slm_width = slm_width
        self.slm_height = slm_height
        self.num_lenslets_x = num_lenslets_x
        self.num_lenslets_y = num_lenslets_y
        
        # Initialize variables
        self.lenslet_array = None
        self.reference_spots = None
        self.spot_detector = None
        self.wavefront = None
        self.simulated_frame = None
        self.true_wavefront = None
        self.captured_frame = None
        
        # Create root window if not provided
        if root is None:
            self.root = tk.Tk()
            self.root.title("Shack-Hartmann Wavefront Sensor")
            self.root.geometry("1400x900")  # Increased window size to accommodate larger plots
        else:
            self.root = root
            
        # Create camera controller if available
        if CAMERA_AVAILABLE:
            try:
                self.camera = CameraController()
            except Exception as e:
                print(f"Error initializing camera: {str(e)}")
                self.camera = None
        else:
            self.camera = None
            
        # Initialize spot detector
        self.spot_detector = SpotDetector(self.num_lenslets_x, self.num_lenslets_y)
        
        # Create lenslet array generator
        self.lenslet_generator = LensletArrayGenerator(
            slm_width=self.slm_width,
            slm_height=self.slm_height
        )
        
        # Create wavefront reconstructor
        self.wavefront_reconstructor = WavefrontReconstructor()
        
        # Create wavefront simulator
        self.wavefront_simulator = WavefrontSimulator(
            width=1456, height=1088,
            num_lenslets_x=self.num_lenslets_x,
            num_lenslets_y=self.num_lenslets_y
        )
        
        # Create GUI
        self._create_widgets()
        
        # Start camera update loop if camera is available
        if self.camera is not None:
            try:
                self.camera.start()
                self.root.after(100, self._update_camera_view)
            except Exception as e:
                print(f"Error starting camera: {str(e)}")
                messagebox.showwarning("Camera Error", f"Could not start camera: {str(e)}")
        else:
            # Still start the camera view update to show "No Camera" message
            self.root.after(100, self._update_camera_view)
            
    def _create_widgets(self):
        """Create the GUI widgets."""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create left and right frames
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create scrollable right frame for controls
        right_frame_outer = ttk.Frame(main_frame)
        right_frame_outer.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        # Create canvas with scrollbar for controls
        control_canvas = tk.Canvas(right_frame_outer, width=250)
        control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add scrollbar to canvas
        scrollbar = ttk.Scrollbar(right_frame_outer, orient=tk.VERTICAL, command=control_canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        control_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Create frame for controls inside canvas
        right_frame = ttk.Frame(control_canvas)
        
        # Add right_frame to canvas
        canvas_frame = control_canvas.create_window((0, 0), window=right_frame, anchor=tk.NW)
        
        # Configure scrolling
        def _on_frame_configure(event):
            control_canvas.configure(scrollregion=control_canvas.bbox("all"))
            control_canvas.itemconfig(canvas_frame, width=control_canvas.winfo_width())
        
        right_frame.bind("<Configure>", _on_frame_configure)
        
        # Make mouse wheel scroll the canvas
        def _on_mousewheel(event):
            control_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        control_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Create main notebook for all visualizations
        self.main_notebook = ttk.Notebook(left_frame)
        self.main_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs for different visualizations
        camera_tab = ttk.Frame(self.main_notebook)
        spots_tab = ttk.Frame(self.main_notebook)
        lenslet_tab = ttk.Frame(self.main_notebook)
        wavefront_2d_tab = ttk.Frame(self.main_notebook)
        wavefront_3d_tab = ttk.Frame(self.main_notebook)
        
        self.main_notebook.add(camera_tab, text="Camera View")
        self.main_notebook.add(spots_tab, text="Spot Detection")
        self.main_notebook.add(lenslet_tab, text="Lenslet Pattern")
        self.main_notebook.add(wavefront_2d_tab, text="2D Wavefront")
        self.main_notebook.add(wavefront_3d_tab, text="3D Wavefront")
        
        # Camera view
        camera_frame = ttk.LabelFrame(camera_tab, text="Camera View")
        camera_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.camera_canvas = tk.Canvas(camera_frame, width=self.slm_width, height=self.slm_height)
        self.camera_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Spots display
        spots_frame = ttk.LabelFrame(spots_tab, text="Detected Spots")
        spots_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.spots_fig = Figure(figsize=(8, 6), dpi=100)
        self.spots_ax = self.spots_fig.add_subplot(111)
        self.spots_canvas = FigureCanvasTkAgg(self.spots_fig, master=spots_frame)
        self.spots_canvas.draw()
        self.spots_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Lenslet pattern display
        lenslet_frame = ttk.LabelFrame(lenslet_tab, text="Lenslet Array Pattern")
        lenslet_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.lenslet_fig = Figure(figsize=(8, 6), dpi=100)
        self.lenslet_ax = self.lenslet_fig.add_subplot(111)
        self.lenslet_canvas = FigureCanvasTkAgg(self.lenslet_fig, master=lenslet_frame)
        self.lenslet_canvas.draw()
        self.lenslet_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initial empty lenslet display
        self.lenslet_ax.set_title("Lenslet Array Pattern")
        self.lenslet_ax.set_xlabel("X (pixels)")
        self.lenslet_ax.set_ylabel("Y (pixels)")
        empty_pattern = np.zeros((self.slm_height, self.slm_width))
        self.lenslet_im = self.lenslet_ax.imshow(empty_pattern, cmap='gray', aspect='equal')
        self.lenslet_fig.colorbar(self.lenslet_im, ax=self.lenslet_ax, label="Grayscale")
        self.lenslet_canvas.draw()
        
        # 2D wavefront plot
        wavefront_2d_frame = ttk.LabelFrame(wavefront_2d_tab, text="2D Wavefront")
        wavefront_2d_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.wavefront_fig = Figure(figsize=(8, 6), dpi=100)
        self.wavefront_ax = self.wavefront_fig.add_subplot(111)
        self.wavefront_canvas = FigureCanvasTkAgg(self.wavefront_fig, master=wavefront_2d_frame)
        self.wavefront_canvas.draw()
        self.wavefront_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 3D wavefront plot
        wavefront_3d_frame = ttk.LabelFrame(wavefront_3d_tab, text="3D Wavefront")
        wavefront_3d_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.wavefront_3d_fig = Figure(figsize=(8, 6), dpi=100)
        self.wavefront_3d_ax = self.wavefront_3d_fig.add_subplot(111, projection='3d')
        self.wavefront_3d_canvas = FigureCanvasTkAgg(self.wavefront_3d_fig, master=wavefront_3d_frame)
        self.wavefront_3d_canvas.draw()
        self.wavefront_3d_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # === Left panel controls ===
        
        # Lenslet array parameters
        lenslet_frame = ttk.LabelFrame(right_frame, text="Lenslet Array Parameters", padding=5)
        lenslet_frame.pack(fill=tk.X, pady=5)
        
        # Number of lenslets
        ttk.Label(lenslet_frame, text="Number of lenslets (X):").pack(anchor=tk.W)
        self.num_lenslets_x_var = tk.StringVar(value=str(self.num_lenslets_x))
        ttk.Entry(lenslet_frame, textvariable=self.num_lenslets_x_var).pack(fill=tk.X, pady=2)
        
        ttk.Label(lenslet_frame, text="Number of lenslets (Y):").pack(anchor=tk.W)
        self.num_lenslets_y_var = tk.StringVar(value=str(self.num_lenslets_y))
        ttk.Entry(lenslet_frame, textvariable=self.num_lenslets_y_var).pack(fill=tk.X, pady=2)
        
        # Lenslet pitch (mm)
        ttk.Label(lenslet_frame, text="Lenslet pitch (mm):").pack(anchor=tk.W)
        self.lenslet_pitch_var = tk.StringVar(value="2.5")
        ttk.Entry(lenslet_frame, textvariable=self.lenslet_pitch_var).pack(fill=tk.X, pady=2)
        
        # Focal length (mm)
        ttk.Label(lenslet_frame, text="Focal length (mm):").pack(anchor=tk.W)
        self.focal_length_var = tk.StringVar(value="100")
        ttk.Entry(lenslet_frame, textvariable=self.focal_length_var).pack(fill=tk.X, pady=2)
        
        # Blazed grating parameters
        ttk.Label(lenslet_frame, text="Shift X (cycles):").pack(anchor=tk.W)
        self.shift_x_var = tk.StringVar(value="0")
        ttk.Entry(lenslet_frame, textvariable=self.shift_x_var).pack(fill=tk.X, pady=2)
        
        ttk.Label(lenslet_frame, text="Shift Y (cycles):").pack(anchor=tk.W)
        self.shift_y_var = tk.StringVar(value="0")
        ttk.Entry(lenslet_frame, textvariable=self.shift_y_var).pack(fill=tk.X, pady=2)
        
        # Generate button
        ttk.Button(lenslet_frame, text="Generate Pattern", command=self._on_generate_pattern).pack(fill=tk.X, pady=5)
        
        # Display on SLM button
        ttk.Button(lenslet_frame, text="Display on SLM", command=self._on_display_slm).pack(fill=tk.X, pady=5)
        
        # Simulation frame
        simulation_frame = ttk.LabelFrame(right_frame, text="Simulation Controls", padding=5)
        simulation_frame.pack(fill=tk.X, pady=5)
        
        # Aberration type
        ttk.Label(simulation_frame, text="Aberration type:").pack(anchor=tk.W)
        self.aberration_type_var = tk.StringVar(value="defocus")
        aberration_combo = ttk.Combobox(simulation_frame, textvariable=self.aberration_type_var)
        aberration_combo['values'] = ('defocus', 'astigmatism', 'coma', 'spherical', 'random', 'custom')
        aberration_combo.pack(fill=tk.X, pady=2)
        
        # Aberration magnitude
        ttk.Label(simulation_frame, text="Magnitude:").pack(anchor=tk.W)
        self.magnitude_var = tk.StringVar(value="1.0")
        ttk.Entry(simulation_frame, textvariable=self.magnitude_var).pack(fill=tk.X, pady=2)
        
        # Load custom wavefront button
        ttk.Button(simulation_frame, text="Load Custom Wavefront", command=self._on_load_custom_wavefront).pack(fill=tk.X, pady=5)
        
        # Generate simulated spots button
        ttk.Button(simulation_frame, text="Generate Simulated Spots", command=self._on_generate_simulated_spots).pack(fill=tk.X, pady=5)
        
        # Camera controls
        camera_frame = ttk.LabelFrame(right_frame, text="Camera Controls", padding=5)
        camera_frame.pack(fill=tk.X, pady=5)
        
        # Capture reference button
        ttk.Button(camera_frame, text="Capture Reference", command=self._on_capture_reference).pack(fill=tk.X, pady=5)
        
        # Capture and process image button
        ttk.Button(camera_frame, text="Capture and Process Image", command=self._on_capture_and_process).pack(fill=tk.X, pady=5)
        
        # Reconstruct wavefront button
        ttk.Button(camera_frame, text="Reconstruct Wavefront", command=self._on_reconstruct_wavefront).pack(fill=tk.X, pady=5)
        
        # Save wavefront button
        ttk.Button(camera_frame, text="Save Wavefront", command=self._on_save_wavefront).pack(fill=tk.X, pady=5)
        
        # Status bar
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, anchor=tk.W, padding=5)
        status_label.pack(side=tk.LEFT, fill=tk.X)
        
    def _update_camera_view(self):
        """Update camera view."""
        if self.camera is not None and hasattr(self.camera, 'is_running') and self.camera.is_running:
            try:
                # Get the latest frame from the camera
                frame = self.camera.get_latest_frame()
                
                if frame is not None:
                    # Convert to RGB for matplotlib
                    if len(frame.shape) == 2:  # If grayscale
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                    else:  # If already RGB
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Update the camera view
                    self.camera_canvas.delete("all")
                    
                    # Resize the frame to fit the canvas
                    resized_frame = cv2.resize(rgb_frame, (self.slm_width, self.slm_height))
                    
                    # Convert OpenCV image to PIL Image
                    pil_image = Image.fromarray(resized_frame)
                    
                    # Convert PIL Image to Tkinter PhotoImage
                    tk_image = ImageTk.PhotoImage(image=pil_image)
                    
                    # Keep a reference to prevent garbage collection
                    self.camera_canvas.image = tk_image
                    
                    # Display the image on the canvas
                    self.camera_canvas.create_image(0, 0, image=tk_image, anchor="nw")
                    
                    # Detect spots if reference spots are available
                    if self.reference_spots is not None and self.spot_detector is not None:
                        # Detect spots in current frame
                        spots = self.spot_detector.detect_spots(frame)
                        
                        # Calculate spot shifts
                        shifts = self.spot_detector.calculate_spot_shifts(spots)
                        
                        # Update spots display
                        self.spots_ax.clear()
                        self.spots_ax.set_title("Detected Spots")
                        self.spots_ax.imshow(rgb_frame, cmap='gray')
                        
                        # Plot reference spots
                        if self.reference_spots is not None:
                            self.spots_ax.scatter(self.reference_spots[:, 0], self.reference_spots[:, 1], 
                                                 color='r', marker='o', s=50, facecolors='none', label='Reference')
                        
                        # Plot current spots
                        if spots is not None:
                            self.spots_ax.scatter(spots[:, 0], spots[:, 1], 
                                                 color='g', marker='+', s=50, label='Current')
                        
                        # Plot shifts as arrows
                        if shifts is not None:
                            for i, ref_spot in enumerate(self.reference_spots):
                                self.spots_ax.arrow(ref_spot[0], ref_spot[1], 
                                                   shifts[i, 0], shifts[i, 1], 
                                                   color='b', width=0.5, head_width=5)
                        
                        self.spots_ax.legend()
                        self.spots_canvas.draw()
            except Exception as e:
                print(f"Error updating camera view: {str(e)}")
        elif self.simulated_frame is not None:
            # Display simulated frame
            if len(self.simulated_frame.shape) == 2:  # If grayscale
                rgb_frame = cv2.cvtColor(self.simulated_frame, cv2.COLOR_GRAY2RGB)
            else:  # If already RGB
                rgb_frame = cv2.cvtColor(self.simulated_frame, cv2.COLOR_BGR2RGB)
                
            # Update the camera view
            self.camera_canvas.delete("all")
            
            # Resize the frame to fit the canvas
            resized_frame = cv2.resize(rgb_frame, (self.slm_width, self.slm_height))
            
            # Convert OpenCV image to PIL Image
            pil_image = Image.fromarray(resized_frame)
            
            # Convert PIL Image to Tkinter PhotoImage
            tk_image = ImageTk.PhotoImage(image=pil_image)
            
            # Keep a reference to prevent garbage collection
            self.camera_canvas.image = tk_image
            
            # Display the image on the canvas
            self.camera_canvas.create_image(0, 0, image=tk_image, anchor="nw")
            
            # If we have reference spots, update the spots display
            if self.reference_spots is not None and self.spot_detector is not None:
                # Detect spots in simulated frame
                spots = self.spot_detector.detect_spots(self.simulated_frame)
                
                # Calculate spot shifts
                shifts = self.spot_detector.calculate_spot_shifts(spots)
                
                # Update spots display
                self.spots_ax.clear()
                self.spots_ax.set_title("Detected Spots (Simulated)")
                self.spots_ax.imshow(rgb_frame, cmap='gray')
                
                # Plot reference spots
                if self.reference_spots is not None:
                    self.spots_ax.scatter(self.reference_spots[:, 0], self.reference_spots[:, 1], 
                                         color='r', marker='o', s=50, facecolors='none', label='Reference')
                
                # Plot current spots
                if spots is not None:
                    self.spots_ax.scatter(spots[:, 0], spots[:, 1], 
                                         color='g', marker='+', s=50, label='Current')
                
                # Plot shifts as arrows
                if shifts is not None:
                    for i, ref_spot in enumerate(self.reference_spots):
                        self.spots_ax.arrow(ref_spot[0], ref_spot[1], 
                                           shifts[i, 0], shifts[i, 1], 
                                           color='b', width=0.5, head_width=5)
                
                self.spots_ax.legend()
                self.spots_canvas.draw()
        else:
            # No camera available, display a message in the camera view
            if hasattr(self, 'camera_ax') and self.camera_ax is not None:
                self.camera_ax.clear()
                self.camera_ax.set_title("Camera Not Available")
                self.camera_ax.text(0.5, 0.5, "No camera connected\nUse simulation controls", 
                                   horizontalalignment='center',
                                   verticalalignment='center',
                                   transform=self.camera_ax.transAxes)
                self.camera_ax.axis('off')
                self.camera_canvas.draw()
                
        # Schedule the next update
        self.root.after(50, self._update_camera_view)
        
    def _update_wavefront_plots(self, wavefront):
        """Update both 2D and 3D wavefront plots."""
        # Update 2D plot
        self.wavefront_ax.clear()
        self.wavefront_ax.set_title("Reconstructed Wavefront")
        self.wavefront_im = self.wavefront_ax.imshow(wavefront, cmap='viridis', interpolation='bilinear')
        self.wavefront_fig.colorbar(self.wavefront_im, ax=self.wavefront_ax, label="Wavefront (waves)")
        self.wavefront_canvas.draw()
        
        # Update 3D plot
        self.wavefront_3d_ax.clear()
        self.wavefront_3d_ax.set_title("3D Wavefront")
        
        # Create coordinate grids for 3D plot
        y_size, x_size = wavefront.shape
        x = np.linspace(0, x_size-1, x_size)
        y = np.linspace(0, y_size-1, y_size)
        X, Y = np.meshgrid(x, y)
        
        # Create 3D surface plot
        # Use a stride to reduce the number of points for better performance
        stride = max(1, min(x_size, y_size) // 50)  # Adaptive stride based on size
        surf = self.wavefront_3d_ax.plot_surface(
            X[::stride, ::stride], 
            Y[::stride, ::stride], 
            wavefront[::stride, ::stride], 
            cmap='viridis',
            linewidth=0,
            antialiased=True,
            rstride=1,
            cstride=1
        )
        
        # Add colorbar
        self.wavefront_3d_fig.colorbar(surf, ax=self.wavefront_3d_ax, shrink=0.5, aspect=5, label="Wavefront (waves)")
        
        # Set labels
        self.wavefront_3d_ax.set_xlabel('X (pixels)')
        self.wavefront_3d_ax.set_ylabel('Y (pixels)')
        self.wavefront_3d_ax.set_zlabel('Phase (waves)')
        
        # Set initial view angle
        self.wavefront_3d_ax.view_init(elev=30, azim=45)
        
        # Draw the 3D plot
        self.wavefront_3d_canvas.draw()
        
    def _on_load_custom_wavefront(self):
        """Load a custom wavefront from an image file."""
        try:
            # Open file dialog
            file_path = filedialog.askopenfilename(
                title="Select Wavefront Image",
                filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.tif;*.tiff"), ("All files", "*.*")]
            )
            
            if not file_path:
                return  # User cancelled
                
            # Load the wavefront
            custom_wavefront = self.wavefront_simulator.load_custom_wavefront(file_path)
            
            if custom_wavefront is None:
                self.status_var.set("Failed to load custom wavefront")
                return
                
            # Store the wavefront
            self.true_wavefront = custom_wavefront
            
            # Display the wavefront
            self.wavefront_ax.clear()
            self.wavefront_ax.set_title("Custom Wavefront (Ground Truth)")
            self.wavefront_im = self.wavefront_ax.imshow(custom_wavefront, cmap='viridis')
            self.wavefront_fig.colorbar(self.wavefront_im, ax=self.wavefront_ax, label="Wavefront (waves)")
            self.wavefront_canvas.draw()
            
            self.status_var.set(f"Custom wavefront loaded from {os.path.basename(file_path)}")
            
            # Set aberration type to custom
            self.aberration_type_var.set("custom")
            
        except Exception as e:
            self.status_var.set(f"Error loading custom wavefront: {str(e)}")
            messagebox.showerror("Error", f"Error loading custom wavefront: {str(e)}")
    
    def _on_generate_simulated_spots(self):
        """Generate simulated spot pattern with specified aberration."""
        try:
            # Get parameters
            aberration_type = self.aberration_type_var.get()
            magnitude = float(self.magnitude_var.get())
            
            # Update number of lenslets
            self.num_lenslets_x = int(self.num_lenslets_x_var.get())
            self.num_lenslets_y = int(self.num_lenslets_y_var.get())
            
            # Update the wavefront simulator
            self.wavefront_simulator = WavefrontSimulator(
                width=1456, height=1088,
                num_lenslets_x=self.num_lenslets_x,
                num_lenslets_y=self.num_lenslets_y
            )
            
            # Generate reference spots first if not already done
            if self.reference_spots is None:
                ref_image, ref_spots = self.wavefront_simulator.generate_reference_spots()
                self.reference_spots = ref_spots
                
                # Save reference image
                cv2.imwrite("reference_spots.png", ref_image)
                
                # Initialize spot detector if not already done
                if self.spot_detector is None:
                    self.spot_detector = SpotDetector(self.num_lenslets_x, self.num_lenslets_y)
                
                # Update the spot detector with the reference spots
                self.spot_detector.reference_spots = self.reference_spots
                
                self.status_var.set(f"Reference spots generated: {len(self.reference_spots)} spots")
            
            # Generate aberrated spots
            aberrated_image, aberrated_spots, wavefront = self.wavefront_simulator.generate_aberrated_spots(
                aberration_type=aberration_type,
                magnitude=magnitude,
                custom_wavefront=self.true_wavefront if aberration_type == 'custom' else None
            )
            
            # Save aberrated image
            cv2.imwrite("aberrated_spots.png", aberrated_image)
            
            # Store the simulated frame and true wavefront
            self.simulated_frame = aberrated_image
            self.true_wavefront = wavefront
            
            # Display the true wavefront
            self.wavefront_ax.clear()
            self.wavefront_ax.set_title("True Wavefront")
            self.wavefront_im = self.wavefront_ax.imshow(wavefront, cmap='viridis')
            self.wavefront_fig.colorbar(self.wavefront_im, ax=self.wavefront_ax, label="Wavefront (waves)")
            self.wavefront_canvas.draw()
            
            self.status_var.set(f"Simulated spots generated with {aberration_type} aberration")
            
        except Exception as e:
            self.status_var.set(f"Error generating simulated spots: {str(e)}")
            messagebox.showerror("Error", f"Error generating simulated spots: {str(e)}")
    
    def _on_reconstruct_wavefront(self):
        """Reconstruct wavefront."""
        if self.reference_spots is None:
            self.status_var.set("Capture reference spots first")
            return
            
        try:
            # Use the captured frame if available, otherwise fall back to the latest frame
            if hasattr(self, 'captured_frame') and self.captured_frame is not None:
                frame = self.captured_frame
            elif self.camera is not None and hasattr(self.camera, 'get_latest_frame') and self.camera.is_running:
                frame = self.camera.get_latest_frame()
            elif self.simulated_frame is not None:
                frame = self.simulated_frame
            else:
                self.status_var.set("No frame available for spot detection")
                return
                
            # Detect spots
            spots = self.spot_detector.detect_spots(frame)
            
            # Calculate spot shifts
            shifts = self.spot_detector.calculate_spot_shifts(spots)
            
            # Get physical parameters
            lenslet_pitch = float(self.lenslet_pitch_var.get()) / 1000  # Convert mm to m
            focal_length = float(self.focal_length_var.get()) / 1000  # Convert mm to m
            
            # Get reconstruction parameters
            upsampling_factor = 4  # Default upsampling factor
            reconstruction_method = "Zonal"  # Default reconstruction method
            
            # Reconstruct wavefront with higher resolution
            self.wavefront_reconstructor = WavefrontReconstructor(lenslet_pitch, focal_length)
            
            if reconstruction_method == "Modal":
                # Use modal reconstruction with Zernike polynomials
                num_modes = 15  # Number of Zernike modes to use
                self.wavefront = self.wavefront_reconstructor.reconstruct_modal(
                    shifts, self.num_lenslets_x, self.num_lenslets_y, 
                    num_modes=num_modes, upsampling_factor=upsampling_factor
                )
            else:
                # Use zonal reconstruction (default)
                self.wavefront = self.wavefront_reconstructor.reconstruct_zonal(
                    shifts, self.num_lenslets_x, self.num_lenslets_y,
                    upsampling_factor=upsampling_factor
                )
            
            # Update both 2D and 3D wavefront plots
            self._update_wavefront_plots(self.wavefront)
            
            # If we have a true wavefront, calculate and display error
            if self.true_wavefront is not None:
                # Resize true wavefront to match reconstructed wavefront if needed
                if self.true_wavefront.shape != self.wavefront.shape:
                    # Create coordinate grids
                    y_orig, x_orig = np.mgrid[0:self.true_wavefront.shape[0], 0:self.true_wavefront.shape[1]]
                    y_new, x_new = np.mgrid[0:self.true_wavefront.shape[0]:self.wavefront.shape[0]*1j, 
                                           0:self.true_wavefront.shape[1]:self.wavefront.shape[1]*1j]
                    
                    # Interpolate true wavefront to match reconstructed wavefront
                    resized_true_wavefront = griddata(
                        (y_orig.flatten(), x_orig.flatten()),
                        self.true_wavefront.flatten(),
                        (y_new, x_new),
                        method='cubic'
                    )
                    
                    # Calculate error
                    error = self.wavefront - resized_true_wavefront
                else:
                    error = self.wavefront - self.true_wavefront
                
                # Display error
                rms_error = np.sqrt(np.mean(np.square(error)))
                self.status_var.set(f"Wavefront reconstructed. RMS error: {rms_error:.6f} waves")
            else:
                self.status_var.set("Wavefront reconstructed")
                
        except Exception as e:
            self.status_var.set(f"Error in wavefront reconstruction: {str(e)}")
            traceback.print_exc()
    
    def _on_capture_and_process(self):
        """Capture a single frame from the camera and process it for spot detection."""
        if self.camera is None or not hasattr(self.camera, 'capture_high_quality_frame'):
            self.status_var.set("Camera not available. Cannot capture image.")
            return
            
        try:
            # Capture a high-quality frame
            frame = self.camera.capture_high_quality_frame()
            
            if frame is None:
                self.status_var.set("Failed to capture frame")
                return
                
            # Store the captured frame
            self.captured_frame = frame
            
            # Convert to RGB for display
            if len(frame.shape) == 2:  # If grayscale
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            else:  # If already RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Update the camera view with the captured frame
            self.camera_canvas.delete("all")
            
            # Resize the frame to fit the canvas
            resized_frame = cv2.resize(rgb_frame, (self.slm_width, self.slm_height))
            
            # Convert OpenCV image to PIL Image
            pil_image = Image.fromarray(resized_frame)
            
            # Convert PIL Image to Tkinter PhotoImage
            tk_image = ImageTk.PhotoImage(image=pil_image)
            
            # Keep a reference to prevent garbage collection
            self.camera_canvas.image = tk_image
            
            # Display the image on the canvas
            self.camera_canvas.create_image(0, 0, image=tk_image, anchor="nw")
            
            # Process the frame for spot detection if reference spots are available
            if self.reference_spots is not None and self.spot_detector is not None:
                # Detect spots in current frame
                spots = self.spot_detector.detect_spots(frame)
                
                # Calculate spot shifts
                shifts = self.spot_detector.calculate_spot_shifts(spots)
                
                # Update spots display
                self.spots_ax.clear()
                self.spots_ax.set_title("Detected Spots")
                self.spots_ax.imshow(rgb_frame, cmap='gray')
                
                # Plot reference spots
                if self.reference_spots is not None:
                    self.spots_ax.scatter(self.reference_spots[:, 0], self.reference_spots[:, 1], 
                                         color='r', marker='o', s=50, facecolors='none', label='Reference')
                
                # Plot current spots
                if spots is not None:
                    self.spots_ax.scatter(spots[:, 0], spots[:, 1], 
                                         color='g', marker='+', s=50, label='Current')
                
                # Plot shifts as arrows
                if shifts is not None:
                    for i, ref_spot in enumerate(self.reference_spots):
                        self.spots_ax.arrow(ref_spot[0], ref_spot[1], 
                                           shifts[i, 0], shifts[i, 1], 
                                           color='b', width=0.5, head_width=5)
                
                self.spots_ax.legend()
                self.spots_canvas.draw()
                
                self.status_var.set(f"Image captured and processed. Detected {len(spots)} spots.")
            else:
                self.status_var.set("Image captured. Capture reference spots first to enable spot detection.")
                
        except Exception as e:
            self.status_var.set(f"Error capturing and processing image: {str(e)}")
            traceback.print_exc()
    
    def _on_capture_reference(self):
        """Capture reference spot pattern."""
        if self.camera is None or not hasattr(self.camera, 'capture_high_quality_frame'):
            # No camera available, use simulated data
            self.status_var.set("Camera not available. Using simulated data.")
            
            try:
                # Create a simulated reference pattern
                # Generate a grid of spots based on lenslet array dimensions
                height, width = 1088, 1456  # Standard camera resolution
                grid_x = np.linspace(50, width-50, self.num_lenslets_x)
                grid_y = np.linspace(50, height-50, self.num_lenslets_y)
                
                # Create meshgrid of spot positions
                xx, yy = np.meshgrid(grid_x, grid_y)
                
                # Create reference spots array
                self.reference_spots = np.column_stack((xx.flatten(), yy.flatten()))
                
                # Create a simulated frame with spots
                frame = np.zeros((height, width), dtype=np.uint8)
                
                # Draw spots on the frame
                for x, y in self.reference_spots:
                    cv2.circle(frame, (int(x), int(y)), 5, 255, -1)
                
                # Apply Gaussian blur to make spots look more realistic
                frame = cv2.GaussianBlur(frame, (15, 15), 3)
                
                # Save frame as reference
                cv2.imwrite("reference_spots.png", frame)
                
                # Initialize spot detector if not already done
                if self.spot_detector is None:
                    self.spot_detector = SpotDetector(self.num_lenslets_x, self.num_lenslets_y)
                
                # Update the spot detector with the reference spots
                self.spot_detector.reference_spots = self.reference_spots
                
                self.status_var.set(f"Simulated reference spots created: {len(self.reference_spots)} spots")
                
                # Update spots display
                if hasattr(self, 'spots_ax') and self.spots_ax is not None:
                    self.spots_ax.clear()
                    self.spots_ax.set_title("Simulated Reference Spots")
                    self.spots_ax.imshow(frame, cmap='gray')
                    self.spots_ax.scatter(self.reference_spots[:, 0], self.reference_spots[:, 1], 
                                         color='r', marker='o', s=50, facecolors='none', label='Reference')
                    self.spots_ax.legend()
                    self.spots_canvas.draw()
                
            except Exception as e:
                self.status_var.set(f"Error creating simulated reference: {str(e)}")
                messagebox.showerror("Error", f"Error creating simulated reference: {str(e)}")
            
            return
            
        try:
            # Capture a high-quality frame
            frame = self.camera.capture_high_quality_frame()
            
            if frame is None:
                self.status_var.set("Failed to capture frame")
                return
                
            # Save frame as reference
            cv2.imwrite("reference_spots.png", frame)
            
            # Initialize spot detector if not already done
            if self.spot_detector is None:
                self.spot_detector = SpotDetector(self.num_lenslets_x, self.num_lenslets_y)
                
            # Detect spots in the reference frame
            self.reference_spots = self.spot_detector.detect_spots(frame)
            
            if self.reference_spots is None or len(self.reference_spots) == 0:
                self.status_var.set("No spots detected in reference image")
                messagebox.showwarning("Warning", "No spots detected in reference image. Try adjusting the camera or lenslet pattern.")
                return
                
            self.status_var.set(f"Reference spots captured: {len(self.reference_spots)} spots detected")
            
            # Update the spot detector with the reference spots
            self.spot_detector.reference_spots = self.reference_spots
            
        except Exception as e:
            self.status_var.set(f"Error capturing reference: {str(e)}")
            messagebox.showerror("Error", f"Error capturing reference: {str(e)}")   
            
    def _on_generate_pattern(self):
        """Generate lenslet array pattern."""
        try:
            # Get parameters from GUI
            self.num_lenslets_x = int(self.num_lenslets_x_var.get())
            self.num_lenslets_y = int(self.num_lenslets_y_var.get())
            self.lenslet_pitch = float(self.lenslet_pitch_var.get()) / 1000  # Convert mm to m
            self.focal_length = float(self.focal_length_var.get()) / 1000  # Convert mm to m
            
            # Generate lenslet array pattern
            self.lenslet_array = self.lenslet_generator.generate_lenslet_array(
                self.lenslet_pitch, self.focal_length)
            
            # Add blazed grating if specified
            shift_x = float(self.shift_x_var.get())
            shift_y = float(self.shift_y_var.get())
            
            if shift_x != 0 or shift_y != 0:
                self.lenslet_array = self.lenslet_generator.add_blazed_grating(
                    self.lenslet_array, shift_x, shift_y)
            
            # Convert to grayscale for display
            grayscale_pattern = self.lenslet_generator.phase_to_grayscale(self.lenslet_array)
            
            # Update display
            self.lenslet_ax.clear()
            self.lenslet_ax.set_title("Lenslet Array Pattern")
            self.lenslet_ax.set_xlabel("X (pixels)")
            self.lenslet_ax.set_ylabel("Y (pixels)")
            self.lenslet_im = self.lenslet_ax.imshow(grayscale_pattern, cmap='gray', aspect='equal')
            self.lenslet_fig.colorbar(self.lenslet_im, ax=self.lenslet_ax, label="Grayscale")
            self.lenslet_canvas.draw()
            
            self.status_var.set("Lenslet pattern generated")
            
        except Exception as e:
            self.status_var.set(f"Error generating pattern: {str(e)}")
            messagebox.showerror("Error", f"Error generating pattern: {str(e)}")
            
    def _on_display_slm(self):
        """Display pattern on SLM."""
        if self.lenslet_array is None:
            self.status_var.set("Generate a pattern first")
            return
            
        try:
            # Convert to grayscale for SLM
            grayscale_pattern = self.lenslet_generator.phase_to_grayscale(self.lenslet_array)
            
            # Save pattern to file
            cv2.imwrite("lenslet_pattern.png", grayscale_pattern)
            
            # Set SDL environment variables for display control before initializing pygame
            os.environ['SDL_VIDEO_WINDOW_POS'] = '1280,0'  # Position at main monitor width
            os.environ['SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS'] = '0'
            
            # Create a thread for SLM display
            self.slm_thread = threading.Thread(target=self._display_slm_pattern, args=(grayscale_pattern,))
            self.slm_thread.daemon = True  # Thread will be terminated when main program exits
            self.slm_thread.start()
            self.status_var.set("Pattern sent to SLM. Press ESC in SLM window to close.")
            
        except Exception as e:
            self.status_var.set(f"Error displaying on SLM: {str(e)}")
            messagebox.showerror("Error", f"Error displaying on SLM: {str(e)}")
            
    def _display_slm_pattern(self, pattern):
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
            pattern_height, pattern_width = pattern.shape
            if (pattern_width, pattern_height) != target_display_size:
                print(f"Resizing pattern from {pattern_width}x{pattern_height} to {target_display_size[0]}x{target_display_size[1]}")
                resized_pattern = cv2.resize(pattern, target_display_size)
            else:
                resized_pattern = pattern
                
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
                
    def _on_reconstruct_wavefront(self):
        """Reconstruct wavefront."""
        if self.reference_spots is None:
            self.status_var.set("Capture reference spots first")
            return
            
        try:
            # Use the captured frame if available, otherwise fall back to the latest frame
            if hasattr(self, 'captured_frame') and self.captured_frame is not None:
                frame = self.captured_frame
            elif self.camera is not None and hasattr(self.camera, 'get_latest_frame') and self.camera.is_running:
                frame = self.camera.get_latest_frame()
            elif self.simulated_frame is not None:
                frame = self.simulated_frame
            else:
                self.status_var.set("No frame available for spot detection")
                return
                
            # Detect spots
            spots = self.spot_detector.detect_spots(frame)
            
            # Calculate spot shifts
            shifts = self.spot_detector.calculate_spot_shifts(spots)
            
            # Get physical parameters
            lenslet_pitch = float(self.lenslet_pitch_var.get()) / 1000  # Convert mm to m
            focal_length = float(self.focal_length_var.get()) / 1000  # Convert mm to m
            
            # Get reconstruction parameters
            upsampling_factor = 4  # Default upsampling factor
            reconstruction_method = "Zonal"  # Default reconstruction method
            
            # Reconstruct wavefront with higher resolution
            self.wavefront_reconstructor = WavefrontReconstructor(lenslet_pitch, focal_length)
            
            if reconstruction_method == "Modal":
                # Use modal reconstruction with Zernike polynomials
                num_modes = 15  # Number of Zernike modes to use
                self.wavefront = self.wavefront_reconstructor.reconstruct_modal(
                    shifts, self.num_lenslets_x, self.num_lenslets_y, 
                    num_modes=num_modes, upsampling_factor=upsampling_factor
                )
            else:
                # Use zonal reconstruction (default)
                self.wavefront = self.wavefront_reconstructor.reconstruct_zonal(
                    shifts, self.num_lenslets_x, self.num_lenslets_y,
                    upsampling_factor=upsampling_factor
                )
            
            # Update both 2D and 3D wavefront plots
            self._update_wavefront_plots(self.wavefront)
            
            # If we have a true wavefront, calculate and display error
            if self.true_wavefront is not None:
                # Resize true wavefront to match reconstructed wavefront if needed
                if self.true_wavefront.shape != self.wavefront.shape:
                    # Create coordinate grids
                    y_orig, x_orig = np.mgrid[0:self.true_wavefront.shape[0], 0:self.true_wavefront.shape[1]]
                    y_new, x_new = np.mgrid[0:self.true_wavefront.shape[0]:self.wavefront.shape[0]*1j, 
                                           0:self.true_wavefront.shape[1]:self.wavefront.shape[1]*1j]
                    
                    # Interpolate true wavefront to match reconstructed wavefront
                    resized_true_wavefront = griddata(
                        (y_orig.flatten(), x_orig.flatten()),
                        self.true_wavefront.flatten(),
                        (y_new, x_new),
                        method='cubic'
                    )
                    
                    # Calculate error
                    error = self.wavefront - resized_true_wavefront
                else:
                    error = self.wavefront - self.true_wavefront
                
                # Display error
                rms_error = np.sqrt(np.mean(np.square(error)))
                self.status_var.set(f"Wavefront reconstructed. RMS error: {rms_error:.6f} waves")
            else:
                self.status_var.set("Wavefront reconstructed")
                
        except Exception as e:
            self.status_var.set(f"Error in wavefront reconstruction: {str(e)}")
            traceback.print_exc()
    
    def _on_save_wavefront(self):
        """Save wavefront to file."""
        if self.wavefront is None:
            self.status_var.set("Reconstruct wavefront first")
            return
            
        try:
            # Save wavefront to file
            np.save("wavefront.npy", self.wavefront)
            
            self.status_var.set("Wavefront saved to wavefront.npy")
            
        except Exception as e:
            self.status_var.set(f"Error saving wavefront: {str(e)}")
            messagebox.showerror("Error", f"Error saving wavefront: {str(e)}")

# Main block to launch the GUI when the script is run directly
if __name__ == "__main__":
    try:
        # Create and run the GUI
        app = ShackHartmannGUI()
        app.root.mainloop()
    except Exception as e:
        print(f"Error starting application: {str(e)}")