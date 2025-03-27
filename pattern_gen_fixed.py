"""
Fixed version of the pattern generator with improved error calculation and convergence criteria.
This version addresses the issues with extremely small errors and improves the convergence behavior.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import time
from tqdm import tqdm
import threading
import os
import json

class PatternGenerator:
    """
    Pattern generator for phase-only spatial light modulators.
    Implements Gerchberg-Saxton and Mixed-Region Amplitude Freedom algorithms.
    """
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
    
    def calculate_error(self, field, algorithm):
        """
        Calculate normalized RMSE error between reconstructed and target intensity.
        This provides a more meaningful error metric than absolute difference.
        """
        recon_intensity = np.abs(field)**2
        
        if algorithm.lower() == 'gs':
            # For GS, calculate error over entire field
            mse = np.mean((recon_intensity - self.target_intensity)**2)
            rmse = np.sqrt(mse)
            # Normalize by target intensity range
            norm_error = rmse / np.max(self.target_intensity)
        else:
            # For MRAF, calculate error only in signal region
            sr_mask = self.signal_region_mask
            if np.sum(sr_mask) > 0:  # Ensure signal region is not empty
                mse = np.mean((recon_intensity[sr_mask == 1] - self.target_intensity[sr_mask == 1])**2)
                rmse = np.sqrt(mse)
                # Normalize by target intensity range
                norm_error = rmse / np.max(self.target_intensity[sr_mask == 1])
            else:
                norm_error = 0.0
                
        return norm_error
    
    def calculate_field_change(self, field, prev_field):
        """
        Calculate the mean change in field intensity between iterations.
        This is a more reliable convergence metric than error delta.
        """
        return np.mean(np.abs(np.abs(field)**2 - np.abs(prev_field)**2))
    
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
        field_change_history = []
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
            field_change_history.append(field_change)
                
            # Print current metrics for debugging
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
        
        # If we reached max iterations, note that
        if i == max_iterations - 1:
            print(stop_reason)
        
        # Calculate final error for comparison
        final_error = self.calculate_error(field, algorithm)
        improvement = initial_error / final_error if final_error > 0 else float('inf')
        print(f"Final error: {final_error:.3e}, Improvement: {improvement:.2f}x")
        
        return field, error_history, stop_reason

# You can copy the rest of your pattern_gen_2.0.py file here, 
# replacing the PatternGenerator class with this improved version
# and updating any references to it in your AdvancedPatternGenerator class.
