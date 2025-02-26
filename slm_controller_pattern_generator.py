"""
Advanced Pattern Generator and Controller for Sony LCX016AL-6 SLM
Combines pattern generation using Gerchberg-Saxton algorithm with direct SLM control.
"""

import cv2
import numpy as np
import pygame
import os
from PIL import Image
import threading
from pathlib import Path
import time
import subprocess
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
import matplotlib.pyplot as plt

from slm_controller import SLMController, Button

class SLMPatternController(SLMController):
    def __init__(self):
        super().__init__()
        # Initialize far-field simulation parameters
        self.wavelength = 532e-9  # 532nm green laser
        self.padding_factor = 2
        self.padded_width = self.width * self.padding_factor
        self.padded_height = self.height * self.padding_factor
        
        # Calculate important parameters
        self.k = 2 * np.pi / self.wavelength  # Wave number
        self.dx = self.pixel_pitch * 1e-6  # Convert to meters
        self.df_x = 1 / (self.padded_width * self.dx)  # Frequency step size x
        self.df_y = 1 / (self.padded_height * self.dx)  # Frequency step size y
        
        # Create coordinate grids
        self.x = np.linspace(-self.padded_width//2, self.padded_width//2-1, self.padded_width) * self.dx
        self.y = np.linspace(-self.padded_height//2, self.padded_height//2-1, self.padded_height) * self.dx
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Additional buttons for pattern generation
        button_width = 150
        button_height = 35
        button_margin = 30
        button_spacing = 10
        
        # Add pattern generation button below load button
        generate_y = self.load_button.rect.bottom + button_spacing
        self.generate_button = Button(
            self.load_button.rect.left,
            generate_y,
            button_width,
            button_height,
            "Generate Pattern",
            self.font,
            (100, 150, 100)
        )

    def generate_input_beam(self):
        """Generate Gaussian input beam profile matching Sony SLM specifications"""
        # Calculate beam parameters based on active area
        beam_width = self.active_area[0] * 1e-3  # Convert mm to meters
        beam_height = self.active_area[1] * 1e-3
        
        # Use larger sigma to ensure beam covers full SLM
        sigma_x = beam_width / 2.355  # FWHM = 2.355 * sigma
        sigma_y = beam_height / 2.355
        
        # Create meshgrid centered on SLM
        x = np.linspace(-self.width/2, self.width/2, self.width) * self.dx
        y = np.linspace(-self.height/2, self.height/2, self.height) * self.dx
        X, Y = np.meshgrid(x, y)
        
        # Calculate centered Gaussian beam
        beam = np.exp(-X**2 / (2 * sigma_x**2) - Y**2 / (2 * sigma_y**2))
        
        # Normalize beam
        beam = beam / np.max(beam)
        
        # Create padded version
        padded_beam = np.zeros((self.padded_height, self.padded_width))
        start_x = (self.padded_width - self.width) // 2
        end_x = start_x + self.width
        start_y = (self.padded_height - self.height) // 2
        end_y = start_y + self.height
        padded_beam[start_y:end_y, start_x:end_x] = beam
        
        return padded_beam

    def load_target_image(self):
        """Load and preprocess target image to match SLM specifications"""
        # Open file dialog
        file_path = filedialog.askopenfilename(
            title="Select Target Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif")]
        )
        
        if not file_path:
            raise ValueError("No image file selected")
            
        # Read image and convert to grayscale
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"Could not load image: {file_path}")
            
        # Convert BGR to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        # Resize to match SLM resolution while maintaining aspect ratio
        target_height = self.height
        target_width = self.width
        
        # Calculate scaling factors
        scale_x = target_width / image.shape[1]
        scale_y = target_height / image.shape[0]
        scale = min(scale_x, scale_y)
        
        # Calculate new dimensions
        new_width = int(image.shape[1] * scale)
        new_height = int(image.shape[0] * scale)
        
        # Resize image
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Create black canvas of SLM size
        canvas = np.zeros((target_height, target_width), dtype=np.uint8)
        
        # Calculate padding to center the image
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        
        # Place resized image in center of canvas
        canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = image
        
        # Normalize to [0, 1]
        target_image = canvas.astype(float) / 255.0
        
        # Zero pad for FFT
        padded_target = np.zeros((self.padded_height, self.padded_width))
        start_x_idx = (self.padded_width - self.width) // 2
        end_x_idx = start_x_idx + self.width
        start_y_idx = (self.padded_height - self.height) // 2
        end_y_idx = start_y_idx + self.height
        padded_target[start_y_idx:end_y_idx, start_x_idx:end_x_idx] = target_image
        
        return padded_target, target_image, start_x_idx, end_x_idx, start_y_idx, end_y_idx

    def gerchberg_saxton(self, target_image, num_iterations=100):
        """Run Gerchberg-Saxton algorithm"""
        # Get input beam profile
        gaussian_beam = self.generate_input_beam()
        
        # Initialize with random phase
        random_phase = np.exp(1j * 2 * np.pi * np.random.rand(self.padded_height, self.padded_width))
        field = gaussian_beam * random_phase
        
        # Track error
        errors = []
        
        # Run iterations with progress bar
        for _ in tqdm(range(num_iterations), desc="Running Gerchberg-Saxton"):
            # Forward FFT to far field
            far_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
            
            # Calculate current error
            current_intensity = np.abs(far_field)**2
            error = np.sum((np.sqrt(current_intensity) - np.sqrt(target_image))**2)
            errors.append(error)
            
            # Keep the phase but replace amplitude with target image
            far_field = np.sqrt(target_image) * np.exp(1j * np.angle(far_field))
            
            # Inverse FFT back to SLM plane
            field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(far_field)))
            
            # Keep the phase and enforce Gaussian amplitude constraint
            field = gaussian_beam * np.exp(1j * np.angle(field))
            
        return np.angle(field), field, errors

    def simulate_far_field(self, field):
        """Simulate propagation to far field"""
        far_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
        intensity = np.abs(far_field)**2
        return intensity

    def generate_pattern(self):
        """Generate a new pattern using Gerchberg-Saxton algorithm"""
        try:
            # Load and preprocess target image
            padded_target, target_image, start_x, end_x, start_y, end_y = self.load_target_image()
            
            # Run Gerchberg-Saxton algorithm
            phase, field, errors = self.gerchberg_saxton(padded_target)
            
            # Extract the phase pattern for the SLM
            slm_phase = phase[start_y:end_y, start_x:end_x]
            
            # Convert phase to 8-bit grayscale
            pattern = ((slm_phase + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
            
            # Apply calibration if available
            if self.is_calibrated:
                pattern = self.apply_calibration(pattern)
            
            # Update the current pattern
            self.current_pattern = pattern
            
            # Update the preview
            self.update_preview()
            
            # Display the pattern on the SLM
            self.update_slm_display()
            
            return True
        except Exception as e:
            print(f"Error generating pattern: {e}")
            return False

    def handle_events(self):
        """Handle UI events including the new generate button"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
                
            # Handle button events
            if self.load_button.handle_event(event):
                self.load_pattern()
            elif self.generate_button.handle_event(event):
                self.generate_pattern()
            elif self.save_preview_button.handle_event(event):
                self.save_preview()
            elif self.save_camera_button.handle_event(event) and self.camera_active:
                self.save_camera()
            elif self.pause_camera_button.handle_event(event) and self.camera_active:
                self.camera_paused = not self.camera_paused
                self.pause_camera_button.text = "Resume Camera" if self.camera_paused else "Pause Camera"
            elif self.calibrate_button.handle_event(event):
                self.calibrate()
                
        return True

    def update_display(self):
        """Update the control window display including the new generate button"""
        # Fill background
        self.control_display.fill((240, 240, 240))
        
        # Draw preview area
        pygame.draw.rect(self.control_display, (200, 200, 200), self.preview_rect)
        if self.current_pattern is not None:
            self.control_display.blit(self.preview_surface, self.preview_rect)
            
        # Draw camera preview area if active
        if self.camera_active:
            pygame.draw.rect(self.control_display, (200, 200, 200), self.camera_rect)
            self.control_display.blit(self.camera_surface, self.camera_rect)
            
        # Draw all buttons
        self.load_button.draw(self.control_display)
        self.generate_button.draw(self.control_display)
        self.save_preview_button.draw(self.control_display)
        if self.camera_active:
            self.save_camera_button.draw(self.control_display)
            self.pause_camera_button.draw(self.control_display)
        self.calibrate_button.draw(self.control_display)
        
        # Update display
        pygame.display.flip()

if __name__ == "__main__":
    controller = SLMPatternController()
    controller.run()
