import numpy as np
from scipy import fftpack
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

class HologramGenerator:
    def __init__(self, slm_width=1920, slm_height=1080):
        """Initialize the hologram generator with SLM dimensions"""
        self.width = slm_width
        self.height = slm_height
        
        # Create hidden root window for file dialogs
        self.root = tk.Tk()
        self.root.withdraw()
        
    def load_target_image(self, image_path):
        """Load and preprocess the target image"""
        # Read image in grayscale
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
            
        # Resize to match SLM dimensions
        img = cv2.resize(img, (self.width, self.height))
        
        # Normalize to [0, 1]
        img = img.astype(float) / 255.0
        
        return img
        
    def gerchberg_saxton(self, target_image, num_iterations=50):
        """Generate a phase-only hologram using the Gerchberg-Saxton algorithm
        
        Args:
            target_image: 2D numpy array of the desired intensity pattern
            num_iterations: Number of iterations for the algorithm
            
        Returns:
            phase_pattern: 2D numpy array of phase values [0, 2π]
        """
        # Initialize random phase
        phase = np.random.random((self.height, self.width)) * 2 * np.pi
        
        # Create initial complex field
        field = np.sqrt(target_image) * np.exp(1j * phase)
        
        for i in range(num_iterations):
            # Forward propagation (FFT)
            propagated = fftpack.fft2(field)
            
            # Keep amplitude of target, but maintain propagated phase
            amplitude = np.abs(propagated)
            phase = np.angle(propagated)
            propagated = np.sqrt(target_image) * np.exp(1j * phase)
            
            # Backward propagation (IFFT)
            field = fftpack.ifft2(propagated)
            
            # Force unit amplitude
            field = np.exp(1j * np.angle(field))
            
        # Return final phase pattern
        return np.angle(field) % (2 * np.pi)
    
    def generate_hologram(self, image_path, output_path=None):
        """Generate and save a hologram for the given image"""
        # Load and preprocess target image
        target = self.load_target_image(image_path)
        
        # Generate phase pattern
        phase_pattern = self.gerchberg_saxton(target)
        
        # Convert to 8-bit grayscale [0, 255]
        hologram = (phase_pattern / (2 * np.pi) * 255).astype(np.uint8)
        
        if output_path:
            # Save hologram
            cv2.imwrite(str(output_path), hologram)
            
        return hologram
    
    def preview_result(self, hologram):
        """Preview the expected reconstruction from the hologram"""
        # Convert to complex field
        field = np.exp(1j * hologram * 2 * np.pi / 255)
        
        # Simulate propagation with FFT
        reconstruction = fftpack.fft2(field)
        intensity = np.abs(reconstruction)**2
        
        # Normalize and display
        intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min())
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(131)
        plt.imshow(hologram, cmap='gray')
        plt.title('Hologram Pattern')
        
        plt.subplot(132)
        plt.imshow(intensity, cmap='gray')
        plt.title('Simulated Reconstruction')
        
        plt.tight_layout()
        plt.show()
        
    def select_input_image(self):
        """Open file dialog to select input image"""
        file_path = filedialog.askopenfilename(
            title="Select Input Image",
            filetypes=[
                ("Image files", "*.png;*.jpg;*.jpeg;*.bmp"),
                ("All files", "*.*")
            ]
        )
        return file_path if file_path else None
        
    def select_output_path(self, default_name="hologram.png"):
        """Open file dialog to select where to save the hologram"""
        file_path = filedialog.asksaveasfilename(
            title="Save Hologram As",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[("PNG files", "*.png")]
        )
        return file_path if file_path else None

if __name__ == "__main__":
    # Create generator
    generator = HologramGenerator()
    
    # Select input image
    image_path = generator.select_input_image()
    if image_path:
        # Select output location
        output_path = generator.select_output_path()
        if output_path:
            try:
                # Generate hologram
                hologram = generator.generate_hologram(image_path, output_path)
                
                # Preview result
                generator.preview_result(hologram)
                
            except Exception as e:
                print(f"Error generating hologram: {e}")
        else:
            print("Save cancelled")
    else:
        print("No image selected")
