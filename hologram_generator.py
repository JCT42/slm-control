import numpy as np
from scipy import fftpack
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm

class HologramGenerator:
    def __init__(self, slm_width=1920, slm_height=1080, wavelength=532e-9, pixel_size=6.4e-6):
        """Initialize the hologram generator
        
        Args:
            slm_width: Width of SLM in pixels
            slm_height: Height of SLM in pixels
            wavelength: Laser wavelength in meters (default: 532nm green laser)
            pixel_size: SLM pixel size in meters (default: 6.4µm)
        """
        self.width = slm_width
        self.height = slm_height
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        
        # Calculate important parameters
        self.k = 2 * np.pi / wavelength  # Wave number
        self.focal_length = 0.5  # Focal length in meters (adjust as needed)
        
        # Create hidden root window for file dialogs
        self.root = tk.Tk()
        self.root.withdraw()
        
    def load_target_image(self, image_path):
        """Load and preprocess the target image"""
        # Read image in grayscale
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
            
        print(f"Original image dimensions: {img.shape}")
            
        # Convert to grayscale if image is color
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        # Resize to match SLM dimensions
        img = cv2.resize(img, (self.width, self.height))
        print(f"Resized to SLM dimensions: {img.shape}")
        
        # Verify dimensions
        if img.shape != (self.height, self.width):
            raise ValueError(f"Image dimensions {img.shape} do not match SLM dimensions ({self.height}, {self.width})")
        
        # Normalize to [0, 1]
        img = img.astype(float) / 255.0
        
        return img
        
    def generate_input_beam(self):
        """Generate Gaussian input beam profile"""
        x = np.linspace(-self.width/2, self.width/2, self.width)
        y = np.linspace(-self.height/2, self.height/2, self.height)
        X, Y = np.meshgrid(x, y)
        
        # Gaussian beam parameters
        w0 = min(self.width, self.height) / 4  # Beam waist (adjust as needed)
        r2 = X**2 + Y**2
        beam = np.exp(-r2 / (2 * w0**2))
        
        return beam
        
    def gerchberg_saxton(self, target_image, num_iterations=50):
        """Generate a phase-only hologram using the Gerchberg-Saxton algorithm
        
        Args:
            target_image: 2D numpy array of the desired intensity pattern
            num_iterations: Number of iterations for the algorithm
            
        Returns:
            phase_pattern: 2D numpy array of phase values [0, 2π]
        """
        # Get input beam profile
        input_beam = self.generate_input_beam()
        
        # Initialize random phase
        phase = np.random.random((self.height, self.width)) * 2 * np.pi
        
        # Create initial complex field with input beam profile
        field = input_beam * np.exp(1j * phase)
        
        # Create progress bar
        pbar = tqdm(range(num_iterations), desc="Generating hologram")
        
        for i in pbar:
            # Forward propagation (FFT)
            propagated = fftpack.fft2(field)
            
            # Keep amplitude of target, but maintain propagated phase
            phase = np.angle(propagated)
            propagated = np.sqrt(target_image) * np.exp(1j * phase)
            
            # Backward propagation (IFFT)
            field = fftpack.ifft2(propagated)
            
            # Force input beam amplitude
            field = input_beam * np.exp(1j * np.angle(field))
            
            # Update progress description with current iteration
            pbar.set_description(f"Generating hologram (iteration {i+1}/{num_iterations})")
        
        # Return final phase pattern
        return np.angle(field) % (2 * np.pi)
    
    def generate_hologram(self, image_path, output_path=None):
        """Generate and optionally save a hologram for the given image"""
        # Load and preprocess target image
        target = self.load_target_image(image_path)
        
        # Generate phase pattern
        phase_pattern = self.gerchberg_saxton(target)
        
        # Convert to 8-bit grayscale [0, 255]
        hologram = (phase_pattern / (2 * np.pi) * 255).astype(np.uint8)
        
        # Save if output path provided
        if output_path:
            cv2.imwrite(str(output_path), hologram)
            
        return hologram
    
    def preview_result(self, hologram, input_image):
        """Preview the input image, hologram and expected reconstruction"""
        # Simulate propagation
        field = np.exp(1j * hologram * 2 * np.pi / 255)
        reconstruction = fftpack.fft2(field)
        intensity = np.abs(reconstruction)**2
        intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min())
        
        # Create figure and subplots
        fig = plt.figure(figsize=(15, 5))
        
        # Input image
        ax1 = plt.subplot(131)
        plt.imshow(input_image, cmap='gray')
        plt.title('Input Image')
        
        # Hologram pattern
        ax2 = plt.subplot(132)
        plt.imshow(hologram, cmap='gray')
        plt.title('Hologram Pattern')
        
        # Simulated reconstruction
        ax3 = plt.subplot(133)
        plt.imshow(intensity, cmap='gray')
        plt.title('Simulated Reconstruction')
        
        # Add save button
        plt.subplots_adjust(bottom=0.2)  # Make room for button
        ax_button = plt.axes([0.4, 0.05, 0.2, 0.075])  # [left, bottom, width, height]
        save_button = Button(ax_button, 'Save Hologram')
        
        def save_hologram(event):
            output_path = self.select_output_path()
            if output_path:
                cv2.imwrite(str(output_path), hologram)
                print(f"Hologram saved to: {output_path}")
        
        save_button.on_clicked(save_hologram)
        
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
        try:
            # Load input image
            input_image = generator.load_target_image(image_path)
            
            # Generate hologram
            hologram = generator.generate_hologram(image_path)
            
            # Preview result with input image
            generator.preview_result(hologram, input_image)
            
        except Exception as e:
            print(f"Error generating hologram: {e}")
    else:
        print("No image selected")
