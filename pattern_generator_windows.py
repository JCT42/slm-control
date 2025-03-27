"""
Pattern Generator for Sony LCX016AL-6 SLM
Generates phase patterns using Gerchberg-Saxton algorithm for far-field image reconstruction.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from tkinter import filedialog
import tkinter as tk
import os
from tqdm import tqdm
from matplotlib.widgets import Button

class FarFieldSimulator:
    def __init__(self, wavelength=650e-9):
        """Initialize the far-field simulator
        
        Sony LCX016AL-6 SLM Specifications:
        - Resolution: 832 x 624 pixels
        - Pixel pitch: 32 µm
        - Active area: 26.6 mm x 20.0 mm
        - Refresh rate: 60 Hz
        - Contrast ratio: 200:1
        
        Args:
            wavelength: Laser wavelength in meters (default: 650nm green laser)
        """
        # Sony LCX016AL-6 specifications
        self.width = 832
        self.height = 624
        self.pixel_size = 32e-6  # 32 µm pixel pitch
        self.active_area = (26.6e-3, 20.0e-3)  # 26.6mm x 20.0mm
        self.refresh_rate = 60  # 60 Hz
        self.contrast_ratio = 200  # 200:1
        
        # Calculate fill factor from active area and resolution
        total_area = self.active_area[0] * self.active_area[1]
        pixel_area = self.width * self.height * (self.pixel_size ** 2)
        self.fill_factor = pixel_area / total_area
        
        # Simulation parameters
        self.padding_factor = 2
        self.padded_width = self.width * self.padding_factor
        self.padded_height = self.height * self.padding_factor
        self.wavelength = wavelength
        self.distance = 0.5  # 0.5m propagation distance
        
        # Calculate important parameters
        self.k = 2 * np.pi / wavelength  # Wave number
        self.dx = self.pixel_size
        self.df_x = 1 / (self.padded_width * self.dx)  # Frequency step size x
        self.df_y = 1 / (self.padded_height * self.dx)  # Frequency step size y
        
        # Create coordinate grids
        self.x = np.linspace(-self.padded_width//2, self.padded_width//2-1, self.padded_width) * self.dx
        self.y = np.linspace(-self.padded_height//2, self.padded_height//2-1, self.padded_height) * self.dx
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Create hidden root window for file dialogs
        self.root = tk.Tk()
        self.root.withdraw()
        
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
        
    def generate_input_beam(self):
        """Generate Gaussian input beam profile matching Sony SLM specifications"""
        # Calculate beam parameters based on active area
        beam_width = self.active_area[0]  # 26.6mm
        beam_height = self.active_area[1]  # 20.0mm
        
        # Use larger sigma to ensure beam covers full SLM
        sigma_x = beam_width / 2.355  # FWHM = 2.355 * sigma
        sigma_y = beam_height / 2.355
        
        # Create meshgrid centered on SLM
        x = np.linspace(-self.width/2, self.width/2, self.width) * self.pixel_size
        y = np.linspace(-self.height/2, self.height/2, self.height) * self.pixel_size
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
        
    def gerchberg_saxton(self, target_image, num_iterations=200):
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
        
    def run_simulation(self, num_iterations=200):
        """Run the complete simulation with 40 iterations by default"""
        # Load and preprocess target image
        padded_target, target_image, start_x, end_x, start_y, end_y = self.load_target_image()
        
        # Generate input beam
        input_beam = self.generate_input_beam()
        input_beam_center = input_beam[start_y:end_y, start_x:end_x]
        
        # Run Gerchberg-Saxton
        phase_mask, final_field, errors = self.gerchberg_saxton(padded_target, num_iterations)
        
        # Simulate reconstruction
        reconstructed = self.simulate_far_field(final_field)
        
        # Extract central portions
        phase_mask_center = phase_mask[start_y:end_y, start_x:end_x]
        reconstructed_center = reconstructed[start_y:end_y, start_x:end_x]
        reconstructed_center = reconstructed_center / np.max(reconstructed_center)
        
        # Create figure with proper aspect ratio
        aspect_ratio = self.width / self.height
        fig = plt.figure(figsize=(15, 10))
        
        # Input image
        ax1 = plt.subplot(231)
        plt.imshow(target_image, cmap="gray", aspect=aspect_ratio)
        plt.title("Target Image")
        plt.colorbar(label="Intensity")
        
        # Input beam profile
        ax2 = plt.subplot(232)
        plt.imshow(input_beam_center, cmap="viridis", aspect=aspect_ratio)
        plt.title("Input Beam Profile")
        plt.colorbar(label="Amplitude")
        
        # Phase mask
        ax3 = plt.subplot(233)
        plt.imshow(phase_mask_center, cmap="twilight", aspect=aspect_ratio)
        plt.title("SLM Phase Mask")
        plt.colorbar(label="Phase (radians)")
        
        # Modulated beam (near-field, at SLM plane)
        ax4 = plt.subplot(234)
        modulated = input_beam_center * np.exp(1j * phase_mask_center)
        plt.imshow(np.abs(modulated), cmap="viridis", aspect=aspect_ratio)
        plt.title("Modulated Beam (Near-field)")
        plt.colorbar(label="Amplitude")
        
        # Reconstructed image (far-field)
        ax5 = plt.subplot(235)
        plt.imshow(reconstructed_center, cmap="gray", aspect=aspect_ratio)
        plt.title("Reconstructed Image (Far-field)")
        plt.colorbar(label="Intensity")
        
        # Error plot
        ax6 = plt.subplot(236)
        plt.plot(errors, 'b-', label='Error')
        
        # Add minimum error annotation
        min_error = min(errors)
        min_idx = errors.index(min_error)
        plt.plot(min_idx, min_error, 'ro')  # Red dot at minimum
        plt.annotate(f'Min Error: {min_error:.2e}', 
                    xy=(min_idx, min_error), 
                    xytext=(10, 10),
                    textcoords='offset points',
                    ha='left',
                    va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.title("Convergence")
        plt.xlabel("Iteration")
        plt.ylabel("Error")
        plt.yscale('log')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.minorticks_on()
        
        # Add save button
        plt.subplots_adjust(bottom=0.2)
        ax_button = plt.axes([0.4, 0.05, 0.2, 0.075])
        save_button = Button(ax_button, 'Save Phase Mask')
        
        def save_mask(event):
            save_path = filedialog.asksaveasfilename(
                title="Save Phase Mask",
                defaultextension=".png",
                filetypes=[("PNG files", "*.png")]
            )
            if save_path:
                # Scale to 8-bit and save
                phase_8bit = ((phase_mask_center + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
                cv2.imwrite(save_path, phase_8bit)
                print(f"Phase mask saved to: {save_path}")
                
                # Also save beam profile and modulated beam
                base_path = os.path.splitext(save_path)[0]
                cv2.imwrite(f"{base_path}_beam.png", (input_beam_center * 255).astype(np.uint8))
                cv2.imwrite(f"{base_path}_modulated.png", (np.abs(modulated) * 255).astype(np.uint8))
                print(f"Beam profile saved to: {base_path}_beam.png")
                print(f"Modulated beam saved to: {base_path}_modulated.png")
        
        save_button.on_clicked(save_mask)
        
        # Add SLM info text
        plt.figtext(0.02, 0.02, 
                   f"Sony LCX016AL-6 SLM Specs:\n" +
                   f"Resolution: {self.width}x{self.height}\n" +
                   f"Pixel pitch: {self.pixel_size*1e6:.1f}µm\n" +
                   f"Active area: {self.active_area[0]*1e3:.1f}mm x {self.active_area[1]*1e3:.1f}mm\n" +
                   f"Refresh rate: {self.refresh_rate}Hz\n" +
                   f"Contrast ratio: {self.contrast_ratio}:1\n" +
                   f"Wavelength: {self.wavelength*1e9:.0f}nm",
                   fontsize=8, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Create simulator
    simulator = FarFieldSimulator()
    
    try:
        # Run simulation
        simulator.run_simulation(num_iterations=200)
    except Exception as e:
        print(f"Error: {e}")
