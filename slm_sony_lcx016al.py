import numpy as np
import cv2
import matplotlib.pyplot as plt
from tkinter import filedialog
import tkinter as tk
from tqdm import tqdm
import os

# Physical parameters for SONY LCX016AL-6 SLM
wavelength = 0.5e-6    # Initial wavelength (500 nm)
pixel_size = 32e-6     # SLM pixel size (32 µm)
distance = 0.1         # Initial propagation distance (10 cm)

# SLM specifications
image_size_x = 832     # SLM width in pixels
image_size_y = 624     # SLM height in pixels
num_iterations = 20    # Number of iterations for optimization
padding_factor = 2     # Padding factor for FFT

# Precompute frequency grids for ASM
def prepare_asm_grids(nx, ny, dx, wavelength):
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dx)
    KX, KY = np.meshgrid(kx, ky)
    k = 2 * np.pi / wavelength
    return KX, KY, k

def propagate_asm(field, z, KX, KY, k):
    """Optimized Angular Spectrum Method using precomputed grids"""
    H = np.exp(1j * z * np.sqrt(k**2 - KX**2 - KY**2 + 0j))
    H[KX**2 + KY**2 > k**2] = 0
    return np.fft.ifft2(np.fft.fft2(field) * H)

def optimize_phase(target, input_beam, z, wavelength, pixel_size, num_iterations, KX, KY, k):
    """Optimized phase pattern calculation"""
    phase = 2 * np.pi * np.random.rand(*input_beam.shape)
    field = input_beam * np.exp(1j * phase)
    
    for i in range(num_iterations):
        # Forward propagation
        prop_field = propagate_asm(field, z, KX, KY, k)
        
        # Apply amplitude constraint in target plane
        prop_field = np.sqrt(target) * np.exp(1j * np.angle(prop_field))
        
        # Back propagation
        field = propagate_asm(prop_field, -z, KX, KY, k)
        
        # Apply amplitude constraint in SLM plane
        field = input_beam * np.exp(1j * np.angle(field))
    
    return np.angle(field)

def main():
    # Create and hide root window
    root = tk.Tk()
    root.withdraw()
    
    # Open file dialog
    file_path = filedialog.askopenfilename(
        title="Choose target image",
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"),
            ("All files", "*.*")
        ]
    )
    
    if not file_path:
        print("No file selected. Exiting...")
        return
    
    # Load and preprocess target image
    target = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if target is None:
        print(f"Error: Could not load image from {file_path}")
        return
    
    # Resize to match SLM resolution
    target = cv2.resize(target, (image_size_x, image_size_y))
    target = target.astype(float) / 255.0
    
    # Zero padding
    padded_size_x = image_size_x * padding_factor
    padded_size_y = image_size_y * padding_factor
    padded_target = np.zeros((padded_size_y, padded_size_x))
    start_idx_x = (padded_size_x - image_size_x) // 2
    start_idx_y = (padded_size_y - image_size_y) // 2
    end_idx_x = start_idx_x + image_size_x
    end_idx_y = start_idx_y + image_size_y
    padded_target[start_idx_y:end_idx_y, start_idx_x:end_idx_x] = target
    
    # Create Gaussian input beam (adjusted for rectangular shape)
    x = np.linspace(-1, 1, padded_size_x)
    y = np.linspace(-1, 1, padded_size_y)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    input_beam = np.exp(-R**2 / 0.5**2)
    
    # Precompute ASM grids
    KX, KY, k = prepare_asm_grids(padded_size_x, padded_size_y, pixel_size, wavelength)
    
    # Create figure and subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot input image
    ax1.imshow(target, cmap='gray')
    ax1.set_title(f'Target Image\n({image_size_x} × {image_size_y} pixels)')
    ax1.axis('equal')
    
    # Optimize phase pattern
    print("\nOptimizing phase pattern...")
    final_phase = optimize_phase(padded_target, input_beam, distance, wavelength,
                                  pixel_size, num_iterations, KX, KY, k)
    
    # Calculate reconstruction
    field = input_beam * np.exp(1j * final_phase)
    reconstructed_field = propagate_asm(field, distance, KX, KY, k)
    reconstructed = np.abs(reconstructed_field)**2
    
    # Plot phase pattern
    phase_plot = ax2.imshow(final_phase[start_idx_y:end_idx_y, start_idx_x:end_idx_x],
                          cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax2.set_title('SLM Phase Pattern\n(32 µm pixel size)')
    plt.colorbar(phase_plot, ax=ax2, label='Phase (rad)')
    ax2.axis('equal')
    
    # Plot reconstructed image
    recon_plot = ax3.imshow(reconstructed[start_idx_y:end_idx_y, start_idx_x:end_idx_x],
                           cmap='gray')
    ax3.set_title(f'Reconstructed Image\n(z = {distance*100:.1f} cm)')
    plt.colorbar(recon_plot, ax=ax3, label='Intensity')
    ax3.axis('equal')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
