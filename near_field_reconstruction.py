import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from tkinter import filedialog
import tkinter as tk
import os

# Physical parameters
wavelength = 0.5e-6  # Initial wavelength (500 nm)
pixel_size = 8e-6    # SLM pixel size (8 µm)
distance = 0.1       # Initial propagation distance (10 cm)

# Algorithm parameters
image_size = 512     # Size of the image
num_iterations = 50  # Number of iterations for optimization
padding_factor = 2   # Padding factor for FFT

def propagate_asm(field, z, dx, wavelength):
    """
    Angular Spectrum Method for near-field propagation
    field: input complex field
    z: propagation distance
    dx: pixel size
    wavelength: wavelength of light
    """
    ny, nx = field.shape
    
    # Spatial frequencies
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dx)
    KX, KY = np.meshgrid(kx, ky)
    
    # Wave number
    k = 2 * np.pi / wavelength
    
    # Transfer function
    H = np.exp(1j * z * np.sqrt(k**2 - KX**2 - KY**2 + 0j))
    H[KX**2 + KY**2 > k**2] = 0  # Filter evanescent waves
    
    # Propagate using angular spectrum
    F = np.fft.fft2(field)
    F_prop = F * H
    return np.fft.ifft2(F_prop)

def optimize_phase(target, input_beam, z, wavelength, pixel_size, num_iterations):
    """
    Optimize phase pattern using Gerchberg-Saxton algorithm
    """
    # Initialize random phase
    phase = 2 * np.pi * np.random.rand(*input_beam.shape)
    field = input_beam * np.exp(1j * phase)
    
    # Gerchberg-Saxton algorithm for near-field
    for i in range(num_iterations):
        # Forward propagation
        prop_field = propagate_asm(field, z, pixel_size, wavelength)
        
        # Apply amplitude constraint in target plane
        prop_field = np.sqrt(target) * np.exp(1j * np.angle(prop_field))
        
        # Back propagation
        field = propagate_asm(prop_field, -z, pixel_size, wavelength)
        
        # Apply amplitude constraint in SLM plane
        field = input_beam * np.exp(1j * np.angle(field))
        
        # Print progress
        if (i + 1) % 10 == 0:
            print(f"Iteration {i + 1}/{num_iterations}")
    
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
    
    # Resize and normalize target image
    target = cv2.resize(target, (image_size, image_size))
    target = target.astype(float) / 255.0
    
    # Zero padding
    padded_size = image_size * padding_factor
    padded_target = np.zeros((padded_size, padded_size))
    start_idx_y = (padded_size - image_size) // 2
    end_idx_y = start_idx_y + image_size
    start_idx_x = (padded_size - image_size) // 2
    end_idx_x = start_idx_x + image_size
    padded_target[start_idx_y:end_idx_y, start_idx_x:end_idx_x] = target
    
    # Create Gaussian input beam
    x = np.linspace(-1, 1, padded_size)
    y = np.linspace(-1, 1, padded_size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    input_beam = np.exp(-R**2 / 0.5**2)
    
    # Create figure and subplots
    plt.ion()  # Turn on interactive mode
    fig = plt.figure(figsize=(15, 7))
    gs = fig.add_gridspec(2, 3, height_ratios=[4, 1])
    
    # Create axes for plots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Create axes for sliders
    ax_wavelength = fig.add_subplot(gs[1, 0])
    ax_distance = fig.add_subplot(gs[1, 1])
    
    # Plot input image
    ax1.imshow(target, cmap='gray')
    ax1.set_title('Target Image\n(832 x 624 pixels)')
    ax1.axis('equal')
    
    # Initial optimization
    final_phase = optimize_phase(padded_target, input_beam, distance, wavelength, 
                               pixel_size, num_iterations)
    
    # Initial reconstruction
    field = input_beam * np.exp(1j * final_phase)
    reconstructed_field = propagate_asm(field, distance, pixel_size, wavelength)
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
    
    # Create sliders
    wavelength_slider = Slider(
        ax=ax_wavelength,
        label='Wavelength (nm)',
        valmin=400,
        valmax=700,
        valinit=wavelength*1e9,
        valfmt='%0.0f'
    )
    
    distance_slider = Slider(
        ax=ax_distance,
        label='Distance (cm)',
        valmin=1,
        valmax=50,
        valinit=distance*100,
        valfmt='%0.1f'
    )
    
    def update(val):
        # Get current values from sliders
        current_wavelength = wavelength_slider.val * 1e-9  # Convert nm to m
        current_distance = distance_slider.val * 1e-2      # Convert cm to m
        
        print(f"Updating: wavelength = {current_wavelength*1e9:.1f} nm, distance = {current_distance*100:.1f} cm")
        
        # Optimize phase pattern with fewer iterations for responsiveness
        new_phase = optimize_phase(padded_target, input_beam, current_distance,
                                 current_wavelength, pixel_size, 10)  # Reduced iterations
        
        # Calculate reconstruction
        field = input_beam * np.exp(1j * new_phase)
        reconstructed_field = propagate_asm(field, current_distance, pixel_size,
                                         current_wavelength)
        reconstructed = np.abs(reconstructed_field)**2
        reconstructed = reconstructed[start_idx_y:end_idx_y, start_idx_x:end_idx_x]
        reconstructed = reconstructed / np.max(reconstructed)  # Normalize
        
        # Update plots
        phase_plot.set_array(new_phase[start_idx_y:end_idx_y, start_idx_x:end_idx_x])
        recon_plot.set_array(reconstructed)
        ax3.set_title(f'Reconstructed Image\n(z = {current_distance*100:.1f} cm)')
        
        # Force redraw
        fig.canvas.draw_idle()
        plt.pause(0.01)
    
    # Register the update function with the sliders
    wavelength_slider.on_changed(update)
    distance_slider.on_changed(update)
    
    plt.tight_layout()
    plt.show(block=True)

if __name__ == "__main__":
    main()
