import numpy as np
import matplotlib.pyplot as plt

def generate_lenslet_array_phase_mask(width=800, height=600, lenslet_pitch=50, wavelength=650e-9, focal_length=0.05):
    """
    Generate a 2D phase mask for a Shack-Hartmann lenslet array simulated on an SLM.

    Parameters:
        width, height: resolution of the SLM in pixels
        lenslet_pitch: size of one lenslet (in pixels)
        wavelength: laser wavelength in meters (e.g., 650e-9 for HeNe)
        focal_length: focal length of each lenslet in meters

    Returns:
        phase_mask: 2D array (height x width) with wrapped phase values [0, 2π]
    """
    k = 2 * np.pi / wavelength  # wave number

    # Create full coordinate grid
    x = np.arange(width) - width // 2
    y = np.arange(height) - height // 2
    X, Y = np.meshgrid(x, y)

    # Local coordinates within each lenslet
    X_tile = X % lenslet_pitch - lenslet_pitch // 2
    Y_tile = Y % lenslet_pitch - lenslet_pitch // 2

    # Quadratic phase pattern (Fresnel lens) for each tile
    r2 = X_tile**2 + Y_tile**2
    phase = (k / (2 * focal_length)) * r2

    # Wrap phase to [0, 2π]
    phase_wrapped = np.mod(phase, 2 * np.pi)

    return phase_wrapped

def save_phase_mask_for_slm(phase_mask, filename, gamma=1.0):
    """
    Save the phase mask as an 8-bit grayscale image for SLM display.
    
    Parameters:
        phase_mask: 2D array with phase values [0, 2π]
        filename: Output filename (should end with .png or .bmp)
        gamma: Gamma correction factor for the SLM's non-linear response
        
    Returns:
        grayscale_image: The converted 8-bit grayscale image
    """
    # Convert phase from [0, 2π] to [-π, π] range as required by the SLM
    phase_shifted = np.mod(phase_mask + np.pi, 2 * np.pi) - np.pi
    
    # Convert phase [-π, π] to grayscale [0, 255] according to the mapping:
    # -π -> 0, 0 -> 128, π -> 255
    normalized_phase = (phase_shifted + np.pi) / (2 * np.pi)
    grayscale_image = (normalized_phase ** gamma * 255).astype(np.uint8)
    
    # Save the image
    plt.imsave(filename, grayscale_image, cmap='gray')
    print(f"Phase mask saved to {filename}")
    
    return grayscale_image

# Example usage
phase_mask = generate_lenslet_array_phase_mask()

# Display the phase mask
plt.figure(figsize=(10, 8))
plt.imshow(phase_mask, cmap='twilight', extent=[0, 800, 0, 600])
plt.colorbar(label='Phase (radians)')
plt.title("SLM Lenslet Array Phase Mask")
plt.xlabel("Pixels")
plt.ylabel("Pixels")
plt.show()

# Save the phase mask for SLM display
# Uncomment to save:
grayscale_image = save_phase_mask_for_slm(phase_mask, "lenslet_array.png")

plt.figure(figsize=(10, 8))
plt.imshow(grayscale_image, cmap='gray', extent=[0, 800, 0, 600])
plt.colorbar(label='Grayscale Value')
plt.title("SLM Lenslet Array (Grayscale for SLM)")
plt.xlabel("Pixels")
plt.ylabel("Pixels")
plt.show()
