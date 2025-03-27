import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

"""
Bessel-Gauss Beam Generator for SLM

This script generates a phase pattern for a Bessel-Gauss beam (axicon + vortex)
suitable for display on a Spatial Light Modulator (SLM).

SLM Specifications:
- Resolution: 800 x 600 pixels
- Pixel Pitch: 32 μm
- Default Wavelength: 650 nm (red laser)
"""

# SLM parameters
H, V = 800, 600  # SLM resolution
lambda_ = 650e-9  # wavelength in meters
k = 2 * np.pi / lambda_
px = 32e-6  # pixel size in meters

# Create coordinate grid
x = (np.arange(-H/2, H/2)) * px
y = (np.arange(-V/2, V/2)) * px
X, Y = np.meshgrid(x, y)
rho = np.sqrt(X**2 + Y**2)  # radial distance
phi = np.angle(X + 1j * Y)  # azimuthal angle

# Bessel-Gauss beam parameters
l = 3  # topological charge (vortex order)
ni = 1.5  # refractive index
gama = 0.4 * np.pi / 180  # axicon angle in radians
kr = k * (ni - 1) * gama  # radial wave number

# Blazed grating parameters (for shifting the beam)
nx, ny = 0, 0  # grating frequency in x and y directions
gx = nx / (H * px)  # spatial frequency in x
gy = ny / (V * px)  # spatial frequency in y

# Generate Bessel-Gauss beam phase pattern
# Combines vortex phase (-l*phi), axicon phase (kr*rho), and blazed grating
bessel_phase = np.mod(-l * phi + kr * rho + 2 * np.pi * (X * gx + Y * gy), 2 * np.pi)

# Convert phase from [0, 2π] to [-π, π] range for SLM
bessel_phase_shifted = np.mod(bessel_phase + np.pi, 2 * np.pi) - np.pi

# Convert phase to grayscale for SLM display
# Map [-π, π] to [0, 255] according to:
# -π → 0 (black), 0 → 128 (gray), π → 255 (white)
grayscale = ((bessel_phase_shifted + np.pi) / (2 * np.pi) * 255).astype(np.uint8)

# Create filenames with parameters
plot_filename = f"bessel_plot_l{l}_gama{gama*180/np.pi:.2f}_nx{nx}_ny{ny}.png"
pattern_filename = f"bessel_pattern_l{l}_gama{gama*180/np.pi:.2f}_nx{nx}_ny{ny}.png"

# Display phase pattern
plt.figure(figsize=(10, 8))
plt.subplot(121)
plt.imshow(bessel_phase, cmap='hsv')
plt.title(f'Bessel-Gauss Beam Phase [0, 2π]\nl={l}, γ={gama*180/np.pi:.2f}°, nx={nx}, ny={ny}')
plt.colorbar(label='Phase [rad]')
plt.axis('off')

plt.subplot(122)
plt.imshow(grayscale, cmap='gray')
plt.title('Bessel-Gauss Beam for SLM')
plt.colorbar(label='Grayscale Value')
plt.axis('off')

plt.tight_layout()
# Save the plot instead of displaying it
plt.savefig(plot_filename, dpi=150)
print(f"Plot saved to {plot_filename}")
# Also display it if running interactively
plt.show()

# Save function for SLM pattern
def save_pattern_for_slm(pattern, filename, gamma=1.0):
    """
    Save the pattern as an 8-bit grayscale image for SLM display.
    
    Parameters:
        pattern: 2D array with grayscale values [0, 255]
        filename: Output filename (should end with .png or .bmp)
        gamma: Gamma correction factor for the SLM's non-linear response
    """
    # Apply gamma correction if needed
    if gamma != 1.0:
        pattern = ((pattern / 255.0) ** gamma * 255).astype(np.uint8)
    
    # Save using PIL for better compatibility
    Image.fromarray(pattern).save(filename)
    print(f"Pattern saved to {filename}")
    
    return pattern

# Save the Bessel-Gauss beam pattern
save_pattern_for_slm(grayscale, pattern_filename)

# Calculate expected Bessel beam parameters
z_max = H * px / (2 * gama * (ni - 1))  # Maximum propagation distance
central_spot_size = 0.383 * lambda_ / (gama * (ni - 1))  # Size of central spot

print(f"Bessel-Gauss Beam Parameters:")
print(f"- Topological charge: {l}")
print(f"- Axicon angle: {gama*180/np.pi:.4f}°")
print(f"- Maximum propagation distance: {z_max*1000:.2f} mm")
print(f"- Central spot diameter: {central_spot_size*1e6:.2f} μm")

# Add a function to generate patterns with different parameters
def generate_bessel_pattern(l_value, gama_degrees, nx_value, ny_value, show_plot=False):
    """
    Generate a Bessel-Gauss beam pattern with specified parameters.
    
    Parameters:
        l_value: Topological charge (vortex order)
        gama_degrees: Axicon angle in degrees
        nx_value, ny_value: Grating frequencies in x and y directions
        show_plot: Whether to display the plot (set to False for headless operation)
    
    Returns:
        grayscale: The grayscale pattern for SLM
    """
    # Convert angle to radians
    gama_rad = gama_degrees * np.pi / 180
    
    # Calculate radial wave number
    kr_value = k * (ni - 1) * gama_rad
    
    # Calculate grating spatial frequencies
    gx_value = nx_value / (H * px)
    gy_value = ny_value / (V * px)
    
    # Generate phase pattern
    bessel_phase = np.mod(-l_value * phi + kr_value * rho + 
                          2 * np.pi * (X * gx_value + Y * gy_value), 2 * np.pi)
    
    # Convert to [-π, π] range
    bessel_phase_shifted = np.mod(bessel_phase + np.pi, 2 * np.pi) - np.pi
    
    # Convert to grayscale
    grayscale = ((bessel_phase_shifted + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
    
    # Create filenames with parameters
    pattern_filename = f"bessel_pattern_l{l_value}_gama{gama_degrees:.2f}_nx{nx_value}_ny{ny_value}.png"
    
    # Save the pattern
    save_pattern_for_slm(grayscale, pattern_filename)
    
    if show_plot:
        plt.figure(figsize=(10, 8))
        plt.subplot(121)
        plt.imshow(bessel_phase, cmap='hsv')
        plt.title(f'Bessel-Gauss Beam Phase [0, 2π]\nl={l_value}, γ={gama_degrees:.2f}°, nx={nx_value}, ny={ny_value}')
        plt.colorbar(label='Phase [rad]')
        plt.axis('off')
        
        plt.subplot(122)
        plt.imshow(grayscale, cmap='gray')
        plt.title('Bessel-Gauss Beam for SLM')
        plt.colorbar(label='Grayscale Value')
        plt.axis('off')
        
        plt.tight_layout()
        plot_filename = f"bessel_plot_l{l_value}_gama{gama_degrees:.2f}_nx{nx_value}_ny{ny_value}.png"
        plt.savefig(plot_filename, dpi=150)
        print(f"Plot saved to {plot_filename}")
        if show_plot:
            plt.show()
    
    return grayscale

# Example of generating multiple patterns with different parameters
# Uncomment to generate additional patterns
"""
print("\nGenerating additional patterns...")
# Different topological charges
generate_bessel_pattern(l_value=2, gama_degrees=0.48, nx_value=1, ny_value=1)
generate_bessel_pattern(l_value=3, gama_degrees=0.48, nx_value=1, ny_value=1)

# Different axicon angles
generate_bessel_pattern(l_value=1, gama_degrees=0.3, nx_value=1, ny_value=1)
generate_bessel_pattern(l_value=1, gama_degrees=0.6, nx_value=1, ny_value=1)

# Different grating frequencies
generate_bessel_pattern(l_value=1, gama_degrees=0.48, nx_value=2, ny_value=2)
generate_bessel_pattern(l_value=1, gama_degrees=0.48, nx_value=3, ny_value=0)
"""
