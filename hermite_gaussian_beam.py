import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial, eval_hermite
from PIL import Image

"""
Hermite-Gaussian Beam Generator for SLM

This script generates a phase pattern for a Hermite-Gaussian beam
suitable for display on a Spatial Light Modulator (SLM).

SLM Specifications:
- Resolution: 800 x 600 pixels
- Pixel Pitch: 32 μm
- Default Wavelength: 650 nm (red laser)
"""

# SLM parameters
H, V = 800, 600  # SLM resolution
lambda_ = 650e-9  # wavelength in meters (converted from mm)
k = 2 * np.pi / lambda_
px = 32e-6  # pixel size in meters (converted from mm)

# Create coordinate grid
x = (np.arange(-H/2, H/2)) * px
y = (np.arange(-V/2, V/2)) * px
X, Y = np.meshgrid(x, y)
rho = np.sqrt(X**2 + Y**2)
phi = np.angle(X + 1j * Y)

# Hermite-Gaussian beam parameters
m, n = 3, 3  # Hermite polynomial orders (m: horizontal, n: vertical)
w0 = 100e-6  # beam waist (in meters)
z = 1e-6  # propagation distance (near waist)
zr = np.pi * w0**2 / lambda_  # Rayleigh range
w = w0 * np.sqrt(1 + (z/zr)**2)  # beam width at distance z
R = z * (1 + (zr/z)**2)  # radius of curvature at distance z

# Blazed grating parameters (for shifting the beam)
nx, ny = 5, 5  # grating frequency in x and y directions
gx = nx / (H * px)  # spatial frequency in x
gy = ny / (V * px)  # spatial frequency in y

# Generate Hermite-Gaussian beam
# Calculate Hermite polynomials
Hx = eval_hermite(m, np.sqrt(2) * X / w)
Hy = eval_hermite(n, np.sqrt(2) * Y / w)

# Normalization constant
rc = np.sqrt(2**(1-n-m)/(np.pi*factorial(n)*factorial(m))) / w

# Full Hermite-Gaussian field
HG = rc * Hx * Hy * np.exp(1j * (n+m+1) * np.arctan(z/zr)) * \
     np.exp(-rho**2 / w**2) * np.exp(-1j * k * rho**2 / (2*R)) * np.exp(1j * k * z)

# Normalize the field
HG = HG / np.sqrt(np.sum(np.abs(HG)**2))

# Extract amplitude and phase
phase = np.angle(HG)
amplitude = np.abs(HG)
amplitude_norm = amplitude / np.max(amplitude)

# Create phase pattern with blazed grating
hg_phase = np.mod(phase + 2 * np.pi * (X * gx + Y * gy), 2 * np.pi)

# Convert phase from [0, 2π] to [-π, π] range for SLM
hg_phase_shifted = np.mod(hg_phase + np.pi, 2 * np.pi) - np.pi

# Convert phase to grayscale for SLM display
# Map [-π, π] to [0, 255] according to:
# -π → 0 (black), 0 → 128 (gray), π → 255 (white)
grayscale = ((hg_phase_shifted + np.pi) / (2 * np.pi) * 255).astype(np.uint8)

# Create filenames with parameters
plot_filename = f"hermite_gaussian_plot_m{m}_n{n}_nx{nx}_ny{ny}.png"
pattern_filename = f"hermite_gaussian_pattern_m{m}_n{n}_nx{nx}_ny{ny}.png"

# Display phase pattern and amplitude
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.imshow(amplitude_norm, cmap='viridis')
plt.title(f'Hermite-Gaussian Amplitude\nMode (m,n) = ({m},{n})')
plt.colorbar(label='Normalized Amplitude')
plt.axis('off')

plt.subplot(132)
plt.imshow(hg_phase, cmap='hsv')
plt.title(f'Hermite-Gaussian Phase [0, 2π]\nMode (m,n) = ({m},{n}), nx={nx}, ny={ny}')
plt.colorbar(label='Phase [rad]')
plt.axis('off')

plt.subplot(133)
plt.imshow(grayscale, cmap='gray')
plt.title('Pattern for SLM')
plt.colorbar(label='Grayscale Value')
plt.axis('off')

plt.tight_layout()
# Save the plot
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

# Save the Hermite-Gaussian beam pattern
save_pattern_for_slm(grayscale, pattern_filename)

# Calculate expected beam parameters
print(f"Hermite-Gaussian Beam Parameters:")
print(f"- Mode indices: (m,n) = ({m},{n})")
print(f"- Beam waist: {w0*1e6:.2f} μm")
print(f"- Rayleigh range: {zr*1000:.2f} mm")
print(f"- Beam width at z={z*1000:.2f} mm: {w*1e6:.2f} μm")

# Add a function to generate patterns with different parameters
def generate_hg_pattern(m_value, n_value, nx_value, ny_value, w0_value=100e-6, show_plot=False):
    """
    Generate a Hermite-Gaussian beam pattern with specified parameters.
    
    Parameters:
        m_value, n_value: Hermite polynomial orders (horizontal, vertical)
        nx_value, ny_value: Grating frequencies in x and y directions
        w0_value: Beam waist in meters
        show_plot: Whether to display the plot (set to False for headless operation)
    
    Returns:
        grayscale: The grayscale pattern for SLM
    """
    # Recalculate beam parameters
    z_value = 1e-6  # near waist
    zr_value = np.pi * w0_value**2 / lambda_
    w_value = w0_value * np.sqrt(1 + (z_value/zr_value)**2)
    R_value = z_value * (1 + (zr_value/z_value)**2)
    
    # Calculate grating spatial frequencies
    gx_value = nx_value / (H * px)
    gy_value = ny_value / (V * px)
    
    # Calculate Hermite polynomials
    Hx = eval_hermite(m_value, np.sqrt(2) * X / w_value)
    Hy = eval_hermite(n_value, np.sqrt(2) * Y / w_value)
    
    # Normalization constant
    rc = np.sqrt(2**(1-n_value-m_value)/(np.pi*factorial(n_value)*factorial(m_value))) / w_value
    
    # Full Hermite-Gaussian field
    HG = rc * Hx * Hy * np.exp(1j * (n_value+m_value+1) * np.arctan(z_value/zr_value)) * \
         np.exp(-rho**2 / w_value**2) * np.exp(-1j * k * rho**2 / (2*R_value)) * np.exp(1j * k * z_value)
    
    # Normalize the field
    HG = HG / np.sqrt(np.sum(np.abs(HG)**2))
    
    # Extract amplitude and phase
    phase = np.angle(HG)
    
    # Create phase pattern with blazed grating
    hg_phase = np.mod(phase + 2 * np.pi * (X * gx_value + Y * gy_value), 2 * np.pi)
    
    # Convert phase from [0, 2π] to [-π, π] range for SLM
    hg_phase_shifted = np.mod(hg_phase + np.pi, 2 * np.pi) - np.pi
    
    # Convert phase to grayscale for SLM display
    grayscale = ((hg_phase_shifted + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
    
    # Create filenames with parameters
    pattern_filename = f"hermite_gaussian_pattern_m{m_value}_n{n_value}_nx{nx_value}_ny{ny_value}.png"
    
    # Save the pattern
    save_pattern_for_slm(grayscale, pattern_filename)
    
    if show_plot:
        amplitude = np.abs(HG)
        amplitude_norm = amplitude / np.max(amplitude)
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        plt.imshow(amplitude_norm, cmap='viridis')
        plt.title(f'Hermite-Gaussian Amplitude\nMode (m,n) = ({m_value},{n_value})')
        plt.colorbar(label='Normalized Amplitude')
        plt.axis('off')
        
        plt.subplot(132)
        plt.imshow(hg_phase, cmap='hsv')
        plt.title(f'Hermite-Gaussian Phase [0, 2π]\nMode (m,n) = ({m_value},{n_value}), nx={nx_value}, ny={ny_value}')
        plt.colorbar(label='Phase [rad]')
        plt.axis('off')
        
        plt.subplot(133)
        plt.imshow(grayscale, cmap='gray')
        plt.title('Pattern for SLM')
        plt.colorbar(label='Grayscale Value')
        plt.axis('off')
        
        plt.tight_layout()
        plot_filename = f"hermite_gaussian_plot_m{m_value}_n{n_value}_nx{nx_value}_ny{ny_value}.png"
        plt.savefig(plot_filename, dpi=150)
        print(f"Plot saved to {plot_filename}")
        if show_plot:
            plt.show()
    
    return grayscale

# Example of generating multiple patterns with different parameters
# Uncomment to generate additional patterns
"""
print("\nGenerating additional patterns...")
# Different mode combinations
generate_hg_pattern(m_value=0, n_value=1, nx_value=5, ny_value=0)
generate_hg_pattern(m_value=1, n_value=1, nx_value=5, ny_value=0)
generate_hg_pattern(m_value=3, n_value=0, nx_value=5, ny_value=0)

# Different grating frequencies
generate_hg_pattern(m_value=2, n_value=0, nx_value=10, ny_value=0)
generate_hg_pattern(m_value=2, n_value=0, nx_value=0, ny_value=10)
"""
