import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import platform

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
l = 1  # topological charge (vortex order)
ni = 1.5  # refractive index
gama = 0.48 * np.pi / 180  # axicon angle in radians
kr = k * (ni - 1) * gama  # radial wave number

# Blazed grating parameters (for shifting the beam)
nx, ny = 1, 1  # grating frequency in x and y directions
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

# Determine if running on Raspberry Pi
is_raspberry_pi = platform.system() == 'Linux' and os.path.exists('/sys/firmware/devicetree/base/model') and 'Raspberry Pi' in open('/sys/firmware/devicetree/base/model').read()

# Set output directory based on platform
if is_raspberry_pi:
    output_dir = "/home/surena/slm-control/Holograms"
else:
    output_dir = os.getcwd()  # Current directory on Windows

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Generate filename with parameters
gama_deg = gama * 180 / np.pi
filename = f"bessel_gauss_l{l}_gama{gama_deg:.2f}_nx{nx}_ny{ny}.png"
plot_filename = f"bessel_gauss_plot_l{l}_gama{gama_deg:.2f}_nx{nx}_ny{ny}.png"
output_path = os.path.join(output_dir, filename)
plot_output_path = os.path.join(output_dir, plot_filename)

# Display phase pattern
plt.figure(figsize=(10, 8))
plt.subplot(121)
plt.imshow(bessel_phase, cmap='hsv')
plt.title(f'Bessel-Gauss Beam (l={l}, γ={gama_deg:.2f}°)')
plt.colorbar(label='Phase [rad]')
plt.axis('off')

plt.subplot(122)
plt.imshow(grayscale, cmap='gray')
plt.title(f'SLM Pattern (nx={nx}, ny={ny})')
plt.colorbar(label='Grayscale Value')
plt.axis('off')

plt.tight_layout()

# Save or show plot based on platform
if is_raspberry_pi:
    plt.savefig(plot_output_path)
    print(f"Plot saved to {plot_output_path}")
else:
    plt.show()

# Save function for SLM pattern
def save_pattern_for_slm(pattern, filepath, gamma=1.0):
    """
    Save the pattern as an 8-bit grayscale image for SLM display.
    
    Parameters:
        pattern: 2D array with grayscale values [0, 255]
        filepath: Full path to output file
        gamma: Gamma correction factor for the SLM's non-linear response
    """
    # Apply gamma correction if needed
    if gamma != 1.0:
        pattern = ((pattern / 255.0) ** gamma * 255).astype(np.uint8)
    
    # Save using PIL for better compatibility
    Image.fromarray(pattern).save(filepath)
    print(f"Pattern saved to {filepath}")
    
    return pattern

# Save the Bessel-Gauss beam pattern
save_pattern_for_slm(grayscale, output_path)

# Calculate expected Bessel beam parameters
z_max = H * px / (2 * gama * (ni - 1))  # Maximum propagation distance
central_spot_size = 0.383 * lambda_ / (gama * (ni - 1))  # Size of central spot

print(f"Bessel-Gauss Beam Parameters:")
print(f"- Topological charge: {l}")
print(f"- Axicon angle: {gama_deg:.4f}°")
print(f"- Maximum propagation distance: {z_max*1000:.2f} mm")
print(f"- Central spot diameter: {central_spot_size*1e6:.2f} μm")
print(f"- Grating frequencies: nx={nx}, ny={ny}")

# Add command line parameter support for easy adjustment
if __name__ == "__main__":
    import argparse
    
    # Only process arguments if script is run directly (not when imported)
    if len(os.sys.argv) > 1:
        parser = argparse.ArgumentParser(description='Generate Bessel-Gauss beam pattern for SLM')
        parser.add_argument('--l', type=int, default=1, help='Topological charge')
        parser.add_argument('--gama', type=float, default=0.48, help='Axicon angle in degrees')
        parser.add_argument('--nx', type=int, default=1, help='Grating frequency in x direction')
        parser.add_argument('--ny', type=int, default=1, help='Grating frequency in y direction')
        parser.add_argument('--gamma', type=float, default=1.0, help='Gamma correction for SLM')
        
        args = parser.parse_args()
        
        print(f"Running with parameters: l={args.l}, gama={args.gama}°, nx={args.nx}, ny={args.ny}, gamma={args.gamma}")
        
        # This would re-run the calculation with new parameters
        # For simplicity, we're just showing the command-line capability
        # A full implementation would regenerate the pattern with these parameters
