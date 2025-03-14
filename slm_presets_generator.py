"""
SLM Presets Generator
Generates common beam pattern presets for an 800x600 SLM

This script creates a set of phase patterns for common beam shapes:
- Gaussian beam
- Super Gaussian beam
- Top Hat beam
- Bessel beam
- Laguerre-Gaussian (LG01) beam
- Vortex beam (with different topological charges)
- Axicon (conical lens)
- Blazed grating (for beam steering)
- Multi-spot pattern
"""

import numpy as np
import cv2
import os
import scipy.special
from scipy.special import factorial
import matplotlib.pyplot as plt
from tqdm import tqdm

# Create output directory if it doesn't exist
output_dir = "slm_presets"
os.makedirs(output_dir, exist_ok=True)

# SLM specifications
width = 800
height = 600
active_area = (26.6e-3, 20.0e-3)  # in meters

# Create coordinate grids
x = np.linspace(-width//2, width//2-1, width)
y = np.linspace(-height//2, height//2-1, height)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
Phi = np.arctan2(Y, X)

# Normalize coordinates for easier parameter adjustments
X_norm = X / (width//2)
Y_norm = Y / (height//2)
R_norm = np.sqrt(X_norm**2 + Y_norm**2)

def phase_to_grayscale(phase):
    """
    Convert phase in [0, 2π] range to grayscale [0, 255]
    
    According to the SLM configuration:
    - Grayscale 0 (black) → Phase -π
    - Grayscale 128 (gray) → Phase 0
    - Grayscale 255 (white) → Phase π
    """
    # First wrap to [0, 2π] range
    phase_wrapped = phase % (2 * np.pi)
    
    # Then convert to [-π, π] range
    phase_shifted = phase_wrapped - np.pi
    
    # Normalize to [0, 1] range for grayscale conversion
    normalized = (phase_shifted + np.pi) / (2 * np.pi)
    
    # Convert to grayscale [0, 255]
    return (normalized * 255).astype(np.uint8)

def save_pattern(pattern, filename, show_preview=False):
    """
    Save the pattern as a grayscale image
    """
    full_path = os.path.join(output_dir, filename)
    cv2.imwrite(full_path, pattern)
    print(f"Saved: {full_path}")
    
    if show_preview:
        plt.figure(figsize=(10, 8))
        plt.imshow(pattern, cmap='gray')
        plt.title(filename)
        plt.colorbar(label='Phase (grayscale)')
        plt.show()

def generate_gaussian_beam(width_factor=1.0):
    """
    Generate a Gaussian beam phase pattern
    """
    sigma_x = width / (2.355 * width_factor)  # FWHM = 2.355 * sigma
    sigma_y = height / (2.355 * width_factor)
    
    # Gaussian phase profile
    phase = 2 * np.pi * np.exp(-(X**2 / (2 * sigma_x**2) + Y**2 / (2 * sigma_y**2)))
    return phase_to_grayscale(phase)

def generate_super_gaussian_beam(n=2, width_factor=1.0):
    """
    Generate a Super Gaussian beam phase pattern
    """
    sigma_x = width / (2.355 * width_factor)
    sigma_y = height / (2.355 * width_factor)
    
    # Super Gaussian phase profile
    phase = 2 * np.pi * np.exp(-((X**2 / (2 * sigma_x**2) + Y**2 / (2 * sigma_y**2)))**n)
    return phase_to_grayscale(phase)

def generate_top_hat_beam(radius_factor=0.5):
    """
    Generate a Top Hat beam phase pattern
    
    Creates a simple phase pattern with a uniform phase inside a circular region
    and zero phase outside. This is the direct phase mask to be displayed on the SLM.
    """
    # Calculate radius in pixels
    radius = min(width, height) * radius_factor / 2
    
    # Create a simple phase pattern
    phase = np.zeros((height, width))
    
    # Set phase inside the circle
    mask = R <= radius
    phase[mask] = np.pi  # Set to π inside the circle
    
    # Create a smooth transition at the edges (optional)
    edge_width = 5  # pixels
    edge_mask = (R > radius - edge_width) & (R <= radius)
    edge_factor = 1 - ((R[edge_mask] - (radius - edge_width)) / edge_width)
    phase[edge_mask] = np.pi * edge_factor
    
    return phase_to_grayscale(phase)

def generate_bessel_beam(k_factor=10.0):
    """
    Generate a Bessel beam phase pattern
    
    Creates a direct phase mask with a radial pattern based on Bessel functions.
    This is the phase mask to be displayed on the SLM.
    """
    k_r = k_factor / (width/2)  # Radial wave number
    
    # Bessel phase profile - direct phase pattern for the SLM
    phase = np.mod(k_r * R, 2 * np.pi)
    return phase_to_grayscale(phase)

def generate_lg_beam(l=1, p=0, width_factor=1.0):
    """
    Generate a Laguerre-Gaussian (LG) beam phase pattern
    
    Creates a direct phase mask for LG modes characterized by:
    - azimuthal index l (orbital angular momentum)
    - radial index p (number of radial nodes)
    
    This is the phase mask to be displayed on the SLM.
    """
    # For LG beams, the phase pattern is primarily determined by the azimuthal index l
    # which creates a spiral phase pattern (orbital angular momentum)
    
    # LG phase profile (spiral phase)
    phase = l * Phi
    
    # Convert to SLM phase pattern
    return phase_to_grayscale(phase)

def generate_vortex_beam(m=1):
    """
    Generate a Vortex beam phase pattern
    
    Creates a direct phase mask with a spiral phase pattern.
    The topological charge m determines the number of 2π phase cycles around the center.
    This is the phase mask to be displayed on the SLM.
    """
    # Vortex phase profile - simple spiral phase
    phase = m * Phi
    return phase_to_grayscale(phase)

def generate_axicon(alpha=5.0):
    """
    Generate an Axicon phase pattern
    
    Creates a direct phase mask with a conical phase profile.
    The angle alpha determines the steepness of the cone.
    This is the phase mask to be displayed on the SLM.
    """
    # Axicon phase profile - conical phase
    k_r = alpha / (width/2)
    phase = k_r * R
    phase = phase % (2 * np.pi)
    return phase_to_grayscale(phase)

def generate_blazed_grating(period=10, angle_degrees=0, depth=1.0):
    """
    Generate a Blazed Grating phase pattern
    
    Creates a direct phase mask with a sawtooth pattern.
    The period determines the spacing between peaks.
    The angle determines the orientation of the grating.
    This is the phase mask to be displayed on the SLM.
    """
    # Convert angle to radians
    angle_rad = np.deg2rad(angle_degrees)
    
    # Create the grating pattern
    grating_x = X * np.cos(angle_rad)
    grating_y = Y * np.sin(angle_rad)
    grating = (grating_x + grating_y) / period
    
    # Create sawtooth pattern with specified depth
    phase = depth * 2 * np.pi * (grating % 1.0)
    
    return phase_to_grayscale(phase)

def generate_multi_spot_pattern(num_spots=3, random=False):
    """
    Generate a Multi-Spot phase pattern
    
    Creates a direct phase mask that will produce multiple focused spots.
    This is the phase mask to be displayed on the SLM.
    """
    phase = np.zeros((height, width))
    
    if random:
        # Random spots within a certain radius
        np.random.seed(42)  # For reproducibility
        num_spots = 10
        max_radius = min(width, height) / 3
        
        for _ in range(num_spots):
            # Random position within a circle
            r = max_radius * np.sqrt(np.random.random())
            theta = 2 * np.pi * np.random.random()
            x_pos = int(width/2 + r * np.cos(theta))
            y_pos = int(height/2 + r * np.sin(theta))
            
            # Add a small blazed grating centered at this position
            dx = X - x_pos
            dy = Y - y_pos
            dist = np.sqrt(dx**2 + dy**2)
            spot_radius = 30
            mask = dist < spot_radius
            
            # Add phase to this spot
            local_phase = np.mod(dx / 5, 2 * np.pi)  # Simple linear phase
            phase[mask] = local_phase[mask]
    else:
        # Evenly spaced spots in a circle
        radius = min(width, height) / 4
        for i in range(num_spots):
            angle = 2 * np.pi * i / num_spots
            x_pos = int(width/2 + radius * np.cos(angle))
            y_pos = int(height/2 + radius * np.sin(angle))
            
            # Add a small blazed grating centered at this position
            dx = X - x_pos
            dy = Y - y_pos
            dist = np.sqrt(dx**2 + dy**2)
            spot_radius = 30
            mask = dist < spot_radius
            
            # Add phase to this spot
            local_phase = np.mod(dx / 5, 2 * np.pi)  # Simple linear phase
            phase[mask] = local_phase[mask]
    
    return phase_to_grayscale(phase)

def generate_fresnel_lens(focal_length_factor=1.0):
    """
    Generate a Fresnel Lens phase pattern
    
    Creates a direct phase mask with a quadratic phase profile.
    The focal_length_factor determines the effective focal length.
    This is the phase mask to be displayed on the SLM.
    """
    # Scale factor for the lens
    scale = focal_length_factor * 10.0 / (width**2 + height**2)
    
    # Quadratic phase profile (standard lens)
    phase = scale * R**2
    phase = phase % (2 * np.pi)
    
    return phase_to_grayscale(phase)

def generate_zernike(index=0, amplitude=1.0):
    """
    Generate a Zernike polynomial phase pattern
    
    Creates a direct phase mask based on Zernike polynomials.
    This is the phase mask to be displayed on the SLM.
    """
    # Normalized coordinates
    rho = R_norm
    theta = Phi
    
    # Limit to unit circle
    mask = rho <= 1.0
    phase = np.zeros((height, width))
    
    # Common Zernike polynomials
    if index == 0:  # Piston
        phase[mask] = amplitude * np.ones_like(rho[mask])
    elif index == 1:  # Tilt X
        phase[mask] = amplitude * rho[mask] * np.cos(theta[mask])
    elif index == 2:  # Tilt Y
        phase[mask] = amplitude * rho[mask] * np.sin(theta[mask])
    elif index == 3:  # Defocus
        phase[mask] = amplitude * (2 * rho[mask]**2 - 1)
    elif index == 4:  # Astigmatism X
        phase[mask] = amplitude * rho[mask]**2 * np.cos(2 * theta[mask])
    elif index == 5:  # Astigmatism Y
        phase[mask] = amplitude * rho[mask]**2 * np.sin(2 * theta[mask])
    elif index == 6:  # Coma X
        phase[mask] = amplitude * (3 * rho[mask]**3 - 2 * rho[mask]) * np.cos(theta[mask])
    elif index == 7:  # Coma Y
        phase[mask] = amplitude * (3 * rho[mask]**3 - 2 * rho[mask]) * np.sin(theta[mask])
    elif index == 8:  # Spherical
        phase[mask] = amplitude * (6 * rho[mask]**4 - 6 * rho[mask]**2 + 1)
    
    # Scale to appropriate phase range
    phase = phase * np.pi
    
    return phase_to_grayscale(phase)

def main():
    """
    Generate and save all SLM preset patterns
    """
    print("Generating SLM preset patterns...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Gaussian beam presets
    for w in [0.5, 1.0, 2.0]:
        pattern = generate_gaussian_beam(width_factor=w)
        save_pattern(pattern, f"gaussian_w{w}.png")
    
    # Super-Gaussian beam presets
    for n in [2, 4, 8]:
        pattern = generate_super_gaussian_beam(n=n)
        save_pattern(pattern, f"super_gaussian_n{n}.png")
    
    # Top hat beam presets
    for r in [0.3, 0.5, 0.7]:
        pattern = generate_top_hat_beam(radius_factor=r)
        save_pattern(pattern, f"top_hat_r{r}.png")
    
    # Bessel beam presets
    for k in [5.0, 10.0, 20.0]:
        pattern = generate_bessel_beam(k_factor=k)
        save_pattern(pattern, f"bessel_k{k}.png")
    
    # LG beam presets
    for l in range(4):
        for p in range(2):
            pattern = generate_lg_beam(l=l, p=p)
            save_pattern(pattern, f"lg_l{l}_p{p}.png")
    
    # Vortex beam presets
    for charge in [1, 2, 3, 5, 10]:
        pattern = generate_vortex_beam(m=charge)
        save_pattern(pattern, f"vortex_m{charge}.png")
    
    # Axicon presets
    for angle in [2.0, 5.0, 10.0]:
        pattern = generate_axicon(alpha=angle)
        save_pattern(pattern, f"axicon_a{angle:.1f}.png")
    
    # Blazed grating presets
    for period in [5, 10, 20]:
        for angle in [0, 45, 90]:
            pattern = generate_blazed_grating(period=period, angle_degrees=angle)
            save_pattern(pattern, f"grating_p{period}_a{angle}.png")
    
    # Multi-spot presets
    for spots in [2, 3, 4, 5]:
        pattern = generate_multi_spot_pattern(num_spots=spots, random=False)
        save_pattern(pattern, f"multi_spot_{spots}.png")
    
    # Random multi-spot
    pattern = generate_multi_spot_pattern(num_spots=10, random=True)
    save_pattern(pattern, "multi_spot_random.png")
    
    # Fresnel lens presets
    for f in [0.25, 0.50, 1.00]:
        pattern = generate_fresnel_lens(focal_length_factor=f)
        save_pattern(pattern, f"fresnel_f{f:.2f}.png")
    
    # Zernike aberration presets
    aberrations = [
        (0, "piston"),
        (1, "tilt_x"),
        (2, "tilt_y"),
        (3, "defocus"),
        (4, "astigmatism_x"),
        (5, "astigmatism_y"),
        (6, "coma_x"),
        (7, "coma_y"),
        (8, "spherical")
    ]
    
    for n, name in aberrations:
        pattern = generate_zernike(index=n, amplitude=1.0)
        save_pattern(pattern, f"zernike_{name}.png")
    
    print(f"All patterns saved to {output_dir}/ directory")

if __name__ == "__main__":
    main()
