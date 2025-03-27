import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# SLM specifications from pattern_gen_3.py
width = 800
height = 600
pixel_pitch = 32e-6  # 32 um
wavelength = 650e-9  # 650 nm (red laser)

# Camera parameters (IMX296 monochrome camera)
camera_width = 1456
camera_height = 1088
camera_pixel_size = 3.45e-6  # 3.45 um

# Optical setup parameters
f = 100e-3  # 100 mm focal length lens
# Calculate maximum deflection angle to keep beam on camera sensor
camera_half_width = (camera_width * camera_pixel_size) / 2
max_angle = np.arctan(camera_half_width / f)
theta_max = max_angle * 0.8  # Use 80% of max to ensure beam stays on sensor

# Create coordinate grids for SLM
x = np.linspace(-width/2, width/2, width) * pixel_pitch
y = np.linspace(-height/2, height/2, height) * pixel_pitch
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)

# Generate axicon phase pattern
k = 2 * np.pi / wavelength
alpha = theta_max  # Cone angle of the axicon
phi = k * alpha * R
phase_mask = np.mod(phi, 2*np.pi)

# Convert phase [0, 2π] to [-π, π] range for SLM
phase_shifted = np.mod(phase_mask + np.pi, 2 * np.pi) - np.pi

# Convert phase to grayscale for SLM display
# Map [-π, π] to [0, 255] according to:
# -π → 0 (black), 0 → 128 (gray), π → 255 (white)
grayscale = ((phase_shifted + np.pi) / (2 * np.pi) * 255).astype(np.uint8)

# Display phase mask
plt.figure(figsize=(10, 8))
plt.subplot(121)
plt.imshow(phase_mask, cmap='hsv', extent=[-width*pixel_pitch*1000/2, width*pixel_pitch*1000/2, 
                                          -height*pixel_pitch*1000/2, height*pixel_pitch*1000/2])
plt.title('Axicon Phase Mask [0, 2π]')
plt.xlabel('mm')
plt.ylabel('mm')
plt.colorbar(label='Phase [rad]')

# Display grayscale pattern for SLM
plt.subplot(122)
plt.imshow(grayscale, cmap='gray')
plt.title('Axicon Pattern for SLM')
plt.xlabel('Pixels')
plt.ylabel('Pixels')
plt.colorbar(label='Grayscale Value')
plt.tight_layout()
plt.show()

# Save the pattern for SLM
# def save_pattern_for_slm(pattern, filename, gamma=1.0):
#     """Save the pattern as an 8-bit grayscale image for SLM display."""
#     # Apply gamma correction if needed
#     if gamma != 1.0:
#         pattern = ((pattern / 255.0) ** gamma * 255).astype(np.uint8)
    
#     # Save using PIL for better compatibility
#     Image.fromarray(pattern).save(filename)
#     print(f"Pattern saved to {filename}")

# Save the axicon pattern
# save_pattern_for_slm(grayscale, "axicon_pattern.png")

# Calculate expected Bessel beam parameters
z_max = width * pixel_pitch / (2 * alpha)  # Maximum propagation distance
central_spot_size = 0.383 * wavelength / alpha  # Size of central spot

print(f"Axicon Parameters:")
print(f"- Cone angle: {alpha:.6f} rad ({alpha*180/np.pi:.4f}°)")
print(f"- Maximum propagation distance: {z_max*1000:.2f} mm")
print(f"- Central spot diameter: {central_spot_size*1e6:.2f} um")
print(f"- Expected on-camera spot diameter: {central_spot_size/camera_pixel_size:.2f} pixels")
