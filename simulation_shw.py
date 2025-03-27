import numpy as np
import matplotlib.pyplot as plt

# Parameters for the simulated lenslet array
H, V = 800, 600  # Resolution
pitch = 40       # pixels per lenslet
f_lenslet = 0.1  # Focal length in meters (100 mm)
wavelength = 633e-9  # Wavelength in meters

# Coordinate grid
x = np.arange(-H//2, H//2)
y = np.arange(-V//2, V//2)
X, Y = np.meshgrid(x, y)

# Create lenslet phase pattern
# Each lenslet is a small parabolic lens: φ(x, y) = -(π/λf)(x² + y²)
phase = np.zeros_like(X, dtype=float)
for i in range(0, H, pitch):
    for j in range(0, V, pitch):
        xi = X[j:j+pitch, i:i+pitch]
        yi = Y[j:j+pitch, i:i+pitch]
        if xi.shape[0] == pitch and xi.shape[1] == pitch:
            r2 = (xi - xi.mean())**2 + (yi - yi.mean())**2
            phase[j:j+pitch, i:i+pitch] = -np.pi / (wavelength * f_lenslet) * r2

# Simulate the far-field (Fourier transform) pattern
field = np.exp(1j * phase)
fft_image = np.fft.fftshift(np.fft.fft2(field))
intensity = np.abs(fft_image) ** 2
intensity /= np.max(intensity)

# Show result
plt.figure(figsize=(6, 5))
plt.imshow(intensity, cmap='inferno', extent=[-1, 1, -1, 1])
plt.title("Simulated CCD Image from Lenslet Array")
plt.xlabel("x [a.u.]")
plt.ylabel("y [a.u.]")
plt.colorbar(label="Normalized Intensity")
plt.tight_layout()
plt.show()
