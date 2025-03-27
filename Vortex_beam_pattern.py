# ========== Phase-only Vortex Beam ==========
import numpy as np
import matplotlib.pyplot as plt

# Constants
H, V = 800  , 600  # SLM resolution
x = np.arange(-H/2, H/2)
y = np.arange(-V/2, V/2)
X, Y = np.meshgrid(x, y)

# Phase-only vortex beam
phi = np.angle(X + 1j * Y)  # Azimuthal angle
n = 1  # Topological charge (matching MATLAB line)
nx, ny = 10, 10  # Number of grooves
gx = nx / H
gy = ny / V

# Phase-only hologram calculation
Hol = np.mod(-n * phi + 2 * np.pi * (gy * Y + gx * X), 2 * np.pi)
SLM = Hol / np.max(Hol) * 255  # Normalize to grayscale

# Display
plt.figure(figsize=(10, 6))
plt.imshow(SLM, cmap='gray')
plt.axis('off')
plt.title("Phase-only Vortex Beam Hologram")
plt.show()