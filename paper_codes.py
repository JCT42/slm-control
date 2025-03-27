import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial, eval_hermite
import scipy.io as sio

# Constants
H, V = 800, 600
lambda_ = 650e-6  # in mm
k = 2 * np.pi / lambda_
px = 32e-3  # pixel size in mm
x = (np.arange(-H/2, H/2)) * px
y = (np.arange(-V/2, V/2)) * px
X, Y = np.meshgrid(x, y)
rho = np.sqrt(X**2 + Y**2)
phi = np.angle(X + 1j * Y)

# ========== Bessel-Gauss (Axicon + Vortex) ==========
ni = 1.5
l = 1
gama = 0.48 * np.pi / 180
kr = k * (ni - 1) * gama
nx = 1
ny = 1
gx = nx / (H * px)
gy = ny / (V * px)
Hol_bessel = np.mod(-l * phi + kr * rho + 2 * np.pi * (X * gx + Y * gy), 2 * np.pi)
SLM_bessel = Hol_bessel / np.max(Hol_bessel) * 255

# ========== Airy Beam ==========
A_airy = (X**3 + Y**3) / 3
Hol_airy = np.mod(A_airy, 2 * np.pi)
SLM_airy = Hol_airy / np.max(Hol_airy) * 255

# ========== Hermite-Gaussian Beam ==========
w0 = 1
z = 0.00001
zr = np.pi * w0**2 / lambda_
w = w0 * np.sqrt(1 + (z/zr)**2)
R = z * (1 + (zr/z)**2)
m, n = 2, 0
Hx = eval_hermite(m, np.sqrt(2) * X / w)
Hy = eval_hermite(n, np.sqrt(2) * Y / w)
rc = np.sqrt(2**(1-n-m)/(np.pi*factorial(n)*factorial(m))) / w
HG = rc * Hx * Hy * np.exp(1j * (n+m+1) * np.arctan(z/zr)) * \
     np.exp(-rho**2 / w**2) * np.exp(-1j * k * rho**2 / (2*R)) * np.exp(1j * k * z)
HG = HG / np.sqrt(np.sum(np.abs(HG)**2))
ph = np.angle(HG)
A = np.abs(HG)
A_norm = A / np.max(A)

# Load fx2.mat (you must provide this file)
try:
    mat = sio.loadmat('fx2.mat')
    fx = mat['fx'].flatten()
    aux = np.round(A_norm * 800).astype(int)
    aux[aux >= len(fx)] = len(fx) - 1
    F = fx[aux]
except:
    F = A_norm  # fallback if fx2.mat not available

nx, ny = 50, 0
gx = nx / (H * px)
gy = ny / (V * px)
Hol_HG = F * np.sin(ph + 2 * np.pi * (X * gx + Y * gy))
Hol_HG -= np.min(Hol_HG)
SLM_HG = Hol_HG / np.max(Hol_HG) * 255

# ========== Digital Propagation ==========
kz = 2 * np.pi * np.sqrt(1 / lambda_**2 - rho**2)
prop_frames = []
for zp in range(1, 6):  # Only showing first 5 frames for speed
    Hol = np.mod(kz * zp + 2 * np.pi * (X * gx + Y * gy), 2 * np.pi)
    Hol -= np.min(Hol)
    SLM = Hol / np.max(Hol) * 255
    prop_frames.append(SLM)

# ========== Digital Focusing ==========
ff = 400  # focal distance in mm
T = np.pi / lambda_ / ff * rho**2
Hol_focus = np.mod(T + 2 * np.pi * (X * gx + Y * gy), 2 * np.pi)
Hol_focus -= np.min(Hol_focus)
SLM_focus = Hol_focus / np.max(Hol_focus) * 255

# ========== Turbulence Simulation ==========
def generate_turbulence(H, V, SR, w0, pixel):
    # Placeholder turbulence generator (real one would require Kolmogorov phase screen)
    phase = np.random.randn(V, H)
    phase = phase / np.max(np.abs(phase)) * 2 * np.pi * SR
    return phase

SR = 0.5
w0 = 1
pixel = 8
phi = np.angle(X + 1j * Y)
n = 1
nx, ny = 50, 50
gx = nx / H
gy = ny / V
turb = generate_turbulence(H, V, SR, w0, pixel)
Hol_turb = np.mod(turb + n * phi + 2 * np.pi * (Y * gy + X * gx), 2 * np.pi)
SLM_turb = Hol_turb / np.max(Hol_turb) * 255

# ========== Display One Example From Each ==========
def show(title, data):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.imshow(data, cmap='gray')
    plt.axis('off')
    plt.show()

#def save_pattern_for_slm(pattern, filename, gamma=1.0, phase_mode=True):
    """
    Save a pattern for display on the SLM.
    
    Parameters:
        pattern: 2D array with pattern data
        filename: Output filename (should end with .png or .bmp)
        gamma: Gamma correction factor for the SLM's non-linear response
        phase_mode: If True, treats pattern as phase data [-π, π] or [0, 2π]
                    If False, treats pattern as already normalized [0, 255]
    """
    if phase_mode:
        # Check if phase is in [0, 2π] range and convert to [-π, π] if needed
        if np.max(pattern) > np.pi and np.min(pattern) >= 0:
            # Convert from [0, 2π] to [-π, π] range
            pattern = np.mod(pattern + np.pi, 2 * np.pi) - np.pi
            
        # Convert phase [-π, π] to grayscale [0, 255] according to the mapping:
        # -π -> 0, 0 -> 128, π -> 255
        normalized_pattern = (pattern + np.pi) / (2 * np.pi)
        output = (normalized_pattern ** gamma * 255).astype(np.uint8)
    else:
        # Pattern is already in [0, 255] range, just apply gamma
        output = ((pattern / 255.0) ** gamma * 255).astype(np.uint8)
    
    # Save the image
    plt.imsave(filename, output, cmap='gray')
    print(f"Pattern saved to {filename}")
    
    return output

show("Bessel-Gauss Beam", SLM_bessel)
show("Airy Beam", SLM_airy)
show("Hermite-Gaussian Beam", SLM_HG)
show("Digital Focus", SLM_focus)
show("Turbulence Phase Screen", SLM_turb)

# Save patterns for SLM display - uncomment to save
save_pattern_for_slm(SLM_bessel, "bessel_gauss.png", phase_mode=False)
save_pattern_for_slm(SLM_airy, "airy_beam.png", phase_mode=False)
save_pattern_for_slm(SLM_HG, "hermite_gaussian.png", phase_mode=False)
save_pattern_for_slm(SLM_focus, "digital_focus.png", phase_mode=False)
save_pattern_for_slm(SLM_turb, "turbulence.png", phase_mode=False)

# Optionally show propagation frames
#for i, frame in enumerate(prop_frames):
#    show(f"Propagation Frame {i+1}", frame)

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

# Save phase-only vortex beam
save_pattern_for_slm(Hol, "phase_only_vortex.png", phase_mode=True)
