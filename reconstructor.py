import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr

def simulate_test_wavefront(shape=(50, 50), type='tilt'):
    """
    Simulates a 2D wavefront.
    type: 'tilt', 'paraboloid', 'sine', 'astigmatism'
    """
    Ny, Nx = shape
    x = np.linspace(-1, 1, Nx)
    y = np.linspace(-1, 1, Ny)
    X, Y = np.meshgrid(x, y)

    if type == 'tilt':
        W = 2 * X  # linear tilt
    elif type == 'paraboloid':
        W = X**2 + Y**2  # spherical defocus
    elif type == 'sine':
        W = 0.5 * np.sin(5 * np.pi * X)  # sinusoidal ripple
    elif type == 'astigmatism':
        W = X**2 - Y**2  # astigmatism mode
    else:
        raise ValueError("Unknown wavefront type.")
    
    return W

def compute_wavefront_slopes(W, dx=1.0, dy=1.0):
    dWdx = np.gradient(W, axis=1) / dx
    dWdy = np.gradient(W, axis=0) / dy
    return dWdx, dWdy

def reconstruct_wavefront(slope_x, slope_y, dx=1.0, dy=1.0):
    Ny, Nx = slope_x.shape
    N = Nx * Ny

    A = lil_matrix((2 * N, N))
    b = np.zeros(2 * N)

    index = lambda x, y: y * Nx + x

    eq = 0
    for y in range(Ny):
        for x in range(Nx):
            i = index(x, y)
            if x < Nx - 1:
                A[eq, i] = -1 / dx
                A[eq, i + 1] = 1 / dx
                b[eq] = slope_x[y, x]
                eq += 1
            if y < Ny - 1:
                A[eq, i] = -1 / dy
                A[eq, i + Nx] = 1 / dy
                b[eq] = slope_y[y, x]
                eq += 1

    # Fix wavefront reference point
    A[eq, 0] = 1
    b[eq] = 0
    eq += 1

    result = lsqr(A.tocsr(), b)[0]
    W_rec = result.reshape((Ny, Nx))
    return W_rec

# Simulate and reconstruct
W_true = simulate_test_wavefront(type='paraboloid')
dWdx, dWdy = compute_wavefront_slopes(W_true)
W_reconstructed = reconstruct_wavefront(dWdx, dWdy)

# Plot results
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
axs[0].imshow(W_true, cmap='viridis')
axs[0].set_title("Original Wavefront")
axs[1].imshow(W_reconstructed, cmap='viridis')
axs[1].set_title("Reconstructed Wavefront")
axs[2].imshow(W_true - W_reconstructed, cmap='bwr')
axs[2].set_title("Reconstruction Error")
plt.tight_layout()
plt.show()
