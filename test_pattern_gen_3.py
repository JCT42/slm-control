import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from pattern_gen_3 import PatternGenerator, OriginalPatternGenerator

# Create a more challenging test pattern with a complex shape
size = 128
pattern = np.zeros((size, size))

# Create a more complex pattern - a "SLM" text
# This will be more challenging for the algorithm to reproduce
# Create a blank image
text_img = np.zeros((size, size))

# Add text using matplotlib
fig = Figure(figsize=(1, 1), dpi=size)
canvas = FigureCanvas(fig)
ax = fig.add_subplot(111)
ax.text(0.5, 0.5, "SLM", fontsize=40, ha='center', va='center')
ax.axis('off')
fig.tight_layout(pad=0)
canvas.draw()

# Convert to numpy array
text_img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(size, size, 4)
text_img = text_img[:, :, 0]  # Take just one channel
text_img = text_img.max() - text_img  # Invert

# Normalize
pattern = text_img / text_img.max()

# Add some noise to make it more challenging
np.random.seed(42)
noise = np.random.normal(0, 0.1, pattern.shape)
pattern = np.abs(pattern + noise)
pattern = pattern / pattern.max()

# Create signal region mask (for MRAF) - slightly larger than the text
signal_mask = ndimage.binary_dilation(pattern > 0.3, iterations=3).astype(float)

# Create both generators for comparison
fixed_gen = PatternGenerator(pattern.copy(), signal_region_mask=signal_mask.copy())
original_gen = OriginalPatternGenerator(pattern.copy(), signal_region_mask=signal_mask.copy())

# Create initial field with random phase
np.random.seed(42)  # For reproducibility
random_phase = np.exp(1j * 2 * np.pi * np.random.rand(size, size))
initial_field_fixed = np.sqrt(pattern) * random_phase
initial_field_original = initial_field_fixed.copy()  # Use same initial field for fair comparison

# Optimize using both algorithms and compare
print("Comparing original vs. fixed GS algorithm...")

# Run original algorithm
print("\nRunning original algorithm:")
field_original, metrics_original, stop_reason_original = original_gen.optimize(
    initial_field_original, algorithm='gs', max_iterations=50, tolerance=1e-4
)

# Run fixed algorithm
print("\nRunning fixed algorithm:")
field_fixed, metrics_fixed, stop_reason_fixed = fixed_gen.optimize(
    initial_field_fixed, algorithm='gs', max_iterations=50, tolerance=1e-4
)

# Compare error histories
plt.figure(figsize=(10, 6))
plt.semilogy(metrics_original['error_history'], 'r-', label='Original Algorithm')
plt.semilogy(metrics_fixed['error_history'], 'b-', label='Fixed Algorithm')
plt.xlabel('Iteration')
plt.ylabel('Error (log scale)')
plt.title('Error History Comparison')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('error_comparison.png')

# Compare field change histories
plt.figure(figsize=(10, 6))
plt.semilogy(metrics_original['field_change_history'], 'r-', label='Original Algorithm')
plt.semilogy(metrics_fixed['field_change_history'], 'b-', label='Fixed Algorithm')
plt.xlabel('Iteration')
plt.ylabel('Field Change (log scale)')
plt.title('Field Change History Comparison')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('field_change_comparison.png')

# Compare correlation histories
plt.figure(figsize=(10, 6))
plt.plot(metrics_original['correlation_history'], 'r-', label='Original Algorithm')
plt.plot(metrics_fixed['correlation_history'], 'b-', label='Fixed Algorithm')
plt.xlabel('Iteration')
plt.ylabel('Correlation')
plt.title('Correlation History Comparison')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('correlation_comparison.png')

# Compare reconstructions
plt.figure(figsize=(15, 10))

# Plot target
plt.subplot(2, 3, 1)
plt.imshow(pattern, cmap='viridis')
plt.title('Target Pattern')
plt.axis('off')

# Plot original reconstruction
plt.subplot(2, 3, 2)
plt.imshow(np.abs(field_original)**2, cmap='viridis')
plt.title(f'Original Algorithm\nCorrelation: {metrics_original["correlation_history"][-1]:.4f}')
plt.axis('off')

# Plot fixed reconstruction
plt.subplot(2, 3, 3)
plt.imshow(np.abs(field_fixed)**2, cmap='viridis')
plt.title(f'Fixed Algorithm\nCorrelation: {metrics_fixed["correlation_history"][-1]:.4f}')
plt.axis('off')

# Plot original phase
plt.subplot(2, 3, 5)
plt.imshow(np.angle(original_gen.propagate(field_original)), cmap='twilight')
plt.title('Original Algorithm Phase')
plt.axis('off')

# Plot fixed phase
plt.subplot(2, 3, 6)
plt.imshow(np.angle(fixed_gen.propagate(field_fixed)), cmap='twilight')
plt.title('Fixed Algorithm Phase')
plt.axis('off')

plt.tight_layout()
plt.savefig('reconstruction_comparison.png')

# Print summary
print("\nComparison Summary:")
print(f"Original algorithm stop reason: {stop_reason_original}")
print(f"Fixed algorithm stop reason: {stop_reason_fixed}")
print(f"Original final error: {metrics_original['error_history'][-1]:.6e}")
print(f"Fixed final error: {metrics_fixed['error_history'][-1]:.6e}")
print(f"Original final correlation: {metrics_original['correlation_history'][-1]:.6f}")
print(f"Fixed final correlation: {metrics_fixed['correlation_history'][-1]:.6f}")
print(f"Original iterations: {len(metrics_original['error_history'])}")
print(f"Fixed iterations: {len(metrics_fixed['error_history'])}")

# Now try MRAF algorithm
print("\nComparing original vs. fixed MRAF algorithm...")

# Reset initial fields
np.random.seed(42)
random_phase = np.exp(1j * 2 * np.pi * np.random.rand(size, size))
initial_field_fixed = np.sqrt(pattern) * random_phase
initial_field_original = initial_field_fixed.copy()

# Run original MRAF algorithm
print("\nRunning original MRAF algorithm:")
field_original_mraf, metrics_original_mraf, stop_reason_original_mraf = original_gen.optimize(
    initial_field_original, algorithm='mraf', max_iterations=50, tolerance=1e-4
)

# Run fixed MRAF algorithm
print("\nRunning fixed MRAF algorithm:")
field_fixed_mraf, metrics_fixed_mraf, stop_reason_fixed_mraf = fixed_gen.optimize(
    initial_field_fixed, algorithm='mraf', max_iterations=50, tolerance=1e-4
)

# Compare MRAF reconstructions
plt.figure(figsize=(15, 10))

# Plot target
plt.subplot(2, 3, 1)
plt.imshow(pattern, cmap='viridis')
plt.title('Target Pattern')
plt.axis('off')

# Plot original MRAF reconstruction
plt.subplot(2, 3, 2)
plt.imshow(np.abs(field_original_mraf)**2, cmap='viridis')
plt.title(f'Original MRAF\nCorrelation: {metrics_original_mraf["correlation_history"][-1]:.4f}')
plt.axis('off')

# Plot fixed MRAF reconstruction
plt.subplot(2, 3, 3)
plt.imshow(np.abs(field_fixed_mraf)**2, cmap='viridis')
plt.title(f'Fixed MRAF\nCorrelation: {metrics_fixed_mraf["correlation_history"][-1]:.4f}')
plt.axis('off')

# Plot original MRAF phase
plt.subplot(2, 3, 5)
plt.imshow(np.angle(original_gen.propagate(field_original_mraf)), cmap='twilight')
plt.title('Original MRAF Phase')
plt.axis('off')

# Plot fixed MRAF phase
plt.subplot(2, 3, 6)
plt.imshow(np.angle(fixed_gen.propagate(field_fixed_mraf)), cmap='twilight')
plt.title('Fixed MRAF Phase')
plt.axis('off')

plt.tight_layout()
plt.savefig('mraf_comparison.png')

# Print MRAF summary
print("\nMRAF Comparison Summary:")
print(f"Original MRAF stop reason: {stop_reason_original_mraf}")
print(f"Fixed MRAF stop reason: {stop_reason_fixed_mraf}")
print(f"Original MRAF final error: {metrics_original_mraf['error_history'][-1]:.6e}")
print(f"Fixed MRAF final error: {metrics_fixed_mraf['error_history'][-1]:.6e}")
print(f"Original MRAF final correlation: {metrics_original_mraf['correlation_history'][-1]:.6f}")
print(f"Fixed MRAF final correlation: {metrics_fixed_mraf['correlation_history'][-1]:.6f}")
print(f"Original MRAF iterations: {len(metrics_original_mraf['error_history'])}")
print(f"Fixed MRAF iterations: {len(metrics_fixed_mraf['error_history'])}")

print("\nDone! Comparison plots saved to disk.")
