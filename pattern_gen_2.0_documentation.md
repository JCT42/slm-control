# Pattern Generator 2.0 Documentation

## Overview

`pattern_gen_2.0.py` is an advanced software tool for generating phase patterns for Spatial Light Modulators (SLMs), specifically designed for the Sony LCX016AL-6 SLM. The software implements multiple algorithms for phase pattern generation, including the Gerchberg-Saxton (GS) algorithm and Mixed-Region Amplitude Freedom (MRAF) algorithm, with a focus on creating high-quality phase-only holograms.

## SLM Specifications

- **Device**: Sony LCX016AL-6 SLM
- **Resolution**: 800 × 600 pixels
- **Pixel Pitch**: 32 μm
- **Refresh Rate**: 60 Hz
- **Contrast Ratio**: 200:1
- **Default Wavelength**: 650 nm (red laser)
- **Phase Range**: 0 to 2π (internally represented as -π to π)

## Key Features

1. **Multiple Modulation Modes**:
   - Phase-only modulation
   - Amplitude-only modulation
   - Combined amplitude and phase modulation

2. **Advanced Pattern Generation Algorithms**:
   - Gerchberg-Saxton (GS) algorithm
   - Mixed-Region Amplitude Freedom (MRAF) algorithm

3. **Input Beam Profile Options**:
   - Gaussian
   - Super Gaussian
   - Top Hat
   - Bessel
   - Laguerre-Gaussian (LG01)
   - Custom (user-provided)

4. **Zero-Order Diffraction Control**:
   - Linear phase ramp implementation
   - Adjustable X and Y shift parameters
   - Real-time preview of shifted patterns

5. **Optimization Metrics**:
   - Normalized Mean Square Error (NMSE) calculation
   - Field change convergence monitoring
   - Error history plotting

6. **Simulation and Visualization**:
   - FFT-based diffraction simulation
   - Logarithmic scaling for better visualization
   - Real-time preview of target, pattern, and reconstruction

7. **Hardware Integration**:
   - Raspberry Pi Camera support
   - Direct SLM control via HDMI output
   - Exposure and gain controls

## Software Architecture

The software is structured around two main classes:

1. **AdvancedPatternGenerator**: The main application class that handles:
   - GUI creation and management
   - Camera integration
   - Pattern generation workflow
   - SLM display
   - File operations (load/save)

2. **PatternGenerator**: The core algorithm implementation class that handles:
   - Optical propagation simulation
   - GS and MRAF algorithm iterations
   - Error calculation
   - Convergence monitoring

## Diffraction Calculation Implementation

The software implements a direct FFT-based diffraction calculation for simulating the optical reconstruction:

1. **Forward Propagation** (SLM to Image Plane):
   ```python
   def inverse_propagate(self, field):
       """Propagate field from SLM plane to image plane"""
       # Use proper FFT shifting for optical propagation
       result = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(field)))
       
       # Apply window on the way back to reduce ringing artifacts
       return result * np.sqrt(self.window)
   ```

2. **Backward Propagation** (Image to SLM Plane):
   ```python
   def propagate(self, field):
       """Propagate field from image plane to SLM plane"""
       # Apply window function to reduce edge artifacts
       windowed_field = field * self.window
       
       # Use proper FFT shifting for optical propagation
       fft_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(windowed_field)))
       
       # Apply DC, high-frequency, and cross filters in Fourier space
       filtered_fft = fft_field * self.dc_filter * self.hf_filter * self.cross_filter
       
       return filtered_fft
   ```

3. **Reconstruction Calculation**:
   ```python
   # Create complex field with phase only
   slm_field = amplitude * np.exp(1j * padded_phase)
   
   # Simulate propagation to far field
   far_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(slm_field)))
   self.reconstruction = np.abs(far_field)**2
   
   # Apply logarithmic scaling for better visualization
   self.reconstruction = np.log1p(self.reconstruction * 10) / np.log1p(10)
   ```

## Phase Shift Implementation

The software implements a linear phase ramp to shift the image away from zero-order diffraction:

```python
# Create coordinate grids for the SLM plane
y, x = np.indices((self.height, self.width))

# Normalize coordinates to [-0.5, 0.5] range
x_norm = (x - self.width // 2) / self.width
y_norm = (y - self.height // 2) / self.height

# Calculate linear phase ramp
phase_ramp = 2 * np.pi * (self.phase_shift_x * x_norm + self.phase_shift_y * y_norm)

# Add phase ramp to SLM phase
self.slm_phase = np.mod(self.slm_phase + phase_ramp, 2 * np.pi) - np.pi
```

## Error Calculation

The software implements a Normalized Mean Square Error (NMSE) calculation:

```python
def calculate_error(self, field, algorithm):
    """
    Calculate Normalized Mean Square Error (NMSE) between reconstructed and target intensity.
    This provides a more meaningful error metric than absolute difference.
    """
    recon_intensity = np.abs(field)**2
    
    if algorithm.lower() == 'gs':
        # For GS, calculate error over entire field
        mse = np.mean((recon_intensity - self.target_intensity)**2)
        # Normalize by mean squared target intensity (NMSE)
        norm_error = mse / np.mean(self.target_intensity**2)
    else:
        # For MRAF, calculate error only in signal region
        sr_mask = self.signal_region_mask
        if np.sum(sr_mask) > 0:  # Ensure signal region is not empty
            mse = np.mean((recon_intensity[sr_mask == 1] - self.target_intensity[sr_mask == 1])**2)
            # Normalize by mean squared target intensity in signal region (NMSE)
            norm_error = mse / np.mean(self.target_intensity[sr_mask == 1]**2)
        else:
            norm_error = 0.0
            
    return norm_error
```

## SLM Phase Representation

The SLM phase is represented in the range [-π to π] in the internal calculations and converted to grayscale [0-255] for display:

- Grayscale 0 (black) → Phase -π
- Grayscale 128 (gray) → Phase 0
- Grayscale 255 (white) → Phase π

The conversion is done using:

```python
normalized_phase = (self.slm_phase + np.pi) / (2 * np.pi)
self.pattern = (normalized_phase ** gamma * 255).astype(np.uint8)
```

Where gamma is a correction factor to compensate for the non-linear response of the SLM hardware.

## Usage Instructions

1. **Loading a Target Image**:
   - Click "Load Target Image" to select an image file
   - The image will be automatically resized to 800×600 pixels

2. **Generating a Pattern**:
   - Select modulation mode (Phase, Amplitude, Combined)
   - Choose algorithm (GS or MRAF)
   - Set number of iterations and tolerance
   - Click "Generate Pattern"

3. **Applying Phase Shift**:
   - Adjust X and Y shift values using sliders or input fields
   - Click "Apply Shift" to update the pattern

4. **Sending to SLM**:
   - Click "Send to SLM" to display the pattern on the connected SLM
   - The pattern will be displayed on HDMI-A-2

5. **Saving Patterns**:
   - Click "Save Pattern" to save the generated pattern as an image file

## Dependencies

- NumPy: For numerical operations
- OpenCV (cv2): For image processing
- Matplotlib: For visualization
- Tkinter: For GUI
- Pygame: For SLM display
- SciPy: For special functions (Bessel functions)
- PiCamera2: For Raspberry Pi camera integration

## Sources and References

The implementation in this code is based on the following scientific sources and references:

1. **Gerchberg-Saxton Algorithm**:
   - Gerchberg, R. W., & Saxton, W. O. (1972). A practical algorithm for the determination of phase from image and diffraction plane pictures. *Optik*, 35, 237-246.
   - Wyrowski, F., & Bryngdahl, O. (1988). Iterative Fourier-transform algorithm applied to computer holography. *Journal of the Optical Society of America A*, 5(7), 1058-1065.

2. **Mixed-Region Amplitude Freedom (MRAF) Algorithm**:
   - Pasienski, M., & DeMarco, B. (2008). A high-accuracy algorithm for designing arbitrary holographic atom traps. *Optics Express*, 16(3), 2176-2190.
   - Harte, T., Bruce, G. D., Keeling, J., & Cassettari, D. (2014). Conjugate gradient minimisation approach to generating holographic traps for ultracold atoms. *Optics Express*, 22(22), 26548-26558.

3. **FFT-Based Diffraction Calculation**:
   - Goodman, J. W. (2005). *Introduction to Fourier Optics* (3rd ed.). Roberts and Company Publishers.
   - Voelz, D. G. (2011). *Computational Fourier Optics: A MATLAB Tutorial*. SPIE Press.

4. **Phase Shift Implementation**:
   - Davis, J. A., Cottrell, D. M., Campos, J., Yzuel, M. J., & Moreno, I. (1999). Encoding amplitude information onto phase-only filters. *Applied Optics*, 38(23), 5004-5013.
   - Arrizon, V., Ruiz, U., Carrada, R., & González, L. A. (2007). Pixelated phase computer holograms for the accurate encoding of scalar complex fields. *Journal of the Optical Society of America A*, 24(11), 3500-3507.

5. **Error Metrics and Convergence**:
   - Fienup, J. R. (1982). Phase retrieval algorithms: a comparison. *Applied Optics*, 21(15), 2758-2769.
   - Soifer, V. A., Kotlyar, V. V., & Doskolovich, L. L. (1997). *Iterative Methods for Diffractive Optical Elements Computation*. Taylor & Francis.

6. **Input Beam Profiles**:
   - Dickey, F. M. (Ed.). (2018). *Laser Beam Shaping: Theory and Techniques* (2nd ed.). CRC Press.
   - Forbes, A. (2019). Structured light from lasers. *Laser & Photonics Reviews*, 13(11), 1900140.

7. **SLM Hardware Specifications and Control**:
   - Sony LCX016AL-6 SLM technical documentation
   - Meadowlark Optics. (2018). *Spatial Light Modulator: Software & Hardware Manual*.
