import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
from matplotlib.widgets import Slider, RadioButtons, Button
from PIL import Image
import tkinter as tk
from tkinter import filedialog
from matplotlib.colors import Normalize
from scipy.special import genlaguerre, hermite, j0

class CustomPatternEditor:
    def __init__(self, size=512):
        self.size = size
        self.pattern = np.zeros((size, size))
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.fig.suptitle('Custom Pattern Editor\nClick to set grayscale values (0 to 255)\nClose window when done')
        self.im = self.ax.imshow(self.pattern, cmap='gray', norm=Normalize(0, 255))
        self.fig.colorbar(self.im, label='Grayscale Value')
        self.current_value = 0
        
        # Add value slider
        ax_slider = self.fig.add_axes([0.2, 0.02, 0.6, 0.03])
        self.value_slider = Slider(ax_slider, 'Grayscale Value', 0, 255, valinit=0)
        self.value_slider.on_changed(self.update_value)
        
        # Connect mouse events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.is_drawing = False
        
        plt.show()
    
    def update_value(self, val):
        self.current_value = int(val)
    
    def on_click(self, event):
        if event.inaxes == self.ax:
            self.is_drawing = event.button == 1
            self.update_pattern(event)
    
    def on_motion(self, event):
        if self.is_drawing and event.inaxes == self.ax:
            self.update_pattern(event)
    
    def update_pattern(self, event):
        if event.xdata is not None and event.ydata is not None:
            x, y = int(event.xdata), int(event.ydata)
            if 0 <= x < self.size and 0 <= y < self.size:
                self.pattern[y, x] = self.current_value
                self.im.set_data(self.pattern)
                self.fig.canvas.draw_idle()

class DiffractionSimulator:
    def __init__(self, size=1024):
        self.size = size
        self.x = np.linspace(-1, 1, size)
        self.y = np.linspace(-1, 1, size)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.R = np.sqrt(self.X**2 + self.Y**2)
        self.Phi = np.arctan2(self.Y, self.X)
        self.wavelength = 0.5  # Default wavelength in micrometers
        self.pixel_size = 8e-6  # SLM pixel size in meters
    
    def set_wavelength(self, wavelength):
        """Set the wavelength for the simulation"""
        self.wavelength = wavelength
    
    def create_input_beam(self, beam_type='gaussian', **kwargs):
        """Create different types of input beams
        
        Available beam types:
        - 'gaussian': Gaussian beam (parameters: w0)
        - 'flat_top': Super-Gaussian beam (parameters: w0, order)
        - 'lg': Laguerre-Gaussian beam (parameters: w0, l, p)
        - 'hg': Hermite-Gaussian beam (parameters: w0, m, n)
        - 'bessel': Bessel beam (parameters: kr)
        """
        if beam_type == 'gaussian':
            w0 = kwargs.get('w0', 0.2)
            return np.exp(-(self.R**2)/(w0**2))
        
        elif beam_type == 'flat_top':
            w0 = kwargs.get('w0', 0.2)
            order = kwargs.get('order', 10)  # Higher order = flatter top
            return np.exp(-(self.R**2/(w0**2))**order)
        
        elif beam_type == 'lg':
            w0 = kwargs.get('w0', 0.2)
            l = kwargs.get('l', 1)    # Azimuthal index
            p = kwargs.get('p', 0)    # Radial index
            
            # Normalized radius
            rho = np.sqrt(2) * self.R / w0
            
            # Associated Laguerre polynomial
            L = genlaguerre(p, abs(l))
            
            # LG mode
            field = (np.sqrt(2*rho**abs(l)) * np.exp(-rho**2/2) * 
                    L(rho**2) * np.exp(1j*l*self.Phi))
            
            return np.abs(field)
        
        elif beam_type == 'hg':
            w0 = kwargs.get('w0', 0.2)
            m = kwargs.get('m', 1)    # Horizontal index
            n = kwargs.get('n', 1)    # Vertical index
            
            # Normalized coordinates
            x = self.X / w0
            y = self.Y / w0
            
            # Hermite polynomials
            Hm = hermite(m)
            Hn = hermite(n)
            
            # HG mode
            field = (Hm(np.sqrt(2)*x) * Hn(np.sqrt(2)*y) * 
                    np.exp(-(x**2 + y**2)))
            
            return np.abs(field)
        
        elif beam_type == 'bessel':
            kr = kwargs.get('kr', 10)
            return j0(kr * self.R)
        
        else:
            return np.ones((self.size, self.size))
    
    def gaussian_beam(self, w0=0.2):
        """Generate a Gaussian beam with waist w0"""
        return np.exp(-(self.R**2)/(w0**2))
    
    def load_image_pattern(self):
        """Load an image file and convert it to grayscale (0-255)"""
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        root.destroy()
        
        if file_path:
            # Load image
            img = Image.open(file_path)
            
            # Convert to grayscale if not already
            if img.mode != 'L':
                img = img.convert('L')
            
            # Resize to match SLM size
            img = img.resize((self.size, self.size), Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            pattern = np.array(img, dtype=np.float32)
            
            # Create preview window
            preview_fig = plt.figure(figsize=(10, 5))
            preview_fig.suptitle('Image Preview')
            
            # Show original image
            ax1 = preview_fig.add_subplot(121)
            ax1.imshow(pattern, cmap='gray')
            ax1.set_title(f'Original (min={pattern.min():.1f}, max={pattern.max():.1f})')
            plt.colorbar(ax1.imshow(pattern, cmap='gray'), ax=ax1)
            
            # Normalize to 0-255 range
            if pattern.max() > 0:  # Avoid division by zero
                pattern = ((pattern - pattern.min()) * 255 / 
                          (pattern.max() - pattern.min()))
            
            # Show normalized image
            ax2 = preview_fig.add_subplot(122)
            ax2.imshow(pattern, cmap='gray', vmin=0, vmax=255)
            ax2.set_title('Normalized (0-255)')
            plt.colorbar(ax2.imshow(pattern, cmap='gray', vmin=0, vmax=255), ax=ax2)
            
            # Add confirmation button
            ax_button = plt.axes([0.4, 0.02, 0.2, 0.04])
            button = Button(ax_button, 'Accept and Close')
            
            def close_preview(event):
                plt.close(preview_fig)
            
            button.on_clicked(close_preview)
            
            plt.show()
            
            return pattern.astype(np.uint8)
        return None
    
    def create_custom_pattern(self):
        """Create a custom pattern using the interactive editor"""
        editor = CustomPatternEditor(self.size)
        plt.close(editor.fig)
        return editor.pattern
    
    def create_phase_pattern(self, pattern_type='grating', **kwargs):
        """Create a grayscale pattern for the SLM (0-255)
        
        Available patterns:
        - 'grating': Linear grating (parameters: period, angle)
        - 'lens': Fresnel lens (parameters: focal_length)
        - 'vortex': Optical vortex (parameters: topological_charge)
        - 'bessel': Bessel beam (parameters: kr)
        - 'axicon': Axicon lens (parameters: cone_angle)
        - 'checkerboard': Checkerboard pattern (parameters: block_size)
        - 'double_slit': Double slit (parameters: slit_width, separation, angle)
        - 'random': Random pattern
        """
        if pattern_type == 'double_slit':
            slit_width = kwargs.get('slit_width', 10)  # width in pixels
            separation = kwargs.get('separation', 50)   # separation in pixels
            angle = kwargs.get('angle', 0)             # angle in degrees
            
            # Create empty pattern
            pattern = np.zeros((self.size, self.size))
            
            # Calculate center of the pattern
            center = self.size // 2
            
            # Create coordinates relative to center
            x = np.arange(-center, center)
            y = np.arange(-center, center)
            X, Y = np.meshgrid(x, y)
            
            # Rotate coordinates
            theta = np.deg2rad(angle)
            X_rot = X * np.cos(theta) + Y * np.sin(theta)
            
            # Create slits
            slit1_center = separation / 2
            slit2_center = -separation / 2
            
            # Define slits with smooth edges using tanh function
            smoothing = 2  # Controls the sharpness of the slit edges
            slit1 = 0.5 * (np.tanh(smoothing * (X_rot - (slit1_center - slit_width/2))) - 
                          np.tanh(smoothing * (X_rot - (slit1_center + slit_width/2))))
            slit2 = 0.5 * (np.tanh(smoothing * (X_rot - (slit2_center - slit_width/2))) - 
                          np.tanh(smoothing * (X_rot - (slit2_center + slit_width/2))))
            
            # Combine slits
            pattern = 255 * (slit1 + slit2)
            
            return pattern
            
        elif pattern_type == 'grating':
            period = kwargs.get('period', 50)
            angle = kwargs.get('angle', 0)  # angle in degrees
            angle_rad = np.deg2rad(angle)
            pattern = np.mod(255 * (self.X * np.cos(angle_rad) + self.Y * np.sin(angle_rad)) / period, 255)
            return pattern
        
        elif pattern_type == 'lens':
            f = kwargs.get('focal_length', 1.0)
            r2 = self.X**2 + self.Y**2
            pattern = np.mod(255 * r2 / f, 255)
            return pattern
        
        elif pattern_type == 'vortex':
            l = kwargs.get('topological_charge', 1)
            phi = np.arctan2(self.Y, self.X)
            pattern = np.mod(255 * (phi / (2*np.pi) + 0.5), 255)
            return pattern
        
        elif pattern_type == 'bessel':
            kr = kwargs.get('kr', 20)
            r = np.sqrt(self.X**2 + self.Y**2)
            pattern = np.mod(255 * kr * r / (2*np.pi), 255)
            return pattern
        
        elif pattern_type == 'axicon':
            cone_angle = kwargs.get('cone_angle', 5)
            r = np.sqrt(self.X**2 + self.Y**2)
            pattern = np.mod(255 * r * np.tan(np.deg2rad(cone_angle)) / (2*np.pi), 255)
            return pattern
        
        elif pattern_type == 'checkerboard':
            block_size = kwargs.get('block_size', 16)
            pattern = np.indices((self.size, self.size)).sum(axis=0)
            return 255 * (np.mod(pattern, block_size) < block_size/2)
        
        elif pattern_type == 'random':
            return np.random.randint(0, 256, (self.size, self.size))
        
        else:
            return np.zeros((self.size, self.size))
    
    def simulate_farfield(self, input_beam, pattern):
        """Simulate the far-field diffraction pattern"""
        # Convert pattern to phase
        phase = 2 * np.pi * pattern / 255
        
        # Apply pattern to input beam
        field = input_beam * np.exp(1j * phase)
        
        # Scale by wavelength
        k = 2 * np.pi / self.wavelength
        field = field * np.exp(1j * k * self.R**2)
        
        # Calculate far-field pattern
        farfield = fftshift(fft2(field))
        return np.abs(farfield)**2
    
    def simulate_nearfield(self, input_beam, pattern):
        """Simulate the near-field diffraction pattern"""
        # Convert pattern to phase
        phase = 2 * np.pi * pattern / 255
        
        # Apply pattern to input beam
        field = input_beam * np.exp(1j * phase)
        
        # Near-field is just the intensity of the field right after the SLM
        return np.abs(field)**2

def main():
    # Create simulator instance with higher resolution
    sim = DiffractionSimulator(size=1024)
    
    # Create figure for interactive visualization
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(3, 4)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[0, 3])
    
    # Available input beam types and their parameters
    input_beams = {
        'Gaussian': {'type': 'gaussian', 'params': {
            'wavelength': (0.4, 0.7, 0.5),  # visible spectrum (normalized)
            'w0': (0.1, 0.5, 0.2)
        }},
        'Flat-Top': {'type': 'flat_top', 'params': {
            'wavelength': (0.4, 0.7, 0.5),
            'w0': (0.1, 0.5, 0.2),
            'order': (2, 20, 10)
        }},
        'Laguerre-Gaussian': {'type': 'lg', 'params': {
            'wavelength': (0.4, 0.7, 0.5),
            'w0': (0.1, 0.5, 0.2),
            'l': (-5, 5, 1),
            'p': (0, 3, 0)
        }},
        'Hermite-Gaussian': {'type': 'hg', 'params': {
            'wavelength': (0.4, 0.7, 0.5),
            'w0': (0.1, 0.5, 0.2),
            'm': (0, 3, 1),
            'n': (0, 3, 1)
        }},
        'Bessel': {'type': 'bessel', 'params': {
            'wavelength': (0.4, 0.7, 0.5),
            'kr': (5, 50, 10)
        }}
    }
    
    # Available patterns and their parameters
    patterns = {
        'Double Slit': {'type': 'double_slit', 'params': {
            'slit_width': (2, 50, 10),
            'separation': (10, 200, 50),
            'angle': (0, 360, 0)
        }},
        'Grating': {'type': 'grating', 'params': {'period': (10, 100, 50), 'angle': (0, 360, 0)}},
        'Lens': {'type': 'lens', 'params': {'focal_length': (0.1, 5.0, 1.0)}},
        'Vortex': {'type': 'vortex', 'params': {'topological_charge': (-5, 5, 1)}},
        'Bessel': {'type': 'bessel', 'params': {'kr': (5, 50, 20)}},
        'Axicon': {'type': 'axicon', 'params': {'cone_angle': (1, 45, 5)}},
        'Checkerboard': {'type': 'checkerboard', 'params': {'block_size': (4, 64, 16)}},
        'Random': {'type': 'random', 'params': {}},
        'Custom': {'type': 'custom', 'params': {}},
        'Load Image': {'type': 'load_image', 'params': {}}
    }
    
    # Initial patterns
    current_pattern = 'Double Slit'
    current_beam = 'Gaussian'
    
    pattern_params = {k: v[2] for k, v in patterns[current_pattern]['params'].items()}
    beam_params = {k: v[2] for k, v in input_beams[current_beam]['params'].items()}
    
    pattern = sim.create_phase_pattern(patterns[current_pattern]['type'], **pattern_params)
    input_beam = sim.create_input_beam(input_beams[current_beam]['type'], **beam_params)
    farfield = sim.simulate_farfield(input_beam, pattern)
    nearfield = sim.simulate_nearfield(input_beam, pattern)
    
    # Plot input beam
    im1 = ax1.imshow(input_beam, cmap='viridis')
    ax1.set_title('Input Beam')
    plt.colorbar(im1, ax=ax1)
    
    # Plot SLM pattern
    im2 = ax2.imshow(pattern, cmap='gray', norm=Normalize(0, 255))
    ax2.set_title('SLM Pattern (Grayscale)')
    plt.colorbar(im2, ax=ax2)
    
    # Calculate zoom window for far-field
    zoom_factor = 4  # Adjust this to change zoom level
    center = sim.size // 2
    window_size = sim.size // (2 * zoom_factor)
    slice_range = slice(center - window_size, center + window_size)
    
    # Plot far-field pattern
    im3 = ax3.imshow(np.log10(farfield[slice_range, slice_range] + 1e-10), 
                     cmap='hot', 
                     extent=[-1/zoom_factor, 1/zoom_factor, -1/zoom_factor, 1/zoom_factor])
    ax3.set_title('Far-field Intensity (log scale)')
    plt.colorbar(im3, ax=ax3)
    
    # Plot near-field pattern
    im4 = ax4.imshow(np.log10(nearfield + 1e-10), cmap='hot')
    ax4.set_title('Near-field Intensity (log scale)')
    plt.colorbar(im4, ax=ax4)
    
    # Add beam selection radio buttons
    ax_radio_beam = fig.add_subplot(gs[1, 0])
    radio_beam = RadioButtons(ax_radio_beam, list(input_beams.keys()))
    ax_radio_beam.set_title('Input Beam')
    
    # Add pattern selection radio buttons
    ax_radio_pattern = fig.add_subplot(gs[1, 1])
    radio_pattern = RadioButtons(ax_radio_pattern, list(patterns.keys()))
    ax_radio_pattern.set_title('SLM Pattern')
    
    # Add parameter sliders
    pattern_sliders = {}
    beam_sliders = {}
    ax_sliders = []
    
    def update(val=None):
        nonlocal pattern, input_beam, farfield, nearfield
        
        # Get current parameters
        pattern_params = {name: slider.val for name, slider in pattern_sliders.items()}
        beam_params = {name: slider.val for name, slider in beam_sliders.items()}
        
        # Update wavelength in simulator if it changed
        if 'wavelength' in beam_params:
            sim.set_wavelength(beam_params['wavelength'])
        
        # Create new pattern and beam
        if current_pattern not in ['Custom', 'Load Image']:
            pattern = sim.create_phase_pattern(patterns[current_pattern]['type'], **pattern_params)
        input_beam = sim.create_input_beam(input_beams[current_beam]['type'], **beam_params)
        
        # Calculate near-field and far-field
        nearfield = sim.simulate_nearfield(input_beam, pattern)
        farfield = sim.simulate_farfield(input_beam, pattern)
        
        # Update plots
        im1.set_data(input_beam)
        im2.set_data(pattern)
        im3.set_data(np.log10(farfield[slice_range, slice_range] + 1e-10))
        im4.set_data(np.log10(nearfield + 1e-10))
        fig.canvas.draw_idle()

    def create_sliders(pattern_name, beam_name):
        # Clear existing sliders
        for ax in ax_sliders:
            ax.remove()
        pattern_sliders.clear()
        beam_sliders.clear()
        ax_sliders.clear()
        
        # Define units for parameters
        param_units = {
            'wavelength': 'µm',
            'w0': 'mm',
            'order': '',
            'l': '',
            'p': '',
            'm': '',
            'n': '',
            'kr': 'rad/mm',
            'slit_width': 'µm',
            'separation': 'µm',
            'angle': '°',
            'period': 'µm',
            'focal_length': 'm',
            'topological_charge': '',
            'cone_angle': '°',
            'block_size': 'px'
        }
        
        # Create new sliders for the selected pattern
        pattern = patterns[pattern_name]
        num_pattern_params = len(pattern['params'])
        
        beam = input_beams[beam_name]
        num_beam_params = len(beam['params'])
        
        # Create beam parameter sliders (left side)
        if num_beam_params > 0:
            slider_height = 0.02
            spacing = 0.01
            start_y = 0.3  # Position in the bottom third of the figure
            for i, (param_name, (min_val, max_val, init_val)) in enumerate(beam['params'].items()):
                y_pos = start_y - (i * (slider_height + spacing))
                ax = fig.add_axes([0.1, y_pos, 0.2, slider_height])
                ax_sliders.append(ax)
                unit = param_units.get(param_name, '')
                label = f'{param_name} [{unit}]' if unit else param_name
                slider = Slider(ax, label, min_val, max_val, valinit=init_val)
                slider.on_changed(update)  # Connect slider to update function
                beam_sliders[param_name] = slider
        
        # Create pattern parameter sliders (middle)
        if num_pattern_params > 0:
            slider_height = 0.02
            spacing = 0.01
            start_y = 0.3  # Position in the bottom third of the figure
            for i, (param_name, (min_val, max_val, init_val)) in enumerate(pattern['params'].items()):
                y_pos = start_y - (i * (slider_height + spacing))
                ax = fig.add_axes([0.4, y_pos, 0.2, slider_height])
                ax_sliders.append(ax)
                unit = param_units.get(param_name, '')
                label = f'{param_name} [{unit}]' if unit else param_name
                slider = Slider(ax, label, min_val, max_val, valinit=init_val)
                slider.on_changed(update)  # Connect slider to update function
                pattern_sliders[param_name] = slider
    
    create_sliders(current_pattern, current_beam)
    
    def update_pattern(pattern_name):
        nonlocal current_pattern, pattern
        current_pattern = pattern_name
        if pattern_name in ['Custom', 'Load Image']:
            if pattern_name == 'Custom':
                pattern = sim.create_custom_pattern()
            else:
                loaded_pattern = sim.load_image_pattern()
                if loaded_pattern is not None:
                    pattern = loaded_pattern
        else:
            create_sliders(pattern_name, current_beam)
            update()
        
        if pattern_name in ['Custom', 'Load Image']:
            farfield = sim.simulate_farfield(input_beam, pattern)
            nearfield = sim.simulate_nearfield(input_beam, pattern)
            im2.set_data(pattern)
            im3.set_data(np.log10(farfield[slice_range, slice_range] + 1e-10))
            im4.set_data(np.log10(nearfield + 1e-10))
        
        fig.canvas.draw_idle()
    
    def update_beam(beam_name):
        nonlocal current_beam, input_beam
        current_beam = beam_name
        create_sliders(current_pattern, beam_name)
        update()
        fig.canvas.draw_idle()
    
    radio_pattern.on_clicked(update_pattern)
    radio_beam.on_clicked(update_beam)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
