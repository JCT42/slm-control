"""
Pattern Generator for Sony LCX016AL-6 SLM - Raspberry Pi Version
Generates phase patterns using Gerchberg-Saxton algorithm for far-field image reconstruction.

SLM Specifications:
- Resolution: 832 x 624 pixels
- Pixel pitch: 32 µm
- Active area: 26.6 x 20.0 mm
- Default wavelength: 532 nm (green laser)
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk, filedialog
from pathlib import Path
from tqdm import tqdm

class PatternGenerator:
    def __init__(self):
        """Initialize the pattern generator with SLM specifications"""
        # Sony LCX016AL-6 specifications - DO NOT MODIFY
        self.width = 832  # pixels
        self.height = 624  # pixels
        self.pixel_size = 32e-6  # 32 µm pixel pitch
        self.active_area = (26.6e-3, 20.0e-3)  # 26.6mm x 20.0mm active area
        
        # Default wavelength (will be adjustable)
        self.wavelength = 532e-9  # 532nm green laser
        
        # Simulation parameters
        self.padding_factor = 2
        self.padded_width = self.width * self.padding_factor
        self.padded_height = self.height * self.padding_factor
        
        # Calculate important parameters
        self.k = 2 * np.pi / self.wavelength  # Wave number
        self.dx = self.pixel_size
        self.df_x = 1 / (self.padded_width * self.dx)  # Frequency step size x
        self.df_y = 1 / (self.padded_height * self.dx)  # Frequency step size y
        
        # Create coordinate grids
        self.x = np.linspace(-self.padded_width//2, self.padded_width//2-1, self.padded_width) * self.dx
        self.y = np.linspace(-self.padded_height//2, self.padded_height//2-1, self.padded_height) * self.dx
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Initialize GUI
        self.setup_gui()
        
    def setup_gui(self):
        """Create the main GUI window and controls"""
        self.root = tk.Tk()
        self.root.title("SLM Pattern Generator")
        self.root.geometry("1200x800")
        
        # Create frames
        self.control_frame = ttk.LabelFrame(self.root, text="Controls", padding="10")
        self.control_frame.pack(fill="x", padx=10, pady=5)
        
        self.preview_frame = ttk.LabelFrame(self.root, text="Preview", padding="10")
        self.preview_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Create parameter controls
        self.create_controls()
        
        # Create preview area
        self.create_preview()
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief="sunken")
        self.status_bar.pack(fill="x", padx=10, pady=5)
        
    def create_controls(self):
        """Create parameter control widgets"""
        # SLM Info frame
        info_frame = ttk.LabelFrame(self.control_frame, text="SLM Specifications")
        info_frame.pack(fill="x", pady=5)
        
        specs = [
            ("Resolution", f"{self.width} x {self.height} pixels"),
            ("Pixel Pitch", "32 µm"),
            ("Active Area", "26.6 x 20.0 mm"),
        ]
        
        for i, (label, value) in enumerate(specs):
            ttk.Label(info_frame, text=f"{label}:").grid(row=i, column=0, padx=5, pady=2, sticky="e")
            ttk.Label(info_frame, text=value).grid(row=i, column=1, padx=5, pady=2, sticky="w")
        
        # File controls
        file_frame = ttk.Frame(self.control_frame)
        file_frame.pack(fill="x", pady=5)
        
        ttk.Button(file_frame, text="Load Image", command=self.load_image).pack(side="left", padx=5)
        ttk.Button(file_frame, text="Save Pattern", command=self.save_pattern).pack(side="left", padx=5)
        
        # Parameter controls
        param_frame = ttk.LabelFrame(self.control_frame, text="Pattern Generation Parameters")
        param_frame.pack(fill="x", pady=5)
        
        # Grid layout for parameters
        params = [
            ("Iterations:", "iterations_var", "100"),
            ("Beam Width Factor:", "beam_width_var", "1.0"),
            ("Phase Range (π):", "phase_range_var", "2.0"),
            ("Wavelength (nm):", "wavelength_var", "532"),
        ]
        
        for i, (label, var_name, default) in enumerate(params):
            ttk.Label(param_frame, text=label).grid(row=i//2, column=i%2*2, padx=5, pady=5, sticky="e")
            setattr(self, var_name, tk.StringVar(value=default))
            ttk.Entry(param_frame, textvariable=getattr(self, var_name), width=10).grid(
                row=i//2, column=i%2*2+1, padx=5, pady=5, sticky="w"
            )
        
        # Generate button
        ttk.Button(param_frame, text="Generate Pattern", command=self.generate_pattern).grid(
            row=len(params)//2+1, column=0, columnspan=4, pady=10
        )
        
    def create_preview(self):
        """Create matplotlib preview area"""
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(15, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.preview_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Add toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, self.preview_frame)
        toolbar.update()
        
        # Initialize axes
        self.ax1.set_title('Target Image')
        self.ax2.set_title('Generated Phase Pattern')
        self.ax3.set_title('Simulated Far Field')
        
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.set_xticks([])
            ax.set_yticks([])
        
    def load_image(self):
        """Load and preprocess target image"""
        try:
            file_path = filedialog.askopenfilename(
                title="Select Target Image",
                filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif")]
            )
            
            if not file_path:
                return
                
            # Read image and convert to grayscale
            image = cv2.imread(file_path)
            if image is None:
                raise ValueError(f"Could not load image: {file_path}")
                
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
            # Resize to match SLM resolution
            scale_x = self.width / image.shape[1]
            scale_y = self.height / image.shape[0]
            scale = min(scale_x, scale_y)
            
            new_width = int(image.shape[1] * scale)
            new_height = int(image.shape[0] * scale)
            
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            
            # Create black canvas of SLM size
            canvas = np.zeros((self.height, self.width), dtype=np.uint8)
            
            # Center the image
            x_offset = (self.width - new_width) // 2
            y_offset = (self.height - new_height) // 2
            canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = image
            
            # Normalize and store
            self.target_image = canvas.astype(float) / 255.0
            
            # Create padded version
            self.padded_target = np.zeros((self.padded_height, self.padded_width))
            start_x = (self.padded_width - self.width) // 2
            end_x = start_x + self.width
            start_y = (self.padded_height - self.height) // 2
            end_y = start_y + self.height
            self.padded_target[start_y:end_y, start_x:end_x] = self.target_image
            
            # Update preview
            self.ax1.clear()
            self.ax1.imshow(self.target_image, cmap='gray')
            self.ax1.set_title('Target Image')
            self.ax1.set_xticks([])
            self.ax1.set_yticks([])
            self.canvas.draw()
            
            self.status_var.set("Image loaded successfully")
            
        except Exception as e:
            self.status_var.set(f"Error loading image: {str(e)}")
            
    def generate_input_beam(self):
        """Generate Gaussian input beam profile"""
        beam_width = float(self.beam_width_var.get())
        sigma_x = self.active_area[0] / (2.355 * beam_width)  # FWHM = 2.355 * sigma
        sigma_y = self.active_area[1] / (2.355 * beam_width)
        
        beam = np.exp(-self.X**2 / (2 * sigma_x**2) - self.Y**2 / (2 * sigma_y**2))
        return beam / np.max(beam)
        
    def gerchberg_saxton(self, target_image):
        """Run Gerchberg-Saxton algorithm"""
        try:
            num_iterations = int(self.iterations_var.get())
            gaussian_beam = self.generate_input_beam()
            
            # Initialize with random phase
            random_phase = np.exp(1j * 2 * np.pi * np.random.rand(self.padded_height, self.padded_width))
            field = gaussian_beam * random_phase
            
            # Run iterations
            for _ in tqdm(range(num_iterations), desc="Running Gerchberg-Saxton"):
                # Forward FFT
                far_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
                
                # Replace amplitude
                far_field = np.sqrt(target_image) * np.exp(1j * np.angle(far_field))
                
                # Inverse FFT
                field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(far_field)))
                
                # Apply input beam constraint
                field = gaussian_beam * np.exp(1j * np.angle(field))
            
            return np.angle(field), field
            
        except Exception as e:
            self.status_var.set(f"Error in pattern generation: {str(e)}")
            return None, None
            
    def generate_pattern(self):
        """Generate pattern using current parameters"""
        if not hasattr(self, 'padded_target'):
            self.status_var.set("Please load an image first")
            return
            
        try:
            # Update wavelength from GUI
            self.wavelength = float(self.wavelength_var.get()) * 1e-9  # Convert nm to m
            self.k = 2 * np.pi / self.wavelength  # Update wave number
            
            # Generate pattern
            phase, field = self.gerchberg_saxton(self.padded_target)
            if phase is None:
                return
                
            # Extract SLM region
            start_x = (self.padded_width - self.width) // 2
            end_x = start_x + self.width
            start_y = (self.padded_height - self.height) // 2
            end_y = start_y + self.height
            
            self.slm_phase = phase[start_y:end_y, start_x:end_x]
            
            # Convert to 8-bit grayscale
            phase_range = float(self.phase_range_var.get())
            self.pattern = ((self.slm_phase + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
            
            # Calculate far field
            far_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
            far_field_intensity = np.abs(far_field)**2
            
            # Update preview
            self.ax2.clear()
            self.ax2.imshow(self.slm_phase, cmap='gray')
            self.ax2.set_title('Generated Phase Pattern')
            self.ax2.set_xticks([])
            self.ax2.set_yticks([])
            
            self.ax3.clear()
            self.ax3.imshow(far_field_intensity, cmap='viridis')
            self.ax3.set_title('Simulated Far Field')
            self.ax3.set_xticks([])
            self.ax3.set_yticks([])
            
            self.canvas.draw()
            self.status_var.set("Pattern generated successfully")
            
        except Exception as e:
            self.status_var.set(f"Error generating pattern: {str(e)}")
            
    def save_pattern(self):
        """Save the generated pattern"""
        if not hasattr(self, 'pattern'):
            self.status_var.set("No pattern to save. Generate a pattern first.")
            return
            
        try:
            file_path = filedialog.asksaveasfilename(
                title="Save Pattern",
                defaultextension=".png",
                filetypes=[("PNG files", "*.png")]
            )
            
            if file_path:
                cv2.imwrite(file_path, self.pattern)
                self.status_var.set(f"Pattern saved to {file_path}")
                
        except Exception as e:
            self.status_var.set(f"Error saving pattern: {str(e)}")
            
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = PatternGenerator()
    app.run()
