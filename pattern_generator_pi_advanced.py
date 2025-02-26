"""
Advanced Pattern Generator for Sony LCX016AL-6 SLM - Raspberry Pi Version
Generates phase patterns using advanced algorithms and optimization techniques.

SLM Specifications:
- Resolution: 832 x 624 pixels
- Pixel Pitch: 32 μm
- Active Area: 26.6 x 20.0 mm
- Refresh Rate: 60 Hz
- Contrast Ratio: 200:1
- Default Wavelength: 532 nm (green laser)

Advanced Features:
- Multiple pattern generation algorithms
- Phase correction and calibration
- Beam shaping and profile control
- Advanced optimization parameters
- Aberration correction
- Multi-plane pattern generation
- Real-time pattern optimization
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import json
from tqdm import tqdm
from scipy.special import jv  # For Bessel functions
from scipy.optimize import minimize  # For pattern optimization
import subprocess
import os
import time
import threading
from pattern_generator_pi import PatternGenerator

class AdvancedSLMPatternGenerator(PatternGenerator):
    def __init__(self):
        """Initialize the advanced pattern generator with extended features"""
        # Initialize base class
        super().__init__()
        
        # Additional instance variables
        self.convergence_threshold = 1e-6
        self.aberration_coefficients = [0.0] * 5  # For Zernike terms
        self.algorithm = "Gerchberg-Saxton"  # Default algorithm
        self.beam_profile = "Gaussian"  # Default beam profile
        self.beam_params = {
            'gaussian_sigma': 0.5,
            'bessel_scale': 5.0,
            'lg_l': 1
        }
        
        # Advanced parameters
        self.max_iterations = 50
        self.phase_correction_enabled = False
        self.phase_correction_map = None
        self.aberration_correction_enabled = False
        self.beam_shaping_enabled = False
        self.multi_plane_enabled = False
        self.num_planes = 1
        self.plane_spacing = 100e-6  # 100 μm between planes
        self.optimization_enabled = False
        self.optimization_metric = "MSE"
        
        # Create the main window
        self.root = tk.Tk()
        self.root.title("Advanced SLM Pattern Generator")
        
        # Create the main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas for scrolling
        self.canvas = tk.Canvas(self.main_frame)
        self.scrollbar = ttk.Scrollbar(self.main_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor=tk.NW)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Pack scrollbar components
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create frames
        self.control_frame = ttk.Frame(self.scrollable_frame)
        self.control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.preview_frame = ttk.Frame(self.scrollable_frame)
        self.preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Status variable
        self.status_var = tk.StringVar(value="Ready")
        
        # Create controls and preview
        self.create_controls()
        self.create_advanced_controls()
        self.create_preview()
        
        # Create status bar
        self.status_bar = ttk.Label(self.scrollable_frame, textvariable=self.status_var)
        self.status_bar.pack(fill=tk.X, padx=10, pady=5)
        
        # Bind keyboard shortcuts
        self.root.bind('<Control-q>', lambda e: self.quit_application())
        self.root.bind('<Escape>', lambda e: self.quit_application())
        
    def create_controls(self):
        """Create basic control panel"""
        # Call base class create_controls first
        super().create_controls()
        
    def create_advanced_controls(self):
        """Create controls for advanced features"""
        advanced_frame = ttk.LabelFrame(self.control_frame, text="Advanced Controls")
        advanced_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Algorithm selection
        ttk.Label(advanced_frame, text="Algorithm:").pack(anchor=tk.W)
        self.algorithm_var = tk.StringVar(value=self.algorithm)
        algorithm_menu = ttk.OptionMenu(advanced_frame, self.algorithm_var,
            self.algorithm,  # Current value
            "Gerchberg-Saxton",
            "Weighted Gerchberg-Saxton",
            "Direct Binary Search",
            "Adaptive-Additive")
        algorithm_menu.pack(fill=tk.X)
        
        # Beam profile selection
        ttk.Label(advanced_frame, text="Beam Profile:").pack(anchor=tk.W)
        self.beam_profile_var = tk.StringVar(value=self.beam_profile)
        profile_menu = ttk.OptionMenu(advanced_frame, self.beam_profile_var,
            self.beam_profile,  # Current value
            "Gaussian",
            "Top-Hat",
            "Bessel",
            "Laguerre-Gaussian",
            "Custom")
        profile_menu.pack(fill=tk.X)
        
        # Phase correction
        self.phase_correction_var = tk.BooleanVar(value=self.phase_correction_enabled)
        ttk.Checkbutton(advanced_frame, text="Enable Phase Correction",
            variable=self.phase_correction_var).pack(anchor=tk.W)
        
        # Zernike coefficients
        zernike_frame = ttk.LabelFrame(advanced_frame, text="Zernike Coefficients")
        zernike_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.zernike_scales = []
        zernike_names = ["Piston", "Tip", "Tilt", "Defocus", "Astigmatism"]
        for i, name in enumerate(zernike_names):
            ttk.Label(zernike_frame, text=name).pack(anchor=tk.W)
            scale = ttk.Scale(zernike_frame, from_=-1.0, to=1.0, orient=tk.HORIZONTAL)
            scale.set(0.0)
            scale.pack(fill=tk.X)
            self.zernike_scales.append(scale)
            
        # Settings buttons
        settings_frame = ttk.Frame(advanced_frame)
        settings_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(settings_frame, text="Save Settings",
            command=self.save_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(settings_frame, text="Load Settings",
            command=self.load_settings).pack(side=tk.LEFT)
        ttk.Button(settings_frame, text="Settings",
            command=self.create_settings_window).pack(side=tk.LEFT)
        
    def create_preview(self):
        """Create preview area"""
        # Create figure with three subplots and increased height
        plt.rcParams['figure.figsize'] = [15, 8]  # Larger figure size
        plt.rcParams['figure.dpi'] = 100
        self.fig = plt.figure()
        
        # Add subplots with proper spacing
        gs = self.fig.add_gridspec(1, 3, hspace=0, wspace=0.3)
        self.ax1 = self.fig.add_subplot(gs[0, 0])
        self.ax2 = self.fig.add_subplot(gs[0, 1])
        self.ax3 = self.fig.add_subplot(gs[0, 2])
        
        # Create canvas with fixed size
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.preview_frame)
        self.canvas.draw()
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.configure(height=700)  # Fixed height
        canvas_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add toolbar at the bottom
        toolbar_frame = ttk.Frame(self.preview_frame)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()
        
        # Set titles and remove ticks
        self.ax1.set_title('Target Image')
        self.ax2.set_title('Generated Pattern')
        self.ax3.set_title('Simulated Reconstruction')
        
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Add SLM specifications below the preview
        specs_text = """SLM Specifications (Sony LCX016AL-6):
• Resolution: 832 x 624 pixels
• Pixel Pitch: 32 μm
• Active Area: 26.6 x 20.0 mm
• Refresh Rate: 60 Hz
• Contrast Ratio: 200:1
• Default Wavelength: 532 nm (green laser)"""
        
        specs_label = ttk.Label(self.preview_frame, text=specs_text, justify=tk.LEFT)
        specs_label.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        
    def create_camera_preview(self):
        """Create camera preview area"""
        self.camera_fig, self.camera_ax = plt.subplots(figsize=(6, 4))
        self.camera_canvas = FigureCanvasTkAgg(self.camera_fig, master=self.camera_frame)
        self.camera_canvas.draw()
        self.camera_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.camera_ax.set_title('Camera Preview')
        self.camera_ax.set_xticks([])
        self.camera_ax.set_yticks([])
        
    def update_camera_preview(self):
        """Update camera preview if camera is active"""
        while self.camera_active and not self.camera_paused:
            ret, frame = self.cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Clear previous image
                self.camera_ax.clear()
                self.camera_ax.imshow(frame)
                self.camera_ax.set_title('Camera Preview')
                self.camera_ax.set_xticks([])
                self.camera_ax.set_yticks([])
                
                # Update canvas
                self.camera_canvas.draw()
            
            # Small delay to prevent high CPU usage
            time.sleep(0.03)
            
    def pause_camera(self):
        """Pause/Resume camera preview"""
        if self.camera_active:
            self.camera_paused = not self.camera_paused
            if self.camera_paused:
                self.pause_camera_button.configure(text="Resume Camera")
                self.status_var.set("Camera paused")
            else:
                self.pause_camera_button.configure(text="Pause Camera")
                self.status_var.set("Camera resumed")
            
    def load_image(self):
        """Load and preprocess target image"""
        try:
            filename = filedialog.askopenfilename(
                title="Select Target Image",
                filetypes=[
                    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                    ("All files", "*.*")
                ]
            )
            
            if filename:
                # Load image
                img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise ValueError("Failed to load image")
                
                # Resize to SLM resolution if needed
                if img.shape != (self.height, self.width):
                    img = cv2.resize(img, (self.width, self.height))
                
                # Normalize to [0, 1]
                self.target = img.astype(float) / 255.0
                
                # Create padded version for FFT
                self.padded_target = np.pad(
                    self.target,
                    ((0, self.height), (0, self.width)),
                    mode='constant'
                )
                
                # Update preview
                self.update_preview()
                self.status_var.set(f"Loaded image: {os.path.basename(filename)}")
                
        except Exception as e:
            self.status_var.set(f"Error loading image: {str(e)}")
            
    def create_settings_window(self):
        """Create settings window for phase image generation"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Phase Image Generation Settings")
        settings_window.geometry("400x600")
        
        # Create settings frame
        settings_frame = ttk.Frame(settings_window, padding="10")
        settings_frame.pack(fill=tk.BOTH, expand=True)
        
        # Algorithm settings
        algo_frame = ttk.LabelFrame(settings_frame, text="Algorithm Settings", padding="5")
        algo_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(algo_frame, text="Max Iterations:").pack(anchor=tk.W)
        max_iter_entry = ttk.Entry(algo_frame)
        max_iter_entry.insert(0, str(self.max_iterations))
        max_iter_entry.pack(fill=tk.X)
        
        ttk.Label(algo_frame, text="Convergence Threshold:").pack(anchor=tk.W)
        conv_entry = ttk.Entry(algo_frame)
        conv_entry.insert(0, str(self.convergence_threshold))
        conv_entry.pack(fill=tk.X)
        
        # Beam profile settings
        beam_frame = ttk.LabelFrame(settings_frame, text="Beam Profile Settings", padding="5")
        beam_frame.pack(fill=tk.X, padx=5, pady=5)
        
        profiles = ["Gaussian", "Top-Hat", "Bessel", "Laguerre-Gaussian"]
        ttk.Label(beam_frame, text="Profile Type:").pack(anchor=tk.W)
        profile_var = tk.StringVar(value=self.beam_profile)
        profile_menu = ttk.OptionMenu(beam_frame, profile_var, self.beam_profile, *profiles)
        profile_menu.pack(fill=tk.X)
        
        # Profile parameters
        param_frame = ttk.LabelFrame(beam_frame, text="Profile Parameters", padding="5")
        param_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Gaussian sigma
        ttk.Label(param_frame, text="Gaussian Sigma:").pack(anchor=tk.W)
        sigma_scale = ttk.Scale(param_frame, from_=0.1, to=2.0, orient=tk.HORIZONTAL)
        sigma_scale.set(0.5)
        sigma_scale.pack(fill=tk.X)
        
        # Bessel scale
        ttk.Label(param_frame, text="Bessel Scale:").pack(anchor=tk.W)
        bessel_scale = ttk.Scale(param_frame, from_=1.0, to=10.0, orient=tk.HORIZONTAL)
        bessel_scale.set(5.0)
        bessel_scale.pack(fill=tk.X)
        
        # LG parameters
        ttk.Label(param_frame, text="LG Orbital Angular Momentum:").pack(anchor=tk.W)
        lg_l_var = tk.StringVar(value="1")
        ttk.Entry(param_frame, textvariable=lg_l_var).pack(fill=tk.X)
        
        # Save button
        def save_settings():
            try:
                self.max_iterations = int(max_iter_entry.get())
                self.convergence_threshold = float(conv_entry.get())
                self.beam_profile = profile_var.get()
                
                # Save profile parameters
                self.beam_params = {
                    'gaussian_sigma': sigma_scale.get(),
                    'bessel_scale': bessel_scale.get(),
                    'lg_l': int(lg_l_var.get())
                }
                
                self.status_var.set("Settings saved successfully")
                settings_window.destroy()
                
            except ValueError as e:
                self.status_var.set(f"Error saving settings: {str(e)}")
        
        ttk.Button(settings_frame, text="Save Settings", command=save_settings).pack(pady=10)
            
    def send_to_slm(self):
        """Send pattern to SLM via HDMI1"""
        if not hasattr(self, 'pattern'):
            self.status_var.set("No pattern to display. Generate or load a pattern first.")
            return
            
        try:
            # Create a fullscreen window on HDMI1
            cv2.namedWindow('SLM Display', cv2.WND_PROP_FULLSCREEN)
            cv2.moveWindow('SLM Display', 1920, 0)  # Move to second display (HDMI1)
            cv2.setWindowProperty('SLM Display', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            
            # Display the pattern
            cv2.imshow('SLM Display', self.pattern)
            self.status_var.set("Pattern sent to SLM (HDMI1)")
            
        except Exception as e:
            self.status_var.set(f"Error sending to SLM: {str(e)}")
            
    def quit_application(self):
        """Clean up and quit the application"""
        if self.camera_active:
            self.camera_active = False
            self.cap.release()
        cv2.destroyAllWindows()
        self.root.quit()
        
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
            
    def weighted_gerchberg_saxton(self, target):
        """Weighted Gerchberg-Saxton algorithm for improved uniformity"""
        try:
            iterations = int(self.max_iter_var.get())
            phase = np.random.uniform(-np.pi, np.pi, target.shape)
            weights = np.ones_like(target)
            
            for i in tqdm(range(iterations), desc="Generating pattern"):
                # Forward propagation
                field = np.exp(1j * phase)
                far_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
                far_field_amp = np.abs(far_field)
                
                # Update weights
                weights = np.where(far_field_amp > 0, target / far_field_amp, weights)
                
                # Apply target amplitude with weights
                far_field = weights * target * np.exp(1j * np.angle(far_field))
                
                # Backward propagation
                field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(far_field)))
                new_phase = np.angle(field)
                
                # Check convergence
                if np.mean(np.abs(new_phase - phase)) < self.convergence_threshold:
                    break
                    
                phase = new_phase
            
            return phase, field
            
        except Exception as e:
            self.status_var.set(f"Error in weighted GS algorithm: {str(e)}")
            return None, None
            
    def direct_binary_search(self, target):
        """Direct Binary Search algorithm for binary phase patterns"""
        try:
            # Initialize binary phase (0 or π)
            phase = np.random.choice([0, np.pi], size=target.shape)
            field = np.exp(1j * phase)
            
            # Initial error
            far_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
            best_error = np.mean((np.abs(far_field)**2 - target)**2)
            
            improved = True
            while improved:
                improved = False
                for i in range(target.shape[0]):
                    for j in range(target.shape[1]):
                        # Flip pixel
                        old_phase = phase[i,j]
                        phase[i,j] = np.pi if old_phase == 0 else 0
                        field = np.exp(1j * phase)
                        
                        # Calculate new error
                        far_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
                        new_error = np.mean((np.abs(far_field)**2 - target)**2)
                        
                        # Keep change if better, revert if worse
                        if new_error < best_error:
                            best_error = new_error
                            improved = True
                        else:
                            phase[i,j] = old_phase
            
            return phase, np.exp(1j * phase)
            
        except Exception as e:
            self.status_var.set(f"Error in DBS algorithm: {str(e)}")
            return None, None
            
    def adaptive_additive(self, target):
        """Adaptive-Additive algorithm for phase retrieval"""
        try:
            iterations = int(self.max_iter_var.get())
            phase = np.random.uniform(-np.pi, np.pi, target.shape)
            
            for i in tqdm(range(iterations), desc="Generating pattern"):
                # Forward propagation
                field = np.exp(1j * phase)
                far_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
                
                # Adaptive-additive update
                far_field_new = (target + np.abs(far_field)) * np.exp(1j * np.angle(far_field))
                
                # Backward propagation
                field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(far_field_new)))
                new_phase = np.angle(field)
                
                # Check convergence
                if np.mean(np.abs(new_phase - phase)) < self.convergence_threshold:
                    break
                    
                phase = new_phase
            
            return phase, field
            
        except Exception as e:
            self.status_var.set(f"Error in adaptive-additive algorithm: {str(e)}")
            return None, None
            
    def calculate_zernike_correction(self):
        """Calculate phase correction using Zernike polynomials"""
        try:
            x = np.linspace(-1, 1, self.width)
            y = np.linspace(-1, 1, self.height)
            X, Y = np.meshgrid(x, y)
            R = np.sqrt(X**2 + Y**2)
            Theta = np.arctan2(Y, X)
            
            correction = np.zeros((self.height, self.width))
            
            # Add Zernike terms
            # Z0,0 = 1 (Piston)
            correction += self.aberration_coefficients[0] * np.ones_like(R)
            
            # Z1,1 = R cos(Theta) (Tip)
            correction += self.aberration_coefficients[1] * R * np.cos(Theta)
            
            # Z1,-1 = R sin(Theta) (Tilt)
            correction += self.aberration_coefficients[2] * R * np.sin(Theta)
            
            # Z2,0 = 2R^2 - 1 (Defocus)
            correction += self.aberration_coefficients[3] * (2*R**2 - 1)
            
            # Z2,2 = R^2 cos(2Theta) (Astigmatism)
            correction += self.aberration_coefficients[4] * R**2 * np.cos(2*Theta)
            
            # Add more Zernike terms as needed...
            
            return correction
            
        except Exception as e:
            self.status_var.set(f"Error calculating Zernike correction: {str(e)}")
            return np.zeros((self.height, self.width))
            
    def get_beam_profile(self):
        """Generate selected beam profile"""
        try:
            x = np.linspace(-1, 1, self.width)
            y = np.linspace(-1, 1, self.height)
            X, Y = np.meshgrid(x, y)
            R = np.sqrt(X**2 + Y**2)
            
            if self.beam_profile == "Gaussian":
                sigma = self.beam_params['gaussian_sigma']
                return np.exp(-R**2 / (2*sigma**2))
                
            elif self.beam_profile == "Top-Hat":
                radius = 0.8  # Adjustable parameter
                return (R <= radius).astype(float)
                
            elif self.beam_profile == "Bessel":
                scale = self.beam_params['bessel_scale']
                return np.abs(jv(0, scale*R))
                
            elif self.beam_profile == "Laguerre-Gaussian":
                l = self.beam_params['lg_l']  # Orbital angular momentum
                p = 0  # Radial index
                sigma = self.beam_params['gaussian_sigma']  # Beam width
                Theta = np.arctan2(Y, X)
                profile = (R/sigma)**(np.abs(l)) * np.exp(-R**2/(2*sigma**2)) * np.exp(1j*l*Theta)
                return np.abs(profile)
                
            else:  # Custom or default
                return np.ones((self.height, self.width))
                
        except Exception as e:
            self.status_var.set(f"Error generating beam profile: {str(e)}")
            return np.ones((self.height, self.width))
            
    def generate_pattern(self):
        """Override pattern generation to use advanced features"""
        if not hasattr(self, 'padded_target'):
            self.status_var.set("Please load an image first")
            return
            
        try:
            # Apply selected algorithm
            if self.algorithm_var.get() == "Weighted GS":
                phase, field = self.weighted_gerchberg_saxton(self.padded_target)
            elif self.algorithm_var.get() == "Direct Binary Search":
                phase, field = self.direct_binary_search(self.padded_target)
            elif self.algorithm_var.get() == "Adaptive-Additive":
                phase, field = self.adaptive_additive(self.padded_target)
            else:
                # Default to original GS algorithm
                phase, field = super().gerchberg_saxton(self.padded_target)
            
            if phase is None:
                return
                
            # Apply corrections if enabled
            if self.phase_correction_enabled and self.phase_correction_map is not None:
                phase += self.phase_correction_map
                
            if self.aberration_correction_enabled:
                phase += self.calculate_zernike_correction()
                
            # Apply beam shaping if enabled
            if self.beam_shaping_enabled:
                field *= self.get_beam_profile()
            
            # Extract SLM region
            start_x = (self.padded_width - self.width) // 2
            end_x = start_x + self.width
            start_y = (self.padded_height - self.height) // 2
            end_y = start_y + self.height
            
            self.slm_phase = phase[start_y:end_y, start_x:end_x]
            
            # Convert to 8-bit grayscale
            self.pattern = ((self.slm_phase + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
            
            # Update preview
            self.ax2.clear()
            self.ax2.imshow(self.slm_phase, cmap='gray')
            self.ax2.set_title('Generated Phase Pattern')
            self.ax2.set_xticks([])
            self.ax2.set_yticks([])
            
            # Calculate and show reconstruction
            far_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
            far_field_intensity = np.abs(far_field)**2
            
            self.ax3.clear()
            self.ax3.imshow(far_field_intensity, cmap='viridis')
            self.ax3.set_title('Simulated Reconstruction')
            self.ax3.set_xticks([])
            self.ax3.set_yticks([])
            
            self.canvas.draw()
            self.status_var.set("Pattern generated successfully")
            
        except Exception as e:
            self.status_var.set(f"Error generating pattern: {str(e)}")
            
    def save_pattern(self):
        """Save the generated pattern"""
        if not hasattr(self, 'pattern'):
            self.status_var.set("No pattern to save")
            return
            
        try:
            # Generate default filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            default_name = f"pattern_{timestamp}.png"
            
            # Use zenity file save dialog
            cmd = ['zenity', '--file-selection', '--save',
                   '--filename=' + default_name,
                   '--file-filter=*.png',
                   '--title=Save Pattern',
                   '--confirm-overwrite']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.status_var.set("Pattern save cancelled")
                return
                
            save_path = result.stdout.strip()
            if not save_path:
                return
                
            # Ensure .png extension
            if not save_path.lower().endswith('.png'):
                save_path += '.png'
                
            # Save the pattern
            cv2.imwrite(save_path, self.pattern)
            self.status_var.set(f"Pattern saved to: {save_path}")
            
        except Exception as e:
            self.status_var.set(f"Error saving pattern: {str(e)}")
            
    def save_settings(self):
        """Save current settings to a JSON file"""
        try:
            settings = {
                'algorithm': self.algorithm_var.get(),
                'beam_profile': self.beam_profile_var.get(),
                'phase_correction': self.phase_correction_var.get(),
                'zernike_coefficients': [scale.get() for scale in self.zernike_scales],
                'convergence_threshold': self.convergence_threshold
            }
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="Save Settings"
            )
            
            if filename:
                with open(filename, 'w') as f:
                    json.dump(settings, f, indent=4)
                self.status_var.set("Settings saved successfully")
                
        except Exception as e:
            self.status_var.set(f"Error saving settings: {str(e)}")
            
    def load_settings(self):
        """Load settings from a JSON file"""
        try:
            filename = filedialog.askopenfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="Load Settings"
            )
            
            if filename:
                with open(filename, 'r') as f:
                    settings = json.load(f)
                
                # Apply loaded settings
                self.algorithm_var.set(settings.get('algorithm', 'Gerchberg-Saxton'))
                self.beam_profile_var.set(settings.get('beam_profile', 'Gaussian'))
                self.phase_correction_var.set(settings.get('phase_correction', False))
                
                # Load Zernike coefficients
                zernike_coeffs = settings.get('zernike_coefficients', [0.0] * 5)
                for scale, coeff in zip(self.zernike_scales, zernike_coeffs):
                    scale.set(coeff)
                
                self.convergence_threshold = settings.get('convergence_threshold', 1e-6)
                
                self.status_var.set("Settings loaded successfully")
                
        except Exception as e:
            self.status_var.set(f"Error loading settings: {str(e)}")
            
    def override_pattern_generation(self):
        """Override pattern generation to use selected algorithm"""
        try:
            # Get selected algorithm
            algorithm = self.algorithm_var.get()
            
            # Update aberration coefficients from sliders
            self.aberration_coefficients = [scale.get() for scale in self.zernike_scales]
            
            # Generate base target
            target = self.get_beam_profile() * self.padded_target
            
            # Apply phase correction if enabled
            if self.phase_correction_var.get():
                correction = self.calculate_zernike_correction()
                target = target * np.exp(1j * correction)
            
            # Choose algorithm
            if algorithm == "Weighted Gerchberg-Saxton":
                phase, field = self.weighted_gerchberg_saxton(target)
            elif algorithm == "Direct Binary Search":
                phase, field = self.direct_binary_search(target)
            elif algorithm == "Adaptive-Additive":
                phase, field = self.adaptive_additive(target)
            else:  # Default to standard Gerchberg-Saxton
                phase, field = super().generate_pattern()
            
            # Update display
            if phase is not None and field is not None:
                self.phase = phase
                self.field = field
                self.update_preview()
                return True
            
            return False
            
        except Exception as e:
            self.status_var.set(f"Error in pattern generation: {str(e)}")
            return False
            
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = AdvancedSLMPatternGenerator()
    app.run()
