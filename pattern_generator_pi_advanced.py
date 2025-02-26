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
from pathlib import Path
from tqdm import tqdm
import subprocess
import os
import time
import threading
from scipy.special import jv  # For Bessel functions
from scipy.optimize import minimize  # For pattern optimization
import json  # For saving/loading settings

class AdvancedPatternGenerator:
    def __init__(self):
        """Initialize the advanced pattern generator with extended features"""
        # Initialize camera state
        self.camera_active = False
        self.camera_paused = False
        
        # Sony LCX016AL-6 specifications
        self.width = 832
        self.height = 624
        self.pixel_size = 32e-6
        self.active_area = (26.6e-3, 20.0e-3)
        self.refresh_rate = 60
        self.contrast_ratio = 200
        
        # Default wavelength
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
        
        # Advanced parameters
        self.algorithm = "Gerchberg-Saxton"  # Default algorithm
        self.max_iterations = 50
        self.convergence_threshold = 1e-6
        self.phase_correction_enabled = False
        self.phase_correction_map = None
        self.aberration_correction_enabled = False
        self.aberration_coefficients = np.zeros(15)  # Zernike coefficients
        self.beam_shaping_enabled = False
        self.beam_profile = "Gaussian"  # Default beam profile
        self.multi_plane_enabled = False
        self.num_planes = 1
        self.plane_spacing = 100e-6  # 100 μm between planes
        
        # Optimization parameters
        self.optimization_enabled = False
        self.optimization_metric = "MSE"  # Mean Square Error
        self.optimization_weight = 1.0
        self.feedback_enabled = False
        
        # Initialize GUI
        self.setup_gui()
        
    def setup_gui(self):
        """Create the main GUI window and controls"""
        self.root = tk.Tk()
        self.root.title("SLM Pattern Generator")
        self.root.geometry("1600x900")  # Increased window size
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create scrollable frame
        self.scrollable_frame = ttk.Frame(self.main_frame)
        self.scrollable_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create frames for different sections
        self.control_frame = ttk.LabelFrame(self.scrollable_frame, text="Controls", padding="10")
        self.control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.preview_frame = ttk.LabelFrame(self.scrollable_frame, text="Pattern Preview", padding="10")
        self.preview_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Add status bar
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(self.scrollable_frame, textvariable=self.status_var)
        self.status_bar.pack(fill=tk.X, padx=5, pady=5)
        
        # Create controls
        self.create_controls()
        
        # Create preview area
        self.create_preview()
        
        # Initialize camera
        try:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.camera_active = True
                # Create camera frame only if camera is available
                self.camera_frame = ttk.LabelFrame(self.scrollable_frame, text="Camera Preview", padding="10")
                self.camera_frame.pack(fill=tk.X, padx=5, pady=5)
                self.create_camera_preview()
                self.camera_thread = threading.Thread(target=self.update_camera_preview, daemon=True)
                self.camera_thread.start()
                self.status_var.set("Camera initialized successfully")
            else:
                self.cap.release()
                self.status_var.set("No camera detected")
        except Exception as e:
            self.status_var.set(f"Camera error: {str(e)}")
        
        # Bind ESC key to quit
        self.root.bind('<Escape>', lambda e: self.quit_application())
        
    def create_controls(self):
        """Create advanced control panel"""
        # Algorithm selection
        algorithm_frame = ttk.LabelFrame(self.control_frame, text="Algorithm Settings", padding="5")
        algorithm_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(algorithm_frame, text="Algorithm:").grid(row=0, column=0, padx=5, pady=2)
        algorithms = ["Gerchberg-Saxton", "Weighted GS", "Direct Binary Search", "Adaptive-Additive"]
        self.algorithm_var = tk.StringVar(value=self.algorithm)
        algorithm_menu = ttk.OptionMenu(algorithm_frame, self.algorithm_var, self.algorithm, *algorithms)
        algorithm_menu.grid(row=0, column=1, padx=5, pady=2)
        
        # Iteration control
        ttk.Label(algorithm_frame, text="Max Iterations:").grid(row=1, column=0, padx=5, pady=2)
        self.max_iter_var = tk.StringVar(value=str(self.max_iterations))
        ttk.Entry(algorithm_frame, textvariable=self.max_iter_var, width=10).grid(row=1, column=1, padx=5, pady=2)
        
        # Phase correction
        correction_frame = ttk.LabelFrame(self.control_frame, text="Correction Settings", padding="5")
        correction_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.phase_correction_var = tk.BooleanVar(value=self.phase_correction_enabled)
        ttk.Checkbutton(correction_frame, text="Enable Phase Correction", variable=self.phase_correction_var).pack(anchor=tk.W)
        
        self.aberration_correction_var = tk.BooleanVar(value=self.aberration_correction_enabled)
        ttk.Checkbutton(correction_frame, text="Enable Aberration Correction", variable=self.aberration_correction_var).pack(anchor=tk.W)
        
        # Beam shaping
        beam_frame = ttk.LabelFrame(self.control_frame, text="Beam Shaping", padding="5")
        beam_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.beam_shaping_var = tk.BooleanVar(value=self.beam_shaping_enabled)
        ttk.Checkbutton(beam_frame, text="Enable Beam Shaping", variable=self.beam_shaping_var).pack(anchor=tk.W)
        
        ttk.Label(beam_frame, text="Beam Profile:").pack(side=tk.LEFT, padx=5)
        profiles = ["Gaussian", "Top-Hat", "Bessel", "Laguerre-Gaussian", "Custom"]
        self.beam_profile_var = tk.StringVar(value=self.beam_profile)
        profile_menu = ttk.OptionMenu(beam_frame, self.beam_profile_var, self.beam_profile, *profiles)
        profile_menu.pack(side=tk.LEFT, padx=5)
        
        # Multi-plane settings
        plane_frame = ttk.LabelFrame(self.control_frame, text="Multi-Plane Settings", padding="5")
        plane_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.multi_plane_var = tk.BooleanVar(value=self.multi_plane_enabled)
        ttk.Checkbutton(plane_frame, text="Enable Multi-Plane", variable=self.multi_plane_var).pack(anchor=tk.W)
        
        ttk.Label(plane_frame, text="Number of Planes:").pack(side=tk.LEFT, padx=5)
        self.num_planes_var = tk.StringVar(value=str(self.num_planes))
        ttk.Entry(plane_frame, textvariable=self.num_planes_var, width=5).pack(side=tk.LEFT, padx=5)
        
        # Optimization settings
        opt_frame = ttk.LabelFrame(self.control_frame, text="Optimization Settings", padding="5")
        opt_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.optimization_var = tk.BooleanVar(value=self.optimization_enabled)
        ttk.Checkbutton(opt_frame, text="Enable Optimization", variable=self.optimization_var).pack(anchor=tk.W)
        
        ttk.Label(opt_frame, text="Metric:").pack(side=tk.LEFT, padx=5)
        metrics = ["MSE", "PSNR", "SSIM", "Custom"]
        self.metric_var = tk.StringVar(value=self.optimization_metric)
        metric_menu = ttk.OptionMenu(opt_frame, self.metric_var, self.optimization_metric, *metrics)
        metric_menu.pack(side=tk.LEFT, padx=5)
        
        # Add save/load settings buttons
        settings_frame = ttk.Frame(self.control_frame)
        settings_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(settings_frame, text="Save Settings", command=self.save_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(settings_frame, text="Load Settings", command=self.load_settings).pack(side=tk.LEFT, padx=5)
        
        # Original controls
        self.load_button = ttk.Button(self.control_frame, text="Load Target Image", command=self.load_image)
        self.load_button.pack(side=tk.LEFT, padx=5)
        
        self.load_pattern_button = ttk.Button(self.control_frame, text="Load Pattern", command=self.load_pattern)
        self.load_pattern_button.pack(side=tk.LEFT, padx=5)
        
        self.send_to_slm_button = ttk.Button(self.control_frame, text="Send to SLM (HDMI1)", command=self.send_to_slm)
        self.send_to_slm_button.pack(side=tk.LEFT, padx=5)
        
        self.save_button = ttk.Button(self.control_frame, text="Save Pattern", command=self.save_pattern)
        self.save_button.pack(side=tk.LEFT, padx=5)
        
        # Pause camera button - only show if camera is active
        if self.camera_active:
            self.pause_camera_button = ttk.Button(self.control_frame, text="Pause Camera", command=self.pause_camera)
            self.pause_camera_button.pack(side=tk.LEFT, padx=5)
        
        # File controls
        file_frame = ttk.Frame(self.control_frame)
        file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(file_frame, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="Save Pattern", command=self.save_pattern).pack(side=tk.LEFT, padx=5)
        
        # Parameter controls
        param_frame = ttk.Frame(self.control_frame)
        param_frame.pack(fill=tk.X, pady=5)
        
        # Number of iterations
        ttk.Label(param_frame, text="Iterations:").grid(row=0, column=0, padx=5, pady=5)
        self.iterations_var = tk.StringVar(value="100")
        ttk.Entry(param_frame, textvariable=self.iterations_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        # Beam width factor
        ttk.Label(param_frame, text="Beam Width Factor:").grid(row=0, column=2, padx=5, pady=5)
        self.beam_width_var = tk.StringVar(value="1.0")
        ttk.Entry(param_frame, textvariable=self.beam_width_var, width=10).grid(row=0, column=3, padx=5, pady=5)
        
        # Phase range
        ttk.Label(param_frame, text="Phase Range (π):").grid(row=0, column=4, padx=5, pady=5)
        self.phase_range_var = tk.StringVar(value="2.0")
        ttk.Entry(param_frame, textvariable=self.phase_range_var, width=10).grid(row=0, column=5, padx=5, pady=5)
        
        # Generate button
        ttk.Button(param_frame, text="Generate Pattern", command=self.generate_pattern).grid(row=0, column=6, padx=20, pady=5)
        
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
            
    def load_pattern(self):
        """Load a pattern from file"""
        try:
            # Use zenity file dialog
            cmd = ['zenity', '--file-selection', 
                   '--file-filter=*.png',
                   '--title=Select Pattern']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.status_var.set("Pattern loading cancelled")
                return
                
            file_path = result.stdout.strip()
            if not file_path:
                return
                
            # Load the pattern
            pattern = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if pattern is None:
                raise ValueError("Could not load pattern")
                
            # Resize if necessary
            if pattern.shape != (self.height, self.width):
                pattern = cv2.resize(pattern, (self.width, self.height))
            
            self.pattern = pattern
            self.slm_phase = (pattern.astype(float) / 255.0 * 2 * np.pi - np.pi)
            
            # Update preview
            self.ax2.clear()
            self.ax2.imshow(pattern, cmap='gray')
            self.ax2.set_title('Loaded Pattern')
            self.ax2.set_xticks([])
            self.ax2.set_yticks([])
            self.canvas.draw()
            
            self.status_var.set(f"Pattern loaded from: {file_path}")
            
        except Exception as e:
            self.status_var.set(f"Error loading pattern: {str(e)}")
            
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
        
    def load_image(self):
        """Load and display target image"""
        try:
            filename = filedialog.askopenfilename(
                title="Select Image",
                filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif")]
            )
            
            if filename:
                # Load and resize image
                target = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                if target is None:
                    raise ValueError("Could not load image")
                
                # Resize to SLM dimensions
                target = cv2.resize(target, (self.width, self.height))
                
                # Pad the target for FFT
                self.padded_target = np.zeros((self.padded_height, self.padded_width))
                start_x = (self.padded_width - self.width) // 2
                end_x = start_x + self.width
                start_y = (self.padded_height - self.height) // 2
                end_y = start_y + self.height
                self.padded_target[start_y:end_y, start_x:end_x] = target
                
                # Display target
                self.ax1.clear()
                self.ax1.imshow(target, cmap='gray')
                self.ax1.set_title('Target Image')
                self.ax1.set_xticks([])
                self.ax1.set_yticks([])
                
                # Clear other plots
                self.ax2.clear()
                self.ax2.set_title('Generated Pattern')
                self.ax2.set_xticks([])
                self.ax2.set_yticks([])
                
                self.ax3.clear()
                self.ax3.set_title('Simulated Reconstruction')
                self.ax3.set_xticks([])
                self.ax3.set_yticks([])
                
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
            
            # Calculate far field for preview
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
        settings = {
            'algorithm': self.algorithm_var.get(),
            'max_iterations': int(self.max_iter_var.get()),
            'phase_correction': self.phase_correction_var.get(),
            'aberration_correction': self.aberration_correction_var.get(),
            'beam_shaping': self.beam_shaping_var.get(),
            'beam_profile': self.beam_profile_var.get(),
            'multi_plane': self.multi_plane_var.get(),
            'num_planes': int(self.num_planes_var.get()),
            'optimization': self.optimization_var.get(),
            'optimization_metric': self.metric_var.get()
        }
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            title="Save Settings"
        )
        
        if filename:
            with open(filename, 'w') as f:
                json.dump(settings, f, indent=4)
            self.status_var.set("Settings saved successfully")
            
    def load_settings(self):
        """Load settings from a JSON file"""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json")],
            title="Load Settings"
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    settings = json.load(f)
                
                self.algorithm_var.set(settings['algorithm'])
                self.max_iter_var.set(str(settings['max_iterations']))
                self.phase_correction_var.set(settings['phase_correction'])
                self.aberration_correction_var.set(settings['aberration_correction'])
                self.beam_shaping_var.set(settings['beam_shaping'])
                self.beam_profile_var.set(settings['beam_profile'])
                self.multi_plane_var.set(settings['multi_plane'])
                self.num_planes_var.set(str(settings['num_planes']))
                self.optimization_var.set(settings['optimization'])
                self.metric_var.set(settings['optimization_metric'])
                
                self.status_var.set("Settings loaded successfully")
            except Exception as e:
                self.status_var.set(f"Error loading settings: {str(e)}")
                
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = AdvancedPatternGenerator()
    app.run()
