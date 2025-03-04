"""
Pattern Generator for Sony LCX016AL-6 SLM - Raspberry Pi Version
Generates phase patterns using Gerchberg-Saxton algorithm for far-field image reconstruction.

SLM Specifications:
- Resolution: 832 x 624 pixels
- Pixel Pitch: 32 μm
- Active Area: 26.6 x 20.0 mm
- Refresh Rate: 60 Hz
- Contrast Ratio: 200:1
- Default Wavelength: 532 nm (green laser)
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk
from pathlib import Path
from tqdm import tqdm
import subprocess
import os
import time
import threading
from tkinter import filedialog
import pygame

class PatternGenerator:
    def __init__(self):
        """Initialize the pattern generator with SLM specifications"""
        # Initialize pygame
        pygame.init()
        
        # Sony LCX016AL-6 specifications
        self.width = 832  # width in pixels
        self.height = 624  # height in pixels
        self.pixel_size = 32e-6  # 32 μm pixel pitch
        self.active_area = (26.6e-3, 20.0e-3)  # 26.6mm x 20.0mm active area
        self.refresh_rate = 60  # Hz
        self.contrast_ratio = 200  # 200:1
        
        # Default wavelength (will be adjustable)
        self.wavelength = 532e-9  # 532nm green laser
        
        # Initialize camera state
        self.camera_active = False
        self.camera_paused = False
        
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
        """Create parameter control widgets"""
        # Add buttons frame
        buttons_frame = ttk.Frame(self.control_frame)
        buttons_frame.pack(fill=tk.X, pady=5)
        
        # Load target image button
        self.load_button = ttk.Button(buttons_frame, text="Load Target Image", command=self.load_image)
        self.load_button.pack(side=tk.LEFT, padx=5)
        
        # Load pattern button
        self.load_pattern_button = ttk.Button(buttons_frame, text="Load Pattern", command=self.load_pattern)
        self.load_pattern_button.pack(side=tk.LEFT, padx=5)
        
        # Send to SLM button
        self.send_to_slm_button = ttk.Button(buttons_frame, text="Send to SLM (HDMI-A-2)", command=self.send_to_slm)
        self.send_to_slm_button.pack(side=tk.LEFT, padx=5)
        
        # Save pattern button
        self.save_button = ttk.Button(buttons_frame, text="Save Pattern", command=self.save_pattern)
        self.save_button.pack(side=tk.LEFT, padx=5)
        
        # Pause camera button - only show if camera is active
        if self.camera_active:
            self.pause_camera_button = ttk.Button(buttons_frame, text="Pause Camera", command=self.pause_camera)
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
            
    def detect_displays(self):
        """Detect and print information about available displays"""
        try:
            if not pygame.display.get_init():
                pygame.display.init()
            num_displays = pygame.display.get_num_displays()
            sizes = pygame.display.get_desktop_sizes()
            print(f"Number of displays: {num_displays}")
            print(f"Display sizes: {sizes}")
            return num_displays, sizes
        except Exception as e:
            print(f"Error detecting displays: {e}")
            return 0, []

    def send_to_slm(self):
        """Send pattern to SLM via HDMI-A-2"""
        if not hasattr(self, 'pattern'):
            self.status_var.set("No pattern to display. Generate or load a pattern first.")
            return
        
        try:
            # Force reinitialize pygame display
            if pygame.display.get_init():
                pygame.display.quit()
            pygame.display.init()
            
            # Set SDL environment variables for display control
            os.environ['SDL_VIDEO_WINDOW_POS'] = '1280,0'  # Position at main monitor width
            os.environ['SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS'] = '0'
            
            # Get display info
            print(f"Number of displays: {pygame.display.get_num_displays()}")
            for i in range(pygame.display.get_num_displays()):
                info = pygame.display.get_desktop_sizes()[i]
                print(f"Display {i}: {info}")
            
            # Create window on second display
            slm_window = pygame.display.set_mode(
                (self.width, self.height),
                pygame.NOFRAME,
                display=1
            )
            
            # Create and show pattern
            pattern_surface = pygame.Surface((self.width, self.height), depth=8)
            pattern_surface.set_palette([(i, i, i) for i in range(256)])
            pygame.surfarray.pixels2d(pattern_surface)[:] = self.pattern.T
            
            # Clear window and display pattern
            slm_window.fill((0, 0, 0))
            slm_window.blit(pattern_surface, (0, 0))
            pygame.display.flip()
            
            self.status_var.set("Pattern sent to HDMI-A-2. Press ESC to close.")
            print("Pattern displayed. Press ESC to close.")
            
            # Wait for ESC
            while True:
                event = pygame.event.wait()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    break
                elif event.type == pygame.QUIT:
                    break
            
            # Cleanup
            pygame.display.quit()
            pygame.display.init()
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            print(f"Detailed error: {str(e)}")
            # Try to recover display
            pygame.display.quit()
            pygame.display.init()
            
    def quit_application(self):
        """Clean up and quit the application"""
        if self.camera_active:
            self.camera_active = False
            self.cap.release()
        cv2.destroyAllWindows()
        self.root.quit()
        
    def load_image(self):
        """Load an image file and convert it to grayscale pattern"""
        filetypes = [
            ('All Image Files', '*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff'),
            ('PNG Files', '*.png'),
            ('JPEG Files', '*.jpg;*.jpeg'),
            ('Bitmap Files', '*.bmp'),
            ('TIFF Files', '*.tif;*.tiff'),
            ('All Files', '*.*')
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=filetypes
        )
        
        if filename:
            try:
                # Read image using cv2
                img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise ValueError("Failed to load image")
                
                # Resize if needed
                if img.shape != (self.height, self.width):
                    img = cv2.resize(img, (self.width, self.height))
                
                self.pattern = img
                self.update_preview()
                self.status_var.set(f"Loaded and converted image: {os.path.basename(filename)}")
                
            except Exception as e:
                self.status_var.set(f"Error loading image: {str(e)}")
                print(f"Detailed error: {str(e)}")
                
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
            
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = PatternGenerator()
    app.run()
