import cv2
import numpy as np
import pygame
import os
from PIL import Image
import threading
from pathlib import Path
import time
import subprocess

class Button:
    def __init__(self, x, y, width, height, text, font, color=(128, 128, 128)):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.font = font
        self.color = color
        self.is_hovered = False
        self.is_clicked = False
        
    def draw(self, surface):
        color = (min(self.color[0] + 20, 255), min(self.color[1] + 20, 255), min(self.color[2] + 20, 255)) if self.is_hovered else self.color
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, (200, 200, 200), self.rect, 2)
        
        text_surface = self.font.render(self.text, True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)
        
    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.is_hovered:
                self.is_clicked = True
                return True
        return False

class SLMController:
    def __init__(self):
        # SLM Parameters for SONY LCX016AL-6
        self.width = 832  # width in pixels
        self.height = 624  # height in pixels
        self.slm_resolution = (self.width, self.height)
        self.pixel_pitch = 32  # μm
        self.refresh_rate = 60  # Hz
        self.contrast_ratio = 200
        self.active_area = (26.6, 20.0)  # mm
        
        # Calibration parameters
        self.calibration_dir = Path('calibration')
        self.calibration_dir.mkdir(exist_ok=True)
        self.lut_file = self.calibration_dir / 'gamma_lut.npy'
        self.flatness_file = self.calibration_dir / 'flatness_correction.npy'
        self.is_calibrated = False
        self.load_calibration()
        
        # Initialize pygame
        pygame.init()
        pygame.font.init()
        
        # Get monitor info
        info = pygame.display.Info()
        self.screen_width = info.current_w
        self.screen_height = info.current_h
        
        # Calculate preview sizes (40% and 30% of screen height)
        preview_height = int(self.screen_height * 0.4)
        preview_width = int(preview_height * self.width / self.height)
        camera_height = int(self.screen_height * 0.3)
        camera_width = int(camera_height * 4 / 3)  # 4:3 aspect ratio
        
        # Create main window
        self.control_display = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption('SLM Control Interface')
        
        # Create SLM window
        os.environ['SDL_VIDEO_WINDOW_POS'] = f'{self.screen_width - self.width - 50},50'
        self.slm_window = pygame.display.set_mode((self.width, self.height), pygame.NOFRAME, display=0)
        pygame.display.set_caption('SLM Output')
        
        # Create output directories
        self.output_dir = Path('output')
        self.patterns_dir = Path('patterns')
        self.output_dir.mkdir(exist_ok=True)
        self.patterns_dir.mkdir(exist_ok=True)
        
        # Pattern management
        self.current_pattern = None
        
        # Calculate positions (centered horizontally)
        pattern_x = (self.screen_width - preview_width - camera_width - 100) // 3
        camera_x = pattern_x * 2 + preview_width
        
        # Preview surface
        self.preview_surface = pygame.Surface((preview_width, preview_height))
        self.preview_rect = pygame.Rect(pattern_x, 50, preview_width, preview_height)
        
        # Camera preview surface
        self.camera_surface = pygame.Surface((camera_width, camera_height))
        self.camera_rect = pygame.Rect(camera_x, 50, camera_width, camera_height)
        
        # Font sizes based on screen height
        self.title_font = pygame.font.SysFont(None, int(self.screen_height * 0.05))
        self.font = pygame.font.SysFont(None, int(self.screen_height * 0.03))
        
        # Button dimensions - make them slightly smaller and increase margins
        button_width = 120
        button_height = 35
        button_margin = 30  # increased margin from edges
        button_spacing = 10  # space between buttons
        
        # Calculate positions for bottom left buttons
        calibrate_x = button_margin * 2
        calibrate_y = self.screen_height - button_height - button_margin * 2
        load_pattern_y = calibrate_y - button_height - button_spacing
        
        # Create buttons
        # Pattern selection - positioned above calibrate button
        self.load_button = Button(calibrate_x, load_pattern_y, button_width, button_height, "Load Pattern", self.font)
        
        # Save buttons - use pattern_x for horizontal alignment
        self.save_preview_button = Button(pattern_x, preview_height + 60, button_width, button_height, "Save Pattern", self.font)
        self.save_camera_button = Button(camera_x, camera_height + 60, button_width, button_height, "Save Camera", self.font)
        
        # Camera control
        self.camera_paused = False
        self.pause_camera_button = Button(camera_x + button_width + 20, camera_height + 60, button_width, button_height, "Pause Camera", self.font)
        
        # Calibrate button (bottom left with increased margin)
        self.calibrate_button = Button(calibrate_x, calibrate_y, button_width, button_height, 'Calibrate', self.font, (100, 150, 100))
        
        # Initialize camera
        self.camera_active = False
        self.camera_paused = False
        try:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.camera_active = True
                self.camera_thread = threading.Thread(target=self.update_camera_preview, daemon=True)
                self.camera_thread.start()
        except:
            print("Warning: Could not initialize camera")
            self.camera_active = False
            
    def load_calibration(self):
        """Load calibration data if available"""
        try:
            if self.lut_file.exists() and self.flatness_file.exists():
                self.gamma_lut = np.load(str(self.lut_file))
                self.flatness_correction = np.load(str(self.flatness_file))
                self.is_calibrated = True
                print("Calibration data loaded successfully")
            else:
                self.gamma_lut = np.arange(256, dtype=np.uint8)  # Linear LUT
                self.flatness_correction = np.ones((self.height, self.width), dtype=np.float32)
                print("No calibration data found, using default values")
        except Exception as e:
            print(f"Error loading calibration: {e}")
            self.gamma_lut = np.arange(256, dtype=np.uint8)
            self.flatness_correction = np.ones((self.height, self.width), dtype=np.float32)
            
    def apply_calibration(self, pattern):
        """Apply gamma and flatness corrections to pattern"""
        if not self.is_calibrated:
            return pattern
            
        # Apply gamma correction
        corrected = self.gamma_lut[pattern]
        
        # Apply flatness correction
        corrected = (corrected.astype(np.float32) * self.flatness_correction).clip(0, 255).astype(np.uint8)
        return corrected
        
    def calibrate_gamma(self):
        """Calibrate the gamma curve using interferometric measurements"""
        print("Starting gamma calibration...")
        # Number of voltage levels to measure
        n_levels = 32
        voltage_levels = np.linspace(0, 255, n_levels, dtype=np.uint8)
        measured_phase = np.zeros(n_levels)
        
        for i, level in enumerate(voltage_levels):
            # Display uniform pattern
            pattern = np.full((self.height, self.width), level, dtype=np.uint8)
            
            # Wait for user to measure phase
            input(f"Measuring phase for voltage level {level}. Press Enter after recording the measurement...")
            phase = float(input("Enter the measured phase shift (in radians): "))
            measured_phase[i] = phase
            
        # Fit the phase response curve
        target_phase = np.linspace(0, 2*np.pi, 256)
        self.gamma_lut = np.interp(target_phase, measured_phase, voltage_levels).astype(np.uint8)
        
        # Save calibration
        np.save(str(self.lut_file), self.gamma_lut)
        print("Gamma calibration completed and saved")
        
    def calibrate_flatness(self):
        """Calibrate for surface non-uniformity using interferometric measurements"""
        print("Starting flatness calibration...")
        
        # Display uniform pattern
        level = 128
        pattern = np.full((self.height, self.width), level, dtype=np.uint8)
        self.display_pattern_raw(pattern)
        
        # Wait for user to capture interferogram
        input("Capture interferogram of uniform pattern. Press Enter when ready...")
        
        # In a real implementation, you would:
        # 1. Capture interferogram using camera
        # 2. Process interferogram to extract phase map
        # 3. Calculate correction map
        
        # For now, we'll use a simple gradient correction as placeholder
        x = np.linspace(-1, 1, self.width)
        y = np.linspace(-1, 1, self.height)
        X, Y = np.meshgrid(x, y)
        r = np.sqrt(X**2 + Y**2)
        self.flatness_correction = 1 / (1 + 0.1*r)  # Simple radial correction
        
        # Save calibration
        np.save(str(self.flatness_file), self.flatness_correction)
        print("Flatness calibration completed and saved")
        
    def start_calibration(self):
        """Start the full calibration procedure"""
        print("\nStarting SLM calibration procedure...")
        print("This will require interferometric measurements.")
        print("Make sure you have set up:")
        print("1. Interferometer")
        print("2. Camera for capturing interferograms")
        print("3. Laser at your operating wavelength")
        
        if input("\nAre you ready to proceed? (y/n): ").lower() == 'y':
            self.calibrate_gamma()
            self.calibrate_flatness()
            self.is_calibrated = True
            print("\nCalibration completed!")
        else:
            print("Calibration cancelled")
            
    def display_pattern_raw(self, pattern):
        """Display pattern without applying calibration"""
        pattern = pattern.astype(np.uint8)
        slm_surface = pygame.Surface((self.width, self.height), depth=8)
        palette = [(i, i, i) for i in range(256)]
        slm_surface.set_palette(palette)
        pygame.surfarray.pixels2d(slm_surface)[:] = pattern.T
        self.slm_window.fill((128, 128, 128))
        self.slm_window.blit(slm_surface, (0, 0))
        pygame.display.update()
        
    def display_pattern(self, pattern_name):
        """Display a pattern on the SLM and preview"""
        pattern_path = self.patterns_dir / f'{pattern_name}.png'
        if pattern_path.exists():
            # Load pattern in grayscale mode
            pattern = cv2.imread(str(pattern_path), cv2.IMREAD_GRAYSCALE)
            
            # Ensure pattern matches SLM dimensions
            if pattern.shape != (self.height, self.width):
                pattern = cv2.resize(pattern, (self.width, self.height))
            
            self.current_pattern = pattern_name
            pattern = pattern.astype(np.uint8)
            
            # Apply calibration if available
            pattern = self.apply_calibration(pattern)
            
            # Create grayscale surface for SLM
            slm_surface = pygame.Surface((self.width, self.height), depth=8)
            palette = [(i, i, i) for i in range(256)]
            slm_surface.set_palette(palette)
            pygame.surfarray.pixels2d(slm_surface)[:] = pattern.T
            
            # Display on SLM window
            self.slm_window.fill((128, 128, 128))
            self.slm_window.blit(slm_surface, (0, 0))
            
            # Create RGB preview (for display only)
            preview_surface = pygame.Surface((self.width, self.height))
            preview_array = np.dstack((pattern, pattern, pattern))
            pygame.surfarray.pixels3d(preview_surface)[:] = np.transpose(preview_array, (1, 0, 2))
            
            # Scale and display preview
            preview_pattern = pygame.transform.scale(preview_surface, (self.preview_surface.get_width(), self.preview_surface.get_height()))
            self.preview_surface.blit(preview_pattern, (0, 0))
            
            pygame.display.update()
        else:
            print(f"Pattern {pattern_name} not found, creating default patterns...")
            self.create_default_patterns()
            self.display_pattern(pattern_name)
            
    def create_default_patterns(self):
        """Create a set of default patterns in the patterns directory"""
        patterns = {
            'blank': np.zeros(self.slm_resolution),
            'binary_grating_vertical': create_binary_grating(self.slm_resolution, period=32, orientation='vertical'),
            'binary_grating_horizontal': create_binary_grating(self.slm_resolution, period=32, orientation='horizontal'),
            'binary_grating_diagonal': create_binary_grating(self.slm_resolution, period=32, orientation='diagonal'),
            'sinusoidal_grating_vertical': create_sinusoidal_grating(self.slm_resolution, period=32, orientation='vertical'),
            'sinusoidal_grating_horizontal': create_sinusoidal_grating(self.slm_resolution, period=32, orientation='horizontal'),
            'sinusoidal_grating_diagonal': create_sinusoidal_grating(self.slm_resolution, period=32, orientation='diagonal'),
            'blazed_grating_vertical': create_blazed_grating(self.slm_resolution, period=32, orientation='vertical'),
            'blazed_grating_horizontal': create_blazed_grating(self.slm_resolution, period=32, orientation='horizontal'),
            'checkerboard_16px': create_checkerboard(self.slm_resolution, square_size=16),
            'checkerboard_32px': create_checkerboard(self.slm_resolution, square_size=32),
            'checkerboard_64px': create_checkerboard(self.slm_resolution, square_size=64),
            'circle_r50': create_circle(self.slm_resolution, radius=50),
            'circle_r100': create_circle(self.slm_resolution, radius=100),
            'circle_r200': create_circle(self.slm_resolution, radius=200),
            'ring_r100': create_ring(self.slm_resolution, radius=100, width=10),
            'ring_r200': create_ring(self.slm_resolution, radius=200, width=20),
            'fresnel_lens_f200mm': create_fresnel_lens(self.slm_resolution, focal_length=200),
            'fresnel_lens_f500mm': create_fresnel_lens(self.slm_resolution, focal_length=500),
            'fresnel_lens_f1000mm': create_fresnel_lens(self.slm_resolution, focal_length=1000),
            'vortex_l1': create_vortex(self.slm_resolution, charge=1),
            'vortex_l2': create_vortex(self.slm_resolution, charge=2),
            'vortex_l3': create_vortex(self.slm_resolution, charge=3)
        }
        
        # Save patterns
        for name, pattern in patterns.items():
            filepath = self.patterns_dir / f"{name}.png"
            cv2.imwrite(str(filepath), pattern)
            print(f"Generated pattern: {name}")
            
    def load_pattern(self):
        """Load a pattern using pcmanfm file dialog"""
        try:
            # Use pcmanfm file dialog
            cmd = ['pcmanfm', str(self.patterns_dir)]
            subprocess.Popen(cmd)
            
            # Wait for user to select file
            pattern_name = input("Enter the pattern filename (or press Enter to cancel): ")
            if pattern_name:
                if not pattern_name.endswith('.png'):
                    pattern_name += '.png'
                self.display_pattern(pattern_name)
                print(f"Loaded pattern: {pattern_name}")
            else:
                print("Load cancelled")
        except Exception as e:
            print(f"Error loading pattern: {e}")

    def save_pattern(self):
        """Save the current pattern preview with pcmanfm file dialog"""
        if self.current_pattern:
            try:
                # Get timestamp for default filename
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                default_filename = f'pattern_{self.current_pattern}_{timestamp}.png'
                
                # Open file manager to output directory
                cmd = ['pcmanfm', str(self.output_dir)]
                subprocess.Popen(cmd)
                
                # Get filename from user
                print(f"Default filename: {default_filename}")
                filename = input("Enter the filename to save (or press Enter to use default): ")
                if not filename:
                    filename = str(self.output_dir / default_filename)
                elif not filename.endswith('.png'):
                    filename = str(self.output_dir / (filename + '.png'))
                else:
                    filename = str(self.output_dir / filename)
                
                # Convert surface to PIL Image and save
                surface_string = pygame.image.tostring(self.preview_surface, 'RGB')
                pil_image = Image.frombytes('RGB', self.preview_surface.get_size(), surface_string)
                pil_image.save(filename)
                print(f"Saved pattern preview to {filename}")
            except Exception as e:
                print(f"Error saving pattern: {e}")

    def save_camera(self):
        """Save the current camera image with pcmanfm file dialog"""
        if self.camera_active:
            try:
                # Get timestamp for default filename
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                default_filename = f'camera_{timestamp}.png'
                
                # Open file manager to output directory
                cmd = ['pcmanfm', str(self.output_dir)]
                subprocess.Popen(cmd)
                
                # Get filename from user
                print(f"Default filename: {default_filename}")
                filename = input("Enter the filename to save (or press Enter to use default): ")
                if not filename:
                    filename = str(self.output_dir / default_filename)
                elif not filename.endswith('.png'):
                    filename = str(self.output_dir / (filename + '.png'))
                else:
                    filename = str(self.output_dir / filename)
                
                # Convert surface to PIL Image and save
                surface_string = pygame.image.tostring(self.camera_surface, 'RGB')
                pil_image = Image.frombytes('RGB', self.camera_surface.get_size(), surface_string)
                pil_image.save(filename)
                print(f"Saved camera image to {filename}")
            except Exception as e:
                print(f"Error saving camera image: {e}")

    def update_camera_preview(self):
        """Update camera preview if camera is active and not paused"""
        if self.camera_active and not self.camera_paused:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.camera_surface.get_width(), self.camera_surface.get_height()))
                pygame_frame = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                self.camera_surface.blit(pygame_frame, (0, 0))

    def toggle_camera_pause(self):
        """Toggle camera pause state"""
        self.camera_paused = not self.camera_paused
        self.pause_camera_button.text = "Resume Camera" if self.camera_paused else "Pause Camera"

    def run(self):
        """Main application loop"""
        running = True
        clock = pygame.time.Clock()
        
        print("Starting SLM Control Interface")
        print("Press ESC to exit")
        print("Available patterns will be loaded from the patterns directory")
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_F11:
                        pygame.display.toggle_fullscreen()
                
                # Handle button clicks
                if self.load_button.handle_event(event):
                    self.load_pattern()
                elif self.save_preview_button.handle_event(event):
                    self.save_pattern()
                elif self.save_camera_button.handle_event(event):
                    self.save_camera()
                elif self.pause_camera_button.handle_event(event):
                    self.toggle_camera_pause()
                elif self.calibrate_button.handle_event(event):
                    self.start_calibration()
            
            # Clear control display
            self.control_display.fill((0, 0, 0))
            
            # Draw buttons
            self.load_button.draw(self.control_display)
            self.save_preview_button.draw(self.control_display)
            self.save_camera_button.draw(self.control_display)
            self.pause_camera_button.draw(self.control_display)
            self.calibrate_button.draw(self.control_display)
            
            # Draw current pattern name
            if self.current_pattern:
                pattern_text = self.font.render(f'Current Pattern: {self.current_pattern}', True, (255, 255, 255))
                text_rect = pattern_text.get_rect(centerx=self.preview_rect.centerx, y=20)
                self.control_display.blit(pattern_text, text_rect)
            
            # Update and draw preview surfaces
            self.control_display.blit(self.preview_surface, self.preview_rect)
            if self.camera_active:
                self.control_display.blit(self.camera_surface, self.camera_rect)
            
            pygame.display.flip()
            clock.tick(60)
        
        self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        if self.camera_active:
            self.cap.release()
        pygame.quit()

def create_binary_grating(resolution, period, orientation):
    height, width = resolution
    if orientation == 'vertical':
        pattern = np.zeros((height, width), dtype=np.uint8)
        pattern[:, ::period] = 255
    elif orientation == 'horizontal':
        pattern = np.zeros((height, width), dtype=np.uint8)
        pattern[::period, :] = 255
    elif orientation == 'diagonal':
        pattern = np.zeros((height, width), dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                if (i + j) % period < period // 2:
                    pattern[i, j] = 255
    return pattern

def create_sinusoidal_grating(resolution, period, orientation):
    height, width = resolution
    if orientation == 'vertical':
        x = np.linspace(0, 2 * np.pi, width)
        pattern = (np.sin(x) + 1) * 127.5
        pattern = np.tile(pattern, (height, 1))
    elif orientation == 'horizontal':
        y = np.linspace(0, 2 * np.pi, height)
        pattern = (np.sin(y) + 1) * 127.5
        pattern = np.tile(pattern[:, np.newaxis], (1, width))
    elif orientation == 'diagonal':
        x = np.linspace(0, 2 * np.pi, width)
        y = np.linspace(0, 2 * np.pi, height)
        X, Y = np.meshgrid(x, y)
        pattern = (np.sin(X + Y) + 1) * 127.5
    return pattern.astype(np.uint8)

def create_blazed_grating(resolution, period, orientation):
    height, width = resolution
    if orientation == 'vertical':
        pattern = np.zeros((height, width), dtype=np.uint8)
        for j in range(width):
            pattern[:, j] = (j % period) * 255 // period
    elif orientation == 'horizontal':
        pattern = np.zeros((height, width), dtype=np.uint8)
        for i in range(height):
            pattern[i, :] = (i % period) * 255 // period
    return pattern

def create_checkerboard(resolution, square_size):
    height, width = resolution
    pattern = np.zeros((height, width), dtype=np.uint8)
    for i in range(0, height, square_size):
        for j in range(0, width, square_size):
            if (i // square_size + j // square_size) % 2 == 0:
                pattern[i:i+square_size, j:j+square_size] = 255
    return pattern

def create_circle(resolution, radius):
    height, width = resolution
    center_y, center_x = height // 2, width // 2
    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    pattern = np.zeros((height, width), dtype=np.uint8)
    pattern[dist_from_center <= radius] = 255
    return pattern

def create_ring(resolution, radius, width):
    height, width = resolution
    center_y, center_x = height // 2, width // 2
    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    pattern = np.zeros((height, width), dtype=np.uint8)
    pattern[(dist_from_center >= radius - width) & (dist_from_center <= radius + width)] = 255
    return pattern

def create_fresnel_lens(resolution, focal_length):
    height, width = resolution
    center_y, center_x = height // 2, width // 2
    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    wavelength = 0.000633  # mm (633nm)
    pattern = np.exp(1j * np.pi * dist_from_center**2 / (wavelength * focal_length))
    return ((np.angle(pattern) + np.pi) * 255 / (2 * np.pi)).astype(np.uint8)

def create_vortex(resolution, charge):
    height, width = resolution
    center_y, center_x = height // 2, width // 2
    Y, X = np.ogrid[:height, :width]
    angle = np.arctan2(Y - center_y, X - center_x)
    pattern = ((angle * charge / (2 * np.pi) + 0.5) * 255).astype(np.uint8)
    return pattern

if __name__ == "__main__":
    controller = SLMController()
    controller.run()
