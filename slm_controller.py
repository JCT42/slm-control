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
        self.show_pattern_list = False
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
        
        # Button dimensions
        button_width = 150
        button_height = 40
        
        # Save buttons
        self.save_preview_button = Button(pattern_x, preview_height + 60, button_width, button_height, "Save Pattern", self.font)
        self.save_camera_button = Button(camera_x, camera_height + 60, button_width, button_height, "Save Camera", self.font)
        
        # Camera control
        self.camera_paused = False
        self.pause_camera_button = Button(camera_x + button_width + 20, camera_height + 60, button_width, button_height, "Pause Camera", self.font)
        
        # Pattern selection
        self.pattern_button = Button(10, 50, button_width, button_height, "Select Pattern", self.font)
        
        # Create pattern buttons
        self.pattern_buttons = []
        patterns = ['blank', 'binary_grating', 'sinusoidal_grating', 'blazed_grating', 'checkerboard', 'circular_aperture', 'lens']
        y = 100  # Starting y position for pattern buttons
        for pattern in patterns:
            btn = Button(10, y, button_width, button_height, pattern, self.font)
            self.pattern_buttons.append(btn)
            y += button_height + 5  # Space between buttons
            
        # Calibrate button (bottom left)
        calibrate_x = 20
        calibrate_y = self.screen_height - button_height - 20
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
            self.display_pattern_raw(pattern)
            
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
        """Create some default SLM patterns"""
        # Create patterns in the correct orientation (832x624)
        width, height = 832, 624  # Fixed dimensions for SLM
        
        # Create blank pattern
        blank = np.ones((height, width), dtype=np.uint8) * 128
        
        # Create binary grating
        x = np.arange(width)
        binary_grating = np.zeros((height, width), dtype=np.uint8)
        period = 20  # pixels
        binary_grating[:, x % period < period/2] = 255
        
        # Create sinusoidal grating
        x = np.arange(width)
        y = np.arange(height)
        X, Y = np.meshgrid(x, y)  # Use default indexing for image coordinates
        period = 40  # pixels
        sinusoidal_grating = np.sin(2 * np.pi * X / period)
        sinusoidal_grating = ((sinusoidal_grating + 1) * 127.5).astype(np.uint8)
        
        # Create blazed grating
        x = np.arange(width)
        blazed_grating = np.zeros((height, width), dtype=np.uint8)
        period = 40  # pixels
        for j in range(width):
            blazed_grating[:, j] = (j % period) * 255 / period
        
        # Create checkerboard
        checkerboard = np.zeros((height, width), dtype=np.uint8)
        square_size = 32  # pixels
        for i in range(0, height, square_size):
            for j in range(0, width, square_size):
                if (i//square_size + j//square_size) % 2 == 0:
                    checkerboard[i:i+square_size, j:j+square_size] = 255
        
        # Create circular aperture
        Y, X = np.ogrid[:height, :width]
        center = (height//2, width//2)
        radius = min(height, width)//4
        dist_from_center = np.sqrt((X - center[1])**2 + (Y - center[0])**2)
        circular_aperture = np.zeros((height, width), dtype=np.uint8)
        circular_aperture[dist_from_center <= radius] = 255
        
        # Create Fresnel lens pattern
        Y, X = np.ogrid[:height, :width]
        center = (height//2, width//2)
        dist_from_center = np.sqrt((X - center[1])**2 + (Y - center[0])**2)
        focal_length = 500  # mm
        wavelength = 0.000633  # mm (633nm)
        lens = np.exp(1j * np.pi * dist_from_center**2 / (wavelength * focal_length))
        lens = ((np.angle(lens) + np.pi) * 255 / (2 * np.pi)).astype(np.uint8)
        
        # Save all patterns
        patterns = {
            'blank': blank,
            'binary_grating': binary_grating,
            'sinusoidal_grating': sinusoidal_grating,
            'blazed_grating': blazed_grating,
            'checkerboard': checkerboard,
            'circular_aperture': circular_aperture,
            'lens': lens
        }
        
        # Ensure patterns directory exists
        self.patterns_dir.mkdir(exist_ok=True)
        
        # Save patterns as grayscale PNGs
        for name, pattern in patterns.items():
            Image.fromarray(pattern, mode='L').save(self.patterns_dir / f'{name}.png')

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

    def save_preview_image(self):
        """Save the current pattern preview with file dialog"""
        if self.current_pattern:
            try:
                # Get timestamp for default filename
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                default_filename = f'pattern_{self.current_pattern}_{timestamp}.png'
                
                # Create a simple file dialog using zenity
                cmd = [
                    'zenity', '--file-selection',
                    '--save',
                    '--filename=' + str(Path.home() / default_filename),
                    '--file-filter=*.png',
                    '--title=Save Pattern Image'
                ]
                
                try:
                    filename = subprocess.check_output(cmd, text=True).strip()
                    if not filename.endswith('.png'):
                        filename += '.png'
                    
                    # Convert surface to PIL Image and save
                    surface_string = pygame.image.tostring(self.preview_surface, 'RGB')
                    pil_image = Image.frombytes('RGB', self.preview_surface.get_size(), surface_string)
                    pil_image.save(filename)
                    print(f"Saved pattern preview to {filename}")
                except subprocess.CalledProcessError:
                    print("Save cancelled")
            except Exception as e:
                print(f"Error saving pattern: {e}")

    def save_camera_image(self):
        """Save the current camera image with file dialog"""
        if self.camera_active:
            try:
                # Get timestamp for default filename
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                default_filename = f'camera_{timestamp}.png'
                
                # Create a simple file dialog using zenity
                cmd = [
                    'zenity', '--file-selection',
                    '--save',
                    '--filename=' + str(Path.home() / default_filename),
                    '--file-filter=*.png',
                    '--title=Save Camera Image'
                ]
                
                try:
                    filename = subprocess.check_output(cmd, text=True).strip()
                    if not filename.endswith('.png'):
                        filename += '.png'
                    
                    # Convert surface to PIL Image and save
                    surface_string = pygame.image.tostring(self.camera_surface, 'RGB')
                    pil_image = Image.frombytes('RGB', self.camera_surface.get_size(), surface_string)
                    pil_image.save(filename)
                    print(f"Saved camera image to {filename}")
                except subprocess.CalledProcessError:
                    print("Save cancelled")
            except Exception as e:
                print(f"Error saving camera image: {e}")

    def run(self):
        """Main application loop"""
        running = True
        clock = pygame.time.Clock()
        
        print("Starting SLM Control Interface")
        print("Press ESC to exit")
        print(f"Available patterns: ")
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_F11:  # Toggle fullscreen
                        pygame.display.toggle_fullscreen()
                
                # Handle pattern selection
                if self.pattern_button.handle_event(event):
                    self.show_pattern_list = not self.show_pattern_list
                
                # Handle pattern list buttons
                if self.show_pattern_list:
                    mouse_pos = pygame.mouse.get_pos()
                    for btn in self.pattern_buttons:
                        # Update hover state
                        btn.handle_event(pygame.event.Event(pygame.MOUSEMOTION, {'pos': mouse_pos}))
                        # Handle click
                        if btn.handle_event(event):
                            self.display_pattern(btn.text)
                            self.show_pattern_list = False
                
                # Handle save buttons
                if self.save_preview_button.handle_event(event):
                    self.save_preview_image()
                if self.save_camera_button.handle_event(event):
                    self.save_camera_image()
                
                # Handle camera pause button
                if self.pause_camera_button.handle_event(event):
                    self.toggle_camera_pause()
                
                # Handle calibrate button
                if self.calibrate_button.handle_event(event):
                    self.start_calibration()
            
            # Clear control display
            self.control_display.fill((0, 0, 0))
            
            # Draw UI
            title = self.title_font.render('SLM Control Interface', True, (255, 255, 255))
            title_rect = title.get_rect(centerx=self.screen_width//2, y=10)
            self.control_display.blit(title, title_rect)
            
            # Draw pattern selection button
            self.pattern_button.draw(self.control_display)
            
            # Draw pattern list if shown
            if hasattr(self, 'show_pattern_list') and self.show_pattern_list:
                for btn in self.pattern_buttons:
                    btn.draw(self.control_display)
            
            # Draw current pattern name
            if hasattr(self, 'current_pattern') and self.current_pattern:
                pattern_text = self.font.render(f'Current: {self.current_pattern}', True, (255, 255, 255))
                text_rect = pattern_text.get_rect(centerx=self.preview_rect.centerx, y=20)
                self.control_display.blit(pattern_text, text_rect)
            
            # Draw preview labels
            preview_label = self.font.render('Pattern Preview', True, (255, 255, 255))
            label_rect = preview_label.get_rect(centerx=self.preview_rect.centerx, bottom=self.preview_rect.top - 5)
            self.control_display.blit(preview_label, label_rect)
            
            # Draw camera status and label
            camera_status = "LIVE" if self.camera_active and not self.camera_paused else "PAUSED"
            status_color = (0, 255, 0) if camera_status == "LIVE" else (255, 165, 0)
            camera_label = self.font.render(f'Camera View ({camera_status})', True, status_color)
            label_rect = camera_label.get_rect(centerx=self.camera_rect.centerx, bottom=self.camera_rect.top - 5)
            self.control_display.blit(camera_label, label_rect)
            
            # Draw preview windows
            pygame.draw.rect(self.control_display, (64, 64, 64), self.preview_rect)
            pygame.draw.rect(self.control_display, (64, 64, 64), self.camera_rect)
            
            # Draw save buttons
            self.save_preview_button.draw(self.control_display)
            self.save_camera_button.draw(self.control_display)
            self.pause_camera_button.draw(self.control_display)
            
            # Draw calibrate button
            self.calibrate_button.draw(self.control_display)
            
            # Update and draw preview surfaces
            self.control_display.blit(self.preview_surface, self.preview_rect)
            
            # Update camera preview
            self.update_camera_preview()
            self.control_display.blit(self.camera_surface, self.camera_rect)
            
            pygame.display.flip()
            clock.tick(60)
        
        self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        if self.camera_active:
            self.cap.release()
        pygame.quit()

if __name__ == "__main__":
    controller = SLMController()
    controller.run()
