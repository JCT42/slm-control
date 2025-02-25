import cv2
import numpy as np
import pygame
import os
from PIL import Image
import threading
from pathlib import Path

class SLMController:
    def __init__(self):
        # SLM Parameters for SONY LCX016AL-6
        self.slm_resolution = (832, 624)  # As per specifications
        self.pixel_pitch = 32  # μm
        self.refresh_rate = 60  # Hz
        self.contrast_ratio = 200  # typical 200:1
        self.active_area = (26.6, 20.0)  # mm x mm (1.3")
        
        # Initialize displays
        os.environ['SDL_VIDEODRIVER'] = 'x11'  # For Raspberry Pi
        pygame.init()
        
        # Get available displays
        num_displays = pygame.display.get_num_displays()
        print(f"Number of displays detected: {num_displays}")
        
        # Control display (HDMI0)
        self.control_display = pygame.display.set_mode((1024, 768))
        pygame.display.set_caption('SLM Control Interface')
        
        # SLM display (HDMI1)
        try:
            # Try to create window on second display
            os.environ['DISPLAY'] = ':0.1'  # Try second X display
            self.slm_display = pygame.display.set_mode(self.slm_resolution, pygame.FULLSCREEN)
        except pygame.error as e:
            print(f"Could not initialize second display: {e}")
            print("Running in single display mode - SLM output will be shown in a window")
            self.slm_display = pygame.display.set_mode(self.slm_resolution, pygame.RESIZABLE)
        
        # Initialize camera
        self.camera = cv2.VideoCapture(0)
        self.camera_active = True
        
        # Load patterns
        self.patterns_dir = Path('patterns')
        self.patterns_dir.mkdir(exist_ok=True)
        self.patterns = []
        self.current_pattern = None
        self.load_patterns()
        
        # Start camera thread
        self.camera_thread = threading.Thread(target=self.update_camera_feed)
        self.camera_thread.daemon = True
        self.camera_thread.start()

    def load_patterns(self):
        """Load grayscale patterns from the patterns directory"""
        if not self.patterns_dir.exists():
            self.create_default_patterns()
        
        for pattern_file in self.patterns_dir.glob('*.png'):
            self.patterns.append(str(pattern_file))

    def create_default_patterns(self):
        """Create some default SLM patterns"""
        # Create basic patterns (examples)
        patterns = {
            'blank': np.zeros(self.slm_resolution, dtype=np.uint8) + 128,
            'horizontal_grating': np.tile(np.linspace(0, 255, self.slm_resolution[0]), 
                                       (self.slm_resolution[1], 1)).astype(np.uint8),
            'vertical_grating': np.tile(np.linspace(0, 255, self.slm_resolution[1]), 
                                     (self.slm_resolution[0], 1)).T.astype(np.uint8),
            'checkerboard': np.kron([[0, 255], [255, 0]], 
                                  np.ones((self.slm_resolution[1]//2, self.slm_resolution[0]//2))).astype(np.uint8)
        }
        
        for name, pattern in patterns.items():
            Image.fromarray(pattern).save(self.patterns_dir / f'{name}.png')

    def update_camera_feed(self):
        """Update camera feed in a separate thread"""
        while self.camera_active:
            ret, frame = self.camera.read()
            if ret:
                # Convert to pygame surface and display on control screen
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (320, 240))
                pygame_frame = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                self.control_display.blit(pygame_frame, (680, 20))
                pygame.display.update()

    def display_pattern(self, pattern):
        """Display a pattern on the SLM"""
        if isinstance(pattern, str):
            pattern = cv2.imread(pattern, cv2.IMREAD_GRAYSCALE)
        
        pattern = cv2.resize(pattern, self.slm_resolution)
        pygame_pattern = pygame.surfarray.make_surface(pattern)
        self.slm_display.blit(pygame_pattern, (0, 0))
        pygame.display.update()

    def run(self):
        """Main application loop"""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Handle pattern selection
                    pass

            # Update control display
            self.control_display.fill((0, 0, 0))
            
            # Draw pattern selection interface
            # TODO: Add pattern selection buttons and preview
            
            pygame.display.flip()

        self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        self.camera_active = False
        self.camera.release()
        pygame.quit()

if __name__ == "__main__":
    controller = SLMController()
    controller.run()
