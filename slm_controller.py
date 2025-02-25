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
        
        # Initialize pygame
        pygame.init()
        pygame.font.init()
        
        # Single monitor mode - create two windows side by side
        self.control_display = pygame.display.set_mode((800, 600))
        pygame.display.set_caption('SLM Control Interface')
        
        # Create SLM window
        self.slm_window = pygame.display.set_mode(self.slm_resolution, pygame.RESIZABLE | pygame.NOFRAME, display=0)
        pygame.display.set_caption('SLM Output')
        
        # Move SLM window to the right of the control window
        os.environ['SDL_VIDEO_WINDOW_POS'] = '810,0'
        
        # Initialize camera
        try:
            self.camera = cv2.VideoCapture(0)
            self.camera_active = True
        except:
            print("Warning: Could not initialize camera")
            self.camera_active = False
        
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
        self.slm_window.blit(pygame_pattern, (0, 0))
        pygame.display.update()

    def run(self):
        """Main application loop"""
        running = True
        # Set up some basic colors
        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)
        GRAY = (128, 128, 128)
        
        # Create a font
        font = pygame.font.SysFont(None, 36)
        
        print("Starting SLM Control Interface")
        print("Press ESC to exit")
        print("Both windows should appear side by side")
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            # Clear both displays
            self.control_display.fill(BLACK)
            self.slm_window.fill(GRAY)  # Gray for visibility
            
            # Draw some text and UI elements
            text = font.render('SLM Control Interface', True, WHITE)
            self.control_display.blit(text, (10, 10))
            
            # Draw a pattern selection area
            pygame.draw.rect(self.control_display, GRAY, (10, 50, 200, 200))
            pattern_text = font.render('Pattern Area', True, WHITE)
            self.control_display.blit(pattern_text, (20, 60))
            
            # Draw a camera preview area
            pygame.draw.rect(self.control_display, GRAY, (220, 50, 320, 240))
            camera_text = font.render('Camera Preview', True, WHITE)
            self.control_display.blit(camera_text, (230, 60))
            
            # Draw some text on SLM window
            slm_text = font.render('SLM Output Window', True, BLACK)
            self.slm_window.blit(slm_text, (10, 10))
            
            # Update both displays
            pygame.display.flip()
            pygame.display.update()
            
        self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        self.camera_active = False
        self.camera.release()
        pygame.quit()

if __name__ == "__main__":
    controller = SLMController()
    controller.run()
