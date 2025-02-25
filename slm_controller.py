import cv2
import numpy as np
import pygame
import os
from PIL import Image
import threading
from pathlib import Path

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
        self.slm_resolution = (832, 624)
        self.pixel_pitch = 32  # μm
        self.refresh_rate = 60  # Hz
        self.contrast_ratio = 200
        self.active_area = (26.6, 20.0)  # mm
        
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
        self.patterns = []
        self.current_pattern = None
        self.create_default_patterns()
        self.load_patterns()
        
        # UI elements
        self.font = pygame.font.SysFont(None, 36)
        self.pattern_button = Button(10, 50, 200, 40, "Select Pattern", self.font)
        self.show_pattern_list = False
        self.pattern_buttons = []
        self.update_pattern_buttons()

    def create_default_patterns(self):
        """Create some default SLM patterns"""
        self.patterns_dir.mkdir(exist_ok=True)
        
        patterns = {
            'blank': np.ones(self.slm_resolution, dtype=np.uint8) * 128,
            
            'binary_grating': np.kron(
                [[0, 255] * (self.slm_resolution[0] // 16)],
                np.ones((self.slm_resolution[1], 8))
            ).astype(np.uint8),
            
            'sinusoidal_grating': (127.5 * (1 + np.sin(np.linspace(0, 8*np.pi, self.slm_resolution[0])))).astype(np.uint8),
            
            'blazed_grating': np.tile(
                np.linspace(0, 255, 32, dtype=np.uint8),
                (self.slm_resolution[1], self.slm_resolution[0] // 32)
            ),
            
            'checkerboard': np.kron(
                [[0, 255] * (self.slm_resolution[0] // 32), [255, 0] * (self.slm_resolution[0] // 32)] * (self.slm_resolution[1] // 32),
                np.ones((16, 16))
            ).astype(np.uint8),
            
            'circular_aperture': np.zeros(self.slm_resolution, dtype=np.uint8),
            
            'lens': np.zeros(self.slm_resolution, dtype=np.uint8)
        }
        
        # Create circular aperture
        center = (self.slm_resolution[0]//2, self.slm_resolution[1]//2)
        radius = min(self.slm_resolution) // 4
        y, x = np.ogrid[-center[1]:self.slm_resolution[1]-center[1], -center[0]:self.slm_resolution[0]-center[0]]
        mask = x*x + y*y <= radius*radius
        patterns['circular_aperture'][mask] = 255
        
        # Create Fresnel lens pattern
        y, x = np.mgrid[-self.slm_resolution[1]//2:self.slm_resolution[1]//2, 
                       -self.slm_resolution[0]//2:self.slm_resolution[0]//2]
        r2 = x*x + y*y
        patterns['lens'] = ((128 * (1 + np.cos(r2 / 1000))) % 256).astype(np.uint8)
        
        # Save all patterns
        for name, pattern in patterns.items():
            if pattern.shape != self.slm_resolution:
                pattern = np.tile(pattern, (self.slm_resolution[1]//pattern.shape[0] + 1,
                                         self.slm_resolution[0]//pattern.shape[1] + 1))
                pattern = pattern[:self.slm_resolution[1], :self.slm_resolution[0]]
            Image.fromarray(pattern).save(self.patterns_dir / f'{name}.png')

    def load_patterns(self):
        """Load all patterns from the patterns directory"""
        self.patterns = []
        for pattern_file in self.patterns_dir.glob('*.png'):
            self.patterns.append(pattern_file.stem)
        self.patterns.sort()

    def update_pattern_buttons(self):
        """Create buttons for each pattern"""
        self.pattern_buttons = []
        for i, pattern in enumerate(self.patterns):
            btn = Button(10, 100 + i*45, 200, 40, pattern, self.font)
            self.pattern_buttons.append(btn)

    def display_pattern(self, pattern_name):
        """Display a pattern on the SLM"""
        pattern_path = self.patterns_dir / f'{pattern_name}.png'
        if pattern_path.exists():
            pattern = cv2.imread(str(pattern_path), cv2.IMREAD_GRAYSCALE)
            self.current_pattern = pattern_name
            self.slm_window.fill((128, 128, 128))
            pygame_pattern = pygame.surfarray.make_surface(pattern)
            self.slm_window.blit(pygame_pattern, (0, 0))
            pygame.display.update()

    def run(self):
        """Main application loop"""
        running = True
        clock = pygame.time.Clock()
        
        print("Starting SLM Control Interface")
        print("Press ESC to exit")
        print(f"Available patterns: {', '.join(self.patterns)}")
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                
                # Handle pattern selection
                if self.pattern_button.handle_event(event):
                    self.show_pattern_list = not self.show_pattern_list
                
                if self.show_pattern_list:
                    for btn in self.pattern_buttons:
                        if btn.handle_event(event):
                            self.display_pattern(btn.text)
                            self.show_pattern_list = False
            
            # Clear control display
            self.control_display.fill((0, 0, 0))
            
            # Draw UI
            title = self.font.render('SLM Control Interface', True, (255, 255, 255))
            self.control_display.blit(title, (10, 10))
            
            # Draw pattern selection button and list
            self.pattern_button.draw(self.control_display)
            if self.show_pattern_list:
                for btn in self.pattern_buttons:
                    btn.handle_event(pygame.event.Event(pygame.MOUSEMOTION, {'pos': pygame.mouse.get_pos()}))
                    btn.draw(self.control_display)
            
            # Draw current pattern name
            if self.current_pattern:
                pattern_text = self.font.render(f'Current: {self.current_pattern}', True, (255, 255, 255))
                self.control_display.blit(pattern_text, (220, 50))
            
            pygame.display.flip()
            clock.tick(60)
        
        self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        if self.camera_active:
            self.camera.release()
        pygame.quit()

if __name__ == "__main__":
    controller = SLMController()
    controller.run()
