import cv2
import numpy as np
import pygame
import os
from PIL import Image
import threading
from pathlib import Path
import time

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
        
        # Initialize pygame
        pygame.init()
        pygame.font.init()
        
        # Single monitor mode - create main window with more space for preview
        self.control_display = pygame.display.set_mode((1000, 800))
        pygame.display.set_caption('SLM Control Interface')
        
        # Create SLM window
        self.slm_window = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE | pygame.NOFRAME, display=0)
        pygame.display.set_caption('SLM Output')
        
        # Move SLM window to the right of the control window
        os.environ['SDL_VIDEO_WINDOW_POS'] = '1010,0'
        
        # Create output directories
        self.output_dir = Path('output')
        self.output_dir.mkdir(exist_ok=True)
        
        # Preview surface
        self.preview_surface = pygame.Surface((400, 300))
        self.preview_rect = pygame.Rect(200, 50, 400, 300)
        
        # Camera preview surface
        self.camera_surface = pygame.Surface((300, 225))
        self.camera_rect = pygame.Rect(650, 50, 300, 225)
        
        # Save buttons
        self.font = pygame.font.SysFont(None, 36)
        self.save_preview_button = Button(200, 360, 200, 40, "Save Pattern", self.font)
        self.save_camera_button = Button(650, 285, 200, 40, "Save Camera", self.font)
        
        # Camera control
        self.camera_paused = False
        self.pause_camera_button = Button(650, 335, 200, 40, "Pause Camera", self.font)
        
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
        self.patterns = ['blank', 'binary_grating', 'sinusoidal_grating', 'blazed_grating', 'checkerboard', 'circular_aperture', 'lens']
        self.current_pattern = None
        
        # Create pattern buttons
        self.pattern_buttons = []
        y = 100  # Starting y position for pattern list
        for pattern in self.patterns:
            btn = Button(10, y, 180, 30, pattern, self.font, color=(100, 100, 100))
            self.pattern_buttons.append(btn)
            y += 35  # Space between buttons
        
        # UI elements
        self.pattern_button = Button(10, 50, 200, 40, "Select Pattern", self.font)
        self.show_pattern_list = False

    def create_default_patterns(self):
        """Create some default SLM patterns"""
        # Create blank pattern
        blank = np.ones((self.height, self.width), dtype=np.uint8) * 128
        
        # Create binary grating
        binary = np.zeros((self.height, self.width), dtype=np.uint8)
        period = 16  # pixels
        for x in range(self.width):
            if (x // period) % 2 == 0:
                binary[:, x] = 255
        
        # Create sinusoidal grating
        x = np.linspace(0, 8*np.pi, self.width)
        sinusoidal = np.zeros((self.height, self.width), dtype=np.uint8)
        for y in range(self.height):
            sinusoidal[y, :] = (127.5 * (1 + np.sin(x))).astype(np.uint8)
        
        # Create blazed grating
        blazed = np.zeros((self.height, self.width), dtype=np.uint8)
        ramp = np.linspace(0, 255, 32, dtype=np.uint8)
        for x in range(0, self.width, 32):
            end = min(x + 32, self.width)
            blazed[:, x:end] = ramp[:end-x, np.newaxis].T
        
        # Create checkerboard
        checkerboard = np.zeros((self.height, self.width), dtype=np.uint8)
        square_size = 16
        for y in range(0, self.height, square_size*2):
            for x in range(0, self.width, square_size*2):
                if y + square_size <= self.height and x + square_size <= self.width:
                    checkerboard[y:y+square_size, x:x+square_size] = 255
                if y + square_size <= self.height and x + square_size*2 <= self.width:
                    checkerboard[y:y+square_size, x+square_size:x+square_size*2] = 0
                if y + square_size*2 <= self.height and x + square_size <= self.width:
                    checkerboard[y+square_size:y+square_size*2, x:x+square_size] = 0
                if y + square_size*2 <= self.height and x + square_size*2 <= self.width:
                    checkerboard[y+square_size:y+square_size*2, x+square_size:x+square_size*2] = 255
        
        # Create circular aperture
        circle = np.zeros((self.height, self.width), dtype=np.uint8)
        center_x, center_y = self.width // 2, self.height // 2
        radius = min(center_x, center_y) // 2
        y, x = np.ogrid[0:self.height, 0:self.width]
        mask = ((x - center_x)**2 + (y - center_y)**2) <= radius**2
        circle[mask] = 255
        
        # Create Fresnel lens pattern
        lens = np.zeros((self.height, self.width), dtype=np.uint8)
        y, x = np.ogrid[0:self.height, 0:self.width]
        r2 = ((x - center_x)**2 + (y - center_y)**2) / 1000.0
        lens = ((128 * (1 + np.cos(r2))) % 256).astype(np.uint8)
        
        # Save all patterns
        patterns = {
            'blank': blank,
            'binary_grating': binary,
            'sinusoidal_grating': sinusoidal,
            'blazed_grating': blazed,
            'checkerboard': checkerboard,
            'circular_aperture': circle,
            'lens': lens
        }
        
        for name, pattern in patterns.items():
            Image.fromarray(pattern).save(self.patterns_dir / f'{name}.png')

    def display_pattern(self, pattern_name):
        """Display a pattern on the SLM and preview"""
        pattern_path = self.patterns_dir / f'{pattern_name}.png'
        if pattern_path.exists():
            pattern = cv2.imread(str(pattern_path), cv2.IMREAD_GRAYSCALE)
            self.current_pattern = pattern_name
            
            # Display on SLM window
            self.slm_window.fill((128, 128, 128))
            pygame_pattern = pygame.surfarray.make_surface(pattern)
            self.slm_window.blit(pygame_pattern, (0, 0))
            
            # Display preview
            preview_pattern = pygame.transform.scale(pygame_pattern, (400, 300))
            self.preview_surface.blit(preview_pattern, (0, 0))
            
            pygame.display.update()
        else:
            # Generate and save the pattern first
            self.create_default_patterns()
            # Then try to display it again
            self.display_pattern(pattern_name)

    def update_camera_preview(self):
        """Update camera preview if camera is active and not paused"""
        if self.camera_active and not self.camera_paused:
            ret, frame = self.camera.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (300, 225))
                pygame_frame = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                self.camera_surface.blit(pygame_frame, (0, 0))

    def toggle_camera_pause(self):
        """Toggle camera pause state"""
        self.camera_paused = not self.camera_paused
        self.pause_camera_button.text = "Resume Camera" if self.camera_paused else "Pause Camera"

    def save_preview_image(self):
        """Save the current pattern preview"""
        if self.current_pattern:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = self.output_dir / f'pattern_{self.current_pattern}_{timestamp}.png'
            pygame.image.save(self.preview_surface, str(filename))
            print(f"Saved pattern preview to {filename}")

    def save_camera_image(self):
        """Save the current camera image"""
        if self.camera_active:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = self.output_dir / f'camera_{timestamp}.png'
            pygame.image.save(self.camera_surface, str(filename))
            print(f"Saved camera image to {filename}")

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
            
            # Clear control display
            self.control_display.fill((0, 0, 0))
            
            # Draw UI
            title = self.font.render('SLM Control Interface', True, (255, 255, 255))
            self.control_display.blit(title, (10, 10))
            
            # Draw pattern selection button
            self.pattern_button.draw(self.control_display)
            
            # Draw pattern list if shown
            if self.show_pattern_list:
                for btn in self.pattern_buttons:
                    btn.draw(self.control_display)
            
            # Draw current pattern name
            if self.current_pattern:
                pattern_text = self.font.render(f'Current: {self.current_pattern}', True, (255, 255, 255))
                self.control_display.blit(pattern_text, (200, 20))
            
            # Draw preview labels
            preview_label = self.font.render('Pattern Preview', True, (255, 255, 255))
            self.control_display.blit(preview_label, (200, 410))
            
            # Draw camera status and label
            camera_status = "LIVE" if self.camera_active and not self.camera_paused else "PAUSED"
            status_color = (0, 255, 0) if camera_status == "LIVE" else (255, 165, 0)
            camera_label = self.font.render(f'Camera View ({camera_status})', True, status_color)
            self.control_display.blit(camera_label, (650, 385))
            
            # Draw preview windows
            pygame.draw.rect(self.control_display, (64, 64, 64), self.preview_rect)
            pygame.draw.rect(self.control_display, (64, 64, 64), self.camera_rect)
            
            # Draw save buttons
            self.save_preview_button.draw(self.control_display)
            self.save_camera_button.draw(self.control_display)
            self.pause_camera_button.draw(self.control_display)
            
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
            self.camera.release()
        pygame.quit()

if __name__ == "__main__":
    controller = SLMController()
    controller.run()
