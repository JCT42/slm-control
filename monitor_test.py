import pygame
import numpy as np
import os

# Force pygame to use the second display
os.environ['DISPLAY'] = ':0.1'
pygame.init()

# Get the resolution of the second display
pygame.display.init()
display_info = pygame.display.Info()
monitor_width = display_info.current_w
monitor_height = display_info.current_h

print(f"Detected monitor resolution: {monitor_width}x{monitor_height}")

# SLM native resolution
slm_width = 832
slm_height = 624

# Initialize display in fullscreen mode on second monitor
screen = pygame.display.set_mode((monitor_width, monitor_height), pygame.NOFRAME | pygame.FULLSCREEN)
pygame.display.set_caption("SLM Test")

# Generate a grayscale gradient at SLM resolution
gradient = np.linspace(0, 255, slm_width, dtype=np.uint8)
pattern = np.tile(gradient, (slm_height, 1))

# Convert to a Pygame surface at SLM resolution
slm_surface = pygame.Surface((slm_width, slm_height), depth=8)
palette = [(i, i, i) for i in range(256)]
slm_surface.set_palette(palette)
pygame.surfarray.pixels2d(slm_surface)[:] = pattern.T

# Scale the surface to monitor resolution while maintaining aspect ratio
scale_factor = min(monitor_width / slm_width, monitor_height / slm_height)
scaled_width = int(slm_width * scale_factor)
scaled_height = int(slm_height * scale_factor)

# Calculate position to center the pattern
x_offset = (monitor_width - scaled_width) // 2
y_offset = (monitor_height - scaled_height) // 2

# Scale the surface
scaled_surface = pygame.transform.scale(slm_surface, (scaled_width, scaled_height))

# Clear screen with black
screen.fill((0, 0, 0))

# Display scaled pattern at center
screen.blit(scaled_surface, (x_offset, y_offset))
pygame.display.flip()

print(f"Pattern scaled to: {scaled_width}x{scaled_height}")
print(f"Pattern positioned at: ({x_offset}, {y_offset})")
print("Press ESC to exit")

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False
            
pygame.quit()