"""
Pattern Generator for Sony LCX016AL-6 SLM - Raspberry Pi Version
Generates phase patterns using the Gerchberg-Saxton algorithm for far-field image reconstruction.

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
from tkinter import ttk, filedialog
import pygame
import os
import time
import threading
from tqdm import tqdm


class PatternGenerator:
    def __init__(self):
        """Initialize the pattern generator with SLM specifications"""
        # SLM Specifications
        self.width = 832
        self.height = 624
        self.pixel_size = 32e-6
        self.wavelength = 532e-9  # 532 nm (green laser)

        # GUI Initialization
        self.root = tk.Tk()
        self.root.title("SLM Pattern Generator")
        self.root.geometry("1200x800")

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Create UI components
        self.setup_ui()

        # Camera state
        self.camera_active = False
        self.cap = None

    def setup_ui(self):
        """Creates UI elements"""
        frame_controls = ttk.LabelFrame(self.root, text="Controls", padding=10)
        frame_controls.pack(fill=tk.X, padx=10, pady=5)

        frame_preview = ttk.LabelFrame(self.root, text="Pattern Preview", padding=10)
        frame_preview.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Buttons
        ttk.Button(frame_controls, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_controls, text="Generate Pattern", command=self.generate_pattern).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_controls, text="Save Pattern", command=self.save_pattern).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_controls, text="Send to SLM (HDMI1)", command=self.send_to_slm).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_controls, text="Quit", command=self.quit_application).pack(side=tk.RIGHT, padx=5)

        # Iterations input
        ttk.Label(frame_controls, text="Iterations:").pack(side=tk.LEFT, padx=5)
        self.iterations_var = tk.StringVar(value="100")
        ttk.Entry(frame_controls, textvariable=self.iterations_var, width=5).pack(side=tk.LEFT, padx=5)

        # Canvas for preview
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 5))
        self.ax1.set_title("Target Image")
        self.ax2.set_title("Generated Pattern")

        self.canvas = FigureCanvasTkAgg(self.fig, master=frame_preview)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, frame_preview)
        self.toolbar.update()

    def load_image(self):
        """Load an image and display it as a target"""
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif")])
        if not file_path:
            return

        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            self.status_var.set("Error loading image")
            return

        # Resize to match SLM
        self.target_image = cv2.resize(image, (self.width, self.height))

        # Display the loaded image
        self.ax1.clear()
        self.ax1.imshow(self.target_image, cmap="gray")
        self.ax1.set_title("Target Image")
        self.ax1.axis("off")

        self.canvas.draw()
        self.status_var.set("Image loaded successfully")

    def generate_pattern(self):
        """Generate a Gerchberg-Saxton phase pattern"""
        if not hasattr(self, "target_image"):
            self.status_var.set("No image loaded. Load an image first.")
            return

        iterations = int(self.iterations_var.get())

        # Initialize with random phase
        field = np.exp(1j * 2 * np.pi * np.random.rand(self.height, self.width))
        target_amplitude = np.sqrt(self.target_image)

        for _ in tqdm(range(iterations), desc="Generating Pattern"):
            far_field = np.fft.fftshift(np.fft.fft2(field))
            far_field = target_amplitude * np.exp(1j * np.angle(far_field))
            field = np.fft.ifft2(np.fft.ifftshift(far_field))
            field = np.exp(1j * np.angle(field))

        # Convert to 8-bit grayscale
        self.generated_pattern = ((np.angle(field) + np.pi) / (2 * np.pi) * 255).astype(np.uint8)

        # Display the generated pattern
        self.ax2.clear()
        self.ax2.imshow(self.generated_pattern, cmap="gray")
        self.ax2.set_title("Generated Pattern")
        self.ax2.axis("off")

        self.canvas.draw()
        self.status_var.set("Pattern generated successfully")

    def save_pattern(self):
        """Save the generated pattern"""
        if not hasattr(self, "generated_pattern"):
            self.status_var.set("No pattern to save")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if not file_path:
            return

        cv2.imwrite(file_path, self.generated_pattern)
        self.status_var.set(f"Pattern saved: {file_path}")

    def send_to_slm(self):
        """Send pattern to SLM via HDMI1 using Pygame"""
        if not hasattr(self, "generated_pattern"):
            self.status_var.set("No pattern to display. Generate or load a pattern first.")
            return

        try:
            # Set HDMI1 as the display
            os.environ["DISPLAY"] = ":0.1"

            # Initialize Pygame and create a fullscreen window on HDMI1
            pygame.init()
            screen = pygame.display.set_mode((self.width, self.height), pygame.NOFRAME | pygame.FULLSCREEN)
            pygame.display.set_caption("SLM Display")

            # Convert pattern to Pygame surface
            slm_surface = pygame.Surface((self.width, self.height), depth=8)
            palette = [(i, i, i) for i in range(256)]  # Grayscale palette
            slm_surface.set_palette(palette)
            pygame.surfarray.pixels2d(slm_surface)[:] = self.generated_pattern.T  # Transpose to match Pygame

            # Display the pattern on HDMI1
            screen.fill((128, 128, 128))
            screen.blit(slm_surface, (0, 0))
            pygame.display.update()

            self.status_var.set("Pattern sent to SLM (HDMI1)")

            # Keep window open until ESC is pressed
            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        running = False

            pygame.quit()

        except Exception as e:
            self.status_var.set(f"Error sending to SLM: {str(e)}")

    def quit_application(self):
        """Exit the application"""
        self.root.quit()


if __name__ == "__main__":
    app = PatternGenerator()
    app.root.mainloop()
