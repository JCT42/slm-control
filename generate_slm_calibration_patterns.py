import numpy as np
from PIL import Image
import os

def generate_blazed_grating(width=800, height=600, num_steps=11, output_dir="blazed_patterns"):
    os.makedirs(output_dir, exist_ok=True)

    # Frequencies in cycles per screen width (low to high spatial frequencies)
    spatial_frequencies = np.linspace(0.5, 5, num_steps)

    for i, fx in enumerate(spatial_frequencies):
        print(f"Generating grating with frequency {fx:.2f} cycles/screen")

        # Phase ramp: from −π to π repeated fx times across the width
        x = np.linspace(0, 1, width)
        phase_ramp = 2 * np.pi * fx * x  # 0 to fx * 2π
        phase_ramp_wrapped = (phase_ramp % (2 * np.pi)) - np.pi  # wrapped to [−π, π)

        # Map to grayscale: −π to π → 0 to 255
        grayscale = ((phase_ramp_wrapped + np.pi) / (2 * np.pi)) * 255
        grayscale_img = np.tile(grayscale, (height, 1)).astype(np.uint8)

        # Save image
        img = Image.fromarray(grayscale_img)
        img.save(os.path.join(output_dir, f"blazed_fx_{fx:.2f}.png"))

generate_blazed_grating()
