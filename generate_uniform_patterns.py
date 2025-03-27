#!/usr/bin/env python3
"""
Generate Uniform Grayscale Patterns for SLM

This script generates three uniform grayscale patterns for an SLM:
1. Value 0 (black) - corresponds to phase -pi
2. Value 127 (mid-gray) - corresponds to phase 0
3. Value 255 (white) - corresponds to phase pi

These patterns are useful for SLM calibration and testing.
"""

import numpy as np
from PIL import Image
import os
import time

def generate_uniform_pattern(width=800, height=600, value=0, output_dir="slm_presets"):
    """
    Generate a uniform grayscale pattern with the specified value.
    
    Args:
        width: Width of the SLM in pixels
        height: Height of the SLM in pixels
        value: Grayscale value (0-255)
        output_dir: Directory to save the pattern
    
    Returns:
        Path to the saved pattern
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create uniform array with the specified value
    pattern = np.full((height, width), value, dtype=np.uint8)
    
    # Create image from array
    img = Image.fromarray(pattern)
    
    # Generate filename with timestamp to avoid overwriting
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"uniform_{value:03d}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    
    # Save image
    img.save(filepath)
    print(f"Generated uniform pattern with value {value} at {filepath}")
    
    return filepath

def generate_all_patterns(width=800, height=600):
    """
    Generate all three uniform patterns (0, 127, 255).
    
    Args:
        width: Width of the SLM in pixels
        height: Height of the SLM in pixels
    
    Returns:
        List of paths to the saved patterns
    """
    # Values corresponding to phases -pi, 0, pi
    values = [0, 127, 255]
    
    # Generate patterns
    paths = []
    for value in values:
        path = generate_uniform_pattern(width, height, value)
        paths.append(path)
    
    return paths

if __name__ == "__main__":
    # Default SLM dimensions (800x600)
    # These can be modified if your SLM has different dimensions
    SLM_WIDTH = 800
    SLM_HEIGHT = 600
    
    print("Generating uniform grayscale patterns for SLM...")
    print("Phase mapping:")
    print("  Value 0   (black)    -> Phase -pi")
    print("  Value 127 (mid-gray) -> Phase 0")
    print("  Value 255 (white)    -> Phase pi")
    
    # Generate all patterns
    paths = generate_all_patterns(SLM_WIDTH, SLM_HEIGHT)
    
    print("\nAll patterns generated successfully!")
    print("You can display these patterns on your SLM for calibration and testing.")
