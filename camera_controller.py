#!/usr/bin/env python3
"""
Camera Controller for IMX296 Monochrome Camera

This module provides a camera controller specifically optimized for the IMX296 monochrome camera
that outputs in RGB3 format (24-bit RGB 8-8-8) at 1456x1088 resolution.

Features:
- Direct access to camera's intensity values from monochrome sensor
- Uses single channel from RGB3 format for true grayscale values
- Histogram generation and analysis of intensity distributions
- Frame capture and saving with metadata
- Thread-safe operation for GUI integration
"""

import os
import time
import threading
import traceback
import subprocess
from typing import Dict, List, Optional, Tuple, Union
import json

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Check if PiCamera2 is available
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("PiCamera2 not available. Install with: pip install picamera2")

class CameraController:
    """
    Camera controller for IMX296 monochrome camera that outputs in RGB3 format.
    Preserves intensity values for scientific analysis.
    """
    
    def __init__(self, 
                 resolution: Tuple[int, int] = (1456, 1088),
                 device: str = "/dev/video0"):
        """
        Initialize the camera controller.
        
        Args:
            resolution: Camera resolution (width, height)
            device: Camera device path
        """
        self.width, self.height = resolution
        self.device = device
        
        # Camera state
        self.camera = None
        self.is_running = False
        self.is_paused = False
        self.thread = None
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_histogram = None
        self.histogram_enabled = True  # Flag to enable/disable histogram generation
        self.auto_adjustments_enabled = False  # Flag to enable/disable auto adjustments
        
        # Camera settings
        self.settings = {
            'exposure': 10.0,  # ms
            'gain': 1.0,       # analog gain
            'brightness': 0,   # -255 to 255
            'contrast': 1.0,   # 0.0 to 2.0
            'sharpness': 0,    # -100 to 100
            'saturation': 0,   # -100 to 100
            'iso': 100,        # 100 to 800
            'exposure_mode': 'auto',  # auto, manual, night, backlight, spotlight
            'white_balance': 'auto',  # auto, sunlight, cloudy, shade, tungsten, fluorescent, incandescent, flash, horizon
            'auto_exposure_enabled': False,  # Enable/disable auto exposure
            'auto_white_balance_enabled': False,  # Enable/disable auto white balance
            'auto_gain_enabled': False,  # Enable/disable auto gain
            'noise_reduction_mode': 0,  # 0=Off, 1=Fast, 2=High Quality, 3=Minimal
        }
        
        # Initialize the camera
        self.initialize()
    
    def initialize(self) -> bool:
        """Initialize the camera with configured settings"""
        try:
            # Check if PiCamera2 is available
            if not PICAMERA2_AVAILABLE:
                print("PiCamera2 is not available. Please install it with: pip install picamera2")
                return False
                
            # Create camera instance
            self.camera = Picamera2()
            
            # Get camera info
            print(f"Camera info: {self.camera.camera_properties}")
            
            # Configure for RGB3 format as detected by v4l2-ctl
            print(f"Using resolution: {self.width}x{self.height}, format: RGB3")
            
            preview_width = int(self.width * 0.5)  # Half size for preview
            preview_height = int(self.height * 0.5)
            
            # Create configuration for still and preview
            try:
                self.camera_config = self.camera.create_still_configuration(
                    main={"size": (self.width, self.height),
                          "format": "RGB888"},  # RGB3 format in PiCamera2
                    lores={"size": (preview_width, preview_height),
                           "format": "YUV420"},
                    display="lores"
                )
            except Exception as config_error:
                print(f"Error creating RGB888 configuration: {str(config_error)}")
                print("Falling back to default configuration")
                self.camera_config = self.camera.create_still_configuration()
                # Update resolution to match what was configured
                self.width = self.camera_config["main"]["size"][0]
                self.height = self.camera_config["main"]["size"][1]
            
            # Apply configuration
            self.camera.configure(self.camera_config)
            
            # Print the actual configuration that was applied
            print(f"Camera configured with: {self.camera_config}")
            
            # Set initial camera controls
            try:
                self.camera.set_controls({
                    "ExposureTime": int(self.settings['exposure'] * 1000),  # Convert ms to μs
                    "AnalogueGain": float(self.settings['gain']),
                    "AeEnable": self.settings['auto_exposure_enabled'],  # Auto exposure
                    "AwbEnable": self.settings['auto_white_balance_enabled'],  # Auto white balance
                    "AgcEnable": self.settings['auto_gain_enabled'],  # Auto gain control
                    "NoiseReductionMode": self.settings['noise_reduction_mode'],  # Noise reduction mode
                })
                print(f"Camera controls set successfully")
                print(f"Auto exposure: {'enabled' if self.settings['auto_exposure_enabled'] else 'disabled'}")
                print(f"Auto white balance: {'enabled' if self.settings['auto_white_balance_enabled'] else 'disabled'}")
                print(f"Auto gain: {'enabled' if self.settings['auto_gain_enabled'] else 'disabled'}")
                print(f"Noise reduction mode: {self.settings['noise_reduction_mode']}")
            except Exception as control_error:
                print(f"Warning: Could not set some camera controls: {str(control_error)}")
                print("Continuing with default controls")
            
            # Get the actual camera resolution after configuration
            actual_width = self.camera_config["main"]["size"][0]
            actual_height = self.camera_config["main"]["size"][1]
            actual_format = self.camera_config["main"]["format"]
            
            print(f"Camera initialized at {actual_width}x{actual_height} in {actual_format} format")
            
            # Start the camera to ensure it's working
            self.camera.start()
            time.sleep(0.5)  # Give it a moment to start
            
            # Capture a test frame to verify camera is working
            try:
                test_frame = self.camera.capture_array()
                print(f"Test frame captured successfully: {test_frame.shape}, dtype: {test_frame.dtype}")
                print(f"Frame values - Min: {np.min(test_frame)}, Max: {np.max(test_frame)}, Mean: {np.mean(test_frame):.1f}")
            except Exception as capture_error:
                print(f"Warning: Test frame capture failed: {str(capture_error)}")
                # Continue anyway as this is just a test
            
            return True
            
        except Exception as e:
            print(f"Camera initialization error: {str(e)}")
            traceback.print_exc()
            return False
    
    def start(self) -> bool:
        """Start the camera capture thread"""
        if self.is_running:
            print("Camera is already running")
            return True
            
        if self.camera is None:
            print("Camera not initialized")
            return False
            
        # Start the camera thread
        self.is_running = True
        self.thread = threading.Thread(target=self._capture_thread, daemon=True)
        self.thread.start()
        print("Camera capture thread started")
        return True
    
    def stop(self) -> None:
        """Stop the camera capture thread"""
        self.is_running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None
        
        # Stop the camera
        if self.camera is not None:
            try:
                self.camera.stop()
                print("Camera stopped")
            except Exception as e:
                print(f"Error stopping camera: {str(e)}")
    
    def _capture_thread(self) -> None:
        """Thread function for continuous frame capture"""
        last_histogram_time = 0
        histogram_interval = 0.5  # Update histogram every 0.5 seconds
        
        while self.is_running:
            try:
                # Skip frame capture if paused
                if self.is_paused:
                    time.sleep(0.1)  # Sleep longer when paused
                    continue
                    
                # Capture a frame
                frame = self.capture_frame()
                
                if frame is not None:
                    # Store the latest frame with thread safety
                    with self.lock:
                        self.latest_frame = frame
                    
                    # Update histogram occasionally
                    current_time = time.time()
                    if current_time - last_histogram_time > histogram_interval and self.histogram_enabled:
                        self.latest_histogram = self.generate_histogram(frame)
                        last_histogram_time = current_time
                
                # Sleep briefly to control frame rate
                time.sleep(0.01)
                
            except Exception as e:
                print(f"Error in capture thread: {str(e)}")
                time.sleep(0.1)  # Sleep longer on error
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame from the camera"""
        try:
            # Capture RGB frame
            rgb_frame = self.camera.capture_array()
            
            # For monochrome camera outputting in RGB3 format,
            # just use the red channel as they're all identical
            if len(rgb_frame.shape) == 3 and rgb_frame.shape[2] == 3:
                # Just extract the red channel (all channels should be identical for monochrome)
                gray = rgb_frame[:, :, 0]  # Use red channel
                print("Using red channel from RGB3 format for monochrome camera") if np.random.random() < 0.001 else None
            else:
                # Already grayscale
                gray = rgb_frame
                
            return gray
                
        except Exception as e:
            print(f"Frame capture error: {str(e)}")
            return None
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the latest captured frame"""
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None
    
    def get_latest_histogram(self) -> Optional[np.ndarray]:
        """Get the latest histogram image"""
        with self.lock:
            if self.latest_histogram is not None:
                return self.latest_histogram.copy()
            return None
    
    def generate_histogram(self, frame: np.ndarray) -> np.ndarray:
        """Generate a histogram image from the frame"""
        try:
            # Create a figure for the histogram with a smaller size
            fig = Figure(figsize=(4.5, 3.5), dpi=100)
            ax = fig.add_subplot(111)
            
            # Add more padding around the plot for axis labels
            fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.9)
            
            # Calculate histogram - ensure we use the full 8-bit range (0-255)
            hist = cv2.calcHist([frame], [0], None, [256], [0, 256])
            
            # Use raw histogram counts instead of normalizing
            # total_pixels = frame.shape[0] * frame.shape[1]
            # normalized_hist = hist / total_pixels
            
            # Plot the raw histogram with logarithmic y-scale
            ax.plot(hist, color='blue')
            ax.set_xlim([0, 256])
            ax.set_yscale('log')  # Set logarithmic scale for y-axis
            ax.set_title('Intensity Histogram (8-bit, Log Scale)')
            ax.set_xlabel('Intensity Value (0-255)')
            ax.set_ylabel('Pixel Count (log)')
            
            # Add vertical lines at min and max values
            min_val = np.min(frame)
            max_val = np.max(frame)
            mean_val = np.mean(frame)
            
            # Add min/max/mean lines
            ax.axvline(x=min_val, color='r', linestyle='--', alpha=0.7, label=f'Min: {min_val:.1f}')
            ax.axvline(x=max_val, color='g', linestyle='--', alpha=0.7, label=f'Max: {max_val:.1f}')
            ax.axvline(x=mean_val, color='y', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.1f}')
            
            # Add legend
            ax.legend(loc='upper right', fontsize='small')
            
            ax.grid(True, alpha=0.3)
            
            # Render the figure to a numpy array
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            buf = canvas.buffer_rgba()
            hist_image = np.asarray(buf)
            plt.close(fig)
            
            return hist_image
            
        except Exception as e:
            print(f"Error generating histogram: {str(e)}")
            # Return a blank image on error
            return np.zeros((350, 450, 4), dtype=np.uint8)
    
    def get_intensity_stats(self, frame: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Get intensity statistics from the frame"""
        if frame is None:
            frame = self.get_latest_frame()
            
        if frame is None:
            return {'min': 0, 'max': 0, 'mean': 0, 'std': 0}
            
        return {
            'min': float(np.min(frame)),
            'max': float(np.max(frame)),
            'mean': float(np.mean(frame)),
            'std': float(np.std(frame))
        }
    
    def capture_high_quality_frame(self, num_frames: int = 5) -> Optional[np.ndarray]:
        """Capture a high-quality frame by averaging multiple frames"""
        try:
            # For PiCamera2, capture a high-quality still
            self.camera.stop()
            time.sleep(0.1)
            
            # Configure for still capture
            self.camera.start()
            time.sleep(0.5)
            
            # Capture the frame
            result = self.camera.capture_array()
            
            # For monochrome camera outputting in RGB3 format,
            # just use the red channel as they're all identical
            if len(result.shape) == 3 and result.shape[2] == 3:
                # Just extract the red channel (all channels should be identical for monochrome)
                result = result[:, :, 0]  # Use red channel
                
            return result
                
        except Exception as e:
            print(f"Error capturing high-quality frame: {str(e)}")
            return None
    
    def save_frame(self, filename: str, frame: Optional[np.ndarray] = None) -> bool:
        """Save a frame to disk with metadata"""
        try:
            # Use provided frame or capture a new high-quality one
            if frame is None:
                save_image = self.capture_high_quality_frame()
                if save_image is None:
                    print("Failed to capture frame for saving")
                    return False
            else:
                save_image = frame.copy()
            
            # Ensure the filename has an appropriate extension
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif')):
                filename += '.png'
            
            # Create directory if it doesn't exist
            directory = os.path.dirname(filename)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            
            # Save the image
            cv2.imwrite(filename, save_image)
            
            # Save metadata
            self._save_metadata(filename, save_image)
            
            print(f"Image saved to {filename}")
            return True
            
        except Exception as e:
            print(f"Error saving frame: {str(e)}")
            return False
    
    def _save_metadata(self, filename: str, image: np.ndarray) -> None:
        """Save metadata about the image"""
        try:
            # Create metadata file
            metadata_filename = os.path.splitext(filename)[0] + '.txt'
            
            with open(metadata_filename, 'w') as f:
                f.write(f"Camera: IMX296 Monochrome RGB3 at {self.width}x{self.height}\n")
                f.write(f"Resolution: {image.shape[1]}x{image.shape[0]}\n")
                f.write(f"Bit Depth: 8-bit\n")
                f.write(f"Value Range: 0-255\n")
                
                f.write(f"Statistics:\n")
                f.write(f"  Min: {np.min(image):.1f}\n")
                f.write(f"  Max: {np.max(image):.1f}\n")
                f.write(f"  Mean: {np.mean(image):.1f}\n")
                f.write(f"  Std Dev: {np.std(image):.1f}\n")
                
                f.write(f"Camera Settings:\n")
                # Format basic settings
                f.write(f"  exposure: {self.settings['exposure']} ms\n")
                f.write(f"  gain: {self.settings['gain']}\n")
                f.write(f"  brightness: {self.settings['brightness']}\n")
                f.write(f"  contrast: {self.settings['contrast']}\n")
                f.write(f"  sharpness: {self.settings['sharpness']}\n")
                f.write(f"  saturation: {self.settings['saturation']}\n")
                f.write(f"  iso: {self.settings['iso']}\n")
                f.write(f"  exposure_mode: {self.settings['exposure_mode']}\n")
                f.write(f"  white_balance: {self.settings['white_balance']}\n")
                
                # Format auto adjustment settings with clear enabled/disabled text
                f.write(f"Auto Adjustment Settings:\n")
                f.write(f"  auto_exposure: {'Enabled' if self.settings['auto_exposure_enabled'] else 'Disabled'}\n")
                f.write(f"  auto_white_balance: {'Enabled' if self.settings['auto_white_balance_enabled'] else 'Disabled'}\n")
                f.write(f"  auto_gain: {'Enabled' if self.settings['auto_gain_enabled'] else 'Disabled'}\n")
                
                # Format noise reduction mode with descriptive text
                noise_mode_desc = {
                    0: "Off",
                    1: "Fast",
                    2: "High Quality",
                    3: "Minimal"
                }
                mode = self.settings['noise_reduction_mode']
                mode_text = noise_mode_desc.get(mode, f"Unknown ({mode})")
                f.write(f"  noise_reduction_mode: {mode} ({mode_text})\n")
                
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                
            print(f"Metadata saved to {metadata_filename}")
            
        except Exception as e:
            print(f"Error saving metadata: {str(e)}")
    
    def set_setting(self, setting: str, value: Union[float, int]) -> bool:
        """Set a camera setting"""
        if setting not in self.settings:
            print(f"Unknown setting: {setting}")
            return False
            
        # Update the setting
        self.settings[setting] = value
        
        # Do not apply the setting to the camera yet
        print(f"Setting {setting} updated to {value} (not applied yet)")
        return True
    
    def apply_all_settings(self) -> bool:
        """Apply all current settings to the camera"""
        try:
            if self.camera is None:
                print("Camera not initialized")
                return False
                
            # Get available controls
            available_controls = {}
            try:
                available_controls = self.camera.camera_controls
                print(f"Available controls: {list(available_controls.keys())}")
            except Exception as control_error:
                print(f"Warning: Could not check available controls: {str(control_error)}")
            
            # Prepare controls dictionary
            controls = {}
            
            # Add exposure time if supported
            if "ExposureTime" in available_controls:
                controls["ExposureTime"] = int(self.settings['exposure'] * 1000)  # Convert ms to μs
            else:
                print("Warning: Camera does not support ExposureTime control")
                
            # Add gain if supported
            if "AnalogueGain" in available_controls:
                controls["AnalogueGain"] = float(self.settings['gain'])
            else:
                print("Warning: Camera does not support AnalogueGain control")
                
            # Add brightness if supported
            if "Brightness" in available_controls:
                controls["Brightness"] = int(self.settings['brightness'])
            else:
                print("Warning: Camera does not support Brightness control")
                
            # Add contrast if supported
            if "Contrast" in available_controls:
                controls["Contrast"] = float(self.settings['contrast'])
            else:
                print("Warning: Camera does not support Contrast control")
            
            # Add sharpness if supported
            if "Sharpness" in available_controls:
                controls["Sharpness"] = int(self.settings['sharpness'])
            else:
                print("Warning: Camera does not support Sharpness control")
            
            # Add saturation if supported
            if "Saturation" in available_controls:
                controls["Saturation"] = int(self.settings['saturation'])
            else:
                print("Warning: Camera does not support Saturation control")
            
            # Add ISO if supported
            if "ISO" in available_controls:
                controls["ISO"] = int(self.settings['iso'])
            else:
                print("Warning: Camera does not support ISO control")
            
            # Add exposure mode if supported
            if "ExposureMode" in available_controls:
                controls["ExposureMode"] = self.settings['exposure_mode']
            else:
                print("Warning: Camera does not support Exposure Mode control")
            
            # Add white balance mode if supported
            if "WhiteBalanceMode" in available_controls:
                controls["WhiteBalanceMode"] = self.settings['white_balance']
            else:
                print("Warning: Camera does not support White Balance Mode control")
                
            # Add auto exposure if supported
            if "AeEnable" in available_controls:
                controls["AeEnable"] = self.settings['auto_exposure_enabled']
            else:
                print("Warning: Camera does not support Auto Exposure control")
                
            # Add auto white balance if supported
            if "AwbEnable" in available_controls:
                controls["AwbEnable"] = self.settings['auto_white_balance_enabled']
            else:
                print("Warning: Camera does not support Auto White Balance control")
                
            # Add auto gain if supported
            if "AgcEnable" in available_controls:
                controls["AgcEnable"] = self.settings['auto_gain_enabled']
            else:
                print("Warning: Camera does not support Auto Gain control")
                
            # Add noise reduction mode if supported
            if "NoiseReductionMode" in available_controls:
                controls["NoiseReductionMode"] = self.settings['noise_reduction_mode']
            else:
                print("Warning: Camera does not support Noise Reduction Mode control")
            
            # Apply the settings
            print(f"Applying camera controls: {controls}")
            self.camera.set_controls(controls)
            
            print("All camera settings applied")
            return True
            
        except Exception as e:
            print(f"Error applying settings: {str(e)}")
            return False
    
    def get_setting(self, setting: str) -> Optional[Union[float, int]]:
        """Get a camera setting"""
        if setting in self.settings:
            return self.settings[setting]
        return None
    
    def pause(self) -> None:
        """Pause the camera capture without stopping the thread"""
        self.is_paused = True
        print("Camera capture paused")
    
    def resume(self) -> None:
        """Resume the camera capture"""
        self.is_paused = False
        print("Camera capture resumed")
    
    def toggle_pause(self) -> bool:
        """Toggle the pause state of the camera capture"""
        if self.is_paused:
            self.resume()
            return False  # No longer paused
        else:
            self.pause()
            return True  # Now paused
            
    def enable_histogram(self) -> None:
        """Enable histogram generation"""
        self.histogram_enabled = True
        print("Histogram generation enabled")
        
    def disable_histogram(self) -> None:
        """Disable histogram generation"""
        self.histogram_enabled = False
        print("Histogram generation disabled")
        
    def toggle_histogram(self) -> bool:
        """Toggle histogram generation on/off"""
        self.histogram_enabled = not self.histogram_enabled
        print(f"Histogram generation {'enabled' if self.histogram_enabled else 'disabled'}")
        return self.histogram_enabled
        
    def is_histogram_enabled(self) -> bool:
        """Check if histogram generation is enabled"""
        return self.histogram_enabled
        
    def enable_auto_adjustments(self) -> None:
        """Enable automatic camera adjustments (auto exposure, white balance, etc.)"""
        self.auto_adjustments_enabled = True
        print("Automatic camera adjustments enabled")
        self.apply_all_settings()  # Apply settings with auto adjustments enabled
        
    def disable_auto_adjustments(self) -> None:
        """Disable automatic camera adjustments (auto exposure, white balance, etc.)"""
        self.auto_adjustments_enabled = False
        print("Automatic camera adjustments disabled")
        self.apply_all_settings()  # Apply settings with auto adjustments disabled
        
    def toggle_auto_adjustments(self) -> bool:
        """Toggle automatic camera adjustments on/off"""
        self.auto_adjustments_enabled = not self.auto_adjustments_enabled
        print(f"Automatic camera adjustments {'enabled' if self.auto_adjustments_enabled else 'disabled'}")
        self.apply_all_settings()  # Apply the change immediately
        return self.auto_adjustments_enabled
        
    def is_auto_adjustments_enabled(self) -> bool:
        """Check if automatic camera adjustments are enabled"""
        return self.auto_adjustments_enabled


class CameraGUI:
    """GUI for camera control and display"""
    
    def __init__(self, root: tk.Tk, camera: Optional[CameraController] = None):
        """Initialize the camera control GUI"""
        self.root = root
        self.root.title("IMX296 Camera Control")
        
        # Create camera controller if not provided
        if camera is None:
            self.camera = CameraController()
        else:
            self.camera = camera
        
        # Preview dimensions
        self.preview_width = 640
        self.preview_height = 480
        
        # Pause state
        self.is_paused = False
        
        # Create the GUI
        self._create_widgets()
        
        # Start the camera
        if not self.camera.is_running:
            self.camera.start()
        
        # Start the update loop
        self._update_preview()
    
    def _create_widgets(self):
        """Create the GUI widgets"""
        # Main notebook
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Preview tab
        preview_tab = ttk.Frame(notebook)
        notebook.add(preview_tab, text="Preview")
        
        # Create preview area
        self._create_preview_area(preview_tab)
        
        # Status bar
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=2)
        
        self.status_var = tk.StringVar(value="Camera ready")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, anchor=tk.W)
        status_label.pack(fill=tk.X, padx=5)
    
    def _create_preview_area(self, parent):
        """Create the camera preview area"""
        # Preview frame
        preview_frame = ttk.Frame(parent)
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Split into preview and histogram/settings
        preview_pane = ttk.Frame(preview_frame)
        preview_pane.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        right_pane = ttk.Frame(preview_frame)
        right_pane.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)
        
        # Preview canvas
        self.preview_canvas = tk.Canvas(preview_pane, 
                                       width=self.preview_width, 
                                       height=self.preview_height,
                                       bg="black")
        self.preview_canvas.pack(pady=5)
        
        # Add intensity info label
        self.intensity_info = ttk.Label(preview_pane, 
                                      text="Intensity (8-bit) - Max: 0, Mean: 0",
                                      font=("Arial", 10))
        self.intensity_info.pack(pady=5)
        
        # Control buttons for preview pane
        control_frame = ttk.Frame(preview_pane)
        control_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(control_frame, text="Capture", command=self._on_capture).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Save", command=self._on_save).pack(side=tk.LEFT, padx=5)
        
        # Add Apply and Reset buttons to control frame
        ttk.Button(control_frame, text="Apply Settings", command=self._on_apply_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Reset Settings", command=self._on_reset).pack(side=tk.LEFT, padx=5)
        
        # Pause button with variable text
        self.pause_text = tk.StringVar(value="Pause")
        self.pause_button = ttk.Button(control_frame, textvariable=self.pause_text, command=self._on_toggle_pause)
        self.pause_button.pack(side=tk.LEFT, padx=5)
        
        # Add checkboxes frame under the control buttons
        checkbox_frame = ttk.Frame(preview_pane)
        checkbox_frame.pack(fill=tk.X, pady=2)
        
        # Auto adjustments toggle checkbox
        self.auto_adjustments_var = tk.BooleanVar(value=self.camera.is_auto_adjustments_enabled())
        self.auto_adjustments_checkbox = ttk.Checkbutton(
            checkbox_frame, 
            text="Enable Auto Adjustments", 
            variable=self.auto_adjustments_var,
            command=self._on_toggle_auto_adjustments
        )
        self.auto_adjustments_checkbox.pack(side=tk.LEFT, padx=5)
        
        # Histogram toggle checkbox
        self.histogram_enabled_var = tk.BooleanVar(value=self.camera.is_histogram_enabled())
        self.histogram_checkbox = ttk.Checkbutton(
            checkbox_frame, 
            text="Enable Histogram", 
            variable=self.histogram_enabled_var,
            command=self._on_toggle_histogram
        )
        self.histogram_checkbox.pack(side=tk.LEFT, padx=5)
        
        # Histogram canvas - reduced size
        histogram_frame = ttk.LabelFrame(right_pane, text="Histogram")
        histogram_frame.pack(pady=5, fill=tk.X)
        
        self.histogram_canvas = tk.Canvas(histogram_frame,
                                        width=450,
                                        height=350,
                                        bg="white")
        self.histogram_canvas.pack(pady=5)
        
        # Create a scrollable frame for camera settings
        settings_frame = ttk.LabelFrame(right_pane, text="Camera Settings")
        settings_frame.pack(fill=tk.X, pady=10, padx=5)
        
        # Create a canvas with scrollbar for the settings
        settings_canvas = tk.Canvas(settings_frame, highlightthickness=0)
        settings_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add a scrollbar to the canvas
        settings_scrollbar = ttk.Scrollbar(settings_frame, orient=tk.VERTICAL, command=settings_canvas.yview)
        settings_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Configure the canvas
        settings_canvas.configure(yscrollcommand=settings_scrollbar.set)
        settings_canvas.bind('<Configure>', lambda e: settings_canvas.configure(scrollregion=settings_canvas.bbox("all")))
        
        # Create a frame inside the canvas which will be scrolled
        scrollable_settings = ttk.Frame(settings_canvas)
        settings_canvas.create_window((0, 0), window=scrollable_settings, anchor="nw", width=settings_canvas.winfo_reqwidth())
        
        # Create numerical entry fields instead of sliders
        # Exposure control
        exposure_frame = ttk.Frame(scrollable_settings)
        exposure_frame.pack(fill=tk.X, pady=2)
        ttk.Label(exposure_frame, text="Exposure (ms):").pack(side=tk.LEFT, padx=5)
        self.exposure_var = tk.StringVar(value=str(self.camera.settings['exposure']))
        exposure_entry = ttk.Entry(exposure_frame, textvariable=self.exposure_var, width=8)
        exposure_entry.pack(side=tk.RIGHT, padx=5)
        
        # Gain control
        gain_frame = ttk.Frame(scrollable_settings)
        gain_frame.pack(fill=tk.X, pady=2)
        ttk.Label(gain_frame, text="Gain:").pack(side=tk.LEFT, padx=5)
        self.gain_var = tk.StringVar(value=str(self.camera.settings['gain']))
        gain_entry = ttk.Entry(gain_frame, textvariable=self.gain_var, width=8)
        gain_entry.pack(side=tk.RIGHT, padx=5)
        
        # Brightness control
        brightness_frame = ttk.Frame(scrollable_settings)
        brightness_frame.pack(fill=tk.X, pady=2)
        ttk.Label(brightness_frame, text="Brightness:").pack(side=tk.LEFT, padx=5)
        self.brightness_var = tk.StringVar(value=str(self.camera.settings['brightness']))
        brightness_entry = ttk.Entry(brightness_frame, textvariable=self.brightness_var, width=8)
        brightness_entry.pack(side=tk.RIGHT, padx=5)
        
        # Contrast control
        contrast_frame = ttk.Frame(scrollable_settings)
        contrast_frame.pack(fill=tk.X, pady=2)
        ttk.Label(contrast_frame, text="Contrast:").pack(side=tk.LEFT, padx=5)
        self.contrast_var = tk.StringVar(value=str(self.camera.settings['contrast']))
        contrast_entry = ttk.Entry(contrast_frame, textvariable=self.contrast_var, width=8)
        contrast_entry.pack(side=tk.RIGHT, padx=5)
        
        # Add additional camera settings that might be available in the camera
        # Sharpness control
        sharpness_frame = ttk.Frame(scrollable_settings)
        sharpness_frame.pack(fill=tk.X, pady=2)
        ttk.Label(sharpness_frame, text="Sharpness:").pack(side=tk.LEFT, padx=5)
        self.sharpness_var = tk.StringVar(value="0")
        sharpness_entry = ttk.Entry(sharpness_frame, textvariable=self.sharpness_var, width=8)
        sharpness_entry.pack(side=tk.RIGHT, padx=5)
        
        # Saturation control
        saturation_frame = ttk.Frame(scrollable_settings)
        saturation_frame.pack(fill=tk.X, pady=2)
        ttk.Label(saturation_frame, text="Saturation:").pack(side=tk.LEFT, padx=5)
        self.saturation_var = tk.StringVar(value="0")
        saturation_entry = ttk.Entry(saturation_frame, textvariable=self.saturation_var, width=8)
        saturation_entry.pack(side=tk.RIGHT, padx=5)
        
        # ISO control
        iso_frame = ttk.Frame(scrollable_settings)
        iso_frame.pack(fill=tk.X, pady=2)
        ttk.Label(iso_frame, text="ISO:").pack(side=tk.LEFT, padx=5)
        self.iso_var = tk.StringVar(value="100")
        iso_entry = ttk.Entry(iso_frame, textvariable=self.iso_var, width=8)
        iso_entry.pack(side=tk.RIGHT, padx=5)
        
        # Exposure mode dropdown
        exp_mode_frame = ttk.Frame(scrollable_settings)
        exp_mode_frame.pack(fill=tk.X, pady=2)
        ttk.Label(exp_mode_frame, text="Exposure Mode:").pack(side=tk.LEFT, padx=5)
        self.exp_mode_var = tk.StringVar(value="auto")
        exp_mode_combo = ttk.Combobox(exp_mode_frame, textvariable=self.exp_mode_var, 
                                     values=["auto", "manual", "night", "backlight", "spotlight"], 
                                     width=10, state="readonly")
        exp_mode_combo.pack(side=tk.RIGHT, padx=5)
        
        # White balance mode dropdown
        wb_mode_frame = ttk.Frame(scrollable_settings)
        wb_mode_frame.pack(fill=tk.X, pady=2)
        ttk.Label(wb_mode_frame, text="White Balance:").pack(side=tk.LEFT, padx=5)
        self.wb_mode_var = tk.StringVar(value="auto")
        wb_mode_combo = ttk.Combobox(wb_mode_frame, textvariable=self.wb_mode_var, 
                                    values=["auto", "sunlight", "cloudy", "shade", "tungsten", "fluorescent", "incandescent", "flash", "horizon"], 
                                    width=10, state="readonly")
        wb_mode_combo.pack(side=tk.RIGHT, padx=5)
        
        # Add auto adjustment controls
        auto_frame = ttk.LabelFrame(scrollable_settings, text="Auto Adjustments")
        auto_frame.pack(fill=tk.X, pady=5)
        
        # Auto exposure toggle
        auto_exp_frame = ttk.Frame(auto_frame)
        auto_exp_frame.pack(fill=tk.X, pady=2)
        self.auto_exposure_var = tk.BooleanVar(value=self.camera.settings['auto_exposure_enabled'])
        auto_exposure_cb = ttk.Checkbutton(
            auto_exp_frame, 
            text="Auto Exposure", 
            variable=self.auto_exposure_var,
            command=lambda: self._on_toggle_auto_setting('auto_exposure_enabled', self.auto_exposure_var)
        )
        auto_exposure_cb.pack(side=tk.LEFT, padx=5)
        
        # Auto white balance toggle
        auto_wb_frame = ttk.Frame(auto_frame)
        auto_wb_frame.pack(fill=tk.X, pady=2)
        self.auto_wb_var = tk.BooleanVar(value=self.camera.settings['auto_white_balance_enabled'])
        auto_wb_cb = ttk.Checkbutton(
            auto_wb_frame, 
            text="Auto White Balance", 
            variable=self.auto_wb_var,
            command=lambda: self._on_toggle_auto_setting('auto_white_balance_enabled', self.auto_wb_var)
        )
        auto_wb_cb.pack(side=tk.LEFT, padx=5)
        
        # Auto gain toggle
        auto_gain_frame = ttk.Frame(auto_frame)
        auto_gain_frame.pack(fill=tk.X, pady=2)
        self.auto_gain_var = tk.BooleanVar(value=self.camera.settings['auto_gain_enabled'])
        auto_gain_cb = ttk.Checkbutton(
            auto_gain_frame, 
            text="Auto Gain", 
            variable=self.auto_gain_var,
            command=lambda: self._on_toggle_auto_setting('auto_gain_enabled', self.auto_gain_var)
        )
        auto_gain_cb.pack(side=tk.LEFT, padx=5)
        
        # Noise reduction mode dropdown
        noise_frame = ttk.Frame(auto_frame)
        noise_frame.pack(fill=tk.X, pady=2)
        ttk.Label(noise_frame, text="Noise Reduction:").pack(side=tk.LEFT, padx=5)
        self.noise_mode_var = tk.StringVar(value=str(self.camera.settings['noise_reduction_mode']))
        noise_mode_combo = ttk.Combobox(
            noise_frame, 
            textvariable=self.noise_mode_var,
            values=["0 (Off)", "1 (Fast)", "2 (High Quality)", "3 (Minimal)"],
            width=15,
            state="readonly"
        )
        noise_mode_combo.pack(side=tk.RIGHT, padx=5)
        noise_mode_combo.bind("<<ComboboxSelected>>", lambda e: self._on_noise_mode_changed())
        
        # Set a fixed height for the settings canvas
        settings_canvas.config(height=300)
    
    def _update_setting_from_entry(self, setting, var):
        """Update a camera setting from an entry field"""
        try:
            value = var.get()
            if setting in ['exposure', 'gain', 'contrast']:
                value = float(value)
            elif setting == 'brightness':
                value = int(value)
            elif setting == 'sharpness':
                value = int(value)
            elif setting == 'saturation':
                value = int(value)
            elif setting == 'iso':
                value = int(value)
                
            # Only update the internal setting value, don't apply to camera yet
            self.camera.settings[setting] = value
            self.status_var.set(f"Setting {setting} ready to apply")
            
        except ValueError:
            self.status_var.set(f"Invalid value for {setting}")
            # Reset to current value
            if setting in ['exposure', 'gain', 'contrast', 'brightness', 'sharpness', 'saturation', 'iso']:
                var.set(str(self.camera.get_setting(setting)))
    
    def _on_capture(self):
        """Handle capture button click"""
        try:
            # Get the current frame
            frame = self.camera.get_latest_frame()
            
            if frame is None:
                self.status_var.set("No frame available to capture")
                return
                
            # Create default capture directory
            default_dir = "/home/surena/slm-control/Captures"
            
            # Create a default filename with timestamp
            default_filename = f"image_{time.strftime('%Y%m%d_%H%M%S')}.png"
            default_path = os.path.join(default_dir, default_filename)
            
            # Make sure the default directory exists
            os.makedirs(default_dir, exist_ok=True)
            
            # Check if zenity is available
            zenity_available = False
            try:
                # Check if zenity is installed
                result = subprocess.run(["which", "zenity"], capture_output=True, text=True)
                zenity_available = result.returncode == 0
            except Exception:
                zenity_available = False
                
            # Use zenity if available
            if zenity_available:
                try:
                    # Run zenity file selection dialog
                    cmd = [
                        "zenity", "--file-selection",
                        "--save",
                        "--filename", default_path,
                        "--title", "Save Captured Image",
                        "--file-filter", "Images | *.png *.jpg *.jpeg *.tif *.tiff"
                    ]
                    
                    # Execute zenity command
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    # Check if user canceled
                    if result.returncode != 0:
                        self.status_var.set("Capture canceled")
                        return
                        
                    # Get the selected filename
                    filename = result.stdout.strip()
                    
                    # If empty, user canceled
                    if not filename:
                        self.status_var.set("Capture canceled")
                        return
                        
                    # Save the frame
                    if self.camera.save_frame(filename, frame):
                        self.status_var.set(f"Image captured and saved to {filename}")
                    else:
                        self.status_var.set("Failed to save captured image")
                        
                except Exception as zenity_error:
                    print(f"Zenity error: {str(zenity_error)}")
                    self.status_var.set(f"Zenity error: {str(zenity_error)}")
                    
                    # Fallback to default path
                    if self.camera.save_frame(default_path, frame):
                        self.status_var.set(f"Image captured and saved to {default_path}")
                    else:
                        self.status_var.set("Failed to save captured image")
            else:
                # Zenity not available, use default path
                print("Zenity not available, saving to default path")
                if self.camera.save_frame(default_path, frame):
                    self.status_var.set(f"Image captured and saved to {default_path}")
                else:
                    self.status_var.set("Failed to save captured image")
                
        except Exception as e:
            self.status_var.set(f"Capture error: {str(e)}")
            print(f"Capture error: {str(e)}")
            traceback.print_exc()
    
    def _on_save(self):
        """Handle save button click"""
        try:
            # Get the current frame
            frame = self.camera.get_latest_frame()
            
            if frame is None:
                self.status_var.set("No frame available to save")
                return
                
            # Create default capture directory
            default_dir = "/home/surena/slm-control/Captures"
            
            # Create a default filename with timestamp
            default_filename = f"image_{time.strftime('%Y%m%d_%H%M%S')}.png"
            default_path = os.path.join(default_dir, default_filename)
            
            # Make sure the default directory exists
            os.makedirs(default_dir, exist_ok=True)
            
            # Check if zenity is available
            zenity_available = False
            try:
                # Check if zenity is installed
                result = subprocess.run(["which", "zenity"], capture_output=True, text=True)
                zenity_available = result.returncode == 0
            except Exception:
                zenity_available = False
                
            # Use zenity if available
            if zenity_available:
                try:
                    # Run zenity file selection dialog
                    cmd = [
                        "zenity", "--file-selection",
                        "--save",
                        "--filename", default_path,
                        "--title", "Save Camera Image",
                        "--file-filter", "Images | *.png *.jpg *.jpeg *.tif *.tiff"
                    ]
                    
                    # Execute zenity command
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    # Check if user canceled
                    if result.returncode != 0:
                        self.status_var.set("Save canceled")
                        return
                        
                    # Get the selected filename
                    filename = result.stdout.strip()
                    
                    # If empty, user canceled
                    if not filename:
                        self.status_var.set("Save canceled")
                        return
                        
                    # Save the frame
                    if self.camera.save_frame(filename, frame):
                        self.status_var.set(f"Image saved to {filename}")
                    else:
                        self.status_var.set("Failed to save image")
                        
                except Exception as zenity_error:
                    print(f"Zenity error: {str(zenity_error)}")
                    self.status_var.set(f"Zenity error: {str(zenity_error)}")
                    
                    # Fallback to default path
                    if self.camera.save_frame(default_path, frame):
                        self.status_var.set(f"Image saved to {default_path}")
                    else:
                        self.status_var.set("Failed to save image")
            else:
                # Zenity not available, use default path
                print("Zenity not available, saving to default path")
                if self.camera.save_frame(default_path, frame):
                    self.status_var.set(f"Image saved to {default_path}")
                else:
                    self.status_var.set("Failed to save image")
                
        except Exception as e:
            self.status_var.set(f"Save error: {str(e)}")
            print(f"Save error: {str(e)}")
            traceback.print_exc()
    
    def _on_reset(self):
        """Handle reset button click"""
        try:
            # Reset camera settings to defaults
            default_settings = {
                'exposure': 20.0,
                'gain': 1.0,
                'brightness': 0,
                'contrast': 1.0,
                'sharpness': 0,
                'saturation': 0,
                'iso': 100,
                'exposure_mode': 'auto',
                'white_balance': 'auto',
            }
            
            for setting, value in default_settings.items():
                self.camera.set_setting(setting, value)
                
            # Update GUI controls
            self.exposure_var.set(str(default_settings['exposure']))
            self.gain_var.set(str(default_settings['gain']))
            self.brightness_var.set(str(default_settings['brightness']))
            self.contrast_var.set(str(default_settings['contrast']))
            self.sharpness_var.set(str(default_settings['sharpness']))
            self.saturation_var.set(str(default_settings['saturation']))
            self.iso_var.set(str(default_settings['iso']))
            self.exp_mode_var.set(str(default_settings['exposure_mode']))
            self.wb_mode_var.set(str(default_settings['white_balance']))
            
            self.status_var.set("Settings reset to defaults")
        except Exception as e:
            self.status_var.set(f"Reset error: {str(e)}")
    
    def _on_toggle_pause(self):
        """Handle pause/resume button click"""
        try:
            # Toggle camera pause state
            is_now_paused = self.camera.toggle_pause()
            
            # Update button text
            if is_now_paused:
                self.pause_text.set("Resume")
                self.status_var.set("Camera paused")
            else:
                self.pause_text.set("Pause")
                self.status_var.set("Camera resumed")
                
            # Update internal state
            self.is_paused = is_now_paused
            
        except Exception as e:
            self.status_var.set(f"Error toggling pause: {str(e)}")
    
    def _update_preview(self):
        """Update the preview display"""
        try:
            # Get the latest frame
            frame = self.camera.get_latest_frame()
            
            if frame is not None:
                # Get intensity stats for display
                stats = self.camera.get_intensity_stats(frame)
                self.intensity_info.config(
                    text=f"Intensity (8-bit) - Min: {stats['min']:.1f}, Max: {stats['max']:.1f}, " +
                         f"Mean: {stats['mean']:.1f}, StdDev: {stats['std']:.1f}"
                )
                
                # Resize if needed
                if frame.shape[1] != self.preview_width or frame.shape[0] != self.preview_height:
                    frame = cv2.resize(frame, (self.preview_width, self.preview_height))
                
                # Convert to RGB format for PIL
                display_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                
                # Convert to PIL Image and then to PhotoImage
                pil_img = Image.fromarray(display_rgb)
                self.tk_img = ImageTk.PhotoImage(image=pil_img)
                self.preview_canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
                
                # Update status
                if self.is_paused:
                    status = "Camera paused"
                else:
                    status = f"Camera running - {frame.shape[1]}x{frame.shape[0]} at 8-bit depth (0-255)"
                self.status_var.set(status)
                
                # Update histogram
                hist_img = self.camera.get_latest_histogram()
                if hist_img is not None:
                    # Convert RGBA to RGB
                    hist_rgb = cv2.cvtColor(hist_img, cv2.COLOR_RGBA2RGB)
                    
                    # Resize to fit canvas
                    hist_rgb = cv2.resize(hist_rgb, (450, 350))
                    
                    # Convert to PIL and then to PhotoImage
                    hist_pil = Image.fromarray(hist_rgb)
                    self.hist_tk_img = ImageTk.PhotoImage(image=hist_pil)
                    self.histogram_canvas.create_image(0, 0, anchor=tk.NW, image=self.hist_tk_img)
            
        except Exception as e:
            self.status_var.set(f"Preview error: {str(e)}")
        
        # Schedule the next update
        self.root.after(50, self._update_preview)
    
    def _on_apply_settings(self):
        """Handle apply settings button click"""
        try:
            # Update all settings from entry fields
            self._update_setting_from_entry('exposure', self.exposure_var)
            self._update_setting_from_entry('gain', self.gain_var)
            self._update_setting_from_entry('brightness', self.brightness_var)
            self._update_setting_from_entry('contrast', self.contrast_var)
            self._update_setting_from_entry('sharpness', self.sharpness_var)
            self._update_setting_from_entry('saturation', self.saturation_var)
            self._update_setting_from_entry('iso', self.iso_var)
            
            # Update dropdown selections
            self.camera.settings['exposure_mode'] = self.exp_mode_var.get()
            self.camera.settings['white_balance'] = self.wb_mode_var.get()
            
            # Update auto adjustment settings
            self.camera.settings['auto_exposure_enabled'] = self.auto_exposure_var.get()
            self.camera.settings['auto_white_balance_enabled'] = self.auto_wb_var.get()
            self.camera.settings['auto_gain_enabled'] = self.auto_gain_var.get()
            
            # Update noise reduction mode
            mode_str = self.noise_mode_var.get()
            if mode_str.startswith("0"):
                self.camera.settings['noise_reduction_mode'] = 0
            elif mode_str.startswith("1"):
                self.camera.settings['noise_reduction_mode'] = 1
            elif mode_str.startswith("2"):
                self.camera.settings['noise_reduction_mode'] = 2
            elif mode_str.startswith("3"):
                self.camera.settings['noise_reduction_mode'] = 3
            
            # Apply all settings to the camera
            success = self.camera.apply_all_settings()
            
            if success:
                self.status_var.set("Camera settings applied successfully")
            else:
                self.status_var.set("Failed to apply some camera settings")
                
        except Exception as e:
            self.status_var.set(f"Error applying settings: {str(e)}")
            print(f"Detailed error: {str(e)}")
            traceback.print_exc()
    
    def _on_toggle_histogram(self):
        """Handle histogram toggle checkbox"""
        try:
            # Update histogram state
            self.camera.histogram_enabled = self.histogram_enabled_var.get()
            self.status_var.set(f"Histogram generation {'enabled' if self.camera.histogram_enabled else 'disabled'}")
        except Exception as e:
            self.status_var.set(f"Histogram toggle error: {str(e)}")
    
    def _on_toggle_auto_adjustments(self):
        """Handle auto adjustments toggle checkbox"""
        try:
            # Update auto adjustments state
            self.camera.auto_adjustments_enabled = self.auto_adjustments_var.get()
            self.status_var.set(f"Auto adjustments {'enabled' if self.camera.auto_adjustments_enabled else 'disabled'}")
            self.camera.apply_all_settings()  # Apply the change immediately
        except Exception as e:
            self.status_var.set(f"Auto adjustments toggle error: {str(e)}")
    
    def _on_toggle_auto_setting(self, setting, var):
        """Handle toggle of an auto setting"""
        try:
            # Update the setting
            self.camera.settings[setting] = var.get()
            self.status_var.set(f"Setting {setting} {'enabled' if var.get() else 'disabled'}")
            self.camera.apply_all_settings()  # Apply the change immediately
        except Exception as e:
            self.status_var.set(f"Error toggling {setting}: {str(e)}")
    
    def _on_noise_mode_changed(self):
        """Handle change in noise reduction mode"""
        try:
            # Get the selected mode
            mode = self.noise_mode_var.get()
            
            # Convert mode to integer
            if mode == "0 (Off)":
                mode = 0
            elif mode == "1 (Fast)":
                mode = 1
            elif mode == "2 (High Quality)":
                mode = 2
            elif mode == "3 (Minimal)":
                mode = 3
            else:
                self.status_var.set("Invalid noise reduction mode")
                return
            
            # Update the setting
            self.camera.settings['noise_reduction_mode'] = mode
            self.status_var.set(f"Noise reduction mode set to {mode}")
            self.camera.apply_all_settings()  # Apply the change immediately
        except Exception as e:
            self.status_var.set(f"Error changing noise reduction mode: {str(e)}")


# Example usage
if __name__ == "__main__":
    print("Starting camera controller...")
    
    try:
        # Create camera controller
        camera = CameraController(
            resolution=(1456, 1088),  # IMX296 native resolution
            device="/dev/video0"      # Default video device
        )
        
        # Create GUI
        root = tk.Tk()
        root.title("IMX296 Camera Control")
        
        # Create camera control GUI
        gui = CameraGUI(root, camera)
        
        # Run the GUI
        root.mainloop()
        
        # Clean up
        camera.stop()
        print("Camera stopped")
        
    except KeyboardInterrupt:
        print("Interrupted by user")
        try:
            camera.stop()
            print("Camera stopped")
        except:
            pass
        
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        try:
            camera.stop()
            print("Camera stopped")
        except:
            pass
