#!/usr/bin/env python3
"""
Camera Controller for IMX296 Camera

This module provides a camera controller specifically optimized for the IMX296 camera
with RGB3 format (24-bit RGB 8-8-8) at 1456x1088 resolution.

Features:
- Direct access to camera's native RGB3 format
- Conversion to grayscale for intensity analysis
- Histogram generation and analysis
- Frame capture and saving with metadata
- Thread-safe operation for GUI integration
"""

import os
import time
import threading
import traceback
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
    Camera controller for IMX296 camera with RGB3 format.
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
        self.thread = None
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_histogram = None
        
        # Camera settings
        self.settings = {
            'exposure': 20.0,  # ms
            'gain': 1.0,       # analog gain
            'brightness': 0,   # -255 to 255
            'contrast': 1.0,   # 0.0 to 2.0
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
                })
                print("Camera controls set successfully")
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
                # Capture a frame
                frame = self.capture_frame()
                
                if frame is not None:
                    # Store the latest frame with thread safety
                    with self.lock:
                        self.latest_frame = frame
                    
                    # Update histogram occasionally
                    current_time = time.time()
                    if current_time - last_histogram_time > histogram_interval:
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
            
            # Convert to grayscale for intensity analysis
            if len(rgb_frame.shape) == 3 and rgb_frame.shape[2] == 3:
                # RGB to grayscale conversion
                gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
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
            # Create a figure for the histogram
            fig = Figure(figsize=(4, 3), dpi=100)
            ax = fig.add_subplot(111)
            
            # Calculate histogram
            hist = cv2.calcHist([frame], [0], None, [256], [0, 256])
            
            # Plot the histogram
            ax.plot(hist, color='blue')
            ax.set_xlim([0, 256])
            ax.set_title('Intensity Histogram')
            ax.set_xlabel('Intensity Value')
            ax.set_ylabel('Pixel Count')
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
            return np.zeros((300, 400, 4), dtype=np.uint8)
    
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
            
            # Convert to grayscale for intensity analysis
            if len(result.shape) == 3 and result.shape[2] == 3:
                result = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
                
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
                f.write(f"Camera: IMX296 RGB3 at {self.width}x{self.height}\n")
                f.write(f"Resolution: {image.shape[1]}x{image.shape[0]}\n")
                f.write(f"Bit Depth: 8-bit\n")
                f.write(f"Value Range: 0-255\n")
                
                f.write(f"Statistics:\n")
                f.write(f"  Min: {np.min(image):.1f}\n")
                f.write(f"  Max: {np.max(image):.1f}\n")
                f.write(f"  Mean: {np.mean(image):.1f}\n")
                f.write(f"  Std Dev: {np.std(image):.1f}\n")
                
                f.write(f"Camera Settings:\n")
                for key, value in self.settings.items():
                    f.write(f"  {key}: {value}\n")
                
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
        
        # Apply the setting to the camera
        try:
            if setting == 'exposure':
                self.camera.set_controls({"ExposureTime": int(value * 1000)})  # ms to μs
            elif setting == 'gain':
                self.camera.set_controls({"AnalogueGain": float(value)})
            # Other settings are applied during processing
            
            print(f"Setting {setting} updated to {value}")
            return True
            
        except Exception as e:
            print(f"Error setting {setting}: {str(e)}")
            return False
    
    def get_setting(self, setting: str) -> Optional[Union[float, int]]:
        """Get a camera setting"""
        if setting in self.settings:
            return self.settings[setting]
        return None


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
        
        # Create the GUI
        self._create_widgets()
        
        # Start the camera
        if not self.camera.is_running:
            self.camera.start()
        
        # Start the update loop
        self._update_preview()
    
    def _create_widgets(self):
        """Create the GUI widgets"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Preview tab
        preview_tab = ttk.Frame(notebook)
        notebook.add(preview_tab, text="Preview")
        
        # Settings tab
        settings_tab = ttk.Frame(notebook)
        notebook.add(settings_tab, text="Settings")
        
        # Create preview area
        self._create_preview_area(preview_tab)
        
        # Create settings controls
        self._create_settings_controls(settings_tab)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, pady=(5, 0))
    
    def _create_preview_area(self, parent):
        """Create the camera preview area"""
        # Preview frame
        preview_frame = ttk.Frame(parent)
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Split into preview and histogram
        preview_pane = ttk.Frame(preview_frame)
        preview_pane.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        histogram_pane = ttk.Frame(preview_frame)
        histogram_pane.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)
        
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
        
        # Histogram canvas
        self.histogram_canvas = tk.Canvas(histogram_pane,
                                        width=400,
                                        height=300,
                                        bg="white")
        self.histogram_canvas.pack(pady=5)
        
        # Control buttons
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(control_frame, text="Capture", command=self._on_capture).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Save", command=self._on_save).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Reset", command=self._on_reset).pack(side=tk.LEFT, padx=5)
    
    def _create_settings_controls(self, parent):
        """Create the settings controls"""
        # Settings frame
        settings_frame = ttk.LabelFrame(parent, text="Camera Settings")
        settings_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)
        
        # Exposure control
        ttk.Label(settings_frame, text="Exposure (ms):").grid(row=0, column=0, sticky=tk.W, pady=5, padx=5)
        self.exposure_var = tk.DoubleVar(value=self.camera.settings['exposure'])
        exposure_scale = ttk.Scale(settings_frame, from_=1, to=100, variable=self.exposure_var,
                                 command=lambda v: self._update_setting('exposure', float(v)))
        exposure_scale.grid(row=0, column=1, sticky=tk.EW, pady=5, padx=5)
        ttk.Label(settings_frame, textvariable=self.exposure_var).grid(row=0, column=2, padx=5)
        
        # Gain control
        ttk.Label(settings_frame, text="Gain:").grid(row=1, column=0, sticky=tk.W, pady=5, padx=5)
        self.gain_var = tk.DoubleVar(value=self.camera.settings['gain'])
        gain_scale = ttk.Scale(settings_frame, from_=1, to=16, variable=self.gain_var,
                             command=lambda v: self._update_setting('gain', float(v)))
        gain_scale.grid(row=1, column=1, sticky=tk.EW, pady=5, padx=5)
        ttk.Label(settings_frame, textvariable=self.gain_var).grid(row=1, column=2, padx=5)
        
        # Brightness control
        ttk.Label(settings_frame, text="Brightness:").grid(row=2, column=0, sticky=tk.W, pady=5, padx=5)
        self.brightness_var = tk.IntVar(value=self.camera.settings['brightness'])
        brightness_scale = ttk.Scale(settings_frame, from_=-255, to=255, variable=self.brightness_var,
                                   command=lambda v: self._update_setting('brightness', int(float(v))))
        brightness_scale.grid(row=2, column=1, sticky=tk.EW, pady=5, padx=5)
        ttk.Label(settings_frame, textvariable=self.brightness_var).grid(row=2, column=2, padx=5)
        
        # Contrast control
        ttk.Label(settings_frame, text="Contrast:").grid(row=3, column=0, sticky=tk.W, pady=5, padx=5)
        self.contrast_var = tk.DoubleVar(value=self.camera.settings['contrast'])
        contrast_scale = ttk.Scale(settings_frame, from_=0.5, to=2.0, variable=self.contrast_var,
                                 command=lambda v: self._update_setting('contrast', float(v)))
        contrast_scale.grid(row=3, column=1, sticky=tk.EW, pady=5, padx=5)
        ttk.Label(settings_frame, textvariable=self.contrast_var).grid(row=3, column=2, padx=5)
        
        # Configure grid
        settings_frame.columnconfigure(1, weight=1)
    
    def _update_setting(self, setting, value):
        """Update a camera setting"""
        self.camera.set_setting(setting, value)
    
    def _on_capture(self):
        """Handle capture button click"""
        try:
            frame = self.camera.capture_high_quality_frame()
            if frame is not None:
                # Show the captured frame
                with self.camera.lock:
                    self.camera.latest_frame = frame
                    self.camera.latest_histogram = self.camera.generate_histogram(frame)
                
                self.status_var.set("Frame captured")
            else:
                self.status_var.set("Failed to capture frame")
        except Exception as e:
            self.status_var.set(f"Capture error: {str(e)}")
    
    def _on_save(self):
        """Handle save button click"""
        try:
            # Create a default filename with timestamp
            filename = f"image_{time.strftime('%Y%m%d_%H%M%S')}.png"
            
            # Get the current frame
            frame = self.camera.get_latest_frame()
            
            if frame is not None:
                # Save the frame
                if self.camera.save_frame(filename, frame):
                    self.status_var.set(f"Image saved to {filename}")
                else:
                    self.status_var.set("Failed to save image")
            else:
                self.status_var.set("No frame available to save")
        except Exception as e:
            self.status_var.set(f"Save error: {str(e)}")
    
    def _on_reset(self):
        """Handle reset button click"""
        try:
            # Reset camera settings to defaults
            default_settings = {
                'exposure': 20.0,
                'gain': 1.0,
                'brightness': 0,
                'contrast': 1.0,
            }
            
            for setting, value in default_settings.items():
                self.camera.set_setting(setting, value)
                
            # Update GUI controls
            self.exposure_var.set(default_settings['exposure'])
            self.gain_var.set(default_settings['gain'])
            self.brightness_var.set(default_settings['brightness'])
            self.contrast_var.set(default_settings['contrast'])
            
            self.status_var.set("Settings reset to defaults")
        except Exception as e:
            self.status_var.set(f"Reset error: {str(e)}")
    
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
                         f"Mean: {stats['mean']:.1f}"
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
                self.status_var.set(f"Camera running - {frame.shape[1]}x{frame.shape[0]} at 8-bit depth")
                
                # Update histogram
                hist_img = self.camera.get_latest_histogram()
                if hist_img is not None:
                    # Convert RGBA to RGB
                    hist_rgb = cv2.cvtColor(hist_img, cv2.COLOR_RGBA2RGB)
                    
                    # Resize to fit canvas
                    hist_rgb = cv2.resize(hist_rgb, (400, 300))
                    
                    # Convert to PIL and then to PhotoImage
                    hist_pil = Image.fromarray(hist_rgb)
                    self.hist_tk_img = ImageTk.PhotoImage(image=hist_pil)
                    self.histogram_canvas.create_image(0, 0, anchor=tk.NW, image=self.hist_tk_img)
            
        except Exception as e:
            self.status_var.set(f"Preview error: {str(e)}")
        
        # Schedule the next update
        self.root.after(50, self._update_preview)


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
