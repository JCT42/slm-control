"""
Camera Control Module for SLM Control System

This module provides a flexible camera interface for capturing and processing
images from various camera types, with a focus on scientific imaging applications.

Features:
- Support for different camera types (PiCamera, OpenCV webcams, etc.)
- 10-bit intensity capture and processing
- Real-time display capabilities using OpenCV
- Histogram and intensity analysis tools
- Thread-safe operation for integration with GUI applications
- Video recording functionality

Dependencies:
- OpenCV (cv2)
- NumPy
- Threading
"""

import cv2
import numpy as np
import threading
import time
import subprocess
import os
from typing import Tuple, Optional, Dict, Any, Callable
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


class CameraController:
    """
    Main camera controller class that provides a unified interface
    for different camera types and processing methods.
    """
    
    def __init__(self, camera_type: str = "opencv", camera_index: int = 0, 
                 resolution: Tuple[int, int] = (1456, 1088),
                 bit_depth: int = 10):
        """
        Initialize the camera controller.
        
        Args:
            camera_type: Type of camera to use ('opencv', 'picamera', etc.)
            camera_index: Camera device index for OpenCV cameras
            resolution: Desired camera resolution (width, height)
            bit_depth: Bit depth for intensity values (default: 10-bit)
        """
        self.camera_type = camera_type
        self.camera_index = camera_index
        self.width, self.height = resolution
        self.bit_depth = bit_depth
        self.max_value = 2**bit_depth - 1  # e.g., 1023 for 10-bit
        
        # Camera state
        self.camera = None
        self.is_running = False
        self.is_paused = False
        self.current_frame = None
        self.last_captured_frame = None
        
        # Video recording state
        self.is_recording = False
        self.video_writer = None
        self.recording_filename = None
        self.recording_start_time = None
        
        # Thread control
        self.capture_thread = None
        self.lock = threading.Lock()
        
        # Camera settings
        self.settings = {
            'exposure': 20.0,  # ms
            'gain': 1.0,
            'brightness': 0,
            'contrast': 0,
            'saturation': 0,
            'hue': 0,
            'auto_exposure': False,
            'auto_gain': False
        }
        
    def initialize(self) -> bool:
        """
        Initialize the camera based on the selected type.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            if self.camera_type == "opencv":
                return self._initialize_opencv()
            elif self.camera_type == "picamera":
                return self._initialize_picamera()
            else:
                print(f"Unsupported camera type: {self.camera_type}")
                return False
        except Exception as e:
            print(f"Camera initialization error: {str(e)}")
            return False
    
    def _initialize_opencv(self) -> bool:
        """Initialize an OpenCV-based camera"""
        try:
            self.camera = cv2.VideoCapture(self.camera_index)
            
            # Set resolution
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            # Try to set 10-bit pixel format if available
            # Note: This may not be supported on all cameras
            self.camera.set(cv2.CAP_PROP_CONVERT_RGB, False)  # Get raw data
            
            # Apply initial settings
            self._apply_opencv_settings()
            
            # Check if camera is opened successfully
            if not self.camera.isOpened():
                print("Failed to open camera")
                return False
                
            # Read actual resolution (may differ from requested)
            actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"Camera initialized at resolution: {actual_width}x{actual_height}")
            
            return True
            
        except Exception as e:
            print(f"OpenCV camera initialization error: {str(e)}")
            return False
    
    def _initialize_picamera(self) -> bool:
        """Initialize a Raspberry Pi Camera"""
        try:
            # Import PiCamera modules here to avoid dependency issues on non-Pi systems
            from picamera2 import Picamera2
            from libcamera import controls
            
            self.camera = Picamera2()
            
            # Configure for 10-bit capture
            config = self.camera.create_still_configuration(
                main={"size": (self.width, self.height),
                      "format": "RGB888"},
                raw={"size": (self.width, self.height),
                     "format": "SRGGB10"}
            )
            self.camera.configure(config)
            
            # Set initial exposure and gain
            self.camera.set_controls({
                "ExposureTime": int(self.settings['exposure'] * 1000),  # Convert ms to μs
                "AnalogueGain": self.settings['gain']
            })
            
            # Start the camera
            self.camera.start()
            
            return True
            
        except ImportError:
            print("PiCamera modules not available. Are you running on a Raspberry Pi?")
            return False
        except Exception as e:
            print(f"PiCamera initialization error: {str(e)}")
            return False
    
    def _apply_opencv_settings(self) -> None:
        """Apply settings to an OpenCV camera"""
        if self.camera is None or self.camera_type != "opencv":
            return
            
        # Set exposure (convert ms to OpenCV's exposure values)
        self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0 if not self.settings['auto_exposure'] else 1)
        if not self.settings['auto_exposure']:
            # OpenCV exposure is logarithmic and platform-dependent
            # This is an approximation that may need adjustment
            self.camera.set(cv2.CAP_PROP_EXPOSURE, np.log(self.settings['exposure']) / np.log(2))
        
        # Set gain
        self.camera.set(cv2.CAP_PROP_GAIN, self.settings['gain'])
        
        # Set other parameters
        self.camera.set(cv2.CAP_PROP_BRIGHTNESS, self.settings['brightness'])
        self.camera.set(cv2.CAP_PROP_CONTRAST, self.settings['contrast'])
        self.camera.set(cv2.CAP_PROP_SATURATION, self.settings['saturation'])
        self.camera.set(cv2.CAP_PROP_HUE, self.settings['hue'])
    
    def start(self) -> bool:
        """
        Start the camera capture thread.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        if self.is_running:
            print("Camera is already running")
            return True
            
        if self.camera is None:
            if not self.initialize():
                return False
        
        self.is_running = True
        self.is_paused = False
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        return True
    
    def stop(self) -> None:
        """Stop the camera capture thread and release resources"""
        self.is_running = False
        
        # Wait for thread to finish
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        
        # Release camera resources
        if self.camera is not None:
            if self.camera_type == "opencv":
                self.camera.release()
            elif self.camera_type == "picamera":
                self.camera.stop()
                self.camera.close()
            
            self.camera = None
    
    def pause(self) -> None:
        """Pause the camera capture"""
        self.is_paused = True
    
    def resume(self) -> None:
        """Resume the camera capture"""
        self.is_paused = False
    
    def _capture_loop(self) -> None:
        """Main camera capture loop that runs in a separate thread"""
        while self.is_running:
            try:
                if not self.is_paused:
                    frame = self._capture_frame()
                    
                    if frame is not None:
                        with self.lock:
                            self.current_frame = frame
                        
                        # Write frame to video if recording
                        if self.is_recording:
                            self.video_writer.write(frame)
                
                # Sleep to control frame rate
                time.sleep(0.033)  # ~30 FPS
                    
            except Exception as e:
                print(f"Camera capture error: {str(e)}")
                time.sleep(0.1)  # Longer sleep on error
    
    def _capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from the camera.
        
        Returns:
            np.ndarray or None: Captured frame as a numpy array, or None on failure
        """
        if self.camera is None:
            return None
            
        try:
            if self.camera_type == "opencv":
                ret, frame = self.camera.read()
                if not ret:
                    print("Failed to capture frame")
                    return None
                
                # Convert to grayscale while preserving bit depth
                if len(frame.shape) == 3:
                    gray = np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140])
                else:
                    gray = frame
                
                # Scale to 10-bit range if needed
                if self.bit_depth == 10 and gray.max() <= 255:
                    gray = gray * (1023/255)
                
                return gray
                
            elif self.camera_type == "picamera":
                frame = self.camera.capture_array()
                
                # Convert to grayscale while preserving bit depth
                if len(frame.shape) == 3:
                    gray = np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140])
                else:
                    gray = frame
                
                return gray
                
            return None
            
        except Exception as e:
            print(f"Frame capture error: {str(e)}")
            return None
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the current frame (thread-safe).
        
        Returns:
            np.ndarray or None: Current frame, or None if not available
        """
        with self.lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return None
    
    def capture_still(self) -> Optional[np.ndarray]:
        """
        Capture a high-quality still image.
        
        Returns:
            np.ndarray or None: Captured still image, or None on failure
        """
        try:
            # Temporarily pause continuous capture
            was_paused = self.is_paused
            self.pause()
            
            # Wait for any in-progress captures to complete
            time.sleep(0.1)
            
            # Capture a frame
            if self.camera_type == "opencv":
                # For OpenCV, we might want to capture multiple frames and average
                # to reduce noise in still captures
                frames = []
                for _ in range(3):
                    ret, frame = self.camera.read()
                    if ret:
                        if len(frame.shape) == 3:
                            gray = np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140])
                        else:
                            gray = frame
                        frames.append(gray)
                    time.sleep(0.05)
                
                if frames:
                    # Average the frames
                    still_frame = np.mean(frames, axis=0).astype(np.float32)
                    
                    # Scale to 10-bit range if needed
                    if self.bit_depth == 10 and still_frame.max() <= 255:
                        still_frame = still_frame * (1023/255)
                    
                    # Store the captured frame
                    self.last_captured_frame = still_frame
                    
                    # Resume if it wasn't paused before
                    if not was_paused:
                        self.resume()
                        
                    return still_frame
            
            elif self.camera_type == "picamera":
                # For PiCamera, we can request a high-quality still
                still_frame = self.camera.capture_array()
                
                # Convert to grayscale while preserving bit depth
                if len(still_frame.shape) == 3:
                    gray = np.dot(still_frame[..., :3], [0.2989, 0.5870, 0.1140])
                else:
                    gray = still_frame
                
                # Store the captured frame
                self.last_captured_frame = gray
                
                # Resume if it wasn't paused before
                if not was_paused:
                    self.resume()
                    
                return gray
            
            # Resume if it wasn't paused before
            if not was_paused:
                self.resume()
                
            return None
            
        except Exception as e:
            print(f"Still capture error: {str(e)}")
            
            # Make sure to resume if it wasn't paused before
            if not was_paused:
                self.resume()
                
            return None
    
    def save_image(self, filename: str, image: Optional[np.ndarray] = None) -> bool:
        """
        Save an image to disk.
        
        Args:
            filename: Path to save the image
            image: Image to save, or None to use last captured frame
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            # Use provided image or last captured frame
            save_image = image if image is not None else self.last_captured_frame
            
            if save_image is None:
                print("No image available to save")
                return False
            
            # Ensure the filename has an extension
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif')):
                filename += '.png'
            
            # Convert to 16-bit for PNG storage (preserving 10-bit values)
            if self.bit_depth == 10:
                save_img = (save_image / self.max_value * 65535).astype(np.uint16)
            else:
                save_img = save_image.astype(np.uint8)
            
            # Save the image
            cv2.imwrite(filename, save_img)
            
            # Save metadata
            self._save_metadata(filename, save_image)
            
            return True
            
        except Exception as e:
            print(f"Error saving image: {str(e)}")
            return False
    
    def _save_metadata(self, image_filename: str, image: np.ndarray) -> None:
        """Save metadata about the image to a text file"""
        try:
            metadata_filename = image_filename + '.txt'
            
            with open(metadata_filename, 'w') as f:
                f.write(f"Camera Type: {self.camera_type}\n")
                f.write(f"Resolution: {image.shape[1]}x{image.shape[0]}\n")
                f.write(f"Bit Depth: {self.bit_depth}-bit\n")
                f.write(f"Value Range: 0-{self.max_value}\n")
                
                if self.bit_depth == 10:
                    f.write(f"Stored as: 16-bit PNG\n")
                    f.write(f"Conversion: Original {self.bit_depth}-bit value * 65535/{self.max_value}\n")
                
                f.write(f"Statistics:\n")
                f.write(f"  Min: {np.min(image):.1f}\n")
                f.write(f"  Max: {np.max(image):.1f}\n")
                f.write(f"  Mean: {np.mean(image):.1f}\n")
                f.write(f"  Median: {np.median(image):.1f}\n")
                f.write(f"  Standard Deviation: {np.std(image):.1f}\n")
                
                # Add camera settings
                f.write(f"Camera Settings:\n")
                for key, value in self.settings.items():
                    f.write(f"  {key}: {value}\n")
                
                # Add timestamp
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                
        except Exception as e:
            print(f"Error saving metadata: {str(e)}")
    
    def set_setting(self, setting: str, value: Any) -> bool:
        """
        Set a camera setting.
        
        Args:
            setting: Name of the setting to change
            value: New value for the setting
            
        Returns:
            bool: True if setting was changed, False otherwise
        """
        if setting not in self.settings:
            print(f"Unknown setting: {setting}")
            return False
        
        # Update the setting
        self.settings[setting] = value
        
        # Apply the setting to the camera
        if self.camera_type == "opencv":
            self._apply_opencv_settings()
        elif self.camera_type == "picamera" and self.camera is not None:
            if setting == 'exposure':
                self.camera.set_controls({"ExposureTime": int(value * 1000)})  # ms to μs
            elif setting == 'gain':
                self.camera.set_controls({"AnalogueGain": value})
        
        return True
    
    def get_histogram_data(self, image: Optional[np.ndarray] = None, 
                          bins: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate histogram data for an image.
        
        Args:
            image: Image to analyze, or None to use current frame
            bins: Number of histogram bins
            
        Returns:
            tuple: (bin_edges, histogram_values)
        """
        # Use provided image or current frame
        if image is None:
            image = self.get_frame()
            
        if image is None:
            return np.array([]), np.array([])
        
        # Calculate histogram
        hist, bins = np.histogram(image.flatten(), bins=bins, range=(0, self.max_value))
        
        return bins, hist
    
    def get_intensity_stats(self, image: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate intensity statistics for an image.
        
        Args:
            image: Image to analyze, or None to use current frame
            
        Returns:
            dict: Dictionary of statistics
        """
        # Use provided image or current frame
        if image is None:
            image = self.get_frame()
            
        if image is None:
            return {
                'min': 0.0,
                'max': 0.0,
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0
            }
        
        # Calculate statistics
        return {
            'min': float(np.min(image)),
            'max': float(np.max(image)),
            'mean': float(np.mean(image)),
            'median': float(np.median(image)),
            'std': float(np.std(image))
        }
    
    def start_recording(self, filename: str) -> bool:
        """
        Start recording video to a file.
        
        Args:
            filename: Path to save the video
        
        Returns:
            bool: True if recording started successfully, False otherwise
        """
        if self.is_recording:
            print("Already recording")
            return False
        
        # Ensure the filename has an extension
        if not filename.lower().endswith(('.mp4', '.avi', '.mov')):
            filename += '.mp4'
        
        # Create a video writer
        self.video_writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 30, (self.width, self.height))
        
        if not self.video_writer.isOpened():
            print("Failed to open video writer")
            return False
        
        self.is_recording = True
        self.recording_filename = filename
        self.recording_start_time = time.time()
        
        return True
    
    def stop_recording(self) -> bool:
        """
        Stop recording video.
        
        Returns:
            bool: True if recording stopped successfully, False otherwise
        """
        if not self.is_recording:
            print("Not recording")
            return False
        
        self.is_recording = False
        
        # Release the video writer
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        
        # Save metadata
        self._save_video_metadata(self.recording_filename, time.time() - self.recording_start_time)
        
        return True
    
    def _save_video_metadata(self, video_filename: str, duration: float) -> None:
        """Save metadata about the video to a text file"""
        try:
            metadata_filename = video_filename + '.txt'
            
            with open(metadata_filename, 'w') as f:
                f.write(f"Camera Type: {self.camera_type}\n")
                f.write(f"Resolution: {self.width}x{self.height}\n")
                f.write(f"Bit Depth: {self.bit_depth}-bit\n")
                f.write(f"Value Range: 0-{self.max_value}\n")
                f.write(f"Duration: {duration:.2f} seconds\n")
                f.write(f"Frame Rate: 30 FPS\n")
                
                # Add camera settings
                f.write(f"Camera Settings:\n")
                for key, value in self.settings.items():
                    f.write(f"  {key}: {value}\n")
                
                # Add timestamp
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                
        except Exception as e:
            print(f"Error saving metadata: {str(e)}")


class CameraControlGUI:
    def __init__(self, parent=None, camera=None):
        """
        Initialize the Camera Control GUI.
        
        Args:
            parent: Parent Tkinter widget (if embedding in another application)
            camera: CameraController instance (will create one if None)
        """
        self.parent = parent
        
        # Create camera controller if not provided
        if camera is None:
            self.camera = CameraController(camera_type="opencv", camera_index=0, 
                                         resolution=(640, 480), bit_depth=10)
            self.camera.initialize()
        else:
            self.camera = camera
        
        # Create main frame if no parent provided
        if parent is None:
            self.root = tk.Tk()
            self.root.title("Camera Control")
            self.main_frame = self.root
        else:
            self.root = None
            self.main_frame = ttk.Frame(parent)
            self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create main layout
        self.create_layout()
        
        # Initialize video recording state
        self.is_recording = False
        
        # Start camera if not already running
        if not self.camera.is_running:
            self.camera.start()
        
        # Start updating the preview
        self.update_preview()
    
    def create_layout(self):
        """Create the main layout of the camera control UI"""
        # Create a frame for the preview and controls
        main_container = ttk.Frame(self.main_frame)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left side: Preview area
        preview_frame = ttk.LabelFrame(main_container, text="Camera Preview")
        preview_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create canvas for video display
        self.preview_width = 640
        self.preview_height = 480
        self.preview_canvas = tk.Canvas(preview_frame, 
                                      width=self.preview_width, 
                                      height=self.preview_height,
                                      bg="black")
        self.preview_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add intensity info label
        self.intensity_info = ttk.Label(preview_frame, 
                                      text="Intensity (10-bit) - Max: 0, Mean: 0",
                                      font=("Arial", 10))
        self.intensity_info.pack(pady=5)
        
        # Right side: Controls
        controls_frame = ttk.Frame(main_container)
        controls_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        # Camera actions section
        actions_frame = ttk.LabelFrame(controls_frame, text="Camera Actions")
        actions_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Capture and save button
        self.capture_button = ttk.Button(actions_frame, text="Capture & Save Image", 
                                       command=self.save_image)
        self.capture_button.pack(fill=tk.X, padx=5, pady=5)
        
        # Record video toggle button
        self.record_text = tk.StringVar(value="Start Recording")
        self.record_button = ttk.Button(actions_frame, textvariable=self.record_text,
                                      command=self.toggle_recording)
        self.record_button.pack(fill=tk.X, padx=5, pady=5)
        
        # Pause/Resume button
        self.pause_text = tk.StringVar(value="Pause Camera")
        self.pause_button = ttk.Button(actions_frame, textvariable=self.pause_text,
                                     command=self.toggle_pause)
        self.pause_button.pack(fill=tk.X, padx=5, pady=5)
        
        # Camera settings section
        settings_frame = ttk.LabelFrame(controls_frame, text="Camera Settings")
        settings_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Exposure control
        ttk.Label(settings_frame, text="Exposure Time (ms):").pack(anchor=tk.W, padx=5, pady=2)
        self.exposure_var = tk.StringVar(value=str(self.camera.settings['exposure']))
        exposure_entry = ttk.Entry(settings_frame, textvariable=self.exposure_var, width=10)
        exposure_entry.pack(anchor=tk.W, padx=5, pady=2)
        
        ttk.Button(settings_frame, text="Set Exposure", 
                 command=lambda: self.set_camera_setting('exposure', float(self.exposure_var.get()))).pack(
                     fill=tk.X, padx=5, pady=5)
        
        # Gain control
        ttk.Label(settings_frame, text="Analog Gain:").pack(anchor=tk.W, padx=5, pady=2)
        self.gain_var = tk.StringVar(value=str(self.camera.settings['gain']))
        gain_entry = ttk.Entry(settings_frame, textvariable=self.gain_var, width=10)
        gain_entry.pack(anchor=tk.W, padx=5, pady=2)
        
        ttk.Button(settings_frame, text="Set Gain",
                 command=lambda: self.set_camera_setting('gain', float(self.gain_var.get()))).pack(
                     fill=tk.X, padx=5, pady=5)
        
        # Brightness control
        ttk.Label(settings_frame, text="Brightness:").pack(anchor=tk.W, padx=5, pady=2)
        self.brightness_var = tk.StringVar(value=str(self.camera.settings['brightness']))
        brightness_entry = ttk.Entry(settings_frame, textvariable=self.brightness_var, width=10)
        brightness_entry.pack(anchor=tk.W, padx=5, pady=2)
        
        ttk.Button(settings_frame, text="Set Brightness",
                 command=lambda: self.set_camera_setting('brightness', int(self.brightness_var.get()))).pack(
                     fill=tk.X, padx=5, pady=5)
        
        # Contrast control
        ttk.Label(settings_frame, text="Contrast:").pack(anchor=tk.W, padx=5, pady=2)
        self.contrast_var = tk.StringVar(value=str(self.camera.settings['contrast']))
        contrast_entry = ttk.Entry(settings_frame, textvariable=self.contrast_var, width=10)
        contrast_entry.pack(anchor=tk.W, padx=5, pady=2)
        
        ttk.Button(settings_frame, text="Set Contrast",
                 command=lambda: self.set_camera_setting('contrast', int(self.contrast_var.get()))).pack(
                     fill=tk.X, padx=5, pady=5)
        
        # Auto exposure checkbox
        self.auto_exposure_var = tk.BooleanVar(value=self.camera.settings['auto_exposure'])
        auto_exposure_check = ttk.Checkbutton(settings_frame, text="Auto Exposure",
                                            variable=self.auto_exposure_var,
                                            command=lambda: self.set_camera_setting('auto_exposure', 
                                                                                 self.auto_exposure_var.get()))
        auto_exposure_check.pack(anchor=tk.W, padx=5, pady=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Camera ready")
        status_bar = ttk.Label(self.main_frame, textvariable=self.status_var, 
                             relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Recording indicator (hidden initially)
        self.recording_indicator = ttk.Label(preview_frame, text="● RECORDING", 
                                          foreground="red", font=("Arial", 12, "bold"))
        # Don't pack it yet - only show when recording
    
    def update_preview(self):
        """Update the preview canvas with the current camera frame"""
        try:
            # Get the current frame
            frame = self.camera.get_frame()
            
            if frame is not None:
                # Get intensity stats for display
                stats = self.camera.get_intensity_stats(frame)
                self.intensity_info.config(
                    text=f"Intensity (10-bit) - Min: {stats['min']:.1f}, Max: {stats['max']:.1f}, " +
                         f"Mean: {stats['mean']:.1f}"
                )
                
                # Convert to 8-bit for display
                display_frame = (frame / self.camera.max_value * 255).astype(np.uint8)
                
                # Resize if needed
                if display_frame.shape[1] != self.preview_width or display_frame.shape[0] != self.preview_height:
                    display_frame = cv2.resize(display_frame, (self.preview_width, self.preview_height))
                
                # Convert to RGB format for PIL
                display_rgb = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2RGB)
                
                # Convert to PIL Image and then to PhotoImage
                pil_img = Image.fromarray(display_rgb)
                self.tk_img = ImageTk.PhotoImage(image=pil_img)
                
                # Update canvas
                self.preview_canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
                
                # Update status
                self.status_var.set(f"Camera running - {frame.shape[1]}x{frame.shape[0]} at 10-bit depth")
            
        except Exception as e:
            self.status_var.set(f"Preview error: {str(e)}")
        
        # Schedule the next update (30 FPS)
        if self.root:
            self.root.after(33, self.update_preview)
        elif self.parent:
            self.parent.after(33, self.update_preview)
    
    def save_image(self):
        """Capture and save an image using zenity file dialog"""
        try:
            # Capture a still image
            still = self.camera.capture_still()
            
            if still is None:
                self.status_var.set("Failed to capture image")
                return
            
            # Use zenity file dialog to get save location
            cmd = ['zenity', '--file-selection', '--save', 
                   '--file-filter=PNG files | *.png', 
                   '--title=Save Captured Image']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            save_path = result.stdout.strip()
            if not save_path:
                self.status_var.set("Save cancelled")
                return
            
            # Add extension if not present
            if not save_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                save_path += '.png'
            
            # Save the image
            if self.camera.save_image(save_path, still):
                self.status_var.set(f"Image saved to {save_path}")
            else:
                self.status_var.set("Failed to save image")
                
        except Exception as e:
            self.status_var.set(f"Error saving image: {str(e)}")
    
    def toggle_recording(self):
        """Toggle video recording on/off using zenity file dialog"""
        try:
            if not self.is_recording:
                # Use zenity file dialog to get save location
                cmd = ['zenity', '--file-selection', '--save', 
                       '--file-filter=MP4 files | *.mp4', 
                       '--title=Save Video Recording']
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                save_path = result.stdout.strip()
                if not save_path:
                    self.status_var.set("Recording cancelled")
                    return
                
                # Add extension if not present
                if not save_path.lower().endswith(('.mp4', '.avi', '.mov')):
                    save_path += '.mp4'
                
                # Start recording
                if self.camera.start_recording(save_path):
                    self.is_recording = True
                    self.record_text.set("Stop Recording")
                    self.status_var.set(f"Recording to {save_path}")
                    
                    # Show recording indicator
                    self.recording_indicator.pack(side=tk.TOP, pady=5)
                else:
                    self.status_var.set("Failed to start recording")
            else:
                # Stop recording
                if self.camera.stop_recording():
                    self.is_recording = False
                    self.record_text.set("Start Recording")
                    self.status_var.set("Recording stopped")
                    
                    # Hide recording indicator
                    self.recording_indicator.pack_forget()
                else:
                    self.status_var.set("Failed to stop recording")
                
        except Exception as e:
            self.status_var.set(f"Error toggling recording: {str(e)}")
    
    def toggle_pause(self):
        """Toggle camera pause/resume"""
        try:
            if self.camera.is_paused:
                self.camera.resume()
                self.pause_text.set("Pause Camera")
                self.status_var.set("Camera resumed")
            else:
                self.camera.pause()
                self.pause_text.set("Resume Camera")
                self.status_var.set("Camera paused")
                
        except Exception as e:
            self.status_var.set(f"Error toggling pause: {str(e)}")
    
    def set_camera_setting(self, setting, value):
        """Set a camera setting"""
        try:
            if self.camera.set_setting(setting, value):
                self.status_var.set(f"{setting.capitalize()} set to {value}")
            else:
                self.status_var.set(f"Failed to set {setting}")
                
        except Exception as e:
            self.status_var.set(f"Error setting {setting}: {str(e)}")
    
    def run(self):
        """Run the GUI (only if created as standalone)"""
        if self.root:
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.root.mainloop()
    
    def on_closing(self):
        """Handle window closing"""
        try:
            # Stop recording if active
            if self.is_recording:
                self.camera.stop_recording()
            
            # Stop the camera
            self.camera.stop()
            
            # Destroy the window
            if self.root:
                self.root.destroy()
                
        except Exception as e:
            print(f"Error during shutdown: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Create a camera controller
    camera = CameraController(camera_type="opencv", camera_index=0, 
                             resolution=(640, 480), bit_depth=10)
    
    # Create a GUI
    gui = CameraControlGUI(camera=camera)
    
    # Run the GUI
    gui.run()
