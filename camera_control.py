"""
Camera Control Module for SLM Control System

This module provides a flexible camera interface for capturing and processing
images from various camera types, with a focus on scientific imaging applications.

Features:
- Support for different camera types (PiCamera2, OpenCV webcams, etc.)
- 8-bit intensity capture and processing
- Real-time display capabilities using OpenCV
- Histogram and intensity analysis tools
- Thread-safe operation for integration with GUI applications
- Video recording functionality

Dependencies:
- OpenCV (cv2)
- NumPy
- Threading
- PiCamera2 (for Raspberry Pi cameras)
"""

import cv2
import numpy as np
import threading
import time
import subprocess
import os
import traceback  # Add traceback for better error reporting
from typing import Tuple, Optional, Dict, Any, Callable
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Try to import picamera2, but don't fail if not available
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("PiCamera2 not available - Raspberry Pi camera support will be limited")


class CameraController:
    """
    Main camera controller class that provides a unified interface
    for different camera types and processing methods.
    """
    
    def __init__(self, camera_type: str = "opencv", camera_index: int = 0, 
                 resolution: Tuple[int, int] = (1456, 1088),
                 bit_depth: int = 8):
        """
        Initialize the camera controller.
        
        Args:
            camera_type: Type of camera to use ('opencv', 'picamera2', etc.)
            camera_index: Camera device index for OpenCV cameras
            resolution: Desired camera resolution (width, height)
            bit_depth: Bit depth for intensity values (default: 8-bit)
        """
        self.camera_type = camera_type
        self.camera_index = camera_index
        self.width, self.height = resolution
        self.bit_depth = bit_depth
        self.max_value = 2**bit_depth - 1  # e.g., 255 for 8-bit
        
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
                return self._initialize_opencv_camera()
            elif self.camera_type == "picamera2":
                return self._initialize_picamera2()
            else:
                print(f"Unsupported camera type: {self.camera_type}")
                return False
        except Exception as e:
            print(f"Camera initialization error: {str(e)}")
            traceback.print_exc()  # Print detailed error message
            return False
    
    def _initialize_opencv_camera(self) -> bool:
        """Initialize an OpenCV camera"""
        try:
            # Try different backends in order of preference
            backends = [
                cv2.CAP_ANY,          # Auto-detect
                cv2.CAP_V4L2,         # Video4Linux2
                cv2.CAP_GSTREAMER,    # GStreamer
                cv2.CAP_DSHOW         # DirectShow (Windows)
            ]
            
            # Try each backend until one works
            for backend in backends:
                print(f"Trying camera backend: {backend}")
                self.camera = cv2.VideoCapture(self.camera_index, backend)
                
                if self.camera.isOpened():
                    print(f"Successfully opened camera with backend: {backend}")
                    break
            
            # If none of the backends worked, try one last time with default
            if not self.camera.isOpened():
                print("All backends failed, trying default...")
                self.camera = cv2.VideoCapture(self.camera_index)
            
            # Final check
            if not self.camera.isOpened():
                print(f"Failed to open camera {self.camera_index}")
                return False
            
            # Set resolution - don't check for success as not all cameras support this
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
            
            # Get actual camera resolution (may differ from requested)
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"Camera initialized with resolution: {actual_width}x{actual_height}")
            
            # Update resolution if different from requested
            if actual_width != self.width or actual_height != self.height:
                print(f"Note: Requested {self.width}x{self.height} but got {actual_width}x{actual_height}")
                self.width = actual_width
                self.height = actual_height
            
            # Try to set camera properties, but don't fail if they're not supported
            try:
                # Try to disable auto exposure
                if not self.settings['auto_exposure']:
                    self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.0)  # 0 = manual
                
                # Try to set exposure
                if 'exposure' in self.settings:
                    self.camera.set(cv2.CAP_PROP_EXPOSURE, float(self.settings['exposure']))
                
                # Try to set gain
                if 'gain' in self.settings:
                    self.camera.set(cv2.CAP_PROP_GAIN, float(self.settings['gain']))
                
                # Try to set brightness
                if 'brightness' in self.settings:
                    self.camera.set(cv2.CAP_PROP_BRIGHTNESS, float(self.settings['brightness']))
                
                # Try to set contrast
                if 'contrast' in self.settings:
                    self.camera.set(cv2.CAP_PROP_CONTRAST, float(self.settings['contrast']))
            except Exception as e:
                print(f"Warning: Could not set some camera properties: {str(e)}")
                # Continue anyway - these are not critical
            
            # Verify camera is working by capturing a test frame
            ret, test_frame = self.camera.read()
            if not ret or test_frame is None:
                print("Warning: Camera opened but test frame capture failed")
                # We'll continue anyway and hope it starts working
            else:
                print(f"Test frame captured successfully: {test_frame.shape}")
            
            return True
            
        except Exception as e:
            print(f"OpenCV camera initialization error: {str(e)}")
            traceback.print_exc()
            return False
    
    def _initialize_picamera2(self) -> bool:
        """Initialize a Raspberry Pi Camera using PiCamera2"""
        try:
            # Check if PiCamera2 is available
            if not PICAMERA2_AVAILABLE:
                print("PiCamera2 is not available. Please install it with: pip install picamera2")
                return False
                
            # Create camera instance - use default camera
            self.camera = Picamera2()
            
            # Get camera info
            print(f"Camera info: {self.camera.camera_properties}")
            
            # Force specific resolution and format based on v4l2-ctl settings
            # This matches: v4l2-ctl -d /dev/video0 --set-fmt-video=width=1456,height=1088,pixelformat=GREY
            self.width = 1456
            self.height = 1088
            pixel_format = "GREY"  # Use GREY format as specified in v4l2-ctl
            
            print(f"Using resolution: {self.width}x{self.height}, format: {pixel_format}")
            
            # Set bit depth to 8 for GREY format
            self.bit_depth = 8
            
            preview_width = int(self.width * 0.5)  # Half size for preview
            preview_height = int(self.height * 0.5)
            
            # Create configuration for still and preview
            try:
                # Try to create a configuration with the GREY format
                self.camera_config = self.camera.create_still_configuration(
                    main={"size": (self.width, self.height),
                          "format": pixel_format},
                    lores={"size": (preview_width, preview_height),
                           "format": "YUV420"},
                    display="lores"
                )
            except Exception as config_error:
                print(f"Error creating GREY format configuration: {str(config_error)}")
                print("Trying alternative format names...")
                
                # Try alternative format names for grayscale
                for alt_format in ["GREY", "Y8", "GRAY", "MONO", "MONO8"]:
                    try:
                        print(f"Trying format: {alt_format}")
                        self.camera_config = self.camera.create_still_configuration(
                            main={"size": (self.width, self.height),
                                  "format": alt_format},
                            lores={"size": (preview_width, preview_height),
                                   "format": "YUV420"},
                            display="lores"
                        )
                        print(f"Format {alt_format} worked!")
                        break
                    except Exception as alt_error:
                        print(f"Format {alt_format} failed: {str(alt_error)}")
                
                # If all alternative formats fail, fall back to default
                if "camera_config" not in locals() or self.camera_config is None:
                    print("All grayscale format attempts failed. Falling back to default configuration")
                    self.camera_config = self.camera.create_still_configuration(
                        main={"size": (self.width, self.height)}
                    )
            
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
            
            print(f"PiCamera2 initialized at {actual_width}x{actual_height} in {actual_format} format")
            print(f"Using 8-bit intensity values (0-255)")
            
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
            print(f"PiCamera2 initialization error: {str(e)}")
            traceback.print_exc()
            return False
    
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
            elif self.camera_type == "picamera2":
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
                traceback.print_exc()  # Print detailed error message
                time.sleep(0.1)  # Longer sleep on error
    
    def _capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from the camera.
        
        Returns:
            np.ndarray: Captured frame as a numpy array, or None if capture failed
        """
        # Check if camera exists
        if self.camera is None:
            return None
            
        try:
            if self.camera_type == "opencv":
                # Capture frame from OpenCV camera
                ret, frame = self.camera.read()
                
                if not ret or frame is None:
                    print("Failed to capture frame")
                    return None
                
                # Convert to grayscale if not already
                if len(frame.shape) > 2:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray = frame
                
                return gray
                
            elif self.camera_type == "picamera2":
                # For Raspberry Pi camera with PiCamera2 and 8-bit mode
                try:
                    # Capture a frame
                    frame = self.camera.capture_array()
                    
                    # No need to convert to grayscale or scale
                    
                    # Print some debug info occasionally
                    if np.random.random() < 0.01:
                        print(f"PiCamera2 frame stats - Min: {np.min(frame)}, Max: {np.max(frame)}, Mean: {np.mean(frame):.1f}")
                    
                    return frame
                    
                except Exception as e:
                    print(f"PiCamera2 frame capture error: {str(e)}")
                    traceback.print_exc()
                    return None
            
            return None
            
        except Exception as e:
            print(f"Frame capture error: {str(e)}")
            traceback.print_exc()
            return None
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the current frame (thread-safe).
        
        Returns:
            np.ndarray: Current frame, or None if not available
        """
        with self.lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return None
    
    def capture_still(self) -> Optional[np.ndarray]:
        """
        Capture a high-quality still image.
        
        Returns:
            np.ndarray: Captured still image, or None on failure
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
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        else:
                            gray = frame
                        frames.append(gray)
                
                if not frames:
                    print("Failed to capture any frames")
                    return None
                
                # Average the frames to reduce noise
                avg_frame = np.mean(frames, axis=0).astype(np.uint8)
                
                result = avg_frame
                
            elif self.camera_type == "picamera2":
                # For PiCamera2, capture a high-quality still
                try:
                    # Configure for a still capture with full resolution
                    result = self.camera.capture_array()
                    
                except Exception as e:
                    print(f"PiCamera2 still capture error: {str(e)}")
                    traceback.print_exc()
                    return None
                    
            else:
                print(f"Unsupported camera type for still capture: {self.camera_type}")
                return None
            
            # Store the captured frame
            with self.lock:
                self.last_captured_frame = result.copy()
            
            # Restore previous pause state
            if not was_paused:
                self.resume()
                
            return result
            
        except Exception as e:
            print(f"Still capture error: {str(e)}")
            traceback.print_exc()
            
            # Restore previous pause state
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
            
            # Save the image
            cv2.imwrite(filename, save_image)
            
            # Save metadata
            self._save_metadata(filename, save_image)
            
            return True
            
        except Exception as e:
            print(f"Error saving image: {str(e)}")
            traceback.print_exc()  # Print detailed error message
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
            traceback.print_exc()  # Print detailed error message
    
    def set_setting(self, setting: str, value: Any) -> bool:
        """
        Set a camera setting.
        
        Args:
            setting: Setting name to change
            value: New value for the setting
            
        Returns:
            bool: True if setting was changed, False otherwise
        """
        try:
            # Update the setting in our dictionary
            if setting not in self.settings:
                print(f"Unknown setting: {setting}")
                return False
                
            # Store the old value
            old_value = self.settings[setting]
            
            # Update with new value
            self.settings[setting] = value
            
            # Apply the setting to the camera
            if self.camera_type == "opencv":
                if self.camera is None:
                    print("Camera not initialized")
                    return False
                    
                if setting == 'exposure':
                    # First disable auto exposure if setting manual exposure
                    if not self.settings['auto_exposure']:
                        self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.0)  # 0 = manual
                    # Then set manual exposure value
                    success = self.camera.set(cv2.CAP_PROP_EXPOSURE, float(value))
                    if not success:
                        print(f"Failed to set exposure to {value}")
                        self.settings[setting] = old_value
                        return False
                    print(f"Set exposure to {value}")
                    
                elif setting == 'gain':
                    success = self.camera.set(cv2.CAP_PROP_GAIN, float(value))
                    if not success:
                        print(f"Failed to set gain to {value}")
                        self.settings[setting] = old_value
                        return False
                    print(f"Set gain to {value}")
                    
                elif setting == 'brightness':
                    success = self.camera.set(cv2.CAP_PROP_BRIGHTNESS, float(value))
                    if not success:
                        print(f"Failed to set brightness to {value}")
                        self.settings[setting] = old_value
                        return False
                        
                elif setting == 'contrast':
                    success = self.camera.set(cv2.CAP_PROP_CONTRAST, float(value))
                    if not success:
                        print(f"Failed to set contrast to {value}")
                        self.settings[setting] = old_value
                        return False
                        
                elif setting == 'auto_exposure':
                    # Convert boolean to float (0.0 or 1.0)
                    auto_value = 1.0 if value else 0.0
                    success = self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_value)
                    if not success:
                        print(f"Failed to set auto_exposure to {value}")
                        self.settings[setting] = old_value
                        return False
                
            elif self.camera_type == "picamera2" and self.camera is not None:
                if setting == 'exposure':
                    # Convert ms to μs
                    self.camera.set_controls({
                        "ExposureTime": int(value * 1000),
                    })
                elif setting == 'gain':
                    self.camera.set_controls({
                        "AnalogueGain": float(value),
                    })
                elif setting == 'brightness':
                    # Convert 0-255 to -100 to 100
                    brightness = int((value / 255) * 200 - 100)
                    self.camera.set_controls({
                        "Brightness": brightness,
                    })
                elif setting == 'contrast':
                    # Convert 0-255 to -100 to 100
                    contrast = int((value / 255) * 200 - 100)
                    self.camera.set_controls({
                        "Contrast": contrast,
                    })
                elif setting == 'auto_exposure':
                    self.camera.set_controls({
                        "AeEnable": value,
                    })
            
            return True
            
        except Exception as e:
            print(f"Error setting {setting}: {str(e)}")
            traceback.print_exc()
            return False
    
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
            traceback.print_exc()  # Print detailed error message


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
                                         resolution=(640, 480), bit_depth=8)
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
                                      text="Intensity (8-bit) - Max: 0, Mean: 0",
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
                
                # Update canvas
                self.preview_canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
                
                # Update status
                self.status_var.set(f"Camera running - {frame.shape[1]}x{frame.shape[0]} at 8-bit depth")
            
        except Exception as e:
            self.status_var.set(f"Preview error: {str(e)}")
            traceback.print_exc()  # Print detailed error message
        
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
            traceback.print_exc()  # Print detailed error message
    
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
            traceback.print_exc()  # Print detailed error message
    
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
            traceback.print_exc()  # Print detailed error message
    
    def set_camera_setting(self, setting, value):
        """Set a camera setting"""
        try:
            if self.camera.set_setting(setting, value):
                self.status_var.set(f"{setting.capitalize()} set to {value}")
            else:
                self.status_var.set(f"Failed to set {setting}")
                
        except Exception as e:
            self.status_var.set(f"Error setting {setting}: {str(e)}")
            traceback.print_exc()  # Print detailed error message
    
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
            traceback.print_exc()  # Print detailed error message


# Example usage
if __name__ == "__main__":
    print("Starting camera control module...")
    
    try:
        # On Raspberry Pi, prioritize PiCamera2
        try_picamera2_first = True  # Set to True for Raspberry Pi
        
        if try_picamera2_first:
            print("Checking for PiCamera2 (recommended for Raspberry Pi)...")
            try:
                import picamera2
                print("PiCamera2 module is available - using PiCamera2")
                camera_type = "picamera2"
                camera_index = 0
            except ImportError:
                print("PiCamera2 module not available, falling back to OpenCV camera")
                try_picamera2_first = False
        
        # If PiCamera2 not available or not prioritized, try OpenCV cameras
        if not try_picamera2_first:
            print("Checking for OpenCV cameras...")
            available_cameras = []
            
            # Try different backends
            backends = [
                (cv2.CAP_ANY, "Auto-detect"),
                (cv2.CAP_V4L2, "Video4Linux2"),
                (cv2.CAP_GSTREAMER, "GStreamer"),
                (cv2.CAP_DSHOW, "DirectShow (Windows)")
            ]
            
            # Try each backend
            for backend_id, backend_name in backends:
                print(f"Checking for cameras with {backend_name} backend...")
                
                # Try first 3 camera indices (0, 1, 2)
                for i in range(3):
                    cap = cv2.VideoCapture(i, backend_id)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret:
                            print(f"Found working camera at index {i} with {backend_name} backend")
                            available_cameras.append((i, backend_id, backend_name))
                        cap.release()
            
            # Use first available camera
            if available_cameras:
                camera_index, backend_id, backend_name = available_cameras[0]
                print(f"Using camera {camera_index} with {backend_name} backend")
                camera_type = "opencv"
            else:
                print("No working OpenCV cameras found.")
                print("Please connect a camera and try again.")
                exit(1)
        
        # Create camera controller with appropriate settings
        print(f"Initializing camera (type: {camera_type}, index: {camera_index})...")
        camera = CameraController(
            camera_type=camera_type, 
            camera_index=camera_index,
            resolution=(1456, 1088),  # Use full resolution for IMX296
            bit_depth=8  # Using 8-bit for simplicity
        )
        
        # Initialize the camera
        if not camera.initialize():
            print("Failed to initialize camera. Please check connections and try again.")
            exit(1)
        
        print("Camera initialized successfully!")
        
        # Create GUI
        root = tk.Tk()
        root.title("Camera Control")
        
        # Create camera control GUI
        gui = CameraControlGUI(root, camera)
        
        # Start the camera
        camera.start()
        
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
