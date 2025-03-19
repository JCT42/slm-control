"""
Advanced Camera Control Module using PiCamera2 for SLM Control System
Designed for IMX296 10-bit camera integration with SLM pattern generation

Features:
- High-precision 10-bit image capture
- Real-time preview with OpenCV integration
- Histogram analysis of intensity distributions
- Exposure and gain control
- Thread-safe operation
"""

import numpy as np
import cv2
import time
import threading
from picamera2 import Picamera2
from libcamera import controls
from PIL import Image, ImageTk

class CameraController:
    """Advanced camera controller for scientific imaging with PiCamera2"""
    
    def __init__(self, config=None):
        """Initialize the camera controller
        
        Args:
            config (dict, optional): Configuration parameters for the camera
        """
        # Default camera configuration
        self.default_config = {
            'width': 1456,           # IMX296 native width
            'height': 1088,          # IMX296 native height
            'bit_depth': 10,         # 10-bit camera
            'pixel_format': 'Y10',   # 10-bit Y-only format
            'exposure_time': 20000,  # 20ms default exposure (in μs)
            'analog_gain': 1.0,      # Default analog gain
            'preview_scale': 0.5,    # Scale factor for preview (0.5 = half size)
            'frame_rate': 30,        # Target frame rate for preview
        }
        
        # Use provided config or default
        self.config = config if config else self.default_config
        
        # Calculate max value based on bit depth
        self.max_value = 2**self.config['bit_depth'] - 1  # 1023 for 10-bit
        
        # Camera state
        self.camera = None
        self.is_initialized = False
        self.is_running = False
        self.is_paused = False
        
        # Thread control
        self.preview_thread = None
        self.preview_lock = threading.Lock()
        self.stop_event = threading.Event()
        
        # Frame storage
        self.current_frame = None
        self.last_captured_frame = None
        
        # Callback for frame updates
        self.frame_callback = None
        self.histogram_callback = None
        
        # Statistics
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        
        # Initialize camera
        self.initialize()
    
    def initialize(self):
        """Initialize the camera with configured settings"""
        try:
            # Create camera instance without device parameter
            self.camera = Picamera2()
            
            # Configure for 10-bit Y10 capture
            preview_width = int(self.config['width'] * self.config['preview_scale'])
            preview_height = int(self.config['height'] * self.config['preview_scale'])
            
            # Create configuration for still and preview
            # For Y10 format, we need to use the correct configuration
            self.camera_config = self.camera.create_still_configuration(
                main={"size": (self.config['width'], self.config['height']),
                      "format": self.config['pixel_format']},
                lores={"size": (preview_width, preview_height),
                       "format": "YUV420"},
                display="lores"
            )
            
            # Apply configuration
            self.camera.configure(self.camera_config)
            
            # Set initial camera controls
            self.camera.set_controls({
                "ExposureTime": self.config['exposure_time'],
                "AnalogueGain": self.config['analog_gain'],
                "FrameDurationLimits": (33333, 33333),  # Target ~30fps
            })
            
            # Mark as initialized
            self.is_initialized = True
            print(f"Camera initialized: {self.config['width']}x{self.config['height']} in {self.config['pixel_format']} format")
            return True
            
        except Exception as e:
            print(f"Camera initialization error: {str(e)}")
            self.is_initialized = False
            return False
    
    def start(self):
        """Start the camera and begin capturing frames"""
        if not self.is_initialized:
            if not self.initialize():
                return False
        
        try:
            # Start the camera
            self.camera.start()
            self.is_running = True
            self.is_paused = False
            self.stop_event.clear()
            
            # Start preview thread if not already running
            if self.preview_thread is None or not self.preview_thread.is_alive():
                self.preview_thread = threading.Thread(target=self._preview_loop, daemon=True)
                self.preview_thread.start()
            
            return True
            
        except Exception as e:
            print(f"Error starting camera: {str(e)}")
            return False
    
    def stop(self):
        """Stop the camera and release resources"""
        self.is_running = False
        self.stop_event.set()
        
        # Wait for thread to finish
        if self.preview_thread and self.preview_thread.is_alive():
            self.preview_thread.join(timeout=1.0)
        
        # Stop the camera
        if self.camera:
            try:
                self.camera.stop()
            except Exception as e:
                print(f"Error stopping camera: {str(e)}")
    
    def close(self):
        """Close the camera and release all resources"""
        self.stop()
        if self.camera:
            try:
                self.camera.close()
                self.camera = None
                self.is_initialized = False
            except Exception as e:
                print(f"Error closing camera: {str(e)}")
    
    def pause(self):
        """Pause the camera preview"""
        self.is_paused = True
    
    def resume(self):
        """Resume the camera preview"""
        self.is_paused = False
    
    def _preview_loop(self):
        """Main preview loop that runs in a separate thread"""
        last_time = time.time()
        frame_count = 0
        
        while self.is_running and not self.stop_event.is_set():
            try:
                if not self.is_paused:
                    # Capture frame
                    frame = self.camera.capture_array()
                    
                    # Process frame
                    if frame is not None:
                        with self.preview_lock:
                            # Store the current frame
                            self.current_frame = frame
                            
                            # Calculate FPS
                            frame_count += 1
                            current_time = time.time()
                            elapsed = current_time - last_time
                            
                            if elapsed >= 1.0:
                                self.fps = frame_count / elapsed
                                frame_count = 0
                                last_time = current_time
                            
                            # Call frame callback if registered
                            if self.frame_callback:
                                self.frame_callback(frame, self.fps)
                
                # Sleep to maintain target frame rate
                time.sleep(1.0 / self.config['frame_rate'])
                    
            except Exception as e:
                print(f"Preview error: {str(e)}")
                time.sleep(0.1)  # Prevent tight loop on error
    
    def capture_frame(self):
        """Capture a high-quality still frame
        
        Returns:
            numpy.ndarray: Grayscale image with 10-bit values (0-1023)
        """
        try:
            # Capture a full-resolution frame
            frame = self.camera.capture_array()
            
            # Store as last captured frame
            self.last_captured_frame = frame
            
            return frame
            
        except Exception as e:
            print(f"Error capturing frame: {str(e)}")
            return None
    
    def get_current_frame(self):
        """Get the most recent frame from the preview
        
        Returns:
            numpy.ndarray: Grayscale image with 10-bit values (0-1023)
        """
        with self.preview_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return None
    
    def set_exposure(self, exposure_ms):
        """Set camera exposure time in milliseconds
        
        Args:
            exposure_ms (float): Exposure time in milliseconds
        
        Returns:
            bool: Success or failure
        """
        try:
            # Convert to microseconds
            exposure_us = int(exposure_ms * 1000)
            self.camera.set_controls({"ExposureTime": exposure_us})
            self.config['exposure_time'] = exposure_us
            return True
        except Exception as e:
            print(f"Error setting exposure: {str(e)}")
            return False
    
    def set_gain(self, gain):
        """Set camera analog gain
        
        Args:
            gain (float): Analog gain value
        
        Returns:
            bool: Success or failure
        """
        try:
            self.camera.set_controls({"AnalogueGain": gain})
            self.config['analog_gain'] = gain
            return True
        except Exception as e:
            print(f"Error setting gain: {str(e)}")
            return False
    
    def get_histogram(self, frame=None):
        """Calculate histogram for the given frame or current frame
        
        Args:
            frame (numpy.ndarray, optional): Frame to analyze. If None, uses current frame.
        
        Returns:
            tuple: (hist, bins) - Histogram data and bin edges
        """
        if frame is None:
            frame = self.get_current_frame()
            
        if frame is None:
            return None, None
            
        try:
            # Calculate histogram with 100 bins across the 10-bit range
            hist, bins = np.histogram(frame.flatten(), bins=100, range=(0, self.max_value))
            return hist, bins
        except Exception as e:
            print(f"Error calculating histogram: {str(e)}")
            return None, None
    
    def get_intensity_stats(self, frame=None):
        """Calculate intensity statistics for the given frame or current frame
        
        Args:
            frame (numpy.ndarray, optional): Frame to analyze. If None, uses current frame.
        
        Returns:
            dict: Dictionary with intensity statistics
        """
        if frame is None:
            frame = self.get_current_frame()
            
        if frame is None:
            return None
            
        try:
            stats = {
                'min': np.min(frame),
                'max': np.max(frame),
                'mean': np.mean(frame),
                'median': np.median(frame),
                'std': np.std(frame)
            }
            return stats
        except Exception as e:
            print(f"Error calculating statistics: {str(e)}")
            return None
    
    def save_image(self, filename, frame=None):
        """Save the current frame or specified frame to a file
        
        Args:
            filename (str): Path to save the image
            frame (numpy.ndarray, optional): Frame to save. If None, uses last captured frame.
        
        Returns:
            bool: Success or failure
        """
        if frame is None:
            if self.last_captured_frame is not None:
                frame = self.last_captured_frame
            else:
                frame = self.get_current_frame()
                
        if frame is None:
            print("No frame available to save")
            return False
            
        try:
            # Ensure the filename has an extension
            if not filename.lower().endswith(('.png', '.jpg', '.tiff', '.tif')):
                filename += '.png'
                
            # For PNG, save as 16-bit to preserve 10-bit values
            if filename.lower().endswith('.png'):
                # Scale 10-bit (0-1023) to 16-bit (0-65535) for storage
                # This preserves the full precision of the original 10-bit values
                save_img = (frame / self.max_value * 65535).astype(np.uint16)
                cv2.imwrite(filename, save_img)
            else:
                # For other formats, save directly
                cv2.imwrite(filename, frame.astype(np.uint16))
                
            # Save metadata file with important information about the intensity values
            metadata_filename = filename + '.txt'
            with open(metadata_filename, 'w') as f:
                f.write(f"Camera: IMX296 Monochrome at {self.config['width']}x{self.config['height']}\n")
                f.write(f"Resolution: {frame.shape[1]}x{frame.shape[0]}\n")
                f.write(f"Bit Depth: {self.config['bit_depth']}-bit\n")
                f.write(f"Pixel Format: {self.config['pixel_format']}\n")
                f.write(f"Original Value Range: 0-{self.max_value}\n")
                f.write(f"Important: These are raw intensity values (0-1023), not phase values.\n")
                
                if filename.lower().endswith('.png'):
                    f.write(f"Storage Format: 16-bit PNG\n")
                    f.write(f"Conversion: Original 10-bit value * 65535/{self.max_value}\n")
                    f.write(f"To recover original values: PNG_value * {self.max_value}/65535\n")
                
                # Add statistics
                stats = self.get_intensity_stats(frame)
                if stats:
                    f.write(f"Statistics:\n")
                    f.write(f"  Min: {stats['min']:.1f}\n")
                    f.write(f"  Max: {stats['max']:.1f}\n")
                    f.write(f"  Mean: {stats['mean']:.1f}\n")
                    f.write(f"  Median: {stats['median']:.1f}\n")
                    f.write(f"  Standard Deviation: {stats['std']:.1f}\n")
                
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Exposure: {self.config['exposure_time']/1000:.2f} ms\n")
                f.write(f"Gain: {self.config['analog_gain']:.2f}\n")
            
            return True
            
        except Exception as e:
            print(f"Error saving image: {str(e)}")
            return False
    
    def create_preview_image(self, frame=None, scale=1.0, min_val=None, max_val=None):
        """Create a preview image suitable for display in a GUI
        
        Args:
            frame (numpy.ndarray, optional): Frame to process. If None, uses current frame.
            scale (float): Scale factor for resizing
            min_val (float, optional): Minimum value for normalization. If None, uses 0.
            max_val (float, optional): Maximum value for normalization. If None, uses max_value.
        
        Returns:
            PIL.ImageTk.PhotoImage: Tkinter-compatible image for display
        """
        if frame is None:
            frame = self.get_current_frame()
            
        if frame is None:
            return None
            
        try:
            # Set normalization range - preserve 10-bit intensity values
            if min_val is None:
                min_val = 0
            if max_val is None:
                max_val = self.max_value
            
            # Y10 format comes as a 10-bit intensity array already
            # We just need to normalize it for display without losing precision
            
            # Normalize to 0-255 for display
            normalized = np.clip((frame - min_val) / (max_val - min_val) * 255, 0, 255).astype(np.uint8)
            
            # Resize if needed
            if scale != 1.0:
                new_width = int(frame.shape[1] * scale)
                new_height = int(frame.shape[0] * scale)
                normalized = cv2.resize(normalized, (new_width, new_height))
            
            # Convert to PIL Image and then to PhotoImage for Tkinter
            pil_img = Image.fromarray(normalized)
            return ImageTk.PhotoImage(image=pil_img)
            
        except Exception as e:
            print(f"Error creating preview image: {str(e)}")
            return None
    
    def register_frame_callback(self, callback):
        """Register a callback function to be called when a new frame is available
        
        Args:
            callback (callable): Function to call with (frame, fps) parameters
        """
        self.frame_callback = callback
    
    def register_histogram_callback(self, callback):
        """Register a callback function to be called when histogram data is updated
        
        Args:
            callback (callable): Function to call with histogram data
        """
        self.histogram_callback = callback


# Example usage
if __name__ == "__main__":
    # Simple test to verify camera functionality
    print("Initializing camera controller for IMX296 camera with Y10 format...")
    
    # Create camera controller with default settings
    # This will use /dev/video0 with Y10 format at 1456x1088 resolution
    camera = CameraController()
    
    if camera.is_initialized:
        print("Camera initialized successfully")
        
        # Start the camera
        camera.start()
        print("Camera started")
        
        # Wait for a few frames
        time.sleep(2)
        
        # Capture a frame
        frame = camera.capture_frame()
        if frame is not None:
            print(f"Frame captured: {frame.shape}, min={np.min(frame)}, max={np.max(frame)}")
            print(f"This is a 10-bit intensity image with values from 0-1023")
            
            # Save the frame
            camera.save_image("test_frame.png", frame)
            print("Frame saved to test_frame.png with metadata")
        
        # Get histogram
        hist, bins = camera.get_histogram()
        if hist is not None:
            print(f"Histogram calculated: {len(hist)} bins")
            print(f"Histogram range: {bins[0]} to {bins[-1]}")
        
        # Get stats
        stats = camera.get_intensity_stats()
        if stats:
            print(f"Intensity stats: min={stats['min']:.1f}, max={stats['max']:.1f}, mean={stats['mean']:.1f}")
            print(f"These are raw intensity values (0-1023), not phase values")
        
        # Stop the camera
        camera.stop()
        camera.close()
        print("Camera stopped and closed")
    else:
        print("Failed to initialize camera")
        
    print("\nTo use this module in your application:")
    print("from camera_control_2 import CameraController")
    print("camera = CameraController()")
    print("camera.start()")
    print("# Get frames with camera.get_current_frame()")
    print("# Capture high-quality stills with camera.capture_frame()")
    print("# Remember: These are 10-bit intensity values (0-1023), not phase values")
