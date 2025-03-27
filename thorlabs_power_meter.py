"""
Thorlabs Power Meter Interface for Raspberry Pi
Connects to Thorlabs power meters via USB and provides a simple interface to read measurements.
"""

import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import threading
import usb.core
import usb.util
import struct

# Thorlabs USB identifiers
THORLABS_VENDOR_ID = 0x1313  # Thorlabs Vendor ID

# Known Thorlabs Power Meter Product IDs
POWER_METER_PRODUCT_IDS = {
    0x8070: "PM100USB",
    0x8071: "PM100D",
    0x8072: "PM100A",
    0x8078: "PM400",
    0x8079: "PM101",
    0x807A: "PM102",
    0x807B: "PM103"
    # Add more product IDs as needed
}

class ThorlabsPowerMeter:
    """Interface for Thorlabs Power Meters using PyUSB"""
    
    def __init__(self):
        """Initialize the power meter connection"""
        self.device = None
        self.connected = False
        self.wavelength = 633.0  # Default wavelength in nm
        self.measurement_thread = None
        self.running = False
        self.power_history = []
        self.time_history = []
        self.start_time = None
        self.max_history_length = 1000  # Maximum number of data points to store
        self.interface_number = 0
        self.endpoint_in = None
        self.endpoint_out = None
        
    def find_devices(self):
        """Find all connected Thorlabs power meter devices"""
        try:
            devices = []
            # Find all Thorlabs devices
            for product_id in POWER_METER_PRODUCT_IDS:
                found_devices = list(usb.core.find(find_all=True, 
                                                  idVendor=THORLABS_VENDOR_ID, 
                                                  idProduct=product_id))
                for dev in found_devices:
                    model = POWER_METER_PRODUCT_IDS.get(product_id, "Unknown")
                    devices.append({
                        'device': dev,
                        'product_id': product_id,
                        'model': model,
                        'serial': self._get_serial_number(dev)
                    })
            
            return devices
        except Exception as e:
            print(f"Error finding devices: {e}")
            return []
    
    def _get_serial_number(self, device):
        """Try to get the serial number of the device"""
        try:
            return usb.util.get_string(device, device.iSerialNumber)
        except:
            return "Unknown"
    
    def connect(self, device_info):
        """Connect to a specific power meter"""
        try:
            self.device = device_info['device']
            
            # Detach kernel driver if active
            if self.device.is_kernel_driver_active(self.interface_number):
                try:
                    self.device.detach_kernel_driver(self.interface_number)
                except usb.core.USBError as e:
                    print(f"Could not detach kernel driver: {e}")
            
            # Set configuration
            self.device.set_configuration()
            
            # Get an endpoint instance
            cfg = self.device.get_active_configuration()
            intf = cfg[(0, 0)]
            
            # Find endpoints
            self.endpoint_out = usb.util.find_descriptor(
                intf,
                custom_match=lambda e: 
                    usb.util.endpoint_direction(e.bEndpointAddress) == 
                    usb.util.ENDPOINT_OUT
            )
            
            self.endpoint_in = usb.util.find_descriptor(
                intf,
                custom_match=lambda e: 
                    usb.util.endpoint_direction(e.bEndpointAddress) == 
                    usb.util.ENDPOINT_IN
            )
            
            if not self.endpoint_out or not self.endpoint_in:
                raise ValueError("Could not find endpoints")
            
            self.connected = True
            
            # Set default wavelength
            self.set_wavelength(self.wavelength)
            
            print(f"Connected to {device_info['model']} (SN: {device_info['serial']})")
            return True
        except Exception as e:
            print(f"Error connecting to device: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the power meter"""
        if self.device and self.connected:
            try:
                # Release the interface
                usb.util.dispose_resources(self.device)
                self.device = None
                self.connected = False
                print("Disconnected from power meter")
                return True
            except Exception as e:
                print(f"Error disconnecting: {e}")
                return False
        return True
    
    def _send_command(self, command, data=None):
        """Send a command to the device and get response"""
        try:
            # Implement the specific command protocol for your power meter model
            # This is a simplified example and may need to be adjusted
            if data:
                self.endpoint_out.write(command + data)
            else:
                self.endpoint_out.write(command)
                
            # Read response
            response = self.endpoint_in.read(64)
            return response
        except Exception as e:
            print(f"Error sending command: {e}")
            return None
    
    def set_wavelength(self, wavelength):
        """Set the wavelength in nm"""
        if not self.connected:
            print("Not connected to a power meter")
            return False
            
        try:
            # Implement the specific command to set wavelength for your model
            # This is a placeholder and needs to be adjusted for your specific device
            command = struct.pack('<BBHH', 0xA2, 0x02, int(wavelength), 0)
            self._send_command(command)
            
            self.wavelength = wavelength
            print(f"Wavelength set to {wavelength} nm")
            return True
        except Exception as e:
            print(f"Error setting wavelength: {e}")
            return False
    
    def get_power(self):
        """Get a single power measurement in watts"""
        if not self.connected:
            print("Not connected to a power meter")
            return None
            
        try:
            # Implement the specific command to get power for your model
            # This is a placeholder and needs to be adjusted for your specific device
            command = struct.pack('<BB', 0xD1, 0x01)
            response = self._send_command(command)
            
            if response and len(response) >= 4:
                # Extract power value from response
                # Format depends on your specific device protocol
                power_value = struct.unpack('<f', response[0:4])[0]
                return power_value
            return None
        except Exception as e:
            print(f"Error measuring power: {e}")
            return None
    
    def start_continuous_measurement(self, interval=0.1):
        """Start continuous power measurements in a separate thread"""
        if self.running:
            print("Measurement already running")
            return False
            
        if not self.connected:
            print("Not connected to a power meter")
            return False
            
        self.running = True
        self.power_history = []
        self.time_history = []
        self.start_time = time.time()
        
        def measurement_loop():
            while self.running:
                try:
                    power = self.get_power()
                    if power is not None:
                        current_time = time.time() - self.start_time
                        self.power_history.append(power)
                        self.time_history.append(current_time)
                        
                        # Limit the history length
                        if len(self.power_history) > self.max_history_length:
                            self.power_history.pop(0)
                            self.time_history.pop(0)
                    
                    time.sleep(interval)
                except Exception as e:
                    print(f"Error in measurement loop: {e}")
                    time.sleep(interval)
        
        self.measurement_thread = threading.Thread(target=measurement_loop)
        self.measurement_thread.daemon = True
        self.measurement_thread.start()
        print(f"Started continuous measurement (interval: {interval}s)")
        return True
    
    def stop_continuous_measurement(self):
        """Stop continuous measurements"""
        if not self.running:
            return
            
        self.running = False
        if self.measurement_thread:
            self.measurement_thread.join(timeout=1.0)
            self.measurement_thread = None
        print("Stopped continuous measurement")
    
    def get_measurement_data(self):
        """Get the current measurement data"""
        return self.time_history.copy(), self.power_history.copy()
    
    def __del__(self):
        """Clean up resources when object is destroyed"""
        self.stop_continuous_measurement()
        self.disconnect()


class PowerMeterGUI:
    """GUI for Thorlabs Power Meter"""
    
    def __init__(self, root=None):
        """Initialize the GUI"""
        self.power_meter = ThorlabsPowerMeter()
        self.create_standalone = root is None
        
        if self.create_standalone:
            self.root = tk.Tk()
            self.root.title("Thorlabs Power Meter")
            self.root.geometry("800x600")
            self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        else:
            self.root = root
            
        self.frame = ttk.Frame(self.root)
        self.frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.create_widgets()
        self.animation = None
        self.devices = []
        
    def create_widgets(self):
        """Create GUI widgets"""
        # Control frame
        control_frame = ttk.LabelFrame(self.frame, text="Controls")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Connection controls
        conn_frame = ttk.Frame(control_frame)
        conn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(conn_frame, text="Device:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.device_combo = ttk.Combobox(conn_frame, state="readonly", width=40)
        self.device_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        self.refresh_btn = ttk.Button(conn_frame, text="Refresh", command=self.refresh_devices)
        self.refresh_btn.grid(row=0, column=2, padx=5, pady=5)
        
        self.connect_btn = ttk.Button(conn_frame, text="Connect", command=self.connect_disconnect)
        self.connect_btn.grid(row=0, column=3, padx=5, pady=5)
        
        # Settings frame
        settings_frame = ttk.Frame(control_frame)
        settings_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(settings_frame, text="Wavelength (nm):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.wavelength_var = tk.DoubleVar(value=633.0)
        self.wavelength_entry = ttk.Entry(settings_frame, textvariable=self.wavelength_var, width=10)
        self.wavelength_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        self.set_wavelength_btn = ttk.Button(settings_frame, text="Set", command=self.set_wavelength)
        self.set_wavelength_btn.grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        
        # Measurement controls
        measurement_frame = ttk.Frame(control_frame)
        measurement_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.current_power_var = tk.StringVar(value="-- W")
        ttk.Label(measurement_frame, text="Current Power:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Label(measurement_frame, textvariable=self.current_power_var, font=("Arial", 12, "bold")).grid(
            row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        self.measure_btn = ttk.Button(measurement_frame, text="Single Measure", command=self.single_measure)
        self.measure_btn.grid(row=0, column=2, padx=5, pady=5)
        
        self.continuous_btn = ttk.Button(measurement_frame, text="Start Continuous", command=self.toggle_continuous)
        self.continuous_btn.grid(row=0, column=3, padx=5, pady=5)
        
        # Plot frame
        plot_frame = ttk.LabelFrame(self.frame, text="Power Measurement")
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Power (W)')
        self.ax.grid(True)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Not connected")
        status_bar = ttk.Label(self.frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, padx=5, pady=5)
        
        # Initial state
        self.set_connected_state(False)
        self.refresh_devices()
        
    def refresh_devices(self):
        """Refresh the list of connected devices"""
        self.device_combo.set("")
        self.device_combo["values"] = []
        self.devices = []
        
        devices = self.power_meter.find_devices()
        if devices:
            device_strings = []
            for i, dev_info in enumerate(devices):
                device_strings.append(f"{dev_info['model']} (SN: {dev_info['serial']})")
                self.devices.append(dev_info)
                
            self.device_combo["values"] = device_strings
            self.device_combo.current(0)
            self.status_var.set(f"Found {len(devices)} device(s)")
        else:
            self.status_var.set("No devices found")
    
    def connect_disconnect(self):
        """Connect to or disconnect from the selected device"""
        if not self.power_meter.connected:
            # Connect
            if not self.device_combo.get():
                self.status_var.set("No device selected")
                return
                
            device_index = self.device_combo.current()
            if device_index < 0 or device_index >= len(self.devices):
                self.status_var.set("Invalid device selection")
                return
                
            if self.power_meter.connect(self.devices[device_index]):
                self.set_connected_state(True)
                self.status_var.set(f"Connected to {self.device_combo.get()}")
            else:
                self.status_var.set("Failed to connect")
        else:
            # Disconnect
            self.stop_continuous()
            if self.power_meter.disconnect():
                self.set_connected_state(False)
                self.status_var.set("Disconnected")
            else:
                self.status_var.set("Failed to disconnect")
    
    def set_connected_state(self, connected):
        """Update UI elements based on connection state"""
        state = "disabled" if not connected else "normal"
        
        self.connect_btn.config(text="Disconnect" if connected else "Connect")
        self.device_combo.config(state="readonly" if not connected else "disabled")
        self.refresh_btn.config(state="normal" if not connected else "disabled")
        
        self.wavelength_entry.config(state=state)
        self.set_wavelength_btn.config(state=state)
        self.measure_btn.config(state=state)
        self.continuous_btn.config(state=state)
    
    def set_wavelength(self):
        """Set the wavelength on the power meter"""
        try:
            wavelength = self.wavelength_var.get()
            if wavelength <= 0:
                self.status_var.set("Invalid wavelength")
                return
                
            if self.power_meter.set_wavelength(wavelength):
                self.status_var.set(f"Wavelength set to {wavelength} nm")
            else:
                self.status_var.set("Failed to set wavelength")
        except Exception as e:
            self.status_var.set(f"Error: {e}")
    
    def single_measure(self):
        """Take a single power measurement"""
        power = self.power_meter.get_power()
        if power is not None:
            self.current_power_var.set(f"{power:.9f} W")
            self.status_var.set("Measurement taken")
        else:
            self.status_var.set("Measurement failed")
    
    def toggle_continuous(self):
        """Toggle continuous measurement mode"""
        if not self.power_meter.running:
            self.start_continuous()
        else:
            self.stop_continuous()
    
    def start_continuous(self):
        """Start continuous measurement"""
        if self.power_meter.start_continuous_measurement():
            self.continuous_btn.config(text="Stop Continuous")
            self.status_var.set("Continuous measurement started")
            
            # Start animation
            self.animation = FuncAnimation(
                self.fig, self.update_plot, interval=100, 
                blit=False, cache_frame_data=False)
            self.canvas.draw()
        else:
            self.status_var.set("Failed to start continuous measurement")
    
    def stop_continuous(self):
        """Stop continuous measurement"""
        if self.animation:
            self.animation.event_source.stop()
            self.animation = None
            
        self.power_meter.stop_continuous_measurement()
        self.continuous_btn.config(text="Start Continuous")
        self.status_var.set("Continuous measurement stopped")
    
    def update_plot(self, frame):
        """Update the plot with new data"""
        times, powers = self.power_meter.get_measurement_data()
        
        if powers:
            # Update current power display
            self.current_power_var.set(f"{powers[-1]:.9f} W")
            
            # Update plot
            self.ax.clear()
            self.ax.plot(times, powers, 'b-')
            self.ax.set_xlabel('Time (s)')
            self.ax.set_ylabel('Power (W)')
            self.ax.grid(True)
            
            # Auto-adjust y-axis limits with some padding
            if len(powers) > 1:
                min_power = min(powers)
                max_power = max(powers)
                range_power = max_power - min_power
                
                if range_power > 0:
                    padding = range_power * 0.1
                    self.ax.set_ylim(min_power - padding, max_power + padding)
                else:
                    # If range is zero (constant power), add some padding
                    self.ax.set_ylim(min_power * 0.9, max_power * 1.1)
            
            self.fig.tight_layout()
        
        return self.ax,
    
    def on_close(self):
        """Handle window close event"""
        self.stop_continuous()
        self.power_meter.disconnect()
        
        if self.create_standalone:
            self.root.destroy()
    
    def run(self):
        """Run the GUI main loop"""
        if self.create_standalone:
            self.root.mainloop()


if __name__ == "__main__":
    app = PowerMeterGUI()
    app.run()
