"""
Thorlabs Power Meter Interface for Raspberry Pi and Windows
Connects to Thorlabs power meters via USB and provides a simple interface to read measurements.
"""

import time
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import threading
import struct
import csv
import pandas as pd
from datetime import datetime
from scipy import stats
import json
import platform
import pyvisa
import subprocess

# Determine if we're on Windows or Linux
IS_WINDOWS = platform.system() == 'Windows'

# Thorlabs USB identifiers
THORLABS_VENDOR_ID = 0x1313  # Thorlabs Vendor ID

POWER_METER_PRODUCT_IDS = {
    0x8070: "PM100USB",
    0x8071: "PM100D",
    0x8072: "PM100A",
    0x8078: "PM400",
    0x8079: "PM101",
    0x807A: "PM102",
    0x807B: "PM103"
}

class ThorlabsPowerMeter:
    """Interface for Thorlabs Power Meters using PyVISA"""
    
    def __init__(self):
        """Initialize the power meter connection"""
        self.rm = pyvisa.ResourceManager()
        self.device = None
        self.connected = False
        self.wavelength = 650.0  # Default wavelength in nm
        self.measurement_thread = None
        self.running = False
        self.power_history = []
        self.time_history = []
        self.start_time = None
        self.max_history_length = 1000  # Maximum number of data points to store
        self.interface_number = 0
        self.endpoint_in = 0x81
        self.endpoint_out = 0x01
        
        # Try to connect to the PM100D
        try:
            self.device = self.rm.open_resource('USB0::1313::8078::INSTR')  # Update the address as necessary
            self.connected = True
            print("Connected to PM100D")
        except Exception as e:
            print(f"Error connecting to device: {e}")
        
        # Simulation parameters
        self.simulation_base_power = 1.0e-3  # 1 mW
        self.simulation_noise = 0.02  # 2% noise
        self.simulation_drift = 0.0001  # 0.01% drift per second
        
        # Statistical analysis variables
        self.stats = {
            'mean': 0.0,
            'median': 0.0,
            'std_dev': 0.0,
            'min': 0.0,
            'max': 0.0,
            'range': 0.0,
            'variance': 0.0,
            'stability': 0.0,  # Coefficient of variation (std_dev/mean)
            'uncertainty': 0.0  # Standard error of the mean
        }
        
        # Reference beam variables
        self.reference_power = None
        self.reference_enabled = False
        self.reference_history = []
        self.relative_power_history = []  # Power relative to reference
        
        # Metadata for measurements
        self.metadata = {
            'device_model': '',
            'device_serial': '',
            'wavelength': self.wavelength,
            'measurement_date': '',
            'measurement_duration': 0,
            'sample_count': 0,
            'notes': ''
        }
        
        # Uncertainty calculation parameters
        self.uncertainty_factors = {
            'calibration': 0.01,  # 1% calibration uncertainty
            'linearity': 0.005,   # 0.5% linearity uncertainty
            'temperature': 0.002, # 0.2% temperature-related uncertainty
            'wavelength': 0.01    # 1% wavelength-dependent uncertainty
        }
    
    def find_devices(self):
        """Find all connected Thorlabs power meter devices"""
        try:
            devices = []
            
            # Use PyVISA to find devices
            for resource in self.rm.list_resources():
                devices.append({
                    'device': resource,
                    'product_id': 0,  # Not available through PyVISA
                    'model': "Thorlabs Power Meter",  # Will be updated after connection
                    'serial': resource,
                    'resource_name': resource
                })
                
            return devices
        except Exception as e:
            print(f"Error finding devices: {e}")
            return []
    
    def connect(self, device_info):
        """Connect to a specific power meter"""
        try:
            # Try to connect to the device
            self.device = self.rm.open_resource(device_info['resource_name'])
            self.connected = True
            print(f"Connected to {device_info['model']} (SN: {device_info['serial']})")
            return True
        except Exception as e:
            print(f"Error connecting to device: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the power meter"""
        if self.device:
            self.device.close()
            self.connected = False
            print("Disconnected from power meter")
    
    def get_power(self):
        """Get the current power measurement"""
        if not self.connected:
            print("Device not connected")
            return None
        
        try:
            power = self.device.query("MEAS:POW?")  # Send query to get power measurement
            return float(power)
        except Exception as e:
            print(f"Error reading power: {e}")
            return None
    
    def set_wavelength(self, wavelength):
        """Set the wavelength in nm"""
        if not self.connected:
            print("Device not connected")
            return False
        
        try:
            self.device.write(f"WAV {wavelength}")  # Command to set wavelength
            self.wavelength = wavelength
            print(f"Wavelength set to {wavelength} nm")
            return True
        except Exception as e:
            print(f"Error setting wavelength: {e}")
            return False
    
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
    
    def calculate_statistics(self):
        """Calculate statistical analysis of the measurement data"""
        if not self.power_history:
            return
        
        self.stats['mean'] = np.mean(self.power_history)
        self.stats['median'] = np.median(self.power_history)
        self.stats['std_dev'] = np.std(self.power_history)
        self.stats['min'] = np.min(self.power_history)
        self.stats['max'] = np.max(self.power_history)
        self.stats['range'] = self.stats['max'] - self.stats['min']
        self.stats['variance'] = np.var(self.power_history)
        self.stats['stability'] = self.stats['std_dev'] / self.stats['mean'] if self.stats['mean'] != 0 else 0
        self.stats['uncertainty'] = self.calculate_uncertainty()
        
    def calculate_uncertainty(self):
        """Calculate the uncertainty of the measurement"""
        # Calculate standard error of the mean
        sem = self.stats['std_dev'] / np.sqrt(len(self.power_history)) if self.power_history else 0
        
        # Calculate total uncertainty using uncertainty factors
        total_uncertainty = sem
        for factor in self.uncertainty_factors.values():
            total_uncertainty = np.sqrt(total_uncertainty**2 + factor**2)
        
        return total_uncertainty
    
    def export_data(self, filename):
        """Export measurement data to a CSV file"""
        if not self.power_history:
            print("No data to export")
            return
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Time (s)", "Power (W)"])
            for time, power in zip(self.time_history, self.power_history):
                writer.writerow([time, power])
        
        print(f"Data exported to {filename}")
    
    def export_statistics(self, filename):
        """Export statistical analysis to a JSON file"""
        self.calculate_statistics()
        
        with open(filename, 'w') as jsonfile:
            json.dump(self.stats, jsonfile, indent=4)
        
        print(f"Statistics exported to {filename}")
    
    def set_reference_power(self, power=None):
        """Set the current power as reference or use provided value"""
        if power is None and self.power_history:
            self.reference_power = self.power_history[-1]
        elif power is not None:
            self.reference_power = power
        else:
            print("No power measurement available to set as reference")
            return False
            
        self.reference_enabled = True
        print(f"Reference power set to {self.reference_power:.9f} W")
        return True
    
    def disable_reference(self):
        """Disable reference beam comparison"""
        self.reference_enabled = False
        print("Reference beam comparison disabled")
    
    def get_relative_power(self):
        """Get power relative to reference (in percentage)"""
        if not self.reference_enabled or self.reference_power is None or self.reference_power == 0:
            return None
            
        if self.power_history:
            relative = (self.power_history[-1] / self.reference_power) * 100
            return relative
        return None
    
    def analyze_stability(self, window_size=None):
        """Analyze beam stability over time"""
        if not self.power_history or len(self.power_history) < 2:
            return None
            
        # Use all data if window_size is None, otherwise use the last window_size points
        data = self.power_history
        if window_size and len(data) > window_size:
            data = data[-window_size:]
            
        # Calculate statistics
        mean = np.mean(data)
        std_dev = np.std(data)
        coefficient_of_variation = std_dev / mean if mean != 0 else float('inf')
        
        # Calculate drift (linear regression)
        times = self.time_history[-len(data):]
        slope, intercept, r_value, p_value, std_err = stats.linregress(times, data)
        
        # Calculate Allan deviation for different time scales
        allan_devs = []
        for tau in [2, 5, 10]:
            if len(data) >= tau * 2:
                # Calculate Allan deviation
                ad = np.sqrt(0.5 * np.mean(np.diff(np.array(data).reshape(-1, tau).mean(axis=1))**2))
                allan_devs.append((tau, ad))
        
        stability_analysis = {
            'mean': mean,
            'std_dev': std_dev,
            'coefficient_of_variation': coefficient_of_variation,
            'drift_slope': slope,  # W/s
            'drift_p_value': p_value,
            'allan_deviations': allan_devs
        }
        
        return stability_analysis
    
    def export_to_excel(self, filename):
        """Export measurement data and statistics to Excel file"""
        if not self.power_history:
            print("No data to export")
            return False
            
        try:
            # Create a Pandas Excel writer
            writer = pd.ExcelWriter(filename, engine='openpyxl')
            
            # Calculate statistics
            self.calculate_statistics()
            
            # Create DataFrame for measurements
            data = {
                'Time (s)': self.time_history,
                'Power (W)': self.power_history
            }
            
            # Add reference data if available
            if self.reference_enabled and len(self.relative_power_history) > 0:
                data['Relative Power (%)'] = self.relative_power_history
            
            df_measurements = pd.DataFrame(data)
            
            # Create DataFrame for statistics
            df_stats = pd.DataFrame({
                'Statistic': list(self.stats.keys()),
                'Value': list(self.stats.values())
            })
            
            # Create DataFrame for metadata
            df_metadata = pd.DataFrame({
                'Property': list(self.metadata.keys()),
                'Value': list(self.metadata.values())
            })
            
            # Write each DataFrame to a different worksheet
            df_measurements.to_excel(writer, sheet_name='Measurements', index=False)
            df_stats.to_excel(writer, sheet_name='Statistics', index=False)
            df_metadata.to_excel(writer, sheet_name='Metadata', index=False)
            
            # Save the Excel file
            writer.close()
            
            print(f"Data exported to {filename}")
            return True
        except Exception as e:
            print(f"Error exporting to Excel: {e}")
            return False
    
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
            self.root.geometry("900x700")
            self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        else:
            self.root = root
            
        self.frame = ttk.Frame(self.root)
        self.frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Variables
        self.device_var = tk.StringVar()
        self.wavelength_var = tk.DoubleVar(value=633.0)
        self.current_power_var = tk.StringVar(value="-- W")
        self.status_var = tk.StringVar(value="Ready")
        self.ref_power_var = tk.StringVar(value="-- W")
        self.relative_power_var = tk.StringVar(value="-- %")
        self.notes_var = tk.StringVar(value="")
        
        # Variables for uncertainty factors
        self.uncertainty_vars = {
            'calibration': tk.DoubleVar(value=1.0),
            'linearity': tk.DoubleVar(value=0.5),
            'temperature': tk.DoubleVar(value=0.2),
            'wavelength': tk.DoubleVar(value=1.0)
        }
        
        # Variable for stability analysis
        self.stability_window_size = tk.IntVar(value=100)
        
        # Dictionary to hold statistics labels
        self.stat_labels = {}
        
        self.create_widgets()
        self.animation = None
        self.devices = []
        
    def create_widgets(self):
        """Create GUI widgets"""
        # Create a notebook (tabbed interface)
        self.notebook = ttk.Notebook(self.frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Main tab
        main_tab = ttk.Frame(self.notebook)
        self.notebook.add(main_tab, text="Main")
        
        # Analysis tab
        analysis_tab = ttk.Frame(self.notebook)
        self.notebook.add(analysis_tab, text="Analysis")
        
        # Scientific tab
        scientific_tab = ttk.Frame(self.notebook)
        self.notebook.add(scientific_tab, text="Scientific")
        
        # Setup main tab
        self.setup_main_tab(main_tab)
        
        # Setup analysis tab
        self.setup_analysis_tab(analysis_tab)
        
        # Setup scientific tab
        self.setup_scientific_tab(scientific_tab)
        
        # Status bar
        status_bar = ttk.Label(self.frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, padx=5, pady=5)
        
        # Initial state
        self.set_connected_state(False)
        self.refresh_devices()
    
    def setup_main_tab(self, parent):
        """Setup the main tab with connection and measurement controls"""
        # Control frame
        control_frame = ttk.LabelFrame(parent, text="Controls")
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
        self.wavelength_entry = ttk.Entry(settings_frame, textvariable=self.wavelength_var, width=10)
        self.wavelength_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        self.set_wavelength_btn = ttk.Button(settings_frame, text="Set", command=self.set_wavelength)
        self.set_wavelength_btn.grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        
        # Measurement controls
        measurement_frame = ttk.Frame(control_frame)
        measurement_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(measurement_frame, text="Current Power:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Label(measurement_frame, textvariable=self.current_power_var, font=("Arial", 12, "bold")).grid(
            row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        self.measure_btn = ttk.Button(measurement_frame, text="Single Measure", command=self.single_measure)
        self.measure_btn.grid(row=0, column=2, padx=5, pady=5)
        
        self.continuous_btn = ttk.Button(measurement_frame, text="Start Continuous", command=self.toggle_continuous)
        self.continuous_btn.grid(row=0, column=3, padx=5, pady=5)
        
        # Plot frame
        plot_frame = ttk.LabelFrame(parent, text="Power Measurement")
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Power (W)')
        self.ax.grid(True)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def setup_analysis_tab(self, parent):
        """Setup the analysis tab with statistical analysis and export options"""
        # Statistics frame
        stats_frame = ttk.LabelFrame(parent, text="Statistical Analysis")
        stats_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create a frame for statistics display
        stats_display = ttk.Frame(stats_frame)
        stats_display.pack(fill=tk.X, padx=5, pady=5)
        
        # Create labels for each statistic
        self.stat_labels = {}
        stats = ['mean', 'median', 'std_dev', 'min', 'max', 'range', 'stability', 'uncertainty']
        stat_names = ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Range', 'Stability', 'Uncertainty']
        
        for i, (stat, name) in enumerate(zip(stats, stat_names)):
            row, col = divmod(i, 4)
            ttk.Label(stats_display, text=f"{name}:").grid(row=row, column=col*2, padx=5, pady=5, sticky=tk.W)
            self.stat_labels[stat] = ttk.Label(stats_display, text="--")
            self.stat_labels[stat].grid(row=row, column=col*2+1, padx=5, pady=5, sticky=tk.W)
        
        # Button to calculate statistics
        self.calc_stats_btn = ttk.Button(stats_frame, text="Calculate Statistics", 
                                         command=self.update_statistics)
        self.calc_stats_btn.pack(padx=5, pady=5)
        
        # Export frame
        export_frame = ttk.LabelFrame(parent, text="Data Export")
        export_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Notes for metadata
        notes_frame = ttk.Frame(export_frame)
        notes_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(notes_frame, text="Notes:").pack(side=tk.LEFT, padx=5, pady=5)
        notes_entry = ttk.Entry(notes_frame, textvariable=self.notes_var, width=50)
        notes_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        
        # Export buttons
        export_buttons = ttk.Frame(export_frame)
        export_buttons.pack(fill=tk.X, padx=5, pady=5)
        
        self.export_csv_btn = ttk.Button(export_buttons, text="Export to CSV", 
                                         command=self.export_to_csv)
        self.export_csv_btn.grid(row=0, column=0, padx=5, pady=5)
        
        self.export_excel_btn = ttk.Button(export_buttons, text="Export to Excel", 
                                          command=self.export_to_excel)
        self.export_excel_btn.grid(row=0, column=1, padx=5, pady=5)
        
        self.export_stats_btn = ttk.Button(export_buttons, text="Export Statistics", 
                                          command=self.export_statistics)
        self.export_stats_btn.grid(row=0, column=2, padx=5, pady=5)
    
    def setup_scientific_tab(self, parent):
        """Setup the scientific tab with reference beam and stability analysis"""
        # Reference beam frame
        ref_frame = ttk.LabelFrame(parent, text="Reference Beam Comparison")
        ref_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Reference controls
        ref_controls = ttk.Frame(ref_frame)
        ref_controls.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(ref_controls, text="Reference Power:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Label(ref_controls, textvariable=self.ref_power_var).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(ref_controls, text="Relative Power:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        ttk.Label(ref_controls, textvariable=self.relative_power_var).grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        
        # Reference buttons
        ref_buttons = ttk.Frame(ref_frame)
        ref_buttons.pack(fill=tk.X, padx=5, pady=5)
        
        self.set_ref_btn = ttk.Button(ref_buttons, text="Set Current as Reference", 
                                     command=self.set_reference)
        self.set_ref_btn.grid(row=0, column=0, padx=5, pady=5)
        
        self.clear_ref_btn = ttk.Button(ref_buttons, text="Clear Reference", 
                                       command=self.clear_reference)
        self.clear_ref_btn.grid(row=0, column=1, padx=5, pady=5)
        
        # Stability analysis frame
        stability_frame = ttk.LabelFrame(parent, text="Beam Stability Analysis")
        stability_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Stability controls
        stability_controls = ttk.Frame(stability_frame)
        stability_controls.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(stability_controls, text="Window Size:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        window_entry = ttk.Entry(stability_controls, textvariable=self.stability_window_size, width=10)
        window_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        self.analyze_stability_btn = ttk.Button(stability_controls, text="Analyze Stability", 
                                               command=self.analyze_beam_stability)
        self.analyze_stability_btn.grid(row=0, column=2, padx=5, pady=5)
        
        # Stability results frame
        self.stability_results_frame = ttk.Frame(stability_frame)
        self.stability_results_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Uncertainty calculation frame
        uncertainty_frame = ttk.LabelFrame(parent, text="Uncertainty Calculation")
        uncertainty_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Uncertainty factors
        uncertainty_factors = ttk.Frame(uncertainty_frame)
        uncertainty_factors.pack(fill=tk.X, padx=5, pady=5)
        
        factors = [
            ('Calibration', 'calibration'), 
            ('Linearity', 'linearity'),
            ('Temperature', 'temperature'),
            ('Wavelength', 'wavelength')
        ]
        
        for i, (name, key) in enumerate(factors):
            ttk.Label(uncertainty_factors, text=f"{name} (%):").grid(row=i//2, column=(i%2)*2, padx=5, pady=5, sticky=tk.W)
            entry = ttk.Entry(uncertainty_factors, textvariable=self.uncertainty_vars[key], width=10)
            entry.grid(row=i//2, column=(i%2)*2+1, padx=5, pady=5, sticky=tk.W)
        
        self.update_uncertainty_btn = ttk.Button(uncertainty_frame, text="Update Uncertainty Factors", 
                                                command=self.update_uncertainty_factors)
        self.update_uncertainty_btn.pack(padx=5, pady=5)
    
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
        
        # Analysis tab buttons
        self.calc_stats_btn.config(state=state)
        self.export_csv_btn.config(state=state)
        self.export_excel_btn.config(state=state)
        self.export_stats_btn.config(state=state)
        
        # Scientific tab buttons
        self.set_ref_btn.config(state=state)
        self.clear_ref_btn.config(state=state)
        self.analyze_stability_btn.config(state=state)
        self.update_uncertainty_btn.config(state=state)
    
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
            
            # Update relative power if reference is enabled
            if self.power_meter.reference_enabled:
                relative = self.power_meter.get_relative_power()
                if relative is not None:
                    self.relative_power_var.set(f"{relative:.2f}%")
                    
                    # Store relative power for export
                    if len(self.power_meter.relative_power_history) < len(powers):
                        self.power_meter.relative_power_history.append(relative)
            
            # Update plot
            self.ax.clear()
            self.ax.plot(times, powers, 'b-')
            self.ax.set_xlabel('Time (s)')
            self.ax.set_ylabel('Power (W)')
            self.ax.grid(True)
            
            # Plot reference line if enabled
            if self.power_meter.reference_enabled and self.power_meter.reference_power is not None:
                self.ax.axhline(y=self.power_meter.reference_power, color='r', linestyle='--', 
                               label=f"Reference: {self.power_meter.reference_power:.9f} W")
                self.ax.legend()
            
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
    
    def update_statistics(self):
        """Update the statistics display"""
        if not self.power_meter.power_history:
            self.status_var.set("No data available for statistics")
            return
            
        self.power_meter.calculate_statistics()
        
        # Update the statistics labels
        for stat, label in self.stat_labels.items():
            value = self.power_meter.stats[stat]
            if stat in ['stability', 'uncertainty']:
                # Show as percentage
                label.config(text=f"{value*100:.4f}%")
            else:
                # Format based on magnitude
                if abs(value) < 0.001:
                    label.config(text=f"{value:.9f}")
                elif abs(value) < 1:
                    label.config(text=f"{value:.6f}")
                else:
                    label.config(text=f"{value:.4f}")
        
        self.status_var.set("Statistics updated")
    
    def export_to_csv(self):
        """Export measurement data to CSV file"""
        if not self.power_meter.power_history:
            self.status_var.set("No data to export")
            return
        
        # Update metadata
        self.update_metadata()
        
        # Use Zenity to prompt for file path
        file_path = subprocess.check_output(['zenity', '--file-selection', '--save', '--confirm-overwrite', '--title="Export to CSV"', '--filename="power_data.csv"']).decode().strip()
        
        if not file_path:
            return
        
        # Export the data
        self.power_meter.export_data(file_path)
        self.status_var.set(f"Data exported to {file_path}")
    
    def export_to_excel(self):
        """Export measurement data to Excel file"""
        if not self.power_meter.power_history:
            self.status_var.set("No data to export")
            return False
            
        try:
            # Update metadata
            self.update_metadata()
            
            # Use Zenity to prompt for file path
            file_path = subprocess.check_output(['zenity', '--file-selection', '--save', '--confirm-overwrite', '--title="Export to Excel"', '--filename="power_data.xlsx"']).decode().strip()
            
            if not file_path:
                return
                
            # Export the data
            if self.power_meter.export_to_excel(file_path):
                self.status_var.set(f"Data exported to {file_path}")
        except Exception as e:
            self.status_var.set(f"Error exporting to Excel: {e}")
    
    def export_statistics(self):
        """Export statistics to JSON file"""
        if not self.power_meter.power_history:
            self.status_var.set("No data for statistics")
            return
            
        # Calculate statistics
        self.power_meter.calculate_statistics()
        
        # Use Zenity to prompt for file path
        file_path = subprocess.check_output(['zenity', '--file-selection', '--save', '--confirm-overwrite', '--title="Export Statistics"', '--filename="statistics.json"']).decode().strip()
        
        if not file_path:
            return
        
        # Export the statistics
        self.power_meter.export_statistics(file_path)
        self.status_var.set(f"Statistics exported to {file_path}")
    
    def update_metadata(self):
        """Update metadata with current information"""
        # Get device info
        if self.device_combo.current() >= 0 and self.device_combo.current() < len(self.devices):
            device_info = self.devices[self.device_combo.current()]
            self.power_meter.metadata['device_model'] = device_info['model']
            self.power_meter.metadata['device_serial'] = device_info['serial']
        
        # Update other metadata
        self.power_meter.metadata['wavelength'] = self.wavelength_var.get()
        self.power_meter.metadata['measurement_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.power_meter.metadata['sample_count'] = len(self.power_meter.power_history)
        
        if self.power_meter.time_history:
            self.power_meter.metadata['measurement_duration'] = max(self.power_meter.time_history)
        
        # Get notes
        self.power_meter.metadata['notes'] = self.notes_var.get()
    
    def set_reference(self):
        """Set current power as reference"""
        if not self.power_meter.connected:
            self.status_var.set("Not connected to a power meter")
            return
            
        if not self.power_meter.power_history:
            self.status_var.set("No power measurements available")
            return
            
        if self.power_meter.set_reference_power():
            self.ref_power_var.set(f"{self.power_meter.reference_power:.9f} W")
            self.status_var.set("Reference power set")
        else:
            self.status_var.set("Failed to set reference power")
    
    def clear_reference(self):
        """Clear reference beam comparison"""
        self.power_meter.disable_reference()
        self.ref_power_var.set("-- W")
        self.relative_power_var.set("-- %")
        self.status_var.set("Reference cleared")
    
    def analyze_beam_stability(self):
        """Analyze beam stability"""
        if not self.power_meter.connected:
            self.status_var.set("Not connected to a power meter")
            return
            
        if not self.power_meter.power_history or len(self.power_meter.power_history) < 10:
            self.status_var.set("Not enough data for stability analysis (need at least 10 points)")
            return
            
        # Get window size
        try:
            window_size = self.stability_window_size.get()
            if window_size <= 0:
                window_size = None
        except:
            window_size = None
            
        # Analyze stability
        stability = self.power_meter.analyze_stability(window_size)
        
        if not stability:
            self.status_var.set("Failed to analyze stability")
            return
            
        # Clear previous results
        for widget in self.stability_results_frame.winfo_children():
            widget.destroy()
            
        # Display results
        row = 0
        
        # Mean and std dev
        ttk.Label(self.stability_results_frame, text="Mean Power:").grid(row=row, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Label(self.stability_results_frame, text=f"{stability['mean']:.9f} W").grid(row=row, column=1, padx=5, pady=2, sticky=tk.W)
        row += 1
        
        ttk.Label(self.stability_results_frame, text="Standard Deviation:").grid(row=row, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Label(self.stability_results_frame, text=f"{stability['std_dev']:.9f} W").grid(row=row, column=1, padx=5, pady=2, sticky=tk.W)
        row += 1
        
        # Coefficient of variation (stability metric)
        ttk.Label(self.stability_results_frame, text="Coefficient of Variation:").grid(row=row, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Label(self.stability_results_frame, text=f"{stability['coefficient_of_variation']*100:.4f}%").grid(row=row, column=1, padx=5, pady=2, sticky=tk.W)
        row += 1
        
        # Drift
        ttk.Label(self.stability_results_frame, text="Power Drift:").grid(row=row, column=0, padx=5, pady=2, sticky=tk.W)
        drift_text = f"{stability['drift_slope']:.9f} W/s"
        if stability['drift_p_value'] < 0.05:
            drift_text += " (Significant)"
        else:
            drift_text += " (Not significant)"
        ttk.Label(self.stability_results_frame, text=drift_text).grid(row=row, column=1, padx=5, pady=2, sticky=tk.W)
        row += 1
        
        # Allan deviations
        if stability['allan_deviations']:
            ttk.Label(self.stability_results_frame, text="Allan Deviations:").grid(row=row, column=0, padx=5, pady=2, sticky=tk.W)
            row += 1
            
            for tau, ad in stability['allan_deviations']:
                ttk.Label(self.stability_results_frame, text=f"  τ = {tau}:").grid(row=row, column=0, padx=5, pady=2, sticky=tk.W)
                ttk.Label(self.stability_results_frame, text=f"{ad:.9f} W").grid(row=row, column=1, padx=5, pady=2, sticky=tk.W)
                row += 1
        
        self.status_var.set("Stability analysis completed")
    
    def update_uncertainty_factors(self):
        """Update uncertainty calculation factors"""
        try:
            # Update the uncertainty factors (convert from percentage to fraction)
            for key, var in self.uncertainty_vars.items():
                self.power_meter.uncertainty_factors[key] = var.get() / 100.0
                
            self.status_var.set("Uncertainty factors updated")
            
            # Recalculate statistics if we have data
            if self.power_meter.power_history:
                self.update_statistics()
                
        except Exception as e:
            self.status_var.set(f"Error updating uncertainty factors: {e}")


if __name__ == "__main__":
    app = PowerMeterGUI()
    app.run()
