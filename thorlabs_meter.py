import pyvisa
import time
import numpy as np
import threading
import csv
import json
from datetime import datetime
import pandas as pd

class ThorlabsPowerMeter:
    def __init__(self):
        self.rm = pyvisa.ResourceManager('@py')  # Use pyvisa-py backend
        self.instrument = None
        self.connected = False
        self.resource_str = None
        self.wavelength = 650.0  # Default wavelength in nm

        # Measurement state
        self.running = False
        self.measurement_thread = None
        self.power_history = []
        self.time_history = []
        self.start_time = None
        self.max_history_length = 1000

        # Statistics
        self.stats = {}
        self.reference_power = None
        self.reference_enabled = False
        self.relative_power_history = []

        # Metadata
        self.metadata = {
            'device_model': '',
            'device_serial': '',
            'wavelength': self.wavelength,
            'measurement_date': '',
            'measurement_duration': 0,
            'sample_count': 0,
            'notes': ''
        }

    def find_devices(self):
        devices = []
        for res in self.rm.list_resources():
            if "USB" in res:
                try:
                    inst = self.rm.open_resource(res)
                    idn = inst.query("*IDN?").strip()
                    if "THORLABS" in idn.upper():
                        devices.append({
                            "resource": res,
                            "idn": idn
                        })
                    inst.close()
                except Exception as e:
                    print(f"Could not query {res}: {e}")
        return devices

    def connect(self, resource_str):
        try:
            self.instrument = self.rm.open_resource(resource_str)
            self.instrument.timeout = 1000
            self.connected = True
            self.resource_str = resource_str
            print(f"Connected to {resource_str}")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def disconnect(self):
        if self.instrument:
            self.instrument.close()
            self.connected = False
            print("Disconnected")

    def set_wavelength(self, wavelength):
        if self.connected:
            try:
                self.instrument.write(f"SENS:CORR:WAV {wavelength}NM")
                self.wavelength = wavelength
                print(f"Wavelength set to {wavelength} nm")
                return True
            except Exception as e:
                print(f"Failed to set wavelength: {e}")
                return False
        return False

    def get_power(self):
        if self.connected:
            try:
                value = self.instrument.query("MEAS:POW?")
                return float(value.strip())
            except Exception as e:
                print(f"Measurement error: {e}")
        return None

    def start_continuous_measurement(self, interval=0.1):
        if self.running or not self.connected:
            return False

        self.running = True
        self.power_history = []
        self.time_history = []
        self.start_time = time.time()

        def loop():
            while self.running:
                power = self.get_power()
                if power is not None:
                    t = time.time() - self.start_time
                    self.power_history.append(power)
                    self.time_history.append(t)

                    if len(self.power_history) > self.max_history_length:
                        self.power_history.pop(0)
                        self.time_history.pop(0)

                time.sleep(interval)

        self.measurement_thread = threading.Thread(target=loop)
        self.measurement_thread.daemon = True
        self.measurement_thread.start()
        return True

    def stop_continuous_measurement(self):
        self.running = False
        if self.measurement_thread:
            self.measurement_thread.join(timeout=1)
            self.measurement_thread = None

    def get_measurement_data(self):
        return self.time_history.copy(), self.power_history.copy()

    def calculate_statistics(self):
        if not self.power_history:
            return

        powers = np.array(self.power_history)
        self.stats = {
            'mean': float(np.mean(powers)),
            'median': float(np.median(powers)),
            'std_dev': float(np.std(powers)),
            'min': float(np.min(powers)),
            'max': float(np.max(powers)),
            'range': float(np.ptp(powers)),
            'variance': float(np.var(powers)),
            'stability': float(np.std(powers) / np.mean(powers)) if np.mean(powers) != 0 else 0,
            'uncertainty': float(np.std(powers) / np.sqrt(len(powers)))
        }

    def set_reference_power(self, power=None):
        if power is None and self.power_history:
            self.reference_power = self.power_history[-1]
        elif power is not None:
            self.reference_power = power
        else:
            return False

        self.reference_enabled = True
        return True

    def disable_reference(self):
        self.reference_enabled = False

    def get_relative_power(self):
        if self.reference_enabled and self.reference_power and self.power_history:
            return (self.power_history[-1] / self.reference_power) * 100
        return None

    def export_data(self, filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Time (s)", "Power (W)"])
            for t, p in zip(self.time_history, self.power_history):
                writer.writerow([t, p])

    def export_statistics(self, filename):
        self.calculate_statistics()
        with open(filename, 'w') as f:
            json.dump(self.stats, f, indent=4)

    def export_to_excel(self, filename):
        self.calculate_statistics()
        df_data = pd.DataFrame({
            'Time (s)': self.time_history,
            'Power (W)': self.power_history
        })
        df_stats = pd.DataFrame({
            'Statistic': list(self.stats.keys()),
            'Value': list(self.stats.values())
        })
        df_metadata = pd.DataFrame({
            'Property': list(self.metadata.keys()),
            'Value': list(self.metadata.values())
        })

        with pd.ExcelWriter(filename) as writer:
            df_data.to_excel(writer, sheet_name='Measurements', index=False)
            df_stats.to_excel(writer, sheet_name='Statistics', index=False)
            df_metadata.to_excel(writer, sheet_name='Metadata', index=False)

    def __del__(self):
        self.stop_continuous_measurement()
        self.disconnect()
