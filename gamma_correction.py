import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class GammaVisualizer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Gamma Correction Visualization")
        
        # Create main figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        # Add slider frame
        slider_frame = tk.Frame(self.root)
        slider_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Create gamma slider
        self.gamma_var = tk.DoubleVar(value=0.7)
        tk.Label(slider_frame, text="Gamma:").pack(side=tk.LEFT)
        tk.Scale(slider_frame, from_=0.1, to=2.0, resolution=0.1,
                orient=tk.HORIZONTAL, variable=self.gamma_var,
                command=self.update_plot).pack(side=tk.LEFT, fill=tk.X, expand=1)
        
        # Initialize plots
        self.setup_plots()
        self.update_plot()
        
    def setup_plots(self):
        """Setup initial plots"""
        # Phase response plot
        self.ax1.set_xlabel('Input Phase (radians)')
        self.ax1.set_ylabel('Output Phase (radians)')
        self.ax1.set_title('Phase Response')
        self.ax1.grid(True)
        
        # Example pattern plot
        self.ax2.set_xlabel('Position (pixels)')
        self.ax2.set_ylabel('Phase')
        self.ax2.set_title('Example Pattern')
        self.ax2.grid(True)
        
    def update_plot(self, *args):
        """Update plots based on current gamma value"""
        gamma = self.gamma_var.get()
        
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        
        # Plot phase response
        x = np.linspace(0, 1, 256)
        y = x ** gamma
        self.ax1.plot([0, 2*np.pi], [0, 2*np.pi], 'r--', label='Linear (γ=1.0)')
        self.ax1.plot(x * 2*np.pi, y * 2*np.pi, 'b-', label=f'γ={gamma:.1f}')
        self.ax1.set_xlabel('Input Phase (radians)')
        self.ax1.set_ylabel('Output Phase (radians)')
        self.ax1.set_title('Phase Response')
        self.ax1.legend()
        self.ax1.grid(True)
        
        # Plot example pattern (grating)
        x = np.linspace(0, 2*np.pi, 200)
        pattern = np.sin(5*x)  # Create a grating pattern
        corrected_pattern = np.sign(pattern) * np.abs(pattern) ** gamma
        
        self.ax2.plot(x, pattern, 'r--', label='Original')
        self.ax2.plot(x, corrected_pattern, 'b-', label='Gamma Corrected')
        self.ax2.set_xlabel('Position (pixels)')
        self.ax2.set_ylabel('Phase')
        self.ax2.set_title('Example Pattern')
        self.ax2.legend()
        self.ax2.grid(True)
        
        # Update canvas
        self.canvas.draw()
        
    def run(self):
        """Start the application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = GammaVisualizer()
    app.run()