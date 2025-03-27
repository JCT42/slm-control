#!/bin/bash
# Script to install Thorlabs Power Meter dependencies on Raspberry Pi
# Transfer this script to your Raspberry Pi and run it with: bash install_thorlabs_pi.sh

echo "Installing dependencies for Thorlabs Power Meter interface..."

# Update package lists
sudo apt-get update

# Install Python and pip if not already installed
sudo apt-get install -y python3 python3-pip

# Install required Python packages
sudo pip3 install numpy matplotlib pyusb pyvisa

# Install libusb for USB device access
sudo apt-get install -y libusb-1.0-0-dev

# Install udev rules for Thorlabs devices
echo 'SUBSYSTEMS=="usb", ATTRS{idVendor}=="1313", MODE="0666"' | sudo tee /etc/udev/rules.d/99-thorlabs.rules

# Reload udev rules
sudo udevadm control --reload-rules
sudo udevadm trigger

echo "Installing Thorlabs Python libraries from GitHub..."

# Create a directory for Thorlabs libraries
mkdir -p ~/thorlabs
cd ~/thorlabs

# Clone the Thorlabs Python libraries from GitHub
git clone https://github.com/thorlabs/thorlabs_apt_device_python.git
git clone https://github.com/thorlabs/thorlabs_tsi_sdk_python.git

# Install the Thorlabs APT device library
cd thorlabs_apt_device_python
sudo pip3 install .
cd ..

echo "Setting up VISA environment..."

# Install Python VISA backend
sudo pip3 install python-vxi11 zeroconf

echo "Installation complete!"
echo "Please connect your Thorlabs Power Meter to the Raspberry Pi via USB."
echo "You may need to restart your Raspberry Pi for all changes to take effect."
