# SLM Control System for SONY LCX016AL-6

This project provides a control interface for the SONY LCX016AL-6 Spatial Light Modulator (SLM) using a Raspberry Pi 4.

## Hardware Requirements

- Raspberry Pi 4
- SONY LCX016AL-6 SLM
- USB Camera
- Two HDMI displays (one for control interface, one for SLM)

## SLM Specifications

- Resolution: 832 x 624 pixels
- Pixel Pitch: 32 μm
- Active Area: 26.6 mm x 20.0 mm (1.3")
- Refresh Rate: max. 60 Hz
- Contrast Ratio: typ. 200:1
- Display Type: Grayscale

## Installation

1. Clone this repository to your Raspberry Pi
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Connect your Raspberry Pi to two displays:
   - HDMI0: Control interface
   - HDMI1: SLM display
2. Connect your USB camera
3. Run the application:
   ```bash
   python slm_controller.py
   ```

## Features

- Real-time camera feed display
- Pattern selection interface
- Pre-loaded SLM patterns
- Custom pattern upload capability
- Real-time SLM control

## Pattern Repository

The `patterns` directory contains pre-generated SLM patterns. You can add your own patterns by placing grayscale PNG files in this directory.
