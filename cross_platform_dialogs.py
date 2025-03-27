"""
Cross-platform file dialog utilities that work on both Windows and Raspberry Pi.
This module provides a unified interface for file dialogs that automatically
uses the appropriate implementation based on the platform.
"""

import os
import platform
import subprocess
import tkinter as tk
from tkinter import filedialog

def get_file_dialog(action='open', title='Select File', 
                   filetypes=None, default_name=None, 
                   confirm_overwrite=True):
    """
    Cross-platform file dialog that works on both Windows and Linux (Raspberry Pi).
    
    Parameters:
    - action: 'open' or 'save'
    - title: Dialog title
    - filetypes: List of tuples (description, pattern) or string for zenity
    - default_name: Default filename for save dialogs
    - confirm_overwrite: Whether to confirm before overwriting existing files
    
    Returns:
    - Selected file path or None if canceled
    """
    # Check if we're on Windows or Linux
    is_windows = platform.system() == 'Windows'
    
    if is_windows:
        # Use tkinter file dialog on Windows
        if action == 'open':
            # Convert filetypes to tkinter format if needed
            tk_filetypes = [('All Files', '*.*')]
            if filetypes and isinstance(filetypes, list):
                tk_filetypes = filetypes
            elif filetypes and isinstance(filetypes, str):
                # Parse zenity-style filter
                parts = filetypes.split('|')
                if len(parts) >= 2:
                    desc = parts[0].strip()
                    exts = parts[1].strip().split()
                    tk_filetypes = [(desc, ' '.join(exts))]
            
            filepath = filedialog.askopenfilename(
                title=title,
                filetypes=tk_filetypes
            )
            return filepath if filepath else None
            
        elif action == 'save':
            # Convert filetypes to tkinter format if needed
            tk_filetypes = [('All Files', '*.*')]
            if filetypes and isinstance(filetypes, list):
                tk_filetypes = filetypes
            elif filetypes and isinstance(filetypes, str):
                # Parse zenity-style filter
                parts = filetypes.split('|')
                if len(parts) >= 2:
                    desc = parts[0].strip()
                    exts = parts[1].strip().split()
                    tk_filetypes = [(desc, ' '.join(exts))]
            
            filepath = filedialog.asksaveasfilename(
                title=title,
                filetypes=tk_filetypes,
                initialfile=default_name
            )
            
            # Add extension if not present (for common image types)
            if filepath and not filepath.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                # Check filetypes for default extension
                default_ext = '.png'  # Default to PNG
                if filetypes and isinstance(filetypes, list) and len(filetypes) > 0:
                    first_ext = filetypes[0][1]
                    if '*.' in first_ext:
                        default_ext = first_ext.split('*.')[1].split()[0]
                filepath += default_ext
            
            return filepath if filepath else None
    else:
        # Use zenity on Linux (Raspberry Pi)
        try:
            cmd = ['zenity', '--file-selection']
            
            if action == 'save':
                cmd.append('--save')
                if default_name:
                    cmd.append('--filename=' + default_name)
                if confirm_overwrite:
                    cmd.append('--confirm-overwrite')
            
            cmd.append('--title=' + title)
            
            if filetypes:
                if isinstance(filetypes, str):
                    cmd.append('--file-filter=' + filetypes)
                elif isinstance(filetypes, list):
                    # Convert tkinter-style filetypes to zenity format
                    filter_str = "All Files | *.*"
                    for desc, patterns in filetypes:
                        if patterns.startswith('*.'):
                            filter_str = f"{desc} | {patterns}"
                            break
                    cmd.append('--file-filter=' + filter_str)
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                return None
                
            filepath = result.stdout.strip()
            if not filepath:
                return None
            
            # Add extension if not present for save dialogs
            if action == 'save' and not filepath.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                filepath += '.png'  # Default to PNG
            
            return filepath
            
        except Exception as e:
            print(f"Error with zenity dialog: {str(e)}")
            # Fall back to tkinter as a last resort
            if action == 'open':
                return filedialog.askopenfilename(title=title)
            else:
                return filedialog.asksaveasfilename(title=title, initialfile=default_name)

def open_file_dialog(title='Open File', filetypes=None):
    """Convenience function for opening files"""
    return get_file_dialog('open', title, filetypes)

def save_file_dialog(title='Save File', filetypes=None, default_name=None):
    """Convenience function for saving files"""
    return get_file_dialog('save', title, filetypes, default_name)
