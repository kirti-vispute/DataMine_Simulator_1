#!/usr/bin/env python3
"""
Data Mining App Launcher
Choose between basic and advanced versions of the application.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import sys
import subprocess
import os

class AppLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Mining App Launcher")
        self.root.geometry("400x300")
        self.root.configure(bg='#f0f0f0')
        
        # Center the window
        self.center_window()
        
        self.setup_ui()
        
    def center_window(self):
        """Center the window on screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
        
    def setup_ui(self):
        """Setup the launcher interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Data Mining Algorithm Explorer", 
                               font=('TkDefaultFont', 16, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Description
        desc_label = ttk.Label(main_frame, 
                              text="Choose the version of the application you want to run:",
                              font=('TkDefaultFont', 10))
        desc_label.pack(pady=(0, 20))
        
        # Version selection frame
        version_frame = ttk.LabelFrame(main_frame, text="Select Version", padding=15)
        version_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Basic version
        basic_frame = ttk.Frame(version_frame)
        basic_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(basic_frame, text="Basic Version", font=('TkDefaultFont', 12, 'bold')).pack(anchor=tk.W)
        ttk.Label(basic_frame, text="• 6 algorithms (K-Means, Decision Tree, etc.)", 
                 font=('TkDefaultFont', 9)).pack(anchor=tk.W)
        ttk.Label(basic_frame, text="• Simple interface", 
                 font=('TkDefaultFont', 9)).pack(anchor=tk.W)
        ttk.Label(basic_frame, text="• Good for beginners", 
                 font=('TkDefaultFont', 9)).pack(anchor=tk.W)
        
        ttk.Button(basic_frame, text="Launch Basic Version", 
                  command=lambda: self.launch_app("data_mining_app.py")).pack(anchor=tk.W, pady=(5, 0))
        
        # Advanced version
        advanced_frame = ttk.Frame(version_frame)
        advanced_frame.pack(fill=tk.X)
        
        ttk.Label(advanced_frame, text="Advanced Version", font=('TkDefaultFont', 12, 'bold')).pack(anchor=tk.W)
        ttk.Label(advanced_frame, text="• 14+ algorithms (DBSCAN, SVM, PCA, etc.)", 
                 font=('TkDefaultFont', 9)).pack(anchor=tk.W)
        ttk.Label(advanced_frame, text="• Enhanced visualizations", 
                 font=('TkDefaultFont', 9)).pack(anchor=tk.W)
        ttk.Label(advanced_frame, text="• Cross-validation, correlation analysis", 
                 font=('TkDefaultFont', 9)).pack(anchor=tk.W)
        
        ttk.Button(advanced_frame, text="Launch Advanced Version", 
                  command=lambda: self.launch_app("advanced_data_mining_app.py")).pack(anchor=tk.W, pady=(5, 0))
        
        # Info frame
        info_frame = ttk.LabelFrame(main_frame, text="Requirements", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(info_frame, text="Make sure you have installed the required packages:", 
                 font=('TkDefaultFont', 9)).pack(anchor=tk.W)
        ttk.Label(info_frame, text="pip install -r requirements.txt", 
                 font=('TkDefaultFont', 9, 'italic')).pack(anchor=tk.W)
        
        # Exit button
        ttk.Button(main_frame, text="Exit", command=self.root.quit).pack(pady=(10, 0))
        
    def launch_app(self, app_file):
        """Launch the selected application"""
        try:
            if not os.path.exists(app_file):
                messagebox.showerror("Error", f"Application file '{app_file}' not found!")
                return
                
            # Launch the application
            subprocess.Popen([sys.executable, app_file])
            self.root.quit()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch application: {str(e)}")

def main():
    """Main function"""
    root = tk.Tk()
    app = AppLauncher(root)
    root.mainloop()

if __name__ == "__main__":
    main()
