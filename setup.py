#!/usr/bin/env python3
"""
Setup script for Data Mining Algorithm Explorer
Installs dependencies and sets up the environment.
"""

import subprocess
import sys
import os
import platform

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("❌ Python 3.7 or higher is required!")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    
    try:
        # Upgrade pip first
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        print("✅ All packages installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False

def test_imports():
    """Test if all required packages can be imported"""
    print("\n🧪 Testing imports...")
    
    packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 
        'sklearn', 'scipy', 'tkinter'
    ]
    
    failed_imports = []
    
    for package in packages:
        try:
            if package == 'sklearn':
                import sklearn
            elif package == 'tkinter':
                import tkinter
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        return False
    
    print("✅ All imports successful!")
    return True

def create_desktop_shortcut():
    """Create desktop shortcut (Windows only)"""
    if platform.system() == "Windows":
        try:
            import winshell
            from win32com.client import Dispatch
            
            desktop = winshell.desktop()
            path = os.path.join(desktop, "Data Mining Explorer.lnk")
            target = os.path.join(os.getcwd(), "launcher.py")
            
            shell = Dispatch('WScript.Shell')
            shortcut = shell.CreateShortCut(path)
            shortcut.Targetpath = sys.executable
            shortcut.Arguments = f'"{target}"'
            shortcut.WorkingDirectory = os.getcwd()
            shortcut.IconLocation = sys.executable
            shortcut.save()
            
            print("✅ Desktop shortcut created!")
            
        except ImportError:
            print("ℹ️  To create desktop shortcuts, install: pip install pywin32 winshell")
        except Exception as e:
            print(f"⚠️  Could not create desktop shortcut: {e}")

def main():
    """Main setup function"""
    print("🚀 Data Mining Algorithm Explorer - Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Install requirements
    if not install_requirements():
        print("\n❌ Setup failed during package installation!")
        return
    
    # Test imports
    if not test_imports():
        print("\n❌ Setup failed during import testing!")
        return
    
    # Create desktop shortcut
    create_desktop_shortcut()
    
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("\n🎉 You can now run the applications:")
    print("   python launcher.py          # Choose version")
    print("   python data_mining_app.py   # Basic version")
    print("   python advanced_data_mining_app.py  # Advanced version")
    print("   python demo.py              # Demo without GUI")
    print("\n💡 For help, check README.md")

if __name__ == "__main__":
    main()
