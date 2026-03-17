import subprocess
import sys
import importlib.util
import importlib

def install_package(package_names):
    """
    Checks if a package is installed, installs it if not, and ensures it's 
    imported into the current environment if the installation succeeds.
    
    Returns True if all packages are ready, False otherwise.
    """
    all_ready = True
    for package_name in package_names:
        spec = importlib.util.find_spec(package_name)
        
        if spec is None:
            print(f"{package_name} not found. Installing now...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package_name], 
                                      stdout=sys.stdout, stderr=sys.stderr)
                print(f"{package_name} successfully installed.")
                
                spec = importlib.util.find_spec(package_name)
                if spec:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    sys.modules[package_name] = module 
                    print(f"Successfully loaded {package_name} into current environment.")
                else:
                    print(f"ERROR: Installation succeeded, but failed to load {package_name}.")
                    all_ready = False
                    
            except subprocess.CalledProcessError as e:
                print(f"Failed to install {package_name}. Error: {e}")
                all_ready = False
        else:
            print(f"{package_name} is already installed.")
            
    return all_ready

if __name__ == "__main__":
    packages = ['Pillow',
                'opencv-python-headless',
                'scikit-learn',
                'matplotlib',
                'seaborn',
                'mlflow',
                'tensorboard',
                'tqdm',
                'timm',
                'pytorch-metric-learning']
    install_package(packages)
