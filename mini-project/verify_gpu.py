
import os
import sys

# Try to find NVIDIA libraries in venv
venv_path = "/home/anudeepth/Documents/mini-project/mini-project/venv"
lib_dirs = []
for root, dirs, files in os.walk(os.path.join(venv_path, "lib/python3.12/site-packages/nvidia")):
    if "lib" in dirs:
        lib_dirs.append(os.path.join(root, "lib"))

if lib_dirs:
    os.environ["LD_LIBRARY_PATH"] = ":".join(lib_dirs) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    print(f"Set LD_LIBRARY_PATH with {len(lib_dirs)} directories")

import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("Python version:", sys.version)

gpus = tf.config.list_physical_devices('GPU')
print("Detected GPUs:", gpus)

if gpus:
    print("SUCCESS: GPU is available for TensorFlow")
else:
    print("FAILURE: No GPU detected by TensorFlow")
