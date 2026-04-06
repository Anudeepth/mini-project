import base64
import urllib.request
import json
import os

diagrams = {
    "architecture.png": """
graph TD
    A[PyQt5 Frontend GUI] --> B{Controller}
    B --> C[Image Preprocessing Module]
    C --> D[OpenCV CLAHE & Filter]
    D --> E[ResNet50V2 Backend Model]
    E --> F[Test-Time Augmentation]
    F --> G[Categorical Prediction]
    G --> B
    B --> H[Update GUI Result]
    """,
    "dfd_level0.png": """
graph TD
    User((User)) -->|Scanned Fingerprint| System[Blood Group Detection System]
    System -->|Predicted Blood Group| User
    System -->|Logs Data| DB[(Local Storage)]
    """,
    "dfd_level1.png": """
graph TD
    A[Raw Original Scan] --> B[Grayscale Conversion]
    B --> C[CLAHE Histogram Equalization]
    C --> D[Gaussian Blur Noise Reduction]
    D --> E[Adaptive Thresholding]
    E --> F[Tensor Resize 128x128]
    F --> G[ResNet Preprocess Input]
    G --> H((To Model Input))
    """
}

def generate_diagram(mermaid_code, filename):
    try:
        encoded = base64.b64encode(mermaid_code.encode('utf-8')).decode('ascii').replace('+', '-').replace('/', '_') # Standard base64url encode for mermaid
        url = f"https://mermaid.ink/img/{encoded}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(filename, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        print(f"Successfully downloaded {filename}")
    except Exception as e:
        print(f"Failed to generate {filename}: {e}")

for filename, code in diagrams.items():
    generate_diagram(code.strip(), filename)
