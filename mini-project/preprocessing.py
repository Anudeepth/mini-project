import cv2
import numpy as np

def preprocess_fingerprint(input_path, output_path="processed_fingerprint.bmp", size=(128, 128)):
    """
    Preprocess fingerprint image:
    1. Convert to grayscale
    2. Enhance ridges using histogram equalization
    3. Apply Gaussian blur and adaptive thresholding
    4. Resize to model input size
    5. Save as BMP
    Returns preprocessed image ready for model prediction.
    """
    # Load image
    img = cv2.imread(input_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast
    gray = cv2.equalizeHist(gray)
    
    # Smooth image to remove noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Extract fingerprint lines using adaptive threshold
    lines = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=11, C=2
    )
    
    # Optional: invert so ridges are white
    lines = cv2.bitwise_not(lines)
    
    # Resize to model input
    lines_resized = cv2.resize(lines, size)
    
    # Save as BMP
    cv2.imwrite(output_path, lines_resized)
    
    # Convert to 3 channels if model expects RGB
    final_img = cv2.cvtColor(lines_resized, cv2.COLOR_GRAY2RGB)
    
    # Normalize for model
    final_img = final_img.astype('float32') / 255.0
    final_img = np.expand_dims(final_img, axis=0)  # Shape: (1, height, width, 3)
    
    return final_img

# Usage:
processed_img = preprocess_fingerprint("fingerprint_input.jpg", "fingerprint_output.bmp")