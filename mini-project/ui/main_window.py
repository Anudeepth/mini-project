import random
import time
import serial.tools.list_ports
from PySide6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QPixmap
from pyfingerprint.pyfingerprint import PyFingerprint

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Keras ImageDataGenerator loads directories in strict alphanumeric order.
# The `dataset/` folder contains A+, A-, AB+, AB-, B+, B-, O+, O-.
# Sorted alphabetically: ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']
blood_groups = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]

# Load the model once when the app starts
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fingerprint_model.keras")
try:
    model = load_model(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def preprocess_fingerprint(img_path, input_shape=(128,128,1), save_roi_path=None):
    """Preprocess image with MinMax stretching and standard equalization"""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image at {img_path}")
        
    # Convert to grayscale if model expects 1 channel
    if input_shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    # Resize to model input size
    img = cv2.resize(img, (input_shape[0], input_shape[1]))
    
    # NEW: Contrast Stretching. This fixes "diminished" or faint fingerprint captures 
    # by stretching the faintest gray to pure black and the lightest to pure white
    # before applying equalization.
    if input_shape[2] == 1:
        img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    # Improve contrast exactly as test_script did
    img_eq = cv2.equalizeHist(img) if input_shape[2]==1 else img
        
    # Save the preview image for the UI
    if save_roi_path:
        cv2.imwrite(save_roi_path, img_eq)

    # Noise reduction
    img_processed = cv2.GaussianBlur(img_eq,(3,3),0)
    
    # Normalize
    img_processed = img_processed / 255.0
    
    # Add channel dimension if needed
    if input_shape[2] == 1:
        img_processed = np.expand_dims(img_processed, axis=-1)
        
    # Add batch dimension
    img_processed = np.expand_dims(img_processed, axis=0)
    
    return img_processed


class FingerprintThread(QThread):

    finished = Signal(bool)
    progress = Signal(str)

    def __init__(self, port_name):
        super().__init__()
        self.port_name = port_name

    def run(self):

        try:
            self.progress.emit("Connecting to Scanner...")
            f = PyFingerprint(self.port_name, 57600, 0xFFFFFFFF, 0x00000000)

            if not f.verifyPassword():
                self.progress.emit("Scanner Password Error")
                self.finished.emit(False)
                return

            self.progress.emit("Place Finger On Scanner")
            while not f.readImage():
                pass

            self.progress.emit("Scanning Fingerprint...")
            f.downloadImage("/home/anudeepth/Documents/fingerprint.bmp")

            # Simulate processing and extracting for UI feedback
            time.sleep(0.5)
            self.progress.emit("Processing Image...")
            time.sleep(0.8)
            self.progress.emit("Extracting Features...")
            time.sleep(0.8)

            self.finished.emit(True)

        except Exception as e:
            print(e)
            self.finished.emit(False)


class MainWindow(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Blood Group Detection Using Fingerprint")
        self.resize(600, 600)
        
        self.is_scanning = False

        # Setup connection timer to monitor scanner presence
        self.connection_timer = QTimer(self)
        self.connection_timer.timeout.connect(self.check_scanner_connection)
        self.connection_timer.start(1500)  # Check every 1.5 seconds

        # Style the main window with a beautiful, clean medical theme
        self.setStyleSheet("""
            QWidget {
                background-color: #F4F9F4;
                color: #2E3B32;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
        """)

        layout = QVBoxLayout()
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(25)

        self.title = QLabel("AI Blood Group Detection")
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet("color: #1B5E20; font-size: 28px; font-weight: bold; background: transparent;")

        self.status = QLabel("Scanner Status : READY")
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setStyleSheet("color: #4CAF50; font-size: 16px; font-weight: bold; background: transparent;")

        # fingerprint image wrapped for centering
        self.image_label = QLabel("Fingerprint\nPreview")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(250, 300)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: #FFFFFF;
                border: 2px dashed #81C784;
                border-radius: 12px;
                color: #81C784;
                font-size: 20px;
                font-weight: bold;
            }
        """)

        image_layout = QHBoxLayout()
        image_layout.addStretch()
        image_layout.addWidget(self.image_label)
        image_layout.addStretch()

        self.result = QLabel("Blood Group : —")
        self.result.setAlignment(Qt.AlignCenter)
        self.result.setStyleSheet("""
            QLabel {
                color: #2E8B57;
                font-size: 24px;
                font-weight: bold;
                background-color: #E8F5E9;
                border: 1px solid #C8E6C9;
                border-radius: 8px;
                padding: 15px;
            }
        """)

        self.scan_btn = QPushButton("Scan Fingerprint")
        self.scan_btn.setMinimumHeight(55)
        self.scan_btn.setCursor(Qt.PointingHandCursor)
        self.scan_btn.setStyleSheet("""
            QPushButton {
                background-color: #2E8B57;
                color: white;
                font-size: 18px;
                font-weight: bold;
                border-radius: 27px;
            }
            QPushButton:hover {
                background-color: #3CB371;
            }
            QPushButton:pressed {
                background-color: #1B5E20;
            }
            QPushButton:disabled {
                background-color: #A5D6A7;
            }
        """)
        self.scan_btn.clicked.connect(self.start_scan)

        from PySide6.QtWidgets import QComboBox

        # Teach Model Control Area
        teach_layout = QHBoxLayout()
        teach_layout.setSpacing(10)
        
        self.teach_label = QLabel("Correct Prediction:")
        self.teach_label.setStyleSheet("color: #1B5E20; font-weight: bold;")
        
        self.teach_combo = QComboBox()
        self.teach_combo.addItems(blood_groups)
        self.teach_combo.setStyleSheet("""
            QComboBox {
                padding: 5px;
                border: 1px solid #C8E6C9;
                border-radius: 4px;
                min-width: 80px;
            }
        """)
        
        self.teach_btn = QPushButton("Teach Model")
        self.teach_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                font-weight: bold;
                border-radius: 4px;
                padding: 8px 15px;
            }
            QPushButton:hover { background-color: #F57C00; }
        """)
        self.teach_btn.clicked.connect(self.teach_model)
        self.teach_btn.setEnabled(False) # Disabled until a scan completes

        teach_layout.addStretch()
        teach_layout.addWidget(self.teach_label)
        teach_layout.addWidget(self.teach_combo)
        teach_layout.addWidget(self.teach_btn)
        teach_layout.addStretch()

        layout.addWidget(self.title)
        layout.addWidget(self.status)
        layout.addLayout(image_layout)
        layout.addWidget(self.result)
        layout.addWidget(self.scan_btn)
        layout.addLayout(teach_layout)

        self.setLayout(layout)

    def check_scanner_connection(self):
        if self.is_scanning:
            return

        ports = [port.device for port in serial.tools.list_ports.comports()]
        self.active_port = next((p for p in ports if 'USB' in p.upper() or 'ACM' in p.upper() or 'COM' in p.upper()), None)

        if self.active_port:
            self.status.setText("Scanner Status : READY")
            self.status.setStyleSheet("color: #4CAF50; font-size: 16px; font-weight: bold; background: transparent;")
            self.scan_btn.setEnabled(True)
            self.scan_btn.setText("Scan Fingerprint")
        else:
            self.status.setText("Scanner Not Connected")
            self.status.setStyleSheet("color: #F44336; font-size: 16px; font-weight: bold; background: transparent;")
            self.scan_btn.setEnabled(False)
            self.scan_btn.setText("Device Offline")

    def start_scan(self):
        self.is_scanning = True
        self.status.setText("Place Finger On Scanner")
        self.status.setStyleSheet("color: #2196F3; font-size: 16px; font-weight: bold; background: transparent;")
        self.scan_btn.setEnabled(False)
        self.scan_btn.setText("Scanning...")

        self.thread = FingerprintThread(self.active_port)
        self.thread.progress.connect(self.update_status)
        self.thread.finished.connect(self.scan_finished)
        self.thread.start()

    def update_status(self, text):
        self.status.setText(text)
        self.status.setStyleSheet("color: #2196F3; font-size: 16px; font-weight: bold; background: transparent;")

    def scan_finished(self, success):
        self.is_scanning = False
        self.scan_btn.setEnabled(True)
        self.scan_btn.setText("Scan Fingerprint")

        if success:

            # display fingerprint image
            img_path = "/home/anudeepth/Documents/fingerprint.bmp"
            roi_path = "/home/anudeepth/Documents/fingerprint_roi.bmp"

            if model is not None:
                try:
                    input_shape = model.input_shape[1:]
                    
                    # Store latest processed img globally for training
                    self.latest_processed_img = preprocess_fingerprint(img_path, input_shape=input_shape, save_roi_path=roi_path)
                    
                    pred = model.predict(self.latest_processed_img)
                    index = np.argmax(pred)
                    blood = blood_groups[index]
                    confidence = int(pred[0][index] * 100)
                    self.result.setText(f"Blood Group : {blood} ({confidence}%)")
                    
                    # Enable teach button & select predicted blood group
                    self.teach_combo.setCurrentText(blood)
                    self.teach_btn.setEnabled(True)
                    
                except Exception as e:
                    print(f"Prediction error: {e}")
                    self.result.setText("Prediction Error")
            else:
                 self.result.setText("Model Not Loaded")

            # display the enhanced fingerprint image instead of raw capture
            if os.path.exists(roi_path):
                pixmap = QPixmap(roi_path)
            else:
                pixmap = QPixmap(img_path)
                
            pixmap = pixmap.scaled(250, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(pixmap)
            self.image_label.setStyleSheet("border: 2px solid #4CAF50; border-radius: 12px; background-color: #FFFFFF;")
            
            self.status.setText("Fingerprint Captured")
            self.status.setText("Fingerprint Captured")

        else:
            self.status.setText("Scanner Error. Please try again.")

    def teach_model(self):
        if model is None or not hasattr(self, 'latest_processed_img'):
            return
            
        true_blood_group = self.teach_combo.currentText()
        true_index = blood_groups.index(true_blood_group)
        
        # Create one-hot encoded label
        y_true = np.zeros((1, len(blood_groups)))
        y_true[0, true_index] = 1.0
        
        self.status.setText(f"Teaching model for {true_blood_group}...")
        self.status.setStyleSheet("color: #FF9800; font-size: 16px; font-weight: bold; background: transparent;")
        self.teach_btn.setEnabled(False)
        self.scan_btn.setEnabled(False)
        
        # Train
        try:
            # Prevent 'catastrophic forgetting' by using a very small learning rate
            # so we only nudge the weights towards this new print, instead of rewriting them entirely.
            from tensorflow.keras.optimizers import Adam
            optimizer = Adam(learning_rate=0.00005)
            
            # Recompile specifically to accept one-hot encoded `categorical_crossentropy`
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
            
            # Use only 2 epochs to prevent the model from overfitting to this single image
            model.fit(self.latest_processed_img, y_true, epochs=2, verbose=1)
            model.save(model_path)
            self.status.setText(f"Model Learned: {true_blood_group}! Model Saved.")
            self.status.setStyleSheet("color: #4CAF50; font-size: 16px; font-weight: bold; background: transparent;")
            
            # Optionally copy fingerprint.bmp to dataset folder
            import shutil
            dataset_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", true_blood_group)
            if not os.path.exists(dataset_dir):
                os.makedirs(dataset_dir)
            
            img_count = len(os.listdir(dataset_dir))
            shutil.copy("/home/anudeepth/Documents/fingerprint.bmp", 
                        os.path.join(dataset_dir, f"live_capture_{img_count+1}.bmp"))
            
        except Exception as e:
            print(f"Teaching error: {e}")
            self.status.setText("Error saving weights.")
            
        self.teach_btn.setEnabled(True)
        self.scan_btn.setEnabled(True)