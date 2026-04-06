import time
import serial.tools.list_ports
from PySide6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QPixmap
from pyfingerprint.pyfingerprint import PyFingerprint
from PySide6.QtWidgets import QFileDialog, QMessageBox

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import shutil

# --- GPU Initialization (Hardware Acceleration) ---
# This prevents TensorFlow from hogging all VRAM and allows 
# the RTX 4050 to work properly without crashing.
def initialize_gpu():
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ Hardware Acceleration: Detected {len(gpus)} GPU(s)")
            # Use Mixed Precision for faster training on RTX 4050
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy('mixed_float16')
            print("🚀 Mixed Precision enabled")
        else:
            print("ℹ️ No GPU detected, using CPU")
    except Exception as e:
        print(f"⚠️ GPU Initialization error: {e}")

initialize_gpu()

# Alphabetically sorted blood group labels (must match training order)
blood_groups = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]

# Paths
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resnet_fingerprint_model.keras")
# Use split_dataset (4200 balanced images, all 8 classes) — NOT dataset/ (84 images, 5 classes)
train_dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "split_dataset", "train")
val_dataset_path   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "split_dataset", "validation")
dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")  # kept for Save-to-Dataset

# Load the ResNet50V2 model
try:
    model = load_model(model_path)
except Exception as e:
    print(f"Error loading ResNet model: {e}")
    model = None

def preprocess_fingerprint(img_path, input_shape=(128,128,3), save_roi_path=None):
    """Preprocess scanner image and fix Zoom/Scale mismatch.
    Smart cropping: Only applies contour-based background cropping if image > 200px (scanner). 
    Dataset uploads which are small remain unharmed.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image at {img_path}")

    # BGR → RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- Smart Scanner ROI CROP ---
    # Scanner images are 288x256 with huge white padding.
    if img_rgb.shape[0] > 200:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 240, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            pad = 5
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(img_rgb.shape[1] - x, w + 2*pad)
            h = min(img_rgb.shape[0] - y, h + 2*pad)
            img_rgb = img_rgb[y:y+h, x:x+w]

    # Resize to exact model input shape directly
    final_img = cv2.resize(img_rgb, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_AREA)

    # Save preview for UI
    if save_roi_path:
        cv2.imwrite(save_roi_path, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))

    # Convert to float32 and pass it directly (Model natively applies preprocessing layer)
    img_array = final_img.astype('float32')
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_with_tta(model, img_array, n_runs=5):
    """Test-Time Augmentation: average predictions over small random augmentations.

    A single noisy scan can push the model the wrong way. By running the image
    through n_runs times with tiny flips/rotations and averaging the softmax
    probabilities we get a much more stable final prediction.
    """
    # Build a lightweight augmentation pipeline (CPU, no training needed)
    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),   # ±18 degrees
        tf.keras.layers.RandomZoom(0.05),
    ])

    preds = []
    for _ in range(n_runs):
        aug = augment(img_array, training=True)  # training=True activates randomness
        preds.append(model.predict(aug, verbose=0))

    # Average all runs
    avg = np.mean(preds, axis=0)
    return avg


class FingerprintThread(QThread):

    finished = Signal(bool)
    progress = Signal(str)

    def __init__(self, port_name):
        super().__init__()
        self.port_name = port_name

    def run(self):
        import os

        NUM_CAPTURES = 3      # fewer captures = less scanner idle time
        CAPTURE_DELAY = 0.15  # very short gap so scanner never sleeps

        try:
            self.progress.emit("Connecting to Scanner...")
            f = PyFingerprint(self.port_name, 57600, 0xFFFFFFFF, 0x00000000)

            if not f.verifyPassword():
                self.progress.emit("Scanner Password Error")
                self.finished.emit(False)
                return

            self.progress.emit("Place Finger On Scanner")

            img_path = "/home/anudeepth/Documents/fingerprint.bmp"
            tmp_paths = []

            for i in range(NUM_CAPTURES):
                self.progress.emit(f"Scanning… ({i+1}/{NUM_CAPTURES}) keep finger still")

                # Keep polling readImage() — this also keeps the sensor active
                while not f.readImage():
                    pass

                tmp_path = f"/home/anudeepth/Documents/fingerprint_tmp_{i}.bmp"
                f.downloadImage(tmp_path)
                tmp_paths.append(tmp_path)

                if i < NUM_CAPTURES - 1:
                    # Very short pause — long enough for next read, short enough
                    # that the scanner does NOT go into its sleep/power-off state
                    time.sleep(CAPTURE_DELAY)

            self.progress.emit("Processing Image...")

            # Average all captures pixel-by-pixel to reduce noise
            frames = [cv2.imread(p).astype(np.float32) for p in tmp_paths]
            averaged = np.mean(frames, axis=0).astype(np.uint8)
            cv2.imwrite(img_path, averaged)

            # Clean up temp files
            for p in tmp_paths:
                try:
                    os.remove(p)
                except Exception:
                    pass

            self.progress.emit("Extracting Features...")
            time.sleep(0.5)

            self.finished.emit(True)

        except Exception as e:
            print(e)
            self.finished.emit(False)



class RetrainThread(QThread):
    """Background thread for proper full-dataset model retraining."""

    progress = Signal(str)
    finished = Signal(bool, str)  # (success, message)

    def run(self):
        try:
            from tensorflow import keras
            from tensorflow.keras import layers
            from tensorflow.keras.applications import ResNet50V2
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

            IMG_SIZE = (128, 128)
            BATCH_SIZE = 32

            self.progress.emit("Loading dataset (split_dataset — all 8 classes)...")

            # Use the pre-split balanced dataset (400–700 images per class, 8 classes)
            # instead of dataset/ which has only 84 images across 5 classes.
            train_ds = tf.keras.utils.image_dataset_from_directory(
                train_dataset_path,
                image_size=IMG_SIZE,
                batch_size=BATCH_SIZE,
                label_mode='categorical',
                shuffle=True,
                seed=123,
            )

            val_ds = tf.keras.utils.image_dataset_from_directory(
                val_dataset_path,
                image_size=IMG_SIZE,
                batch_size=BATCH_SIZE,
                label_mode='categorical',
                shuffle=False,
            )

            class_names = train_ds.class_names
            print("Retrain - Classes identified:", class_names)

            self.progress.emit("No enhancements applied. Using raw robust pipeline...")
            
            # Use datasets directly without mismatch mappings
            train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
            val_ds   = val_ds.prefetch(tf.data.AUTOTUNE)

            self.progress.emit("Building fresh ResNet50V2 model...")

            # ResNet50V2 preprocessing
            preprocess_input = tf.keras.applications.resnet_v2.preprocess_input

            # Data augmentation: teaches model to handle zoom, rotation, and shifts
            data_augmentation = keras.Sequential([
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.2),      # ±36 degrees
                layers.RandomZoom(0.5),          # CRITICAL: Teach model to handle zoomed scanner captures
                layers.RandomTranslation(0.1, 0.1),
            ])

            # Create FRESH base model from pre-trained ImageNet weights
            base_model = ResNet50V2(input_shape=(128, 128, 3),
                                    include_top=False,
                                    weights='imagenet')

            # Fine-tune: unfreeze top 50 layers
            base_model.trainable = True
            fine_tune_at = len(base_model.layers) - 50
            for layer_item in base_model.layers[:fine_tune_at]:
                layer_item.trainable = False

            # Build complete model
            inputs = keras.Input(shape=(128, 128, 3))
            x = data_augmentation(inputs)
            x = preprocess_input(x)
            x = base_model(x, training=False)
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dropout(0.2)(x)
            outputs = layers.Dense(len(blood_groups), activation='softmax')(x)
            new_model = keras.Model(inputs, outputs)

            new_model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            early_stopping = EarlyStopping(
                monitor='val_accuracy', patience=5,
                restore_best_weights=True, verbose=1
            )
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss', factor=0.5,
                patience=3, min_lr=1e-6, verbose=1
            )

            self.progress.emit("Phase 2: Fine-tuning top 50 ResNet layers...")

            history = new_model.fit(
                train_ds,
                epochs=30,
                validation_data=val_ds,
                callbacks=[early_stopping, reduce_lr]
            )

            # Get final accuracy
            final_val_acc = history.history['val_accuracy'][-1] * 100

            # Save the newly trained model
            new_model.save(model_path)

            self.finished.emit(True, f"Retrain complete! Val accuracy: {final_val_acc:.1f}%")

        except Exception as e:
            print(f"Retrain error: {e}")
            self.finished.emit(False, str(e))


class MainWindow(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Blood Group Detection Using Fingerprint")
        self.resize(600, 700)
        
        self.is_scanning = False
        self.is_retraining = False
        self.new_images_count = 0  # Track images collected since last training
        self.active_port = None   # Set by check_scanner_connection(); init here to avoid AttributeError

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

        from PySide6.QtWidgets import QProgressBar, QGridLayout
        
        # Probability chart
        self.chart_layout = QGridLayout()
        self.bars = {}
        for i, bg in enumerate(blood_groups):
            lbl = QLabel(bg)
            lbl.setStyleSheet("font-weight: bold; color: #2E3B32;")
            bar = QProgressBar()
            bar.setTextVisible(True)
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #81C784;
                    border-radius: 4px;
                    text-align: center;
                    color: black;
                }
                QProgressBar::chunk {
                    background-color: #4CAF50;
                }
            """)
            self.bars[bg] = bar
            
            row = i // 2
            col = (i % 2) * 2
            self.chart_layout.addWidget(lbl, row, col)
            self.chart_layout.addWidget(bar, row, col + 1)

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
        self.scan_btn.setMinimumHeight(45)
        self.scan_btn.setCursor(Qt.PointingHandCursor)
        self.scan_btn.setStyleSheet("""
            QPushButton {
                background-color: #2E8B57;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 22px;
            }
            QPushButton:hover { background-color: #3CB371; }
            QPushButton:pressed { background-color: #1B5E20; }
            QPushButton:disabled { background-color: #A5D6A7; }
        """)
        self.scan_btn.clicked.connect(self.start_scan)

        self.upload_btn = QPushButton("Upload Image")
        self.upload_btn.setMinimumHeight(45)
        self.upload_btn.setCursor(Qt.PointingHandCursor)
        self.upload_btn.setStyleSheet("""
            QPushButton {
                background-color: #0288D1;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 22px;
            }
            QPushButton:hover { background-color: #03A9F4; }
            QPushButton:pressed { background-color: #01579B; }
        """)
        self.upload_btn.clicked.connect(self.upload_image)

        scan_layout = QHBoxLayout()
        scan_layout.addWidget(self.scan_btn)
        scan_layout.addWidget(self.upload_btn)

        from PySide6.QtWidgets import QComboBox

        # Teach Model Control Area (data collection only — no more model.fit!)
        teach_layout = QHBoxLayout()
        teach_layout.setSpacing(10)
        
        self.teach_label = QLabel("Correct Blood Group:")
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
        
        self.teach_btn = QPushButton("Save to Dataset")
        self.teach_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                font-weight: bold;
                border-radius: 4px;
                padding: 8px 15px;
            }
            QPushButton:hover { background-color: #F57C00; }
            QPushButton:disabled { background-color: #FFCC80; }
        """)
        self.teach_btn.clicked.connect(self.teach_model)
        self.teach_btn.setEnabled(False) # Disabled until a scan completes

        teach_layout.addStretch()
        teach_layout.addWidget(self.teach_label)
        teach_layout.addWidget(self.teach_combo)
        teach_layout.addWidget(self.teach_btn)
        teach_layout.addStretch()

        # New images counter + Retrain button
        retrain_layout = QHBoxLayout()
        retrain_layout.setSpacing(10)

        self.new_images_label = QLabel("New images collected: 0")
        self.new_images_label.setStyleSheet("color: #555; font-size: 13px;")

        self.retrain_btn = QPushButton("🔄 Retrain Model on Full Dataset")
        self.retrain_btn.setMinimumHeight(45)
        self.retrain_btn.setCursor(Qt.PointingHandCursor)
        self.retrain_btn.setStyleSheet("""
            QPushButton {
                background-color: #1565C0;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 22px;
                padding: 8px 20px;
            }
            QPushButton:hover { background-color: #1976D2; }
            QPushButton:pressed { background-color: #0D47A1; }
            QPushButton:disabled { background-color: #90CAF9; }
        """)
        self.retrain_btn.clicked.connect(self.start_retrain)

        retrain_layout.addWidget(self.new_images_label)
        retrain_layout.addStretch()
        retrain_layout.addWidget(self.retrain_btn)

        layout.addWidget(self.title)
        layout.addWidget(self.status)
        layout.addLayout(image_layout)
        layout.addLayout(self.chart_layout)
        layout.addWidget(self.result)
        layout.addLayout(scan_layout)
        layout.addLayout(teach_layout)
        layout.addLayout(retrain_layout)

        self.setLayout(layout)

    def check_scanner_connection(self):
        if self.is_scanning or self.is_retraining:
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

                    # Store latest image path for dataset collection
                    self.latest_img_path = img_path
                    self.latest_processed_img = preprocess_fingerprint(img_path, input_shape=input_shape, save_roi_path=roi_path)

                    # TTA: average 5 augmented runs for a more stable prediction
                    pred = predict_with_tta(model, self.latest_processed_img, n_runs=5)

                    # Log the full probability distribution to terminal
                    print("-" * 30)
                    print("TTA-Averaged Prediction Probabilities:")
                    for i, prob in enumerate(pred[0]):
                        percent_val = int(prob * 100)
                        print(f"  {blood_groups[i]}: {prob * 100:.2f}%")
                        self.bars[blood_groups[i]].setValue(percent_val)
                    print("-" * 30)

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

        else:
            self.status.setText("Scanner Error. Please try again.")

    def upload_image(self):
        """Allows user to upload a fingerprint image directly without scanning."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Fingerprint Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.status.setText("Processing Uploaded Image...")
            self.status.setStyleSheet("color: #2196F3; font-size: 16px; font-weight: bold; background: transparent;")
            # Process the image directly instead of scanning
            img_path = "/home/anudeepth/Documents/fingerprint.bmp"
            
            # Copy to standardize pipeline
            img = cv2.imread(file_path)
            cv2.imwrite(img_path, img)

            # Manually trigger success flow
            self.scan_finished(True)

    def teach_model(self):
        """Save the scanned fingerprint to the dataset folder (NO model training).
        
        This only collects data. To actually improve the model, click "Retrain Model"
        after collecting enough new images.
        """
        if not hasattr(self, 'latest_img_path'):
            return
            
        true_blood_group = self.teach_combo.currentText()
        
        try:
            # Copy fingerprint to the correct blood group dataset folder
            bg_dataset_dir = os.path.join(dataset_path, true_blood_group)
            if not os.path.exists(bg_dataset_dir):
                os.makedirs(bg_dataset_dir)
            
            img_count = len(os.listdir(bg_dataset_dir))
            dest_path = os.path.join(bg_dataset_dir, f"live_capture_{img_count+1}.bmp")
            shutil.copy(self.latest_img_path, dest_path)
            
            self.new_images_count += 1
            self.new_images_label.setText(f"New images collected: {self.new_images_count}")
            
            self.status.setText(f"✅ Saved fingerprint as {true_blood_group} to dataset ({self.new_images_count} new)")
            self.status.setStyleSheet("color: #4CAF50; font-size: 16px; font-weight: bold; background: transparent;")
            
            print(f"Saved fingerprint to {dest_path}")
            
        except Exception as e:
            print(f"Save error: {e}")
            self.status.setText(f"Error saving: {e}")
            
        self.teach_btn.setEnabled(False)  # Disable until next scan

    def start_retrain(self):
        """Start full-dataset retraining in a background thread."""
        global model

        self.is_retraining = True
        self.retrain_btn.setEnabled(False)
        self.retrain_btn.setText("⏳ Retraining...")
        self.scan_btn.setEnabled(False)
        self.teach_btn.setEnabled(False)

        self.status.setText("Retraining model on full dataset — please wait...")
        self.status.setStyleSheet("color: #1565C0; font-size: 16px; font-weight: bold; background: transparent;")

        self.retrain_thread = RetrainThread()
        self.retrain_thread.progress.connect(self.update_status)
        self.retrain_thread.finished.connect(self.retrain_finished)
        self.retrain_thread.start()

    def retrain_finished(self, success, message):
        """Called when the retrain thread finishes."""
        global model

        self.is_retraining = False
        self.retrain_btn.setEnabled(True)
        self.retrain_btn.setText("🔄 Retrain Model on Full Dataset")
        self.scan_btn.setEnabled(True)

        if success:
            # Reload the freshly trained model
            try:
                model = load_model(model_path)
                print("Reloaded retrained model successfully.")
            except Exception as e:
                print(f"Error reloading model: {e}")

            self.new_images_count = 0
            self.new_images_label.setText("New images collected: 0")

            self.status.setText(f"✅ {message}")
            self.status.setStyleSheet("color: #4CAF50; font-size: 16px; font-weight: bold; background: transparent;")
        else:
            self.status.setText(f"❌ Retrain failed: {message}")
            self.status.setStyleSheet("color: #F44336; font-size: 16px; font-weight: bold; background: transparent;")