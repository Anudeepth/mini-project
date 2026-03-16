import time
import serial.tools.list_ports
from PySide6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QPixmap
from pyfingerprint.pyfingerprint import PyFingerprint

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import shutil

# Keras ImageDataGenerator loads directories in strict alphanumeric order.
# The `dataset/` folder contains A+, A-, AB+, AB-, B+, B-, O+, O-.
# Sorted alphabetically: ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']
blood_groups = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]

# Paths
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resnet_fingerprint_model.keras")
dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")

# Load the ResNet50V2 model
try:
    model = load_model(model_path)
except Exception as e:
    print(f"Error loading ResNet model: {e}")
    model = None

def preprocess_fingerprint(img_path, input_shape=(128,128,3), save_roi_path=None):
    """Preprocess scanner image to match dataset quality before prediction.
    
    Scanner images are faded (mean~239, std~31) while dataset images are
    high-contrast (mean~150, std~100). We apply CLAHE + histogram equalization
    to bridge this quality gap.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image at {img_path}")
    
    # --- Enhancement: fix the faded scanner output ---
    # Convert to grayscale for enhancement
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step 1: CLAHE — boosts local contrast to reveal faint fingerprint ridges
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Step 2: Global histogram equalization — normalizes overall brightness
    # to match the dataset distribution (mean~150, std~100)
    enhanced = cv2.equalizeHist(enhanced)
    
    # Step 3: Light Gaussian blur to reduce scanner noise
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # Convert back to 3-channel RGB (model expects 3 channels)
    img_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    
    # Save the enhanced preview image for the UI
    if save_roi_path:
        cv2.imwrite(save_roi_path, cv2.cvtColor(img_enhanced, cv2.COLOR_RGB2BGR))
    
    # Log stats so we can verify the enhancement is working
    print(f"  Preprocessing: original mean={gray.mean():.0f} std={gray.std():.0f} → enhanced mean={enhanced.mean():.0f} std={enhanced.std():.0f}")
        
    # Resize to model input size
    img_resized = cv2.resize(img_enhanced, (input_shape[0], input_shape[1]))

    # Add batch dimension
    # The image is kept as [0, 255] floats because the Keras model ALREADY has
    # the tensorflow.keras.applications.resnet_v2.preprocess_input layer embedded inside of it!
    img_processed = img_resized.astype('float32')
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

            self.progress.emit("Loading dataset...")

            train_ds = tf.keras.utils.image_dataset_from_directory(
                dataset_path,
                validation_split=0.2,
                subset="training",
                seed=123,
                image_size=IMG_SIZE,
                batch_size=BATCH_SIZE,
                label_mode='categorical'
            )

            val_ds = tf.keras.utils.image_dataset_from_directory(
                dataset_path,
                validation_split=0.2,
                subset="validation",
                seed=123,
                image_size=IMG_SIZE,
                batch_size=BATCH_SIZE,
                label_mode='categorical'
            )

            class_names = train_ds.class_names
            print("Retrain - Classes identified:", class_names)

            self.progress.emit("Applying CLAHE+HistEq preprocessing to dataset...")

            # ---- CRITICAL: Apply the SAME enhancement used at inference time ----
            # This ensures the model trains on images that look like what the scanner produces.
            def enhance_image_batch(images, labels):
                """Apply CLAHE + histogram equalization to a batch of images (via numpy)."""
                def _enhance_batch_np(img_batch):
                    batch = img_batch.numpy().astype(np.uint8)
                    enhanced_batch = []
                    for img in batch:
                        # Convert to grayscale
                        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                        # CLAHE
                        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                        enhanced = clahe.apply(gray)
                        # Global histogram equalization
                        enhanced = cv2.equalizeHist(enhanced)
                        # Gaussian blur
                        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
                        # Back to 3-channel
                        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
                        enhanced_batch.append(enhanced_rgb)
                    return np.array(enhanced_batch, dtype=np.float32)

                enhanced = tf.py_function(_enhance_batch_np, [images], tf.float32)
                enhanced.set_shape(images.shape)
                return enhanced, labels

            # Apply enhancement to both train and validation sets
            train_ds = train_ds.map(enhance_image_batch, num_parallel_calls=tf.data.AUTOTUNE)
            val_ds = val_ds.map(enhance_image_batch, num_parallel_calls=tf.data.AUTOTUNE)

            # Prefetch for performance
            train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
            val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

            self.progress.emit("Building fresh ResNet50V2 model...")

            # ResNet50V2 preprocessing
            preprocess_input = tf.keras.applications.resnet_v2.preprocess_input

            # Data augmentation
            data_augmentation = keras.Sequential([
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.1),
            ])

            # Create FRESH base model from pre-trained ImageNet weights
            base_model = ResNet50V2(input_shape=(128, 128, 3),
                                    include_top=False,
                                    weights='imagenet')

            # Fine-tune: unfreeze top 30 layers
            base_model.trainable = True
            fine_tune_at = len(base_model.layers) - 30
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
                monitor='val_accuracy', patience=4,
                restore_best_weights=True, verbose=1
            )
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss', factor=0.5,
                patience=2, min_lr=1e-6, verbose=1
            )

            self.progress.emit("Training on full dataset (this may take a while)...")

            history = new_model.fit(
                train_ds,
                epochs=20,
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
        layout.addWidget(self.scan_btn)
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
                    
                    pred = model.predict(self.latest_processed_img)
                    
                    # Log the full probability distribution to terminal
                    print("-" * 30)
                    print("Raw Prediction Probabilities:")
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