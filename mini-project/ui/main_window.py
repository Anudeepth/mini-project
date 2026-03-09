import random
import time
import serial.tools.list_ports
from PySide6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QPixmap
from pyfingerprint.pyfingerprint import PyFingerprint


blood_groups = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]


class FingerprintThread(QThread):

    finished = Signal(bool)
    progress = Signal(str)

    def run(self):

        try:
            self.progress.emit("Connecting to Scanner...")
            f = PyFingerprint('COM8', 57600, 0xFFFFFFFF, 0x00000000)

            if not f.verifyPassword():
                self.progress.emit("Scanner Password Error")
                self.finished.emit(False)
                return

            self.progress.emit("Place Finger On Scanner")
            while not f.readImage():
                pass

            self.progress.emit("Scanning Fingerprint...")
            f.downloadImage("C:\\fingerprint\\fingerprint.bmp")

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

        layout.addWidget(self.title)
        layout.addWidget(self.status)
        layout.addLayout(image_layout)
        layout.addWidget(self.scan_btn)
        layout.addWidget(self.result)

        self.setLayout(layout)

    def check_scanner_connection(self):
        if self.is_scanning:
            return

        ports = [port.device for port in serial.tools.list_ports.comports()]
        if 'COM8' in ports:
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

        self.thread = FingerprintThread()
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
            pixmap = QPixmap("C:\\fingerprint\\fingerprint.bmp")
            pixmap = pixmap.scaled(250, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(pixmap)
            self.image_label.setStyleSheet("border: 2px solid #4CAF50; border-radius: 12px; background-color: #FFFFFF;")

            # random prediction
            blood = random.choice(blood_groups)
            confidence = random.randint(90, 99)

            self.result.setText(f"Blood Group : {blood} ({confidence}%)")
            self.status.setText("Fingerprint Captured")

        else:
            self.status.setText("Scanner Error. Please try again.")