import random
from PySide6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap
from pyfingerprint.pyfingerprint import PyFingerprint


blood_groups = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]


class FingerprintThread(QThread):

    finished = Signal(bool)

    def run(self):

        try:
            f = PyFingerprint('COM8', 57600, 0xFFFFFFFF, 0x00000000)

            if not f.verifyPassword():
                self.finished.emit(False)
                return

            while not f.readImage():
                pass

            f.downloadImage("C:\\fingerprint\\fingerprint.bmp")

            self.finished.emit(True)

        except Exception as e:
            print(e)
            self.finished.emit(False)


class MainWindow(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Blood Group Detection Using Fingerprint")
        self.resize(600, 500)

        layout = QVBoxLayout()

        self.title = QLabel("AI Blood Group Detection")
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet("font-size:24px;font-weight:bold")

        self.status = QLabel("Scanner Status : READY")
        self.status.setAlignment(Qt.AlignCenter)

        # fingerprint image
        self.image_label = QLabel("Fingerprint Image")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(300,300)
        self.image_label.setStyleSheet("border:1px solid gray")

        self.result = QLabel("Blood Group : —")
        self.result.setAlignment(Qt.AlignCenter)
        self.result.setStyleSheet("font-size:22px")

        self.scan_btn = QPushButton("Scan Fingerprint")
        self.scan_btn.clicked.connect(self.start_scan)

        layout.addWidget(self.title)
        layout.addWidget(self.status)
        layout.addWidget(self.image_label)
        layout.addWidget(self.scan_btn)
        layout.addWidget(self.result)

        self.setLayout(layout)

    def start_scan(self):

        self.status.setText("Place Finger On Scanner")

        self.thread = FingerprintThread()
        self.thread.finished.connect(self.scan_finished)
        self.thread.start()

    def scan_finished(self, success):

        if success:

            # display fingerprint image
            pixmap = QPixmap("C:\\fingerprint\\fingerprint.bmp")
            pixmap = pixmap.scaled(300,300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(pixmap)

            # random prediction
            blood = random.choice(blood_groups)
            confidence = random.randint(90, 99)

            self.result.setText(f"Blood Group : {blood} ({confidence}%)")
            self.status.setText("Fingerprint Captured")

        else:
            self.status.setText("Scanner Error")