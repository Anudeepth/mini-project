import random
from pyfingerprint.pyfingerprint import PyFingerprint

blood_groups = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]

def scan_fingerprint():
    try:
        import serial.tools.list_ports
        ports = [port.device for port in serial.tools.list_ports.comports()]
        active_port = next((p for p in ports if 'USB' in p.upper() or 'ACM' in p.upper() or 'COM' in p.upper()), None)
        
        if not active_port:
            return None
            
        f = PyFingerprint(active_port, 57600)

        if not f.verifyPassword():
            return None

        while f.readImage() == False:
            pass

        f.convertImage(0x01)

        return random.choice(blood_groups)

    except:
        return None