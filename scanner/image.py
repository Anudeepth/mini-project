from pyfingerprint.pyfingerprint import PyFingerprint
from PIL import Image
import numpy as np
import time

try:
    
    # Connect to fingerprint sensor
    f = PyFingerprint('COM8', 57600, 0xFFFFFFFF, 0x00000000)

    if (f.verifyPassword() == False):
        raise ValueError('Wrong sensor password')

    print('Fingerprint sensor connected')
    print('Waiting for finger...')

    # Wait until finger is placed
    while (f.readImage() == False):
        pass

    print('Fingerprint captured')

    # Download image from sensor
    f.downloadImage('C:/fingerprint/fingerprint.bmp')

    print('Fingerprint image saved as fingerprint.bmp')

except Exception as e:
    print('Operation failed!')
    print('Error:', str(e))