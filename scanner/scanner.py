import serial
import time

ser = serial.Serial('COM8', 57600, timeout=3)

cmd = bytes.fromhex("EF01FFFFFFFF0100071300000000001B")

ser.write(cmd)

time.sleep(1)

response = ser.read_all()

print("Response:", response.hex())

ser.close()