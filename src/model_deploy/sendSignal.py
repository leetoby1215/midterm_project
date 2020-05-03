import numpy as np
import serial
import time
waitTime = 0.001

signalTable = []
signalTable.append(20501)

# send the waveform table to K66F
serdev = '/dev/ttyACM0'
s = serial.Serial(serdev)
for data in signalTable:
    s.write(bytes(str(data), 'UTF-8'))
    time.sleep(waitTime)
s.close()