import numpy as np
import serial
import time
waitTime = 0.001

# generate the waveform table
signalLength = 4096
t = np.linspace(0, 2 * np.pi * (1.0 - 1.0 / signalLength), signalLength)
signalTable = (np.sin(t) + 1.0) / 2.0 * 0.5
# output formatter
formatter = lambda x: "%.5f" % x

# send the waveform table to K66F
serdev = '/dev/ttyACM0'
s = serial.Serial(serdev)
for data in signalTable:
    s.write(bytes(formatter(data), 'UTF-8'))
    time.sleep(waitTime)
s.close()