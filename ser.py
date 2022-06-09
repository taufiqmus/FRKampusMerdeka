import serial
import time

id_list = ['abu']

def check(id):
    if id in id_list:
        ser.write(b'1')
        time.sleep(3)
        ser.write(b'0')


with serial.Serial('COM3', 9600, timeout=1) as ser:
    while True:
        id = input()
        check(id)

