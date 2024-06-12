import time

import serial
import pynmea2


class pothole_GPS:
    def __init__(self):
        self.latitude = 35.830650
        self.logitude = 128.755062

    def read_gps_data(self):
        while True:
            # 시리얼 포트 설정 (적절한 포트와 보드레이트로 설정)
            port = '/dev/ttyTHS1'  # Jetson Nano의 UART 포트
            baudrate = 9600  # 일반적으로 Neo-7M의 보드레이트는 9600

            # 시리얼 포트 열기
            ser = serial.Serial(port, baudrate, timeout=1)

            try:
                while True:
                    line = ser.readline().decode('ascii', errors='replace')
                    if line.startswith('$GNGGA') or line.startswith('$GPGGA'):
                        msg = pynmea2.parse(line)
                        self.latitude = msg.latitude
                        self.logitude = msg.longitude
            except Exception as e:
                print("System::Can't get GPS pos")
            ser.close()
            time.sleep(1)


