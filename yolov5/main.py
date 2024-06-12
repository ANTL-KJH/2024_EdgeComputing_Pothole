import threading
import time

import pothole_module.Pothole_GPS
import pothole_module.Pothole_Yolo
import pothole_module.Pothole_information

class Pothole_detector:
    def __init__(self):
        self.info=pothole_module.Pothole_information.information()
        self.GPS = pothole_module.Pothole_GPS.pothole_GPS()
        self.PotholeDetectorYOLO = pothole_module.Pothole_Yolo.PotholeDetector(self.info)


    def run(self):
        yolov5_thread = threading.Thread(target=self.PotholeDetectorYOLO.run)
        yolov5_thread.start()




def main():
    pth_detector = Pothole_detector()
    pth_detector.run()


if __name__ == "__main__":
    main()
