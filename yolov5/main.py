import threading
import pothole_module.Pothole_GPS
from pothole_module.Pothole_Yolo import run_yolov5


class Pothole_detector:
    def __init__(self):
        self.GPS = pothole_module.Pothole_GPS.pothole_GPS()



    def run(self):
        yolov5_thread = threading.Thread(target=run_yolov5)
        yolov5_thread.start()



def main():
    pth_detector = Pothole_detector()
    pth_detector.run()


if __name__ == "__main__":
    main()
