import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import cv2
import time
import ctypes
from dector_trt import Detector
import pycuda.autoinit


def detect(engine_file_path):
    detector = Detector(engine_file_path)
    capture = cv2.VideoCapture(0)
    # capture = cv2.VideoCapture(0)
    fps = 0.0
    while True:
        ret, img = capture.read()
        if img is None:
            print('No image input!')
            break

        t1 = time.time()
        result_img = detector.detect(img)

        fps = (fps + (1. / (time.time() - t1))) / 2
        cv2.putText(result_img, 'FPS: {:.2f}'.format(fps), (50, 30), 0, 1, (0, 255, 0), 2)
        cv2.putText(result_img, 'Time: {:.3f}'.format(time.time() - t1), (50, 60), 0, 1, (0, 255, 0), 2)
        if ret == True:
            cv2.imshow('frame', result_img)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        else:
            break

    capture.release()
    cv2.destroyAllWindows()
    detector.destroy()


if __name__ == '__main__':

    PLUGIN_LIBRARY = "weights/libmyplugins.so"
    ctypes.CDLL(PLUGIN_LIBRARY)
    engine_file_path = 'weights/yolov5s.engine'
    detect(engine_file_path)