import cv2
import pyyolo
import time
def main():
    detector = pyyolo.YOLO("./cfg/yolov4-tiny-3l_fruit.cfg",
                           "./backup/yolov4-tiny-3l_fruit_last.weights",
                           "./data/fruit.data",
                           detection_threshold = 0.05,
                           hier_threshold = 0.5,
                           nms_threshold = 0.45)

    cap = cv2.VideoCapture("./videos/T06_1_right.MP4")
    while True:
        prev_time = time.time()
        ret, frame = cap.read()
        if ret:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            dets = detector.detect(frame, rgb=False)
            for i, det in enumerate(dets):
                #print(f'Detection: {i}, {det}')
                xmin, ymin, xmax, ymax = det.to_xyxy()
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255))
            frame = cv2.resize(frame, (576, 1080),interpolation=cv2.INTER_LINEAR)    
            cv2.imshow('cvwindow', frame)
            print(1/(time.time()-prev_time))
            if cv2.waitKey(1) == 27:
                break

if __name__ == '__main__':
    main()