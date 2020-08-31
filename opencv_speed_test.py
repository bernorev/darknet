import cv2
import time

cap = cv2.VideoCapture("MS88_left_3.mp4")


while(cap.isOpened()):
    ret, frame_read = cap.read()
    
    if ret == True:
        prev_time = time.time()
        frame_read = cv2.rotate(frame_read, cv2.ROTATE_90_COUNTERCLOCKWISE)

        cv2.imshow('Demo', frame_read)
        cv2.waitKey(1)
        print("FPS : " , str(int(1/(time.time()-prev_time))))