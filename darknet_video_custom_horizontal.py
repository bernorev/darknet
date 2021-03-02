from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet

#from imutils.video import FileVideoStream
from imutils.video import FPS
from queue import Queue
from threading import Thread
stored_exception=None

from darknet import set_gpu
def video_capture(frame_queue, darknet_image_queue ,width,height):
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            ###
            ###
            vid_width = 1920

            net_width_ideal = int(height * 16 / 9)
            net_width = width #736

            #net_width_ratio = round(net_width/net_width_ideal,2)
            net_vid_width_ratio = net_width_ideal/vid_width

            center_line = ((vid_width*net_vid_width_ratio)/2)

            startPixel = round(center_line-(net_width/2))
            endPixel = round(center_line+(net_width/2))

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (net_width_ideal, height),interpolation=cv2.INTER_CUBIC)    

            frame_resized = frame_resized[:,startPixel:endPixel]
            #colour correction
            lab = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2LAB)
            lab_planes = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0,tileGridSize=(16,16))
            lab_planes[0] = clahe.apply(lab_planes[0])
            lab = cv2.merge(lab_planes)
            frame_resized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            frame_queue.put(frame_resized)
        else :
            print("readerror")
            #break
    cap.release()

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax



netMain = None
metaMain = None
altNames = None
set_gpu(0)

def YOLO():

    global metaMain, netMain, altNames, cap, darknet_image

    #configPath = "./cfg/yolov4-fruit.cfg"
    #weightPath = "./backup/yolov4-fruit_last.weights"
    #metaPath = "./data/fruit.data"

    #configPath = "./cfg/yolov4-tiny_fruit.cfg"
    #weightPath = "./backup/yolov4-tiny_fruit_15000.weights"
    #metaPath = "./data/fruit.data"

    #configPath = "./cfg/fruit-tiny-3l.cfg"
    #weightPath = "./backup/fruit-tiny-3l_last.weights"
    #metaPath = "./data/fruit.data"
##### GRAPES
    configPath = "./cfg/grapes-tiny.cfg"
    weightPath = "./backup/grapes-tiny_last.weights"
    metaPath = "./data/fruit.data"


    
    
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")



    darknet_image_queue = Queue(maxsize=1)
    frame_queue = Queue(maxsize=3)

    network, class_names, class_colors = darknet.load_network(
        configPath,
        metaPath,
        weightPath,
        batch_size=1
    )
    width = darknet.network_width(network)
    height = darknet.network_height(network)

    
    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(width, height, 3)  


    cap = cv2.VideoCapture("/media/berno/Transcend/Fruit_counting/tafeldruiwe_12_feb_2021/1_01_L.MP4")
    #cap = cv2.VideoCapture("/media/berno/sata_disk/gopro2/100GOPRO/GX010103.MP4")

    Thread(target=video_capture, args=(frame_queue, darknet_image_queue,width,height)).start()
    #cap = cv2.VideoCapture(0)
    #cap = FileVideoStream("F:/FruitCounting_Videos/tweefontein_videos/T10_1_left.MP4").start()
    #cap = cv2.VideoCapture("/media/berno/sata_disk/FruitCounting Videos/goedgegun_videos/green_apples_left_1.MP4")

    #cap = FileVideoStream("T06_1_right.MP4").start()

    #cap = FileVideoStream("/media/berno/TOSHIBA EXT/FruitCounting_Videos/tweefontein_videos/T10_1_left.MP4").start()
    #cap = FileVideoStream("/home/berno/Videos/MS93_left_2.mp4").start()
    time.sleep(1.0)

    #cap.set(3, 1280)
    #cap.set(4, 1024)
    out = cv2.VideoWriter(
        "output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 60.0,
        (1080, 576))
    
    print("Starting the YOLO loop...")
    print(class_colors)
    key = ''
    class_colors = {'fruit': (221, 0, 105)}
    while key != 113:
        prev_time = time.time()

        frame_read = frame_queue.get()

        darknet.copy_image_from_bytes(darknet_image, frame_read.tobytes())

        detections = darknet.detect_image(network, class_names, darknet_image, thresh=0.2)

        image = darknet.draw_boxes(detections, frame_read, class_colors)
        #print(detections[1])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #cv2.putText(image, "Queue Size: {}".format(cap.Q.qsize()),
        #            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        #print(1/(time.time()-prev_time))
        image = cv2.resize(image, (896,512),interpolation=cv2.INTER_LINEAR) 
        cv2.imshow('Demo', image)
        key = cv2.waitKey(1)
        #out.write(image)
        print(1/(time.time()-prev_time))
    cv2.destroyAllWindows()
    cap.release()
    out.release()

if __name__ == "__main__":
    YOLO()
