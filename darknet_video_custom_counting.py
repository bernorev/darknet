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
from sort import *

def video_capture(frame_queue, darknet_image_queue ,width,height):
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            ###
            vid_width = 1080

            net_width_ideal = height * 9 / 16
            net_width = width #736

            #net_width_ratio = round(net_width/net_width_ideal,2)
            net_vid_width_ratio = net_width_ideal/vid_width

            center_line = ((vid_width*net_vid_width_ratio)/2)

            startPixel = round(center_line-(net_width/2))
            endPixel = round(center_line+(net_width/2))
            frame = frame[startPixel:endPixel,:]
            #frame = frame[400:680,:]
            ###
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE) 
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (width, height),interpolation=cv2.INTER_LINEAR)    
            #colour correction
            lab = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2LAB)
            lab_planes = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
            lab_planes[0] = clahe.apply(lab_planes[0])
            lab = cv2.merge(lab_planes)
            frame_resized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            frame_queue.put(frame_resized)
            #darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
            #darknet_image_queue.put(darknet_image)
            print("Frame queue size : " + str(frame_queue.qsize()))
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


# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
	return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
Sort()

from darknet import set_gpu

set_gpu(0)

def YOLO():

    COLORS = np.random.randint(0, 255, size=(200, 3),
        dtype="uint8")
    #ct = CentroidTracker(maxDisappeared=0, maxDistance=200)
    ct = Sort(max_age=2)
    trackers = []
    trackableObjects = {}
    memory = {}
    counter = 0

    global metaMain, netMain, altNames, cap, darknet_image

    configPath = "./cfg/yolov4-fruit.cfg"
    weightPath = "./backup/yolov4-fruit_last.weights"
    metaPath = "./data/fruit.data"

    #configPath = "./cfg/yolov4-tiny_fruit.cfg"
    #weightPath = "./backup/yolov4-tiny_fruit_last.weights"
    #metaPath = "./data/fruit.data"

    #configPath = "./cfg/yolov4-tiny-3l_fruit.cfg"
    #weightPath = "./backup/yolov4-tiny-3l_fruit_last.weights"
    #metaPath = "./data/fruit.data"
    
    
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

    #cap = cv2.VideoCapture("boontjieskloof_l_8.MP4")
    #cap = cv2.VideoCapture("E:/FruitCounting_Videos/tweefontein_videos/T10_1_left.MP4")
    cap = cv2.VideoCapture("videos/GH010053.MP4")

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
        "output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 120.0,
        (width, height))
    
    print("Starting the YOLO loop...")
    W = None
    H = None
    tracks = ct.update([])
    while True:
        prev_time = time.time()

        frame_read = frame_queue.get()
        darknet.copy_image_from_bytes(darknet_image, frame_read.tobytes())
        #
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=0.03,nms=0.6)
        
        
        #print(detections[1])
        image = frame_read
        image = darknet.draw_boxes(detections, image, class_colors)
        if W is None or H is None:
            (H, W) = image.shape[:2]
        line = [(W // 2,0), (W // 2,H)]
        rects = []
        for detection in detections:
          #  if (detection[2][2] * detection[2][3]) > 100 : 
                x, y, w, h = detection[2][0],\
                    detection[2][1],\
                    detection[2][2],\
                    detection[2][3]
                xmin, ymin, xmax, ymax = convertBack(
                    float(x), float(y), float(w), float(h))
                pt1 = (xmin, ymin)
                pt2 = (xmax, ymax)
                #cv2.rectangle(image, pt1, pt2, (0, 255, 0), 1)
                rects.append(convertBack(float(x), float(y), float(w), float(h)))


        #rects = []
        ###FOR SORT
        rects = np.asarray(rects)
        ####
        #print(rects)
        if len(rects) > 0:
            tracks = ct.update(rects)

        boxes = []
        indexIDs = []
        c = []
        previous = memory.copy()
        memory = {}
        for track in tracks:
            boxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track[4]))
            memory[indexIDs[-1]] = boxes[-1]
        if len(boxes) > 0:
            i = int(0)
            for box in boxes:
                # extract the bounding box coordinates
                (x, y) = (int(box[0]), int(box[1]))
                (w, h) = (int(box[2]), int(box[3]))
                # draw a bounding box rectangle and label on the image
                # color = [int(c) for c in COLORS[classIDs[i]]]
                # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
                cv2.rectangle(image, (x, y), (w, h), color, 2)
                if indexIDs[i] in previous:
                    previous_box = previous[indexIDs[i]]
                    (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                    (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                    p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                    p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
                    cv2.line(image, p0, p1, color, 3)
                    if intersect(p0, p1, line[0], line[1]):
                        counter += 1
                # text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                #text = "{}".format(indexIDs[i])
                #cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                i += 1


        # draw line
        cv2.line(image, line[0], line[1], (255, 0, 0), 3)

        # draw counter
        cv2.putText(image, str(counter), ((50),200), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 0, 0), 3)       


        #cv2.putText(image, "Queue Size: {}".format(cap.Q.qsize()),
        #            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   
        
        cv2.imshow('Demo', image)
        k = cv2.waitKey(1)
        if k == 27:
            break
        #out.write(image)
        print(1/(time.time()-prev_time))
    cap.release()
    out.release()

if __name__ == "__main__":
    YOLO()
