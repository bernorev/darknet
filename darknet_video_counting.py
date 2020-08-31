from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
import imutils
#import dlib
from sort import *
from datetime import datetime
import csv


#from imutils.video import FileVideoStream
from imutils.video import FPS
from queue import Queue
from threading import Thread
stored_exception=None

class FileVideoStream:
    def __init__(self, path, queue_size=100):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(path)
        self.stopped = False

        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queue_size)
        # intialize thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True

    def start(self):
        # start a thread to read frames from the file video stream
        self.thread.start()
        return self

    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                break

            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file
                (grabbed, frame) = self.stream.read()

                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                #if not grabbed:
                #    self.stopped = True
                    

                if grabbed:

                    #frame = frame[400:680,: ]
                    
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE) 
                    #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                    lab_planes = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
                    lab_planes[0] = clahe.apply(lab_planes[0])
                    lab = cv2.merge(lab_planes)
                    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                    
                    #frame = frame[:,380:700 ]
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    
                    self.Q.put(frame)
                else:
                    print("readerror")
                
                
            else:
                time.sleep(0.1)  # Rest for 10ms, we have a full queue

        self.stream.release()

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    # Insufficient to have consumer use while(more()) which does
    # not take into account if the producer has reached end of
    # file stream.
    def running(self):
        return self.more() or not self.stopped

    def more(self):
        # return True if there are still frames in the queue. If stream is not stopped, try to wait a moment
        tries = 0
        while self.Q.qsize() == 0 and not self.stopped and tries < 5:
            time.sleep(0.1)
            tries += 1

        return self.Q.qsize() > 0

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        # wait until stream resources are released (producer thread might be still grabbing frame)
        self.thread.join()  

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
    timeElapsed = 0.0
    startTime = datetime.now()
    workdir = os.getcwd()
    memory = {}
    counter = 0
    # initialize a list of colors to represent each possible class label
    np.random.seed(42)


    COLORS = np.random.randint(0, 255, size=(200, 3),
        dtype="uint8")
    #ct = CentroidTracker(maxDisappeared=0, maxDistance=200)
    ct = Sort(max_age=3)
    trackers = []
    trackableObjects = {}

    # initialize the total number of frames processed thus far, along
    # with the total number of objects that have moved either up or down
    totalFrames = 0         
    totalDown = 0
    totalUp = 0

    global metaMain, netMain, altNames

    #configPath = "./cfg/yolov3_tiny_3l_fruit_counting_default_anchors.cfg"
    #weightPath = "./backup/yolov3_tiny_3l_fruit_counting_default_anchors_last.weights"
    #metaPath = "./data/fruit.data"

    #configPath = "./cfg/yolov3-tiny-prn_fruitcounting.cfg"
    #weightPath = "./backup/yolov3-tiny-prn_fruitcounting_last.weights"
    #metaPath = "./data/fruit.data"
    
    #configPath = "./cfg/yolov4_fruit.cfg"
    #weightPath = "./models/yolov4_fruit_last.weights"
    #metaPath = "./data/fruit.data"

    configPath = "./cfg/yolov4-tiny_fruit.cfg"
    weightPath = "./backup/yolov4-tiny_fruit_last.weights"
    metaPath = "./data/fruit.data"

    #configPath = "./cfg/yolov3_tiny_pan_fruit.cfg"
    #weightPath = "./backup/yolov3_tiny_pan_fruit_last.weights"
    #metaPath = "./data/fruit.data"

    #configPath = "./cfg/yolov4-tiny_3lSPP.txt"
    #weightPath = "./backup/yolov4-tiny_3lSPP_last.weights"
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
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    #cap = cv2.VideoCapture(0)
    W = None
    H = None


    #cap = cv2.VideoCapture("/media/berno/sata_disk/FruitCounting Videos/goedgegun_videos/green_apples_left_1.MP4")
    #cap = FileVideoStream("T06_1_right.MP4").start()
    cap = FileVideoStream("E:/FruitCounting_Videos/tweefontein_videos/T10_1_left.MP4").start()
    #cap = FileVideoStream("/home/berno/Videos/MS93_left_2.mp4").start()

    #cap = FileVideoStream(0).start()
    time.sleep(1.0)
    #cap = cv2.VideoCapture(os.path.join(workdir, "videos", "VID_20191125_141350.mp4"))
    #cap = cv2.VideoCapture(os.path.join(workdir, "videos", "GX020083.MP4"))
    #cap = cv2.VideoCapture("./videos/971.mov")
    #cap = cv2.VideoCapture("/media/berno/New Volume/Glen Elgin Videos/MS93_left_1.MP4")
    #cap.set(3, 1280)
    #cap.set(4, 1024)
    #out = cv2.VideoWriter(
    #    "output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 30.0,
    #    (darknet.network_width(netMain), darknet.network_height(netMain)))
    out = cv2.VideoWriter(
        "output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 120.0,
        (darknet.network_width(netMain), darknet.network_height(netMain)))


    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)
    totalFrames=1

    i = 0
    video_end_frames = 0
    while True:
        totalFrames += 1
        #if totalFrames == 200 :
        #    break
        prev_time = time.time()
        frame_read = cap.read()
        

        if True :
            #frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
            ####IF THE THING IS SIDEWAYS from phone
            #frame_rgb = imutils.rotate_bound(frame_rgb, -90)
            ####
            #frame_rgb = frame_rgb[:, 380:700]
            frame_resized = cv2.resize(frame_read,
                                       (darknet.network_width(netMain),
                                        darknet.network_height(netMain)),
                                       interpolation=cv2.INTER_LINEAR)






            if W is None or H is None:
              (H, W) = frame_resized.shape[:2]

            #cv2.line(image, (W // 2,0), (W // 2,H), (255, 0, 0), 2)
            #line = [(W*3 // 4,0), (W * 3// 4,H)]
            line = [(W // 2,0), (W // 2,H)]

            darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

            #if (totalFrames % 1 == 0) or (totalFrames == 1):
            detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.1,nms=0.55)

            #image = cvDrawBoxes(detections, frame_resized)
            image = frame_resized
            rects = []
            for detection in detections:
                if (detection[2][2] * detection[2][3]) > 100 : 
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
            if detections is not None:
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
                        #cv2.line(image, p0, p1, color, 3)

                        if intersect(p0, p1, line[0], line[1]):
                            counter += 1

                    # text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                    #text = "{}".format(indexIDs[i])
                    #cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    i += 1

            # draw line
            #cv2.line(image, line[0], line[1], (255, 0, 0), 3)

            # draw counter
            cv2.putText(image, str(counter), (300,200), cv2.FONT_HERSHEY_DUPLEX, 4, (255, 0, 0), 4)
            # counter += 1
            elapsed_time = (datetime.now() - startTime).total_seconds()
            #cv2.putText(image, str(elapsed_time), (0,100), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 0, 0), 4)
            #timeElapsed = (datetime.now() - startTime).total_seconds()
            #cv2.putText(image, str(totalFrames), (0,300), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 0, 0), 4)
            #print(round(elapsed_time % 5))
            #cv2.putText(image, "Queue Size: {}".format(cap.Q.qsize()),
            #    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)    
            # construct a tuple of information we will be displaying on the
            # frame

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            print("FPS : " , str(int(1/(time.time()-prev_time))))
            cv2.imshow('Demo', image)
            cv2.waitKey(1)
            out.write(image)
            video_end_frames = 0

    cap.release()
    out.release()


if __name__ == "__main__":
    YOLO()
