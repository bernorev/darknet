from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
import imutils
from sort import *
from datetime import datetime, timedelta
import time
import csv
import math
import pandas as pd
import argparse
from darknet import set_gpu



#from imutils.video import FileVideoStream
from imutils.video import FPS
from queue import Queue
from threading import Thread
stored_exception=None

class FileVideoStream:
    def __init__(self, path, queue_size=200):
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
                    frame = frame[380:700,: ]
                    
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
                    #quit()
                
                
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

ap = argparse.ArgumentParser()

ap.add_argument("-i", "--input", type=str,required=True,help="path to video file")
ap.add_argument("-gps", "--gps", type=str,required=True,help="path to gps file")
ap.add_argument("-gpu", "--gpu", type=int,required=True,help="GPU to use")
args = vars(ap.parse_args())

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

def YOLO():
    workdir = os.getcwd()


    timeElapsed = 0.0
    startTime = datetime.now()

    memory = {}
    counter = 0
    # initialize a list of colors to represent each possible class label
    np.random.seed(42)


    COLORS = np.random.randint(0, 255, size=(200, 3),
        dtype="uint8")
    #ct = CentroidTracker(maxDisappeared=0, maxDistance=200)
    ct = Sort()
    trackers = []
    trackableObjects = {}

    # initialize the total number of frames processed thus far, along
    # with the total number of objects that have moved either up or down
    totalFrames = 0
    totalDown = 0
    totalUp = 0

    global metaMain, netMain, altNames

    #configPath = "./cfg/yolov3-spp-fruit-counting.cfg"
    #weightPath = "./backup/yolov3-spp-fruit-counting_last.weights"
    #metaPath = "./data/fruit.data"

    #configPath = "./cfg/yolov3_tiny_pan_fruit_counting.cfg"
    #weightPath = "./backup/yolov3_tiny_pan_fruit_counting_last.weights"
    #metaPath = "./data/fruit.data"

    configPath = "./cfg/yolov3_tiny_pan_fruit.cfg"
    weightPath = "./backup/yolov3_tiny_pan_fruit_last.weights"
    metaPath = "./data/fruit.data"

    #configPath = "./cfg/yolov3-tiny_3l_fruit_cropped.cfg"
    #weightPath = "./backup/yolov3-tiny_3l_fruit_last.weights"
    #metaPath = "./data/fruit.data"

    #configPath = "./cfg/yolov3-tiny-prn-fruit_cropped.cfg"
    #weightPath = "./backup/yolov3-tiny-prn-fruit_last.weights"
    #metaPath = "./data/fruit.data"

    #configPath = "./cfg/yolo_v3_tiny_pan3_fruit_cropped.cfg"
    #weightPath = "./backup/yolo_v3_tiny_pan3_fruit_cropped_last.weights"
    #metaPath = "./data/fruit.data"

    #configPath = "./cfg/yolo_v3_tiny_pan3_fruit_cropped.cfg"
    #weightPath = "./backup/yolo_v3_tiny_pan3_fruit_last.weights"
    #metaPath = "./data/fruit.data"

    #configPath = "./cfg/yolo_v3_tiny_pan3_fruit_fullres.cfg"
    #weightPath = "./backup/yolo_v3_tiny_pan3_fruit_last.weights"
    #metaPath = "./data/fruit.data"

    #configPath = "./cfg/xyolo_cropped.cfg"
    #weightPath = "./backup/xyolo_last.weights"
    #metaPath = "./data/fruit.data"
    set_gpu(args["gpu"])

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
    #cap = cv2.VideoCapture("/media/berno/New Volume/jakkalriver Videos/32_left_1.MP4")
    print("[INFO] starting video file thread...")
    cap = FileVideoStream(args["input"]).start()
    


    cap2 = cv2.VideoCapture(args["input"])

    print(args["input"])
    print("Total Frame")
    total_vid_frames = cap2.get(cv2.CAP_PROP_FRAME_COUNT)
    print(total_vid_frames)
    #cap = cv2.VideoCapture(os.path.join(workdir, "videos", "GX030093.MP4"))
    #cap.set(3, 1280)
    #cap.set(4, 1024)
    
    result_video_name = args["input"].rsplit('.')[-2].rsplit('/',1)[-1] + "_counting.avi"
    #out = cv2.VideoWriter(
    #    os.path.join(workdir, "results", result_video_name), cv2.VideoWriter_fourcc(*"MJPG"), 120.0,
    #    (darknet.network_width(netMain), darknet.network_height(netMain)))
    
    
    print("Starting the YOLO loop...")
    print()
    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)
    totalFrames=1

    print(args["input"])
    counting_file_name = args["input"].rsplit('.')[-2].rsplit('/',1)[-1] + "_counting.csv"
    #counting_file_name = "32_left_1_counts.csv"
    counting_file = open(os.path.join(workdir, "results", counting_file_name), mode='w', newline='')
    counting_writer = csv.writer(counting_file, delimiter=',')
    counting_writer.writerow(["elapsed_time", "counter","section_count",'longitude','latitude'])
    counting_file.close()
    #counting_file = open('counting.csv','w')
    i = 0
    prev_time = 0
    startTime =time.time()
    elapsed_time = time.time()
    frames_count = 0
    section_count = 0
    video_end_frames = 0
    while True:
        #totalFrames += 1
        #frames_count += 1

        fps_time = time.time()
        frame_read = cap.read()

        if True:

            #frame_rgb = frame_rgb[:, 400:600]
            frame_resized = cv2.resize(frame_read,
                                       (darknet.network_width(netMain),
                                        darknet.network_height(netMain)),
                                       interpolation=cv2.INTER_LINEAR)



            totalFrames += 1
            frames_count += 1


            if W is None or H is None:
              (H, W) = frame_resized.shape[:2]

            #cv2.line(image, (W // 2,0), (W // 2,H), (255, 0, 0), 2)

            #line = [(W*3 // 4,0), (W * 3// 4,H)]
            line = [(W // 2,0), (W // 2,H)]
            
            #print("Do detection",time.time())


            image = frame_resized
            rects = []

            darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())
            detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.05,nms=.55)


            #image = cvDrawBoxes(detections, frame_resized)
            #print("Go over detections",time.time())
            for detection in detections:
                if (detection[2][2] * detection[2][3]) > 50 : 
                    x, y, w, h = detection[2][0],\
                        detection[2][1],\
                        detection[2][2],\
                        detection[2][3]
                    xmin, ymin, xmax, ymax = convertBack(
                        int(x), int(y), int(w), int(h))
                    pt1 = (xmin, ymin)
                    pt2 = (xmax, ymax)
                    #cv2.rectangle(image, pt1, pt2, (0, 255, 0), 1)
                    rects.append(convertBack(int(x), int(y), int(w), int(h)))
                    
            #print("start sort",time.time())


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
                        cv2.line(image, p0, p1, color, 3)

                        if intersect(p0, p1, line[0], line[1]):
                            counter += 1
                            section_count += 1

                    #text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                    #text = "{}".format(indexIDs[i])
                    #cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    i += 1
            #print("draw frame stuff",time.time())
            ###### draw line
            cv2.line(image, line[0], line[1], (255, 0, 0), 3)
#
            ####### draw counter
            cv2.putText(image, str(counter), (0,50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 0), 4)
#
#
            cv2.putText(image, str(elapsed_time), (0,150), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 0), 4)
            #timeElapsed = (datetime.now() - startTime).total_seconds()
            #cv2.putText(image, str(totalFrames), (0,300), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 0, 0), 4)
            #print(round(elapsed_time % 5))
            cv2.putText(image, str(frames_count), (0,200), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 0), 4)
            cv2.putText(image, str(section_count), (0,250), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 0), 4)
            cv2.putText(image, str(frames_count % 120), (0,300), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 0), 4)

            #gps_data = pd.read_csv(os.path.join(workdir, "videos", "GX030093_1hz.csv"))
            #gps_data = pd.read_csv("/media/berno/New Volume/jakkalriver Videos/32_1.csv")
            gps_data = pd.read_csv(args["gps"])
            #print(args["input"])


            #print("write csv",time.time())
            if (frames_count % 120) == 0  :

                with open(os.path.join(workdir, "results", counting_file_name), mode='a', newline='') as counting_file:
                    counting_writer = csv.writer(counting_file, delimiter=',')
                    counting_writer.writerow([elapsed_time, counter,section_count,gps_data.loc[frames_count/120-1,'longitude'],gps_data.loc[frames_count/120-1,'latitude']])
                    #counting_writer.writerow([elapsed_time, counter,section_count,gps_data.loc[frames_count/120,'X'],gps_data.loc[frames_count/120,'Y']])

                print("writing points data")
                section_count = 0

                #prev_time = prev_time + 4


            elapsed_time = (time.time() - startTime)


            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #cv2.imshow('Demo', image)
            #cv2.waitKey(1)
            
            
            #if k==27:    # Esc key to stop
            #    break


            #out.write(image)
            #video_end_frames = 0
            print(1/(time.time()-fps_time))
            print("Progress : " + str(int(frames_count*100/total_vid_frames)) + "%" + " | Total Count : " + str(counter) + "| Section Count : " + str(section_count) )

    cap.release()
    out.release()
    counting_file.close()

if __name__ == "__main__":
    YOLO()
