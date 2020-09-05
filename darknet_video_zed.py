from ctypes import *
import math
import random
from random import randint

import os
import sys
import logging

import cv2
import numpy as np
import time
import darknet

#from imutils.video import FileVideoStream
from imutils.video import FPS
from queue import Queue
from threading import Thread
stored_exception=None

import statistics
import getopt
import pyzed.sl as sl

from darknet import set_gpu
from darknet import bbox2points

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def draw_boxes(detections, image, colors):
    import cv2
    for label, confidence, bbox in detections:
        left, top, right, bottom = bbox2points(bbox)

        x = int((left + right)/2)
        y = int((top + bottom)/2)
        #print(x)
        #print(y)

        depth_val = depth.get_value(y, x)

        distance = depth_val

        print(f'width {left-right} height {bottom-top} distance {distance}')

        #angle = 2*

        cv2.rectangle(image, (left, top), (right, bottom), colors[label], 1)
        cv2.putText(image, "{} [{:.2f}] {}".format(label, float(confidence),distance),
                    (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colors[label], 2)
        
        '''
        cv2.rectangle(image, (left, top), (right, bottom), colors[label], 1)
        cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
                    (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colors[label], 2)
        '''
    return image

depth = sl.Mat()
mirror_ref = sl.Transform()
mirror_ref.set_translation(sl.Translation(2.75,4.0,0))
tr_np = mirror_ref.m
def video_capture(frame_queue, darknet_image_queue ,width,height):

    # Create a Camera object
    zed = sl.Camera()
    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Use PERFORMANCE depth mode
    init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD720 video mode
    init_params.camera_fps = 30   # Set fps at 15
    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)
    # Capture 50 frames and stop

    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.sensing_mode = sl.SENSING_MODE.FILL# Use STANDARD sensing mode
    # Setting the depth confidence parameters
    runtime_parameters.confidence_threshold = 100
    runtime_parameters.textureness_confidence_threshold = 100

    image = sl.Mat()
    depth_for_display = sl.Mat()

    while True:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            ###
            #frame = frame[400:680,:]
            ###

            zed.retrieve_image(image, sl.VIEW.LEFT)
            image_data = image.get_data()

            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

            zed.retrieve_image(depth_for_display, sl.VIEW.DEPTH)
            depth_image = depth_for_display.get_data()
            cv2.imshow('Depth', depth_image)

     
            
            #cv2.imshow('Demo1', frame_resized)
            #frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE) 
            frame_rgb = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (width, height),
                                    interpolation=cv2.INTER_LINEAR)
            frame_queue.put(frame_resized)
            darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
            darknet_image_queue.put(darknet_image)
            #print(darknet_image_queue.qsize())
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
    
    #configPath = "./cfg/yolov3_tiny_3l_fruit_counting_default_anchors.cfg"
    #weightPath = "./backup/yolov3_tiny_3l_fruit_counting_default_anchors_last.weights"
    #metaPath = "./data/fruit.data"

    #configPath = "./cfg/yolov4_fruit.cfg"
    #weightPath = "./models/yolov4_fruit_last.weights"
    #metaPath = "./data/fruit.data"

    #configPath = "./cfg/yolov3_tiny_pan_fruit.cfg"
    #weightPath = "./backup/yolov3_tiny_pan_fruit_last.weights"
    #metaPath = "./data/fruit.data"

    configPath = "./cfg/yolov4-tiny_fruit.cfg"
    weightPath = "./backup/yolov4-tiny_fruit_last.weights"
    metaPath = "./data/fruit.data"

    #configPath = "./cfg/yolov4-tiny.cfg"
    #weightPath = "./models/yolov4-tiny.weights"
    #metaPath = "./cfg/coco.data"
    

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

    darknet_image_queue = Queue(maxsize=2)
    frame_queue = Queue(maxsize=2)

    Thread(target=video_capture, args=(frame_queue, darknet_image_queue,width,height)).start()


    time.sleep(1.0)

    #cap.set(3, 1280)
    #cap.set(4, 1024)
    out = cv2.VideoWriter(
        "output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 30.0,
        (width, height))
    
    print("Starting the YOLO loop...")

    key=''
    
    while key != 113:#q
        prev_time = time.time()
        
        frame_read = frame_queue.get()
        detections = darknet.detect_image(network, class_names, darknet_image_queue.get(), thresh=0.2   )
        image = draw_boxes(detections, frame_read,class_colors)
    #   print(detections[1])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



        depth_image=0
        #cv2.putText(image, "Queue Size: {}".format(cap.Q.qsize()),
        #            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        print(1/(time.time()-prev_time))
        cv2.imshow('Demo', image)

        key = cv2.waitKey(5)
        out.write(image)
    
        # Close the camera
    zed.close()
    #cap.release()
    out.release()

if __name__ == "__main__":
    YOLO()
