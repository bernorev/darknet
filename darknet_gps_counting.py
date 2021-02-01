from ctypes import *
import math
import random
import os
import sys
import cv2
import numpy as np
import time
import darknet
import imutils
import csv
import pandas as pd
import argparse
from darknet import set_gpu
#from imutils.video import FileVideoStream
from imutils.video import FPS
from queue import Queue
from threading import Thread
stored_exception=None
from sort import *
from gpx_to_csv import Converter
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
            #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) 
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (width, height),interpolation=cv2.INTER_CUBIC)    
            #colour correction
            lab = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2LAB)
            lab_planes = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8))
            lab_planes[0] = clahe.apply(lab_planes[0])
            lab = cv2.merge(lab_planes)
            frame_resized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            frame_queue.put(frame_resized)
            #darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
            #darknet_image_queue.put(darknet_image)
            #print("Frame queue size : " + str(frame_queue.qsize()))
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

ap = argparse.ArgumentParser()

ap.add_argument("-i", "--input", type=str,required=True,help="path to video file")
#ap.add_argument("-gps", "--gps", type=str,required=True,help="path to gps file")
ap.add_argument("-gpu", "--gpu",default=0 ,type=int,required=False,help="GPU to use")
args = vars(ap.parse_args())


set_gpu(args["gpu"])

workdir = os.getcwd()
#Extract GPS coordinates
#gopro2gpx -s -vvv T06_1_right.MP4 T06


#gps extract string for gopro2gpx 
#videos/GH011086.MP4

######EXTRACT GPS DATA
extract_str = f'gopro2gpx -vvv {args["input"]} {args["input"][:-4]}'
os.system(extract_str)

#convert GPX to csv
if Converter(input_file=f'{args["input"][:-4]}.gpx').gpx_to_csv(output_file=f'{args["input"][:-4]}.csv') == False:
    sys.exit("GPS file conversion failed")
#gps file to use


gps_csv_file = f'{args["input"][:-4]}.csv'

gps_csv_file = gps_csv_file.replace("_R","_L")
#gps_csv_file = gps_csv_file.replace("_L","")
print(f'GPS file = ' + gps_csv_file)

time.sleep(1)
#gps_csv_file = 'videos/SRblock2_1_L.csv'


def YOLO():
    timeElapsed = 0.0
    set_gpu(args["gpu"])

    COLORS = np.random.randint(0, 255, size=(200, 3),
        dtype="uint8")
    #ct = CentroidTracker(maxDisappeared=0, maxDistance=200)
    ct = Sort(max_age=1, min_hits=1, iou_threshold=0.01)
    trackers = []
    memory = {}
    counter = 0

    global metaMain, netMain, altNames, cap, darknet_image

    #configPath = "./cfg/yolov4-fruit.cfg"
    #weightPath = "./backup/yolov4-fruit_last.weights"
    #metaPath = "./data/fruit.data"

    configPath = "./cfg/yolov4-tiny_fruit.cfg"
    weightPath = "./backup/yolov4-tiny_fruit_last.weights"
    metaPath = "./data/fruit.data"

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

    #set input video and get video metadata
    cap = cv2.VideoCapture(args["input"])
    print(args["input"])
    print("Total Frame")
    total_vid_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(total_vid_frames)
    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    print(vid_fps)

    result_video_name = args["input"].rsplit('.')[-2].rsplit('/',1)[-1] + "_counting.avi"



    Thread(target=video_capture, args=(frame_queue, darknet_image_queue,width,height)).start()



    totalFrames=1
    #set counting output file
    print(args["input"])
    counting_file_name = args["input"].rsplit('.')[-2].rsplit('/',1)[-1] + "_counting.csv"
    counting_file = open(os.path.join(workdir, "results", counting_file_name), mode='w', newline='')
    counting_writer = csv.writer(counting_file, delimiter=',')
    counting_writer.writerow(["elapsed_time", "counter","section_count",'Longitude','Latitude'])
    counting_file.close()

    #wait for everything to initialize
    time.sleep(1.0)


    i = 0
    prev_time = 0
    startTime =time.time()
    elapsed_time = time.time()
    frames_count = 0
    section_count = 0
    video_end_frames = 0


    time.sleep(1.0)
    #cap.set(3, 1280)
    #cap.set(4, 1024)

    #output video file

    #counting_file_name = args["input"].rsplit('.')[-2].rsplit('/',1)[-1] + "_counting.csv"

    out = cv2.VideoWriter(
        "output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 120.0,
        (width, height))
    
    print("Starting the YOLO loop...")
    W = None
    H = None
    tracks = ct.update([])

    #for EMI data
    gps_data = pd.read_csv(gps_csv_file,sep='\t')

    #for GOPRO data
    gps_data = pd.read_csv(gps_csv_file,sep=',')
    #gps_data = gps_data.drop_duplicates(subset=['time'], keep='first')


    gps_point_count = len(gps_data.index)
    print("gps points count :" + str(gps_point_count))

    vid_frames_per_gps_point = total_vid_frames/gps_point_count
    print("vid_frames_per_gps_point :" + str(vid_frames_per_gps_point))

    vid_length = total_vid_frames/vid_fps
    print("vid_length :" + str(vid_length))

    gps_points_per_second = gps_point_count/vid_length
    print("gps_points_per_second :" + str(gps_points_per_second))


    last_fps = 0
    current_fps = 0
    while True:
        prev_time = round(time.time(),2)

        frame_read = frame_queue.get()
        darknet.copy_image_from_bytes(darknet_image, frame_read.tobytes())
        #
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=0.05,nms=0.6)
        
        
        #print(detections[1])
        image = frame_read
        image = darknet.draw_boxes(detections, image, class_colors)

        totalFrames += 1
        frames_count += 1

        if W is None or H is None:
            (H, W) = image.shape[:2]
        line = [(W // 2,0), (W // 2,H)]
        rects = []
        for detection in detections:
            if (detection[2][2] * detection[2][3]) > 300 : 
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
                        section_count += 1
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
        
        cv2.putText(image, str(elapsed_time), (0,150), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 0), 4)
        #timeElapsed = (datetime.now() - startTime).total_seconds()
        #cv2.putText(image, str(totalFrames), (0,300), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 0, 0), 4)
        #print(round(elapsed_time % 5))
        cv2.putText(image, str(frames_count), (0,200), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 0), 4)
        cv2.putText(image, str(section_count), (0,250), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 0), 4)
        cv2.putText(image, str(frames_count % 120), (0,300), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 0), 4)
        
        #print(args["input"])
        #print("write csv",time.time())
        if (frames_count % int(vid_fps)) == 0  :
            with open(os.path.join(workdir, "results", counting_file_name), mode='a', newline='') as counting_file:
                counting_writer = csv.writer(counting_file, delimiter=',')
                #gps_point_count = len(gps_data.index)
                #vid_frames_per_gps_point = total_vid_frames/gps_point_count
                #counting_writer.writerow([elapsed_time, counter,section_count,gps_data.iloc[int(frames_count/vid_fps)]['longitude'],gps_data.iloc[int(frames_count/vid_fps)]['latitude']])
                #counting_writer.writerow([elapsed_time, counter,section_count,gps_data.loc[frames_count/120,'X'],gps_data.loc[frames_count/120,'Y']])
                
                
                #vid_file_number = int(gps_csv_file[-5])-1
                print(frames_count)
                print(vid_fps)
                print(vid_frames_per_gps_point)
                #print("GPS row " + str(int(round(((frames_count+vid_file_number*1502*vid_fps) /  vid_fps)*2.5  - 1))))

                
                #print(f'vid {vid_file_number}')
                #gps_file_start_offset = int((vid_file_number - 1) * ((25 * 60) + 2) * 2.5)
                #print(f'gps_file_start_offset {gps_file_start_offset}')
                #print(f'row {int(round((frames_count /  vid_fps)*2.5 ) )- 1 + gps_file_start_offset}')
                counting_writer.writerow([elapsed_time, counter,section_count,gps_data.iloc[int(round((frames_count /  vid_fps)*gps_points_per_second + 1))]['longitude'],gps_data.iloc[int(round((frames_count / vid_fps)*gps_points_per_second + 1))]['latitude']])
                #counting_writer.writerow([elapsed_time, counter,section_count,gps_data.iloc[int(round(((frames_count+vid_file_number*1502*vid_fps) /  vid_fps)*2.5 ) )- 1 ]['Longitude'],gps_data.iloc[int(round(((frames_count+vid_file_number*1502*vid_fps) / vid_fps)*2.5 ) )- 1]['Latitude']])
            print("writing points data")
            section_count = 0
            #prev_time = prev_time + 4
        elapsed_time = (time.time() - startTime)
        



        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   
        
        #cv2.imshow('Demo', image)
        #k = cv2.waitKey(1)
        #if k == 27:
        #    break
        #out.write(image)
        current_fps = round(1/(time.time()-prev_time))
        avg_fps = (current_fps + last_fps) / 2
        print("FPS : " + str(avg_fps) + " | Time Left : " + str((total_vid_frames-frames_count)/avg_fps/60) + "mins")
        print("Progress : " + str(int(frames_count*100/total_vid_frames)) + "%" + " | Total Count : " + str(counter) + "| Section Count : " + str(section_count) )
        last_fps = current_fps
    cap.release()
    out.release()

if __name__ == "__main__":
    YOLO()
