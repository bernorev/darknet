import cv2
import numpy as np
import time

confidence_threshold = 0.2
nms_threshold = 0.4
num_classes = 1

net = cv2.dnn.readNet("./cfg/yolov4-tiny_fruit.cfg","./backup/yolov4-tiny_fruit_last.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

frame = np.random.randint(255, size=(800, 1440, 3), dtype=np.uint8)
blob = cv2.dnn.blobFromImage(frame, 0.00392, (800, 1440), [0, 0, 0], True, False)

# warmup
for i in range(3):
    net.setInput(blob)
    detections = net.forward(net.getUnconnectedOutLayersNames())

# benchmark
start = time.time()
for i in range(100):
    net.setInput(blob)
    detections = net.forward(net.getUnconnectedOutLayersNames())
end = time.time()

ms_per_image = (end - start) * 1000 / 100

print("Time per inference: %f ms" % (ms_per_image))
print("FPS: ", 1000.0 / ms_per_image)
