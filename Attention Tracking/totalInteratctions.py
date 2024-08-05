import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

video_path = 'WhatsApp Video 2024-08-05 at 10.01.12_281b1a61.mp4'  # Replace with your video file path or 0 for webcam
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

model = YOLO('./run3/train/weights/best.pt')

classNames = {
    0: 'hand-raising',
    1: 'reading',
    2: 'writing'
}

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
# totalCount = []
num_inter = 0

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if conf > 0.5:
                cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
                cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                                   scale=2, thickness=2, offset=2)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)
    #
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        # cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
        #                    scale=2, thickness=3, offset=10)
        num_inter = max(num_inter, int(id))
        print(num_inter)

    cv2.imshow("Image", img)
    cv2.waitKey(1)