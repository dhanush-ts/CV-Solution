import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import time
import requests

video_path = 'WhatsApp Video 2024-08-05 at 10.01.12_281b1a61.mp4'  # Replace with your video file path or 0 for webcam
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

phone = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "diningtable", "toilet", "TV monitor", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

idx = phone.index("cell phone")

def dataGet(ra):
    data = {
      "subjectID": 1,
      "rating": ra
    }

    headers = {
        "Authorization": f"Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IkNTMTIzIiwidXNlcl90eXBlIjoic3RhZmYiLCJleHAiOjE3NTQ0NDk2Mzh9.OsIlhB88ZZMHK9_tlI06E3J2D3r4Alj14OopsUfLmVM",  # Replace with your actual token
        "Content-Type": "application/json"
    }


    response = requests.post("http://localhost:8000/api/features/interaction/", json=data,headers=headers)

    print(f"Status Code: {response.status_code}")

model = YOLO('./run3/train/weights/best.pt')
model2 = YOLO("./yolov8l.pt")

prev = 0
classNames = {
    0: 'hand-raising',
    1: 'reading',
    2: 'writing'
}
start_time = time.time()
arr = []
# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
# totalCount = []
num_inter = 0

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    results2 = model2(img, stream=True)

    detections = np.empty((0, 5))

    for r in results2:
        boxes = r.boxes
        for box in boxes:
            if int(box.cls[0])==idx:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
                w, h = x2 - x1, y2 - y1

                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class Name
                cls = int(box.cls[0])
                currentClass = phone[cls]

                if conf > 0.5:
                    cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
                    cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                                       scale=2, thickness=2, offset=2)

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
    print(num_inter,arr)

    elapsed_time = time.time() - start_time
    if elapsed_time > 5:
        print(f"Maximum interactions in last 30 seconds: {num_inter}")
        dataGet(num_inter-prev)
        arr.append(num_inter-prev)
        tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
        prev = num_inter
        start_time = time.time()


    cv2.imshow("Image", img)
    cv2.waitKey(1)