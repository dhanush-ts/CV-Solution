classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

import cv2
from ultralytics import YOLO

# Load a pre-trained YOLOv8 model (e.g., yolov8n for nano, yolov8s for small, etc.)
model = YOLO('../yolov8l.pt')  # replace with 'yolov8s.pt' or other versions as needed

# Open a video file or capture from camera
video_path = './SECURUS CCTV - 2 Megapixel IP Camera with Audio Classroom Solution.mp4'  # Replace with your video file path or 0 for webcam
cap = cv2.VideoCapture(video_path)

# Define the codec and create VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.avi', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model(frame)

    # Draw bounding boxes and labels on the frame
    boxes = results[0].boxes  # Get the Boxes object from results

    # Iterate over detected objects and draw bounding boxes
    for box in boxes:
        # Extract bounding box coordinates, confidence, and class ID
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = box.conf[0]
        class_id = int(box.cls[0])
        label = model.names[class_id]

        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Draw label
        cv2.putText(frame, f'{label} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                    2)

    # Display the resulting frame
    cv2.imshow('YOLOv8 Detection', frame)
    # Display the resulting frame
    cv2.imshow('YOLOv8 Detection', frame)

    # Write the frame to the output video file
    out.write(frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

