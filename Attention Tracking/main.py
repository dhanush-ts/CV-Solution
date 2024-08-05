import cv2
from ultralytics import YOLO

# Define class names
classNames = {
    0: 'hand-raising',
    1: 'reading',
    2: 'writing'
}

# Load a pre-trained YOLOv8 model
model = YOLO('./run3/train/weights/best.pt')  # Replace with your model path

# Open a video file or capture from camera
video_path = 'WhatsApp Video 2024-08-05 at 10.01.12_281b1a61.mp4'  # Replace with your video file path or 0 for webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame from video or video ended.")
        break

    # Perform object detection
    results = model(frame)  # Perform inference
    detections = results[0].boxes.data  # Get detections for the first image

    # Draw bounding boxes and labels on the frame
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection.cpu().numpy()
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = f"{classNames[int(cls)]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('YOLOv8 Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
