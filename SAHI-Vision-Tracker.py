import cv2
import numpy as np
from sahi import AutoDetectionModel
from sahi.utils import visualization as sahi_visualization
from ultralytics import YOLO


model = YOLO('yolov8n.pt')

# Initialize the AutoDetectionModel using the YOLO model
detection_model = AutoDetectionModel(model=model)


cap = cv2.VideoCapture(
    r"C:\Users\DELL\Downloads\Telegram Desktop\pythone\openCv\vidoies\2293417-sd_640_360_24fps.mp4")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = detection_model.detect(frame)

    # Draw the bounding boxes
    frame = sahi_visualization.draw_bboxes(
        frame, results, class_names=model.names)

    # Display the frame
    cv2.imshow('Real-time Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
