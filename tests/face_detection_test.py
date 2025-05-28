import cv2
import supervision as sv
from ultralytics import YOLO

image = cv2.imread("../test.jpg")
model = YOLO("../yolov8n-face.pt")
result = model(image)[0]
detections = sv.Detections.from_ultralytics(result)

print(len(detections))

box_annotator = sv.BoxAnnotator()
annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections)

cv2.imwrite("result.jpg", annotated_frame)
