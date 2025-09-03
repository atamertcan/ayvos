import cv2
import cvzone
import numpy as np
from ultralytics import YOLO
import cv2 as cv
from sort import *

capture = cv.VideoCapture("../Videos/people.mp4")


model1 = YOLO("YOLO_weights/yolov8n.pt")
mask = cv.imread("../Photos/maskPeople.png")

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
#limits = [550,700,1750,700] car için
limits = [100,800,1900,400]
totalCount = []

def distance(px, py, x1, y1, x2, y2):
    return abs((y2 - y1) * px - (x2 - x1) * py + x2*y1 - y2*x1) / (
        ((y2 - y1)**2 + (x2 - x1)**2) ** 0.5
    )

while True:
    isTrue, frame = capture.read()
    imgRegion = cv2.bitwise_and(frame,mask)
    results = model1(imgRegion,stream=True,imgsz=320)

    detections = np.empty((0,5))

    names = model1.names
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)

            conf = round(float(box.conf[0]),2)

            className = names[int(box.cls)]
            if className=="person" and conf > 0.3:
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections,currentArray))

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

    resultsTracker = tracker.update(detections)
    cv.line(frame,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),5)
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
        cvzone.putTextRect(frame, f' {id}',
                           (max(0, x1), max(35, y1)), scale=1, thickness=1, offset=4)
        cx, cy = int((x2+x1) / 2), int((y1+y2) / 2)
        cv.circle(frame,(cx,cy),5,(255,0,255),cv.FILLED)

        dist = distance(cx, cy, limits[0], limits[1], limits[2], limits[3])

        if dist < 20:
            if id not in totalCount:
                totalCount.append(id)

    cvzone.putTextRect(frame, f' Count:{len(totalCount)}',
                       (50, 50))
    cv.imshow("Video", frame)
