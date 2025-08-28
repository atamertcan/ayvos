import cvzone
from ultralytics import YOLO
import cv2 as cv

capture = cv.VideoCapture("Videos/people.mp4")
#capture.set(3,1280)
#capture.set(4,720)

model1 = YOLO("YOLO_weights/yolov8n.pt")

while True:
    isTrue, frame = capture.read()
    results = model1(frame,stream=True)
    names = model1.names
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)

            #print(x1,y1,x2,y2)
            cv.rectangle(frame,(x1,y1),(x2,y2),(255,0,255),3)
            conf = round(float(box.conf[0]),2)
            #print(conf)
            className = names[int(box.cls)]
            cvzone.putTextRect(frame,f'{className} {conf}',
                         (max(0,x1), max(35,y1)),scale=0.7,thickness=1)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

    cv.imshow("Video", frame)
