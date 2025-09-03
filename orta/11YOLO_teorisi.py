"""
Genel olarak YOLO versiyonları arasındaki fark hız, detaylandırma ve doğruluk olmuş.
İlk versiyonlarında görseli 1x1 ızgaralara bölerek ızgaralardaki sınıf olasılıklarını hesaplayarak bounding box tahminleri yaptı.
son versiyonlarına doğru mimarileri değiştirmişler, pytorch destekli hale getirmişler(esneklik artmış)
ve ızgaradan sonra box + anchorlu yapıya geçerken daha ileride anchor free(YOLOX) hale gelmiştir.
YOLO R modeliyle ise multi-task learning kullanılmıştır

"""

from ultralytics import YOLO
import cv2 as cv

model = YOLO("YOLO_weights/yolov8n.pt")


img = cv.imread("../Photos/cars.jpg")
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
img1 = cv.GaussianBlur(img,(5,5),0)
img2 = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,15,5)
img2gray = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)

result1=model(img2gray)
result2=model(img1)
result3=model("Photos/cars.jpg")

cv.imshow("Threshold", result1[0].plot())
cv.imshow("Blur", result2[0].plot())
cv.imshow("Original", result3[0].plot())

cv.waitKey(0)