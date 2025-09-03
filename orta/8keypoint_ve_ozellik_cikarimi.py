import cv2 as cv
import numpy as np

#Keypoints bulmak için tanımlamak için kullanılan bir algoritma. Görüntü eşleştirme, nesne tanıma gibi alanlarda kullanılır
#Görsel Gaussian blur ile farklı derecelerde bulanıklaştırılır ve aralarındaki farklar alınır(DoG).
#Her piksel komşu piksellerle karşılaştırılır ve buna göre keypointler belirlenir.

img = cv.imread("../Photos/character.jpg", 0)

sift = cv.SIFT_create()
#keypoints=sift.detect(img, None)
keypoints, des = sift.detectAndCompute(img, None)
img1 = cv.drawKeypoints(img, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#img2 = cv.drawKeypoints(img, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)
#img3 = cv.drawKeypoints(img, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)


#eşleştirme için kullanılıyor 
print("Keypoint sayısı:", len(keypoints))
print("Descriptor boyutu:", des.shape)
print("İlk keypoint'in descriptor'u:\n", des[0])


cv.imshow("comparison", np.hstack([img1]))

cv.waitKey(0)
cv.destroyAllWindows()