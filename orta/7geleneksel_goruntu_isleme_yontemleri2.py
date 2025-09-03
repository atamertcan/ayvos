import cv2 as cv
import numpy as np

img = cv.imread("../Photos/logo.png")
kernel = np.ones((5,5),np.uint8)

dilation = cv.dilate(img, kernel, iterations=1) #genişleme yapar. Kesikli bölgeleri doldurmada kullanılır.Iteration genişlemeyi belirliyor.
dilation2 = cv.dilate(img, kernel, iterations=2)
erosion = cv.erode(img, kernel, iterations=2) #Gürültü gidermede kullanılır. Aşındırma yapar yani kalınlığı veya boyutu azaltır
#Normalde gürültü var ise önce erosion üstüne dilation yapılır ki gürültü giderilip ana görüntü belirginlik hale getirilsin.


cv.imshow("erosion", erosion)
cv.imshow("dilation", dilation)
cv.imshow("dilation2", dilation2)
cv.imshow("original", img)

cv.waitKey(0)
cv.destroyAllWindows()


