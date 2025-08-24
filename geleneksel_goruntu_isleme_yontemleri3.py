import cv2 as cv
#Piksellerin eşit değerlerini belirler. Eşik değerin altında kalanlar 0'a dönüştürülür.
#Üstünde kalanlar ise max olarak belirlediğimiz değere çıkarılır.
#Gri tonlamalı görüntüler tercih edilir.


#SIMPLE TRESHOLDING
img = cv.imread("Photos/blackandwhitecat.jpeg")

ret, thresh1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
ret, thresh2 = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
ret, thresh3 = cv.threshold(img, 50, 255, cv.THRESH_BINARY)

cv.imshow("original", img)
cv.imshow("thresh1", thresh1)
cv.imshow("thresh2", thresh2)
cv.imshow("thresh3", thresh3)

cv.waitKey(0)
cv.destroyAllWindows()
