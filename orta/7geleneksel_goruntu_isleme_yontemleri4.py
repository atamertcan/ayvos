import cv2 as cv

img = cv.imread("../Photos/blackandwhitecat.jpeg", 0)

thresh1 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,15,5)
thresh2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,51,11) #daha da kalınlaşmış detay azalmış
thresh3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,15,5) #biraz daha ince çizmiş

cv.imshow("original", img)
cv.imshow("mean", thresh1)
cv.imshow("mean2", thresh2)
cv.imshow("gaussian", thresh3)

cv.waitKey(0)
cv.destroyAllWindows()