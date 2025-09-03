import cv2 as cv

img = cv.imread("../Photos/cat.jpeg")

res = cv.resize(img, None, fx = 1.5, fy = 0.7, interpolation=cv.INTER_CUBIC) 
#interpolation değişen pikseller için ortalama bulma konusunda kullanılan yöntem
#cubic olan en yavaş fakat en düzgün sonuç veren
cv.imshow("cat", img)
cv.imshow("resizedcat", res)

cv.waitKey()
cv.destroyAllWindows()