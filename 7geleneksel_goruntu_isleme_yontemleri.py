import cv2 as cv
import numpy as np
#Önce blurlama yaparak başlıyoruz ki detaylarda bulunan kenarları da almasın 

img = cv.imread("Photos/character.jpg", 0)
img = cv.resize(img, None, fx=0.5, fy=0.5)
blur = cv.GaussianBlur(img, (5,5), cv.BORDER_DEFAULT)

def autoCanny(blur, sigma=0.33):
    median = np.median(blur)
    lower = int(max(0,(1.0-sigma)*median))
    upper = int(min(255,(1.0+sigma)*median))
    canny = cv.Canny(blur, lower, upper)
    return canny
    

canny = cv.Canny(blur, 75, 200)
auto = autoCanny(blur)

cv.imshow("original", img)
cv.imshow("canny", canny)
cv.imshow("autocanny", auto)
cv.imshow("sum", np.hstack([auto,canny]))

cv.waitKey(0)
cv.destroyAllWindows()
