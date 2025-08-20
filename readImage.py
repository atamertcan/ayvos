import cv2 as cv
import numpy as np
from matplotlib import pyplot


img = cv.imread("Photos/cat.jpeg", 0) # 0 yaptığımızda siyah beyaz olur

cv.imshow("Cat", img)

cv.imwrite("Photos/blackandwhitecat.jpeg", img)
cv.waitKey(0) #bir tuşa basana kadar açık kalır
cv.destroyAllWindows

