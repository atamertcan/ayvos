import cv2 as cv

img = cv.imread("Photos/cat.jpeg")

blur = cv.GaussianBlur(img, (3,3), cv.BORDER_DEFAULT) #Buradaki kernel size'ı arttırarak daha fazla blur elde edilir
cv.imshow("normal", img)
cv.imshow("blurred", blur)

cv.waitKey()
cv.destroyAllWindows()