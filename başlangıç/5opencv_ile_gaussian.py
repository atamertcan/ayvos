import cv2 as cv

img = cv.imread("../Photos/cat.jpeg")

blur = cv.GaussianBlur(img, (3,3), cv.BORDER_DEFAULT) #Buradaki kernel size'ı arttırarak daha fazla blur elde edilir

blur2 = cv.GaussianBlur(img, (7,7), cv.BORDER_DEFAULT)  #Kernelin bir kısmı görüntü dışına taşar bunu engellemek için borderdefault var.
                                                        #Aynı zamanda gaussian blur varsayılanıdır.
blur3 = cv.GaussianBlur(img, (7,7),cv.BORDER_REPLICATE)                  
cv.imshow("normal", img)
cv.imshow("blurred", blur)
cv.imshow("more blurred", blur2)
cv.imshow("more blurred replicate", blur3)

cv.waitKey()
cv.destroyAllWindows()