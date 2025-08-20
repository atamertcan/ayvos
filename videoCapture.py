import cv2 as cv

cam = cv.VideoCapture(0)

while True:
    isTrue, frame = cam.read()

    if not isTrue:
        print("Could not access camera")
        break
    cv.imshow("cam", frame)
    if cv.waitKey(1) & 0xFF == ord("q"):    #waitkey(1) demek video için 1ms bekle demek akıcı video olması için önemli
        print("Show has ended")             #bu if bloğu aslında kareyi güncelle tuş kontrolü yap tuş q ise çık demek
        break

cam.release()
cv.destroyAllWindows()