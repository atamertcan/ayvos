import cv2 as cv


def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)  #frame.shape[1] shows width of the image
    height = int(frame.shape[0] * scale)  #frame.shape[0] shows height of the image
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

cv.waitKey(0)

capture = cv.VideoCapture('Videos/catvideo.mp4')

while True:
    isTrue, frame = capture.read()

    frame_resized = rescaleFrame(frame)

    cv.imshow('Video', frame)
    cv.imshow('Video Resized', frame_resized)

    if cv.waitKey(12) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()