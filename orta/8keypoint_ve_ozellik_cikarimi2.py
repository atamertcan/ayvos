import cv2 as cv

#Kullanım olarak ve genel mantık olarak SIFT ile benzer. SIFT'e göre çok daha hızlı.

img = cv.imread("../Photos/character.jpg", cv.IMREAD_GRAYSCALE)

orb = cv.ORB_create()
keypoints = orb.detect(img, None)
keypoints,_ = orb.compute(img, keypoints)#descriptor için
img = cv.drawKeypoints(img,keypoints,img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv.imshow("keypoints", img)

cv.waitKey(0)
cv.destroyAllWindows()

