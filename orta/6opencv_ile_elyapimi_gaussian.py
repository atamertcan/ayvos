import cv2 as cv
import numpy as np
import time

# Elle Gaussian kernel üret
def gaussian_kernel(size: int, sigma: float):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / np.sum(kernel)  # normalize et
    return kernel

# Görüntü oku
img = cv.imread("../Photos/cat.jpeg")

# Kernel oluştur
kernel = gaussian_kernel(11, 2)

# Zaman ölçerek uygula
start = time.time()
manual_blur = cv.filter2D(img, -1, kernel)
end = time.time()
print("Elle kernel süresi:", end - start)

start = time.time()
opencv_blur = cv.GaussianBlur(img, (11,11), 2)
end = time.time()
print("OpenCV GaussianBlur süresi:", end - start)

cv.imshow("Orijinal", img)
cv.imshow("Elle Kernel", manual_blur)
cv.imshow("OpenCV GaussianBlur", opencv_blur)
cv.waitKey(0)
cv.destroyAllWindows()