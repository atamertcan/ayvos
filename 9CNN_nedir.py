#Filtre mantığı derin öğrenmede convolution olarak adlandırılıyor.
#Aslında görüntü işlemedeki filtre hatta kullanılan yöntem bile benzerdir.
#Convolution da bir filtreleme metodudur. Her bir uzun veri dizisinde daha küçük veri dizileriyle filtreleme yapılır.
#Genel mantık: Filtre boyu kadar değer al, filtrede üstüne denk gelen sayılarla çarpıp çıkan değerleri topla. Stride kadar kaydır.
#Convolution sonrası fotoğrafta featurlar ortaya çıkarılmış olur.
#Padding: Fotoğrafta kenar kısımlarda kalan pikseller filtreye düzgün denk gelmediği için padding ile ekstra piksel eklenir.
"""
CNN bu layerlardan oluşur:
Convolution layer: Filtre uygulayıp layer tespiti yapar.
Pooling layer: Feature haritalarının uzaysal uzunluğunu düşürür.Maxpooling(en büyüğü seçme) çok kullanılıyor.
En önemli featurelara erişilebiliyor.Boyutu küçültür.
Fully connected layer: Bu katmandaki tüm nöronlar bir önceki katmandaki nöronlara bağlı.
Poolingden çıkan özellikler vektöre dönüştürülür.
Activation layer: Ağa nonlineerlik ekleyerek karmaşık ilişkileri öğrenebilmesini sağlar.Negatif değerleri 0 yapar.
"""
import cv2 as cv
import numpy as np
 
img = cv.imread("Photos/character.jpg", cv.IMREAD_COLOR)
img = cv.resize(img, None, fx = 0.5, fy = 0.5)

kernel = np.ones((10,10),dtype=float)/100 #ortalama aldık. Box blur.
convolution = cv.filter2D(img, -1, kernel)


cv.imshow("original",img)
cv.imshow("convolved",convolution)

cv.waitKey(0)
cv.destroyAllWindows()

