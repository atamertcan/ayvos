"""
Genel olarak YOLO versiyonları arasındaki fark hız, detaylandırma ve maliyet olmuş.
İlk versiyonlarında görseli 1x1 ızgaralara bölerek ızgaralardaki sınıf olasılıklarını hesaplayarak bounding box tahminleri yaptı.
son versiyonlarına doğru mimarileri değiştirmişler, pytorch destekli hale getirmişler(esneklik artmış)
ve ızgaradan sonra anchorlu yapıya geçerken daha ileride anchor free(YOLOX) hale gelmiştir.
YOLO R modeliyle ise multi-task learning kullanılmıştırç

"""