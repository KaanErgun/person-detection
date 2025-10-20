"""
YOLOv10 ile Gerçek Zamanlı İnsan Tespiti
Mac kamerasını kullanarak insan tespiti yapar ve tespit edilen kişileri kare içine alır.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import sys
import time

class PersonDetector:
    def __init__(self, model_path='yolov10/yolov10n.pt'):
        """
        YOLOv10 model ile insan tespit sınıfı
        
        Args:
            model_path: YOLOv10 model dosyasının yolu
        """
        print("Model yükleniyor...")
        try:
            self.model = YOLO(model_path)
            print("Model başarıyla yüklendi!")
        except Exception as e:
            print(f"Model yükleme hatası: {e}")
            sys.exit(1)
        
        # COCO dataset'te 'person' sınıfı 0 indeksindedir
        self.person_class_id = 0
        
        # Renk ayarları (BGR formatında)
        self.box_color = (0, 255, 0)  # Yeşil
        self.text_color = (255, 255, 255)  # Beyaz
        self.box_thickness = 2
        
    def detect_persons(self, frame):
        """
        Verilen frame'de insan tespiti yapar
        
        Args:
            frame: OpenCV ile yakalanmış görüntü frame'i
            
        Returns:
            İşlenmiş frame ve tespit edilen kişi sayısı
        """
        # YOLOv10 ile tespit yap
        results = self.model(frame, verbose=False)
        
        person_count = 0
        
        # Sonuçları işle
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Sınıf ID'sini kontrol et (sadece insan tespitlerini al)
                class_id = int(box.cls[0])
                
                if class_id == self.person_class_id:
                    person_count += 1
                    
                    # Güven skoru
                    confidence = float(box.conf[0])
                    
                    # Bounding box koordinatları
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Dikdörtgen çiz
                    cv2.rectangle(frame, (x1, y1), (x2, y2), 
                                self.box_color, self.box_thickness)
                    
                    # Etiket metni (İnsan + güven skoru)
                    label = f'Insan {confidence:.2f}'
                    
                    # Metin boyutunu hesapla
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    
                    # Metin arka planı için dikdörtgen çiz
                    cv2.rectangle(frame, 
                                (x1, y1 - text_height - baseline - 10),
                                (x1 + text_width, y1),
                                self.box_color, -1)
                    
                    # Metni yaz
                    cv2.putText(frame, label, (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                              self.text_color, 2)
        
        return frame, person_count
    
    def run(self):
        """
        Kamera akışını başlatır ve insan tespiti yapar
        """
        print("Kamera açılıyor...")
        cap = cv2.VideoCapture(0)  # 0 = varsayılan kamera
        
        if not cap.isOpened():
            print("HATA: Kamera açılamadı!")
            print("Lütfen kamera izinlerini kontrol edin.")
            return
        
        print("Kamera başarıyla açıldı!")
        print("Çıkmak için 'q' tuşuna basın...")
        
        # Kamera çözünürlüğünü ayarla
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Gerçek FPS hesaplama için değişkenler
        fps = 0
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Frame okunamadı!")
                break
            
            # İnsan tespiti yap
            processed_frame, person_count = self.detect_persons(frame)
            
            # Gerçek FPS hesapla
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 1.0:  # Her saniyede bir güncelle
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
            
            # Bilgi metni ekle
            info_text = f'Tespit Edilen Kisi Sayisi: {person_count}'
            cv2.putText(processed_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Gerçek FPS göster
            fps_text = f'FPS: {fps:.1f}'
            cv2.putText(processed_frame, fps_text, (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Görüntüyü göster
            cv2.imshow('YOLOv10 Insan Tespiti', processed_frame)
            
            # 'q' tuşuna basılırsa çık
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Uygulama kapatılıyor...")
                break
        
        # Kaynakları serbest bırak
        cap.release()
        cv2.destroyAllWindows()
        print("Kamera kapatıldı.")

def main():
    """
    Ana fonksiyon
    """
    print("=" * 50)
    print("YOLOv10 İnsan Tespit Sistemi")
    print("=" * 50)
    
    # PersonDetector sınıfını başlat
    detector = PersonDetector(model_path='yolov10/yolov10n.pt')
    
    # Kamera ile tespiti başlat
    detector.run()

if __name__ == "__main__":
    main()
