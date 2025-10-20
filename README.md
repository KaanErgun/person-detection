# YOLOv10 İnsan Tespit Sistemi

Mac kamerasını kullanarak gerçek zamanlı insan tespiti yapan Python uygulaması.

## Özellikler

- ✅ Gerçek zamanlı kamera görüntüsü işleme
- ✅ YOLOv10m modeli ile yüksek doğrulukta insan tespiti
- ✅ Tespit edilen kişileri yeşil kare içine alma
- ✅ Güven skoru gösterimi
- ✅ Tespit edilen kişi sayısı sayacı
- ✅ FPS göstergesi

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

## Kullanım

Uygulamayı başlatmak için:
```bash
python person_detection.py
```

### Kontroller

- **q**: Uygulamadan çıkış

## Kamera İzinleri (macOS)

macOS'ta kamera kullanabilmek için:

1. **Sistem Tercihleri** > **Güvenlik ve Gizlilik** > **Gizlilik** > **Kamera**
2. Terminal veya Python için kamera erişimini etkinleştirin

Eğer izin sorunu yaşarsanız, Terminal'i kamera erişimi listesine ekleyin.

## Model Bilgisi

Uygulama `yolov10/yolov10m.pt` model dosyasını kullanır. Bu dosyanın doğru konumda olduğundan emin olun.

## Sorun Giderme

### Kamera açılmıyor
- Kamera izinlerini kontrol edin
- Başka bir uygulama kamerayı kullanıyor olabilir

### Model yüklenemiyor
- `yolov10/yolov10m.pt` dosyasının var olduğundan emin olun
- Ultralytics kütüphanesinin doğru yüklendiğini kontrol edin

### Düşük FPS
- Daha hafif bir model (yolov10n.pt) kullanmayı deneyin
- Kamera çözünürlüğünü azaltın

## Gereksinimler

- Python 3.8+
- OpenCV
- Ultralytics YOLOv10
- NumPy
- PyTorch
