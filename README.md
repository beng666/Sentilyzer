# Sentilyzer

**Sentilyzer**, Türkçe metinlerde entity (varlık) bazlı duygu analizi (sentiment analysis) yapmayı sağlayan bir projedir. Bu proje, BERT tabanlı bir model ile çalışmakta ve çeşitli araçlar ile zenginleştirilmiş bir GUI (Grafik Kullanıcı Arayüzü) sunmaktadır. Ayrıca, FastAPI ile sağlanan bir API üzerinden Swagger UI kullanarak da test edilebilir.

## Proje Bileşenleri

### 1. `utils.py`
Bu modül, metin işleme, duygu tahmini, entity tespiti gibi temel işlevleri içerir.

- **SpaCy** modelini kullanarak metinlerdeki entity'leri tanır.
- **BERT** modelini kullanarak metinlerin vektör temsillerini çıkarır.
- MLPClassifier modelini kullanarak entity'lerin context'lerini analiz ederek sentiment tahmini yapar.

### 2. `main.py`
Bu dosya, FastAPI framework'ü ile yazılmış bir API uygulamasıdır.

- `POST /predict` endpoint'i, bir metin girdisi alır ve bu metindeki entity'lerin bağlamını (context) analiz eder ve sentiment tahminlerini döner.
- FastAPI ile entegre edilmiş Swagger UI aracılığıyla bu endpoint'leri doğrudan web tarayıcınızdan test edebilirsiniz.
- Arka planda, `utils.py` dosyasındaki fonksiyonları kullanarak sentiment analizini gerçekleştirir.

### 3. `train_for_sentiment_analysis.py`
Bu dosya, bir veri kümesini kullanarak modeli eğitmek için kullanılan Python script'idir.

- Veriyi temizler, stop word'leri çıkarır ve özellik vektörleri çıkarır.
- MLPClassifier kullanarak sentiment sınıflandırma modeli eğitir ve bu modeli kaydeder.

### 4. `main_Tkinter(GUI).py`
Bu dosya, Tkinter kullanarak bir GUI uygulaması sunar.

- Kullanıcıdan JSON formatında bir metin alır, bunu işler ve sonuçları kullanıcıya gösterir.


### 5. `matplotlib_ile_değerlendirme.py`
Bu dosya, eğitim ve test sonuçlarını değerlendirmek için matplotlib kullanarak grafikler oluşturur.

## Kurulum

### Gereksinimler
Proje Python 3.8+ ile uyumludur. Aşağıdaki bağımlılıkları kurmanız gerekmektedir:

- `torch`
- `transformers`
- `spacy`
- `joblib`
- `scikit-learn`
- `pandas`
- `matplotlib`
- `tkinter`
- `nltk`
- `fastapi`
- `uvicorn`

Tüm bağımlılıkları yüklemek için:

```bash
pip install -r requirements.txt
