# Matrix Factorization Algorithms - Comprehensive Application

Bu proje, matris faktörizasyon algoritmalarının detaylı kullanımını içeren kapsamlı bir Python uygulamasıdır.

## Algoritmalar

### Klasik Algoritmalar

#### 1. SVD (Singular Value Decomposition)
- **Açıklama**: Matrisi tekil değerlerine ayırır. Matematiksel olarak en kesin yöntemdir.
- **En İyi Kullanım Alanı**: Öneri sistemleri, Gürültü temizleme

#### 2. PCA (Principal Component Analysis)
- **Açıklama**: Veriyi daha düşük boyutlu bir uzaya izdüşürür.
- **En İyi Kullanım Alanı**: Veri görselleştirme, Özellik seçimi

#### 3. NMF (Non-negative Matrix Factorization)
- **Açıklama**: Matrisleri sadece pozitif değerlerle ayırır.
- **En İyi Kullanım Alanı**: Görüntü işleme, Metin madenciliği (Topic Modeling)

#### 4. ALS (Alternating Least Squares)
- **Açıklama**: Büyük ölçekli (Spark gibi) sistemlerde paralel çalışmaya çok uygundur.
- **En İyi Kullanım Alanı**: Büyük ölçekli öneri motorları

### Modern Deep Learning Algoritmalar 🚀

#### 5. NCF (Neural Collaborative Filtering)
- **Açıklama**: SVD'nin Deep Learning versiyonu. Doğrusal olmayan ilişkileri öğrenir.
- **En İyi Kullanım Alanı**: Netflix, YouTube gibi modern öneri sistemleri
- **Gereksinim**: TensorFlow

#### 6. Autoencoder (Denoising & VAE)
- **Açıklama**: SVD ve PCA'in Deep Learning karşılığı. Gürültü temizleme ve öneri sistemleri için.
- **En İyi Kullanım Alanı**: Görüntü gürültü temizleme, Variational Autoencoder ile öneriler
- **Gereksinim**: TensorFlow

#### 7. Factorization Machines (FM) & DeepFM
- **Açıklama**: Context-aware öneri sistemi. Yan bilgileri (saat, cihaz, vb.) kullanır.
- **En İyi Kullanım Alanı**: Reklam tıklama tahmini (CTR), Context-aware öneriler
- **Gereksinim**: TensorFlow

#### 8. Transformer (BERT4Rec/SASRec)
- **Açıklama**: ChatGPT mimarisinin öneri sistemlerine uyarlanmış hali. Zaman ve sıra bilgisini kullanır.
- **En İyi Kullanım Alanı**: TikTok, YouTube gibi sequential öneriler
- **Gereksinim**: TensorFlow

#### 9. GNN (Graph Neural Network)
- **Açıklama**: Veriyi tablo değil, ağ (graph) olarak görür. İlişkisel veriyi en iyi işleyen yöntem.
- **En İyi Kullanım Alanı**: Pinterest, Uber Eats, Sosyal ağ tabanlı öneriler
- **Gereksinim**: PyTorch, PyTorch Geometric

## Kurulum

### Temel Kurulum (Klasik Algoritmalar)

```bash
pip install -r requirements.txt
```

### Modern Algoritmalar İçin (Opsiyonel)

Modern Deep Learning algoritmalarını kullanmak için:

```bash
# TensorFlow (NCF, Autoencoder, FM, Transformer için)
pip install tensorflow

# PyTorch ve PyTorch Geometric (GNN için)
pip install torch torch-geometric
# Windows için: pip install torch torch-geometric -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
```

**Not**: Modern algoritmalar opsiyoneldir. Klasik algoritmalar TensorFlow/PyTorch olmadan çalışır.

## Kullanım

### Web Uygulaması (Streamlit)
```bash
streamlit run app.py
```

### Jupyter Notebook
```bash
jupyter notebook analysis.ipynb
```

## Proje Yapısı

```
matrix-factorization-app/
├── app.py                 # Streamlit web uygulaması
├── algorithms/            # Algoritma implementasyonları
│   ├── svd.py            # Klasik: SVD
│   ├── pca.py            # Klasik: PCA
│   ├── nmf.py            # Klasik: NMF
│   ├── als.py            # Klasik: ALS
│   ├── ncf.py            # Modern: Neural Collaborative Filtering
│   ├── autoencoder.py    # Modern: Denoising Autoencoder & VAE
│   ├── fm.py             # Modern: Factorization Machines & DeepFM
│   ├── transformer.py    # Modern: Transformer-based Recommendation
│   └── gnn.py            # Modern: Graph Neural Network
├── examples/              # Kullanım örnekleri
│   ├── recommendation_system.py
│   ├── noise_reduction.py
│   ├── visualization.py
│   ├── topic_modeling.py
│   └── image_processing.py
├── utils/                 # Yardımcı fonksiyonlar
│   ├── data_loader.py
│   └── visualization.py
├── analysis.ipynb         # Detaylı analiz notebook'u
└── requirements.txt
```

## Özellikler

- ✅ Her algoritma için detaylı implementasyon
- ✅ **Gerçek veri seti yükleme desteği** (CSV, Excel)
- ✅ Gerçek dünya kullanım örnekleri
- ✅ İnteraktif görselleştirmeler
- ✅ Performans karşılaştırmaları
- ✅ Web tabanlı kullanıcı arayüzü (Streamlit)
- ✅ Jupyter notebook ile detaylı analiz
- ✅ Benchmark ve performans test modülü
- ✅ 5+ farklı kullanım örneği

## 📁 Gerçek Veri Seti Yükleme

Uygulama artık dışarıdan veri seti yükleme özelliğine sahip! Kendi veri setinizi yükleyip algoritmaları test edebilirsiniz.

### Desteklenen Formatlar

- **CSV** (.csv)
- **Excel** (.xlsx, .xls)

### Veri Formatı

#### 1. Long Format (Önerilen) 📊

Her satır bir kullanıcı-ürün-rating üçlüsü:

```csv
user_id,item_id,rating
1,5,4.5
1,12,3.0
2,5,5.0
2,8,2.5
...
```

**Sütun İsimleri:**
- Kullanıcı sütunu: `user_id`, `user`, `userId`, `CustomerID`, vb. (otomatik tespit)
- Ürün sütunu: `item_id`, `item`, `itemId`, `product_id`, `movie_id`, vb. (otomatik tespit)
- Rating sütunu: `rating`, `Rating`, `score`, `value`, vb. (otomatik tespit)

#### 2. Matrix Format 📈

Zaten rating matrisi formatında:

```csv
,item_1,item_2,item_3,...
user_1,4.5,NaN,3.0,...
user_2,5.0,2.5,NaN,...
...
```

### Kullanım

1. **Streamlit Uygulamasında:**
   - Herhangi bir algoritma sayfasına gidin (örn: SVD - Öneri Sistemi)
   - "📁 Dosyadan Yükle" seçeneğini seçin
   - Veri dosyanızı yükleyin
   - Veri formatını seçin (Long Format veya Matrix Format)
   - Sütun isimlerini belirtin (veya otomatik tespit edilmesine izin verin)
   - "📥 Veriyi Yükle" butonuna tıklayın

2. **Python Kodunda:**
```python
from utils.data_loader import load_rating_data_from_file

# Dosya yükleme (Streamlit file_uploader'dan)
rating_matrix, user_mapping, item_mapping = load_rating_data_from_file(
    file,
    user_col='user_id',      # Opsiyonel: otomatik tespit edilir
    item_col='item_id',      # Opsiyonel: otomatik tespit edilir
    rating_col='rating'      # Opsiyonel: otomatik tespit edilir
)

# Artık rating_matrix'i kullanabilirsiniz
print(f"Yüklenen veri: {rating_matrix.shape[0]} kullanıcı, {rating_matrix.shape[1]} ürün")
```

### Örnek Veri Dosyası

Proje kök dizininde `example_ratings.csv` örnek dosyası bulunmaktadır. Bu dosyayı test için kullanabilirsiniz.

### Önemli Notlar

- Rating değerleri otomatik olarak **1-5 aralığına normalize** edilir
- Eksik veriler (NaN) korunur
- Veri yoğunluğu otomatik hesaplanır ve gösterilir
- Büyük veri setleri için işlem süresi artabilir

## Detaylı Kullanım Örnekleri

### 1. Öneri Sistemi (SVD & ALS)
```python
from algorithms.svd import SVDRecommender
from utils.data_loader import generate_rating_matrix

# Rating matrisi oluştur
rating_matrix = generate_rating_matrix(n_users=100, n_items=50)

# Model eğit
model = SVDRecommender(n_components=20)
model.fit(rating_matrix)

# Tahmin yap
prediction = model.predict(user_idx=0, item_idx=0)
```

### 2. Gürültü Temizleme (SVD)
```python
from algorithms.svd import SVDNoiseReducer

# Gürültülü veriyi temizle
reducer = SVDNoiseReducer(n_components=None, threshold=0.95)
reducer.fit(noisy_data)
denoised_data = reducer.denoise(noisy_data)
```

### 3. Veri Görselleştirme (PCA)
```python
from algorithms.pca import PCAAnalyzer

# PCA uygula
pca = PCAAnalyzer(n_components=None)
X_transformed = pca.fit_transform(X)

# 2D görselleştirme
pca.plot_2d_projection(X, y=labels)
```

### 4. Topic Modeling (NMF)
```python
from algorithms.nmf import NMFTopicModeler

# Topic'leri bul
model = NMFTopicModeler(n_topics=5)
model.fit(documents)
topics = model.get_topics(n_words=10)
```

### 5. Görüntü İşleme (NMF)
```python
from algorithms.nmf import NMFImageProcessor

# Görüntüleri işle
processor = NMFImageProcessor(n_components=20)
processor.fit(images)
reconstructed = processor.reconstruct()
```

## Benchmark ve Performans Testi

```bash
python benchmark.py
```

Bu komut tüm algoritmaların performansını test eder ve karşılaştırma grafikleri oluşturur.

## Örnek Script'ler

Her kullanım alanı için ayrı örnek script'ler:

```bash
# Öneri sistemi
python examples/recommendation_system.py

# Gürültü temizleme
python examples/noise_reduction.py

# Veri görselleştirme
python examples/visualization.py

# Topic modeling
python examples/topic_modeling.py

# Görüntü işleme
python examples/image_processing.py
```

## Gereksinimler

Tüm bağımlılıklar `requirements.txt` dosyasında listelenmiştir. Python 3.8+ gereklidir.

## Lisans

Bu proje eğitim amaçlıdır.

