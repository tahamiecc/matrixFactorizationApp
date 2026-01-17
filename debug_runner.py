"""
Otomatik Matrix Faktörizasyon Algoritmaları Test Scripti
Tüm algorithms/ klasöründeki algoritmaları otomatik olarak test eder.
"""

import numpy as np
import pandas as pd
import traceback
import warnings
warnings.filterwarnings('ignore')

# Algoritmaları ayrı ayrı import et (bir tanesi başarısız olsa bile diğerleri test edilebilsin)
CLASSIC_ALGORITHMS = {}

# Her klasik algoritmayı ayrı ayrı import et
classic_algorithm_list = [
    ('SVDRecommender', 'svd'),
    ('SVDNoiseReducer', 'svd'),
    ('PCAAnalyzer', 'pca'),
    ('NMFImageProcessor', 'nmf'),
    ('NMFTopicModeler', 'nmf'),
    ('ALSRecommender', 'als'),
]

for class_name, module_name in classic_algorithm_list:
    try:
        module = __import__(f'algorithms.{module_name}', fromlist=[class_name])
        CLASSIC_ALGORITHMS[class_name] = getattr(module, class_name)
    except ImportError as e:
        print(f"⚠️ {class_name} import edilemedi: {e}")
    except AttributeError as e:
        print(f"⚠️ {class_name} modülde bulunamadı: {e}")

CLASSIC_AVAILABLE = len(CLASSIC_ALGORITHMS) > 0

# Modern algoritmalar (PyTorch gerekli) - ayrı ayrı import et
MODERN_ALGORITHMS = {}
MODERN_AVAILABLE = False

# Önce PyTorch'un yüklü olup olmadığını kontrol et
try:
    import torch
    
    modern_algorithm_list = [
        ('NCFRecommender', 'ncf'),
        ('DenoisingAutoencoder', 'autoencoder'),
        ('VAERecommender', 'autoencoder'),
        ('FactorizationMachine', 'fm'),
        ('DeepFM', 'fm'),
        ('TransformerRecommender', 'transformer'),
        ('GNNRecommender', 'gnn'),
    ]
    
    for class_name, module_name in modern_algorithm_list:
        try:
            module = __import__(f'algorithms.{module_name}', fromlist=[class_name])
            MODERN_ALGORITHMS[class_name] = getattr(module, class_name)
        except ImportError as e:
            print(f"⚠️ {class_name} import edilemedi: {e}")
        except AttributeError as e:
            print(f"⚠️ {class_name} modülde bulunamadı: {e}")
    
    MODERN_AVAILABLE = len(MODERN_ALGORITHMS) > 0
except ImportError:
    print("⚠️ PyTorch yüklü değil. Modern algoritmalar test edilmeyecek.")


def generate_positive_data(shape=(50, 10), min_val=0.1, max_val=5.0):
    """
    Pozitif değerler içeren test verisi oluşturur (NMF için gerekli)
    
    Args:
        shape: Veri boyutu (n_samples, n_features)
        min_val: Minimum değer
        max_val: Maksimum değer
    
    Returns:
        numpy array - pozitif değerler içeren matris
    """
    np.random.seed(42)
    data = np.random.uniform(min_val, max_val, size=shape)
    return data


def generate_rating_matrix(shape=(50, 10), min_rating=1, max_rating=5):
    """
    Rating matrisi oluşturur (öneri sistemleri için)
    
    Args:
        shape: Matris boyutu (n_users, n_items)
        min_rating: Minimum rating
        max_rating: Maksimum rating
    
    Returns:
        numpy array - rating matrisi
    """
    np.random.seed(42)
    # Bazı değerler NaN olsun (gerçekçi rating matrisi)
    rating_matrix = np.random.uniform(min_rating, max_rating, size=shape)
    # %30'u NaN yap (henüz rating verilmemiş)
    mask = np.random.random(shape) < 0.3
    rating_matrix[mask] = np.nan
    return rating_matrix


def generate_text_corpus(n_documents=50, n_words_per_doc=20):
    """
    Topic modeling için metin verisi oluşturur
    
    Args:
        n_documents: Doküman sayısı
        n_words_per_doc: Her dokümandaki kelime sayısı
    
    Returns:
        list - doküman listesi
    """
    np.random.seed(42)
    words = ['python', 'machine', 'learning', 'data', 'science', 'algorithm',
             'matrix', 'factorization', 'deep', 'neural', 'network', 'model',
             'training', 'prediction', 'analysis', 'visualization', 'recommendation',
             'system', 'user', 'item', 'rating', 'collaborative', 'filtering']
    
    documents = []
    for _ in range(n_documents):
        doc = ' '.join(np.random.choice(words, size=n_words_per_doc))
        documents.append(doc)
    
    return documents


def generate_user_sequences(n_users=20, n_items=10, seq_length=5):
    """
    Transformer için kullanıcı sequence'leri oluşturur
    
    Args:
        n_users: Kullanıcı sayısı
        n_items: Ürün sayısı
        seq_length: Sequence uzunluğu
    
    Returns:
        dict - {user_id: [item_id1, item_id2, ...]}
    """
    np.random.seed(42)
    sequences = {}
    for user_id in range(n_users):
        sequences[user_id] = np.random.randint(0, n_items, size=seq_length).tolist()
    return sequences


def test_algorithm(algorithm_name, algorithm_class, test_func, *args, **kwargs):
    """
    Bir algoritmayı test eder
    
    Args:
        algorithm_name: Algoritma adı (gösterim için)
        algorithm_class: Algoritma sınıfı
        test_func: Test fonksiyonu (lambda veya fonksiyon)
        *args, **kwargs: Test fonksiyonuna geçirilecek argümanlar
    """
    try:
        print(f"\n{'='*60}")
        print(f"🧪 TEST: {algorithm_name}")
        print(f"{'='*60}")
        
        # Algoritma instance'ını oluştur
        instance = algorithm_class()
        
        # Test fonksiyonunu çalıştır
        result = test_func(instance, *args, **kwargs)
        
        print(f"✅ {algorithm_name} BAŞARILI")
        if result is not None:
            print(f"   Sonuç tipi: {type(result).__name__}")
            if hasattr(result, 'shape'):
                print(f"   Sonuç boyutu: {result.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ {algorithm_name} HATA VERDİ")
        print(f"   Hata: {str(e)}")
        print(f"\n   Detaylı Traceback:")
        traceback.print_exc()
        return False


def main():
    """Ana test fonksiyonu"""
    print("="*60)
    print("🚀 MATRIX FAKTÖRİZASYON ALGORİTMALARI TEST SCRIPTİ")
    print("="*60)
    
    results = {
        'success': [],
        'failed': []
    }
    
    # Test verilerini oluştur
    print("\n📊 Test verileri oluşturuluyor...")
    positive_data = generate_positive_data(shape=(50, 10))
    rating_matrix = generate_rating_matrix(shape=(50, 10))
    text_corpus = generate_text_corpus(n_documents=50)
    user_sequences = generate_user_sequences(n_users=20, n_items=10)
    
    print(f"   ✅ Pozitif veri: {positive_data.shape}")
    print(f"   ✅ Rating matrisi: {rating_matrix.shape}")
    print(f"   ✅ Metin korpusu: {len(text_corpus)} doküman")
    print(f"   ✅ Kullanıcı sequence'leri: {len(user_sequences)} kullanıcı")
    
    # ========== KLASİK ALGORİTMALAR ==========
    if CLASSIC_AVAILABLE:
        print("\n" + "="*60)
        print("📊 KLASİK ALGORİTMALAR TEST EDİLİYOR")
        print("="*60)
        
        # Test konfigürasyonları
        classic_tests = [
            ("SVDRecommender", lambda alg, data: alg.fit(data), rating_matrix),
            ("SVDNoiseReducer", lambda alg, data: alg.fit(data), positive_data),
            ("PCAAnalyzer", lambda alg, data: alg.fit(data, standardize=True), positive_data),
            ("NMFImageProcessor", lambda alg, data: alg.fit(data), positive_data),
            ("NMFTopicModeler", lambda alg, docs: alg.fit(docs, max_features=100, min_df=1, max_df=0.95), text_corpus),
            ("ALSRecommender", lambda alg, data: alg.fit(data, implicit=False), rating_matrix),
        ]
        
        for alg_name, test_func, test_data in classic_tests:
            if alg_name in CLASSIC_ALGORITHMS:
                if test_algorithm(alg_name, CLASSIC_ALGORITHMS[alg_name], test_func, test_data):
                    results['success'].append(alg_name)
                else:
                    results['failed'].append(alg_name)
            else:
                print(f"⚠️ {alg_name} import edilemediği için test edilemedi")
                results['failed'].append(alg_name)
    
    # ========== MODERN ALGORİTMALAR (PyTorch) ==========
    if MODERN_AVAILABLE:
        print("\n" + "="*60)
        print("🚀 MODERN ALGORİTMALAR TEST EDİLİYOR (PyTorch)")
        print("="*60)
        
        # FM ve DeepFM için özel veri hazırla
        n_samples = 100
        user_ids = np.random.randint(0, 50, size=n_samples)
        item_ids = np.random.randint(0, 10, size=n_samples)
        ratings = np.random.uniform(1, 5, size=n_samples)
        
        # Test konfigürasyonları
        modern_tests = [
            ("NCFRecommender", lambda alg, data: alg.fit(data, epochs=2, batch_size=32, verbose=0), rating_matrix),
            ("DenoisingAutoencoder", lambda alg, data: alg.fit(data, epochs=2, batch_size=16, verbose=0), positive_data),
            ("VAERecommender", lambda alg, data: alg.fit(data, epochs=2, batch_size=32, verbose=0), rating_matrix),
            ("FactorizationMachine", lambda alg, u_ids, i_ids, r: alg.fit(u_ids, i_ids, r, epochs=2, batch_size=32, verbose=0), user_ids, item_ids, ratings),
            ("DeepFM", lambda alg, u_ids, i_ids, r: alg.fit(u_ids, i_ids, r, epochs=2, batch_size=32, verbose=0), user_ids, item_ids, ratings),
            ("TransformerRecommender", lambda alg, seqs: alg.fit(seqs, epochs=2, batch_size=16, verbose=0), user_sequences),
            ("GNNRecommender", lambda alg, data: alg.fit(data, epochs=2, verbose=False), rating_matrix),
        ]
        
        for test_config in modern_tests:
            alg_name = test_config[0]
            test_func = test_config[1]
            test_data = test_config[2:]  # Kalan argümanlar
            
            if alg_name in MODERN_ALGORITHMS:
                if test_algorithm(alg_name, MODERN_ALGORITHMS[alg_name], test_func, *test_data):
                    results['success'].append(alg_name)
                else:
                    results['failed'].append(alg_name)
            else:
                print(f"⚠️ {alg_name} import edilemediği için test edilemedi")
                results['failed'].append(alg_name)
    
    # ========== ÖZET ==========
    print("\n" + "="*60)
    print("📊 TEST ÖZETİ")
    print("="*60)
    print(f"✅ Başarılı: {len(results['success'])} algoritma")
    for name in results['success']:
        print(f"   ✓ {name}")
    
    print(f"\n❌ Başarısız: {len(results['failed'])} algoritma")
    for name in results['failed']:
        print(f"   ✗ {name}")
    
    total = len(results['success']) + len(results['failed'])
    success_rate = (len(results['success']) / total * 100) if total > 0 else 0
    print(f"\n📈 Başarı Oranı: {success_rate:.1f}% ({len(results['success'])}/{total})")
    print("="*60)


if __name__ == "__main__":
    main()
