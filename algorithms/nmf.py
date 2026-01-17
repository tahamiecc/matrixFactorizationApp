"""
Non-negative Matrix Factorization (NMF) Implementation
Kullanım: Görüntü işleme, Metin madenciliği (Topic Modeling)
"""

import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')


class NMFImageProcessor:
    """
    NMF kullanarak görüntü işleme ve analiz
    """
    
    def __init__(self, n_components=10, random_state=42, max_iter=200):
        """
        Args:
            n_components: Kullanılacak bileşen sayısı
            random_state: Rastgelelik için seed
            max_iter: Maksimum iterasyon sayısı
        """
        self.n_components = n_components
        self.random_state = random_state
        self.max_iter = max_iter
        self.nmf = None
        self.W = None  # Basis matrisi
        self.H = None  # Coefficient matrisi
        
    def fit(self, image_matrix):
        """
        Görüntü matrisini NMF ile faktörize eder
        
        Args:
            image_matrix: Görüntü matrisi (n_images x n_pixels)
                          Her satır bir görüntüyü temsil eder
        """
        # Negatif değerleri 0 yap (NMF için gerekli)
        image_matrix = np.maximum(image_matrix, 0)
        
        self.nmf = NMF(
            n_components=self.n_components,
            random_state=self.random_state,
            max_iter=self.max_iter,
            init='random'
        )
        
        self.W = self.nmf.fit_transform(image_matrix)
        self.H = self.nmf.components_
        
        return self
    
    def reconstruct(self, image_idx=None):
        """
        Görüntüleri yeniden oluşturur
        
        Args:
            image_idx: Belirli bir görüntü indeksi (None ise tümü)
            
        Returns:
            Yeniden oluşturulmuş görüntü(ler)
        """
        if self.W is None or self.H is None:
            raise ValueError("Model henüz eğitilmemiş.")
        
        if image_idx is not None:
            reconstructed = np.dot(self.W[image_idx:image_idx+1], self.H)
            return reconstructed[0]
        else:
            return np.dot(self.W, self.H)
    
    def get_basis_images(self, image_shape):
        """
        Temel görüntüleri (basis images) döndürür
        
        Args:
            image_shape: Orijinal görüntü boyutu (height, width)
            
        Returns:
            Temel görüntüler (n_components x height x width)
        """
        if self.H is None:
            raise ValueError("Model henüz eğitilmemiş.")
        
        basis_images = self.H.reshape(self.n_components, *image_shape)
        return basis_images
    
    def compress_image(self, image, image_shape):
        """
        Bir görüntüyü sıkıştırır
        
        Args:
            image: Sıkıştırılacak görüntü (1D array)
            image_shape: Görüntü boyutu (height, width)
            
        Returns:
            Sıkıştırılmış görüntü
        """
        if self.nmf is None:
            raise ValueError("Model henüz eğitilmemiş.")
        
        # Görüntüyü düzleştir ve negatif değerleri temizle
        image_flat = np.maximum(image.flatten(), 0)
        
        # NMF ile dönüştür
        w = self.nmf.transform(image_flat.reshape(1, -1))
        
        # Yeniden oluştur
        compressed = np.dot(w, self.H)
        
        return compressed.reshape(image_shape)
    
    def get_compression_ratio(self, original_shape):
        """
        Sıkıştırma oranını hesaplar
        
        Args:
            original_shape: Orijinal görüntü boyutu
            
        Returns:
            Sıkıştırma oranı
        """
        original_size = np.prod(original_shape)
        compressed_size = self.n_components * (original_shape[0] + original_shape[1])
        return original_size / compressed_size


class NMFTopicModeler:
    """
    NMF kullanarak topic modeling
    """
    
    def __init__(self, n_topics=5, random_state=42, max_iter=200):
        """
        Args:
            n_topics: Bulunacak topic sayısı
            random_state: Rastgelelik için seed
            max_iter: Maksimum iterasyon sayısı
        """
        self.n_topics = n_topics
        self.random_state = random_state
        self.max_iter = max_iter
        self.nmf = None
        self.vectorizer = None
        self.feature_names = None
        self.W = None  # Document-topic matrisi
        self.H = None  # Topic-word matrisi
        
    def fit(self, documents, max_features=1000, min_df=2, max_df=0.95):
        """
        Dokümanlardan topic'leri çıkarır
        
        Args:
            documents: Metin dokümanları listesi
            max_features: Kullanılacak maksimum kelime sayısı
            min_df: Minimum doküman frekansı
            max_df: Maksimum doküman frekansı
        """
        # TF-IDF vektörizasyonu
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            stop_words='english'  # İngilizce stop words
        )
        
        X = self.vectorizer.fit_transform(documents)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # NMF uygula
        self.nmf = NMF(
            n_components=self.n_topics,
            random_state=self.random_state,
            max_iter=self.max_iter,
            init='random'
        )
        
        self.W = self.nmf.fit_transform(X)
        self.H = self.nmf.components_
        
        return self
    
    def get_topics(self, n_words=10):
        """
        Her topic için en önemli kelimeleri döndürür
        
        Args:
            n_words: Her topic için gösterilecek kelime sayısı
            
        Returns:
            Topic'ler ve kelimeleri içeren dictionary
        """
        if self.H is None or self.feature_names is None:
            raise ValueError("Model henüz eğitilmemiş.")
        
        topics = {}
        for topic_idx in range(self.n_topics):
            # En yüksek ağırlıklı kelimeleri bul
            top_indices = self.H[topic_idx].argsort()[-n_words:][::-1]
            top_words = [(self.feature_names[i], self.H[topic_idx][i]) 
                        for i in top_indices]
            topics[f'Topic {topic_idx + 1}'] = top_words
        
        return topics
    
    def get_document_topics(self, document_idx=None):
        """
        Dokümanların topic dağılımını döndürür
        
        Args:
            document_idx: Belirli bir doküman indeksi (None ise tümü)
            
        Returns:
            Doküman-topic dağılımı
        """
        if self.W is None:
            raise ValueError("Model henüz eğitilmemiş.")
        
        if document_idx is not None:
            # Normalize et (olasılık dağılımı gibi)
            topic_dist = self.W[document_idx]
            topic_dist = topic_dist / topic_dist.sum()
            return topic_dist
        else:
            # Tüm dokümanlar için normalize et
            topic_dist = self.W / self.W.sum(axis=1, keepdims=True)
            return topic_dist
    
    def predict_topic(self, new_document):
        """
        Yeni bir doküman için en uygun topic'i tahmin eder
        
        Args:
            new_document: Yeni doküman metni
            
        Returns:
            En uygun topic indeksi ve skorları
        """
        if self.nmf is None or self.vectorizer is None:
            raise ValueError("Model henüz eğitilmemiş.")
        
        # Dokümanı vektörize et
        X_new = self.vectorizer.transform([new_document])
        
        # Topic dağılımını hesapla
        w_new = self.nmf.transform(X_new)
        w_new = w_new / w_new.sum()  # Normalize et
        
        # En yüksek skorlu topic'i bul
        top_topic = np.argmax(w_new[0])
        top_score = w_new[0][top_topic]
        
        return top_topic, w_new[0]
    
    def get_topic_keywords(self, topic_idx, n_words=10):
        """
        Belirli bir topic için anahtar kelimeleri döndürür
        
        Args:
            topic_idx: Topic indeksi
            n_words: Gösterilecek kelime sayısı
            
        Returns:
            (kelime, skor) tuple'ları listesi
        """
        if self.H is None or self.feature_names is None:
            raise ValueError("Model henüz eğitilmemiş.")
        
        top_indices = self.H[topic_idx].argsort()[-n_words:][::-1]
        keywords = [(self.feature_names[i], self.H[topic_idx][i]) 
                   for i in top_indices]
        
        return keywords
    
    def get_topic_coherence(self):
        """
        Topic tutarlılığını hesaplar (basit bir ölçüm)
        
        Returns:
            Ortalama topic tutarlılığı
        """
        if self.H is None:
            raise ValueError("Model henüz eğitilmemiş.")
        
        # Her topic için en yüksek skorlu kelimelerin ortalaması
        top_scores = []
        for topic_idx in range(self.n_topics):
            top_10_scores = np.sort(self.H[topic_idx])[-10:]
            top_scores.append(np.mean(top_10_scores))
        
        return np.mean(top_scores)

