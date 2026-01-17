"""
Singular Value Decomposition (SVD) Implementation
Kullanım: Öneri sistemleri, Gürültü temizleme
"""

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class SVDRecommender:
    """
    SVD tabanlı öneri sistemi
    """
    
    def __init__(self, n_components=50, random_state=42):
        """
        Args:
            n_components: Kullanılacak tekil değer sayısı
            random_state: Rastgelelik için seed
        """
        self.n_components = n_components
        self.random_state = random_state
        self.svd = None
        self.user_factors = None
        self.item_factors = None
        self.mean_rating = None
        self.source_topic_matrix = None  # Orijinal pivot table matrisi (NaN'lerle)
        self.filled_matrix = None  # NaN'ler doldurulmuş matris
        
    def fit(self, rating_matrix, fill_na_with_mean=True):
        """
        Rating matrisini SVD ile faktörize eder
        
        Args:
            rating_matrix: Kullanıcı-ürün rating matrisi (n_users x n_items)
            fill_na_with_mean: NaN değerleri ortalama rating ile doldur (varsayılan: True)
        """
        # Orijinal matrisi sakla (NaN'lerle)
        if isinstance(rating_matrix, pd.DataFrame):
            self.source_topic_matrix = rating_matrix.copy()
            rating_matrix = rating_matrix.values
        else:
            self.source_topic_matrix = rating_matrix.copy() if not hasattr(rating_matrix, 'toarray') else rating_matrix.toarray()
        
        # Büyük matrisler için sparse matrix kullan
        is_sparse = hasattr(rating_matrix, 'toarray') or isinstance(rating_matrix, (csr_matrix, csc_matrix))
        
        if is_sparse:
            # Sparse matrix ise dense'e çevir (NaN doldurma için)
            rating_matrix = rating_matrix.toarray()
            is_sparse = False
        
        # Ortalama rating'i hesapla (sadece mevcut değerler üzerinden)
        mask = ~np.isnan(rating_matrix)
        if np.sum(mask) > 0:
            self.mean_rating = np.mean(rating_matrix[mask])
        else:
            self.mean_rating = 0.0
        
        # NaN değerleri ortalama rating ile doldur (eğer isteniyorsa)
        if fill_na_with_mean:
            self.filled_matrix = rating_matrix.copy()
            self.filled_matrix[~mask] = self.mean_rating
        else:
            self.filled_matrix = rating_matrix.copy()
            # NaN'leri 0 ile doldur (SVD için gerekli)
            self.filled_matrix[~mask] = 0.0
        
        # Matrisi center et (ortalama rating'i çıkar)
        matrix_centered = self.filled_matrix - self.mean_rating
        
        # Truncated SVD kullan (büyük matrisler için)
        self.svd = TruncatedSVD(
            n_components=self.n_components,
            random_state=self.random_state
        )
        
        # Faktörize et - fit_transform() ile latent matrix oluştur
        self.user_factors = self.svd.fit_transform(matrix_centered)
        self.item_factors = self.svd.components_.T
        
        return self
    
    def predict(self, user_idx, item_idx):
        """
        Belirli bir kullanıcı-ürün çifti için rating tahmin eder
        
        Args:
            user_idx: Kullanıcı indeksi
            item_idx: Ürün indeksi
            
        Returns:
            Tahmin edilen rating
        """
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model henüz eğitilmemiş. Önce fit() çağrılmalı.")
        
        prediction = np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        return prediction + self.mean_rating
    
    def predict_all(self):
        """
        Tüm kullanıcı-ürün çiftleri için rating tahmin eder
        
        Returns:
            Tahmin matrisi (n_users x n_items)
        """
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model henüz eğitilmemiş. Önce fit() çağrılmalı.")
        
        predictions = np.dot(self.user_factors, self.item_factors.T)
        return predictions + self.mean_rating
    
    def evaluate(self, test_matrix):
        """
        Test seti üzerinde model performansını değerlendirir
        
        Args:
            test_matrix: Test rating matrisi
            
        Returns:
            RMSE değeri
        """
        predictions = self.predict_all()
        mask = ~np.isnan(test_matrix)
        rmse = np.sqrt(mean_squared_error(
            test_matrix[mask],
            predictions[mask]
        ))
        return rmse
    
    def get_singular_values(self):
        """
        Tekil değerleri döndürür
        
        Returns:
            Tekil değerler dizisi
        """
        if self.svd is None:
            raise ValueError("Model henüz eğitilmemiş.")
        return self.svd.singular_values_
    
    def get_latent_matrix(self):
        """
        Latent matrix'i (kaynakların gizli özellik skorları) döndürür
        Bu, fit_transform() ile oluşturulan matristir.
        
        Returns:
            Latent matrix (n_users x n_components)
        """
        if self.user_factors is None:
            raise ValueError("Model henüz eğitilmemiş. Önce fit() çağrılmalı.")
        return self.user_factors
    
    def get_user_similarity(self, metric='cosine'):
        """
        Kaynaklar (kullanıcılar) arası benzerlik matrisini hesaplar
        Latent matrix üzerinden kosinüs benzerliği kullanır.
        
        Args:
            metric: Benzerlik metriği ('cosine' varsayılan)
            
        Returns:
            Benzerlik matrisi (n_users x n_users)
        """
        if self.user_factors is None:
            raise ValueError("Model henüz eğitilmemiş. Önce fit() çağrılmalı.")
        
        if metric == 'cosine':
            similarity_matrix = cosine_similarity(self.user_factors)
        else:
            raise ValueError(f"Desteklenmeyen metrik: {metric}. Sadece 'cosine' destekleniyor.")
        
        return similarity_matrix
    
    def get_item_similarity(self, metric='cosine'):
        """
        Konular (ürünler) arası benzerlik matrisini hesaplar
        svd.components_ üzerinden kosinüs benzerliği kullanır.
        
        Args:
            metric: Benzerlik metriği ('cosine' varsayılan)
            
        Returns:
            Benzerlik matrisi (n_items x n_items)
        """
        if self.svd is None:
            raise ValueError("Model henüz eğitilmemiş. Önce fit() çağrılmalı.")
        
        if metric == 'cosine':
            # components_ shape: (n_components x n_items)
            similarity_matrix = cosine_similarity(self.svd.components_.T)
        else:
            raise ValueError(f"Desteklenmeyen metrik: {metric}. Sadece 'cosine' destekleniyor.")
        
        return similarity_matrix
    
    def get_predicted_scores(self, as_dataframe=False, user_ids=None, item_ids=None):
        """
        Tahmini puanları döndürür (latent_matrix @ svd.components_ ile yeniden oluşturulmuş)
        Bu matris, orijinalde NaN olan hücreler için SVD modelinin tahmin ettiği puanları içerir.
        
        Args:
            as_dataframe: DataFrame olarak döndür (varsayılan: False)
            user_ids: Kullanıcı ID'leri (DataFrame için)
            item_ids: Ürün ID'leri (DataFrame için)
            
        Returns:
            Tahmin matrisi (numpy array veya DataFrame)
        """
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model henüz eğitilmemiş. Önce fit() çağrılmalı.")
        
        # latent_matrix @ svd.components_ ile yeniden oluştur
        # user_factors shape: (n_users x n_components)
        # item_factors shape: (n_items x n_components)
        # components_ shape: (n_components x n_items)
        predicted_scores = np.dot(self.user_factors, self.svd.components_)
        
        # Ortalama rating'i geri ekle
        predicted_scores = predicted_scores + self.mean_rating
        
        if as_dataframe:
            if user_ids is None:
                user_ids = [f"User_{i+1}" for i in range(predicted_scores.shape[0])]
            if item_ids is None:
                item_ids = [f"Item_{i+1}" for i in range(predicted_scores.shape[1])]
            
            return pd.DataFrame(
                predicted_scores,
                index=user_ids,
                columns=item_ids
            )
        
        return predicted_scores


class SVDNoiseReducer:
    """
    SVD kullanarak gürültü temizleme
    """
    
    def __init__(self, n_components=None, threshold=0.95):
        """
        Args:
            n_components: Kullanılacak tekil değer sayısı (None ise otomatik)
            threshold: Varyansın korunması için eşik değeri (0-1)
        """
        self.n_components = n_components
        self.threshold = threshold
        self.svd = None
        self.singular_values = None
        
    def fit(self, data_matrix):
        """
        Veri matrisini analiz eder ve optimal bileşen sayısını belirler
        
        Args:
            data_matrix: Temizlenecek veri matrisi
        """
        # Full SVD hesapla
        U, s, Vt = np.linalg.svd(data_matrix, full_matrices=False)
        self.singular_values = s
        
        # Eğer n_components belirtilmemişse, threshold'a göre belirle
        if self.n_components is None:
            cumulative_variance = np.cumsum(s**2) / np.sum(s**2)
            self.n_components = np.argmax(cumulative_variance >= self.threshold) + 1
        
        # Truncated SVD ile yeniden hesapla (daha hızlı)
        self.svd = TruncatedSVD(n_components=self.n_components)
        self.svd.fit(data_matrix)
        
        return self
    
    def denoise(self, data_matrix):
        """
        Veri matrisindeki gürültüyü temizler
        
        Args:
            data_matrix: Temizlenecek veri matrisi
            
        Returns:
            Temizlenmiş veri matrisi
        """
        if self.svd is None:
            self.fit(data_matrix)
        
        # Düşük rank yaklaşımı
        transformed = self.svd.transform(data_matrix)
        denoised = self.svd.inverse_transform(transformed)
        
        return denoised
    
    def get_noise_reduction_ratio(self):
        """
        Gürültü azaltma oranını hesaplar
        
        Returns:
            Gürültü azaltma oranı (0-1)
        """
        if self.singular_values is None:
            raise ValueError("Model henüz eğitilmemiş.")
        
        total_variance = np.sum(self.singular_values**2)
        retained_variance = np.sum(self.singular_values[:self.n_components]**2)
        
        return retained_variance / total_variance
    
    def get_optimal_components(self, data_matrix):
        """
        Farklı bileşen sayıları için varyans korunma oranlarını hesaplar
        
        Args:
            data_matrix: Veri matrisi
            
        Returns:
            (component_counts, variance_ratios) tuple
        """
        U, s, Vt = np.linalg.svd(data_matrix, full_matrices=False)
        total_variance = np.sum(s**2)
        
        component_counts = np.arange(1, min(len(s), 100) + 1)
        variance_ratios = []
        
        for k in component_counts:
            retained_variance = np.sum(s[:k]**2)
            variance_ratios.append(retained_variance / total_variance)
        
        return component_counts, np.array(variance_ratios)

