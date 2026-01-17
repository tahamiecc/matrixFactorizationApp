"""
Alternating Least Squares (ALS) Implementation
Kullanım: Büyük ölçekli öneri motorları
"""

import numpy as np
from sklearn.metrics import mean_squared_error
import warnings
import scipy.sparse
warnings.filterwarnings('ignore')


class ALSRecommender:
    """
    ALS tabanlı öneri sistemi
    Büyük ölçekli sistemler için optimize edilmiş
    """
    
    def __init__(self, n_factors=50, regularization=0.1, 
                 iterations=15, random_state=42, alpha=40):
        """
        Args:
            n_factors: Latent faktör sayısı
            regularization: Regularizasyon parametresi (lambda)
            iterations: Alternatif optimizasyon iterasyon sayısı
            random_state: Rastgelelik için seed
            alpha: Güven ağırlığı (implicit feedback için)
        """
        self.n_factors = n_factors
        self.regularization = regularization
        self.iterations = iterations
        self.random_state = random_state
        self.alpha = alpha
        
        self.user_factors = None
        self.item_factors = None
        self.user_biases = None
        self.item_biases = None
        self.global_mean = None
        
    def _init_factors(self, n_users, n_items):
        """Faktör matrislerini başlat"""
        np.random.seed(self.random_state)
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)
    
    def fit(self, rating_matrix, implicit=False):
        """
        Rating matrisini ALS ile faktörize eder
        
        Args:
            rating_matrix: Kullanıcı-ürün rating matrisi (n_users x n_items)
                          Dense numpy array, sparse matrix veya numpy matrix olabilir
            implicit: Implicit feedback kullan (True) veya explicit (False)
        """
        # Sparse matrix desteği
        is_sparse = scipy.sparse.issparse(rating_matrix)
        
        if is_sparse:
            # Sparse matrix'i dense'e çevir (ALS için gerekli)
            # Büyük matrisler için uyarı verilebilir ama şimdilik dense kullanıyoruz
            n_users, n_items = rating_matrix.shape
            # Sparse matrix'i dense numpy array'e çevir
            if hasattr(rating_matrix, 'toarray'):
                rating_matrix = rating_matrix.toarray()
            elif hasattr(rating_matrix, 'A'):
                rating_matrix = np.array(rating_matrix.A)
            else:
                rating_matrix = np.array(rating_matrix)
            # Sparse matrix'te 0 değerleri eksik veriyi temsil eder, NaN'a çevir
            rating_matrix = np.where(rating_matrix == 0, np.nan, rating_matrix)
            is_sparse = False  # Artık dense oldu
        else:
            # Dense matrix veya numpy matrix
            if hasattr(rating_matrix, 'A'):
                # numpy.matrix objesi
                rating_matrix = np.array(rating_matrix.A)
            else:
                rating_matrix = np.array(rating_matrix)
        
        # Shape kontrolü
        if rating_matrix.ndim != 2:
            raise ValueError(f"Rating matrix 2 boyutlu olmalı, {rating_matrix.ndim} boyutlu alındı")
        
        n_users, n_items = rating_matrix.shape
        
        if n_users == 0 or n_items == 0:
            raise ValueError(f"Rating matrix boş olamaz: shape = {rating_matrix.shape}")
        
        # Global ortalama
        self.global_mean = np.nanmean(rating_matrix)
        
        # Faktörleri başlat
        self._init_factors(n_users, n_items)
        
        # Implicit feedback için confidence matrisi
        if implicit:
            # Rating varsa yüksek güven, yoksa düşük
            confidence = np.where(np.isnan(rating_matrix), 1, 1 + self.alpha * rating_matrix)
            preference = np.where(np.isnan(rating_matrix), 0, 1)
        else:
            # Explicit feedback - sadece mevcut rating'leri kullan
            confidence = np.where(np.isnan(rating_matrix), 0, 1)
            preference = np.where(np.isnan(rating_matrix), 0, rating_matrix)
        
        # Alternatif optimizasyon
        for iteration in range(self.iterations):
            # User faktörlerini güncelle
            for u in range(n_users):
                # Bu kullanıcının rating verdiği item'lar
                rated_items = ~np.isnan(rating_matrix[u, :])
                
                if np.sum(rated_items) == 0:
                    continue
                
                if implicit:
                    # Implicit feedback için weighted ALS
                    Cu = np.diag(confidence[u, rated_items])
                    Pu = preference[u, rated_items]
                    items_f = self.item_factors[rated_items, :]
                    
                    # A = items_f^T * Cu * items_f + lambda * I
                    A = items_f.T @ Cu @ items_f + self.regularization * np.eye(self.n_factors)
                    # b = items_f^T * Cu * Pu
                    b = items_f.T @ Cu @ Pu
                    
                    self.user_factors[u] = np.linalg.solve(A, b)
                else:
                    # Explicit feedback için standart ALS
                    items_f = self.item_factors[rated_items, :]
                    ratings_u = rating_matrix[u, rated_items]
                    
                    # Bias'ları çıkar
                    ratings_u_centered = ratings_u - self.global_mean - self.item_biases[rated_items]
                    
                    # A = items_f^T * items_f + lambda * I
                    A = items_f.T @ items_f + self.regularization * np.eye(self.n_factors)
                    # b = items_f^T * ratings
                    b = items_f.T @ ratings_u_centered
                    
                    self.user_factors[u] = np.linalg.solve(A, b)
            
            # Item faktörlerini güncelle
            for i in range(n_items):
                # Bu item'a rating veren kullanıcılar
                rated_by = ~np.isnan(rating_matrix[:, i])
                
                if np.sum(rated_by) == 0:
                    continue
                
                if implicit:
                    # Implicit feedback için weighted ALS
                    Ci = np.diag(confidence[rated_by, i])
                    Pi = preference[rated_by, i]
                    users_f = self.user_factors[rated_by, :]
                    
                    # A = users_f^T * Ci * users_f + lambda * I
                    A = users_f.T @ Ci @ users_f + self.regularization * np.eye(self.n_factors)
                    # b = users_f^T * Ci * Pi
                    b = users_f.T @ Ci @ Pi
                    
                    self.item_factors[i] = np.linalg.solve(A, b)
                else:
                    # Explicit feedback için standart ALS
                    users_f = self.user_factors[rated_by, :]
                    ratings_i = rating_matrix[rated_by, i]
                    
                    # Bias'ları çıkar
                    ratings_i_centered = ratings_i - self.global_mean - self.user_biases[rated_by]
                    
                    # A = users_f^T * users_f + lambda * I
                    A = users_f.T @ users_f + self.regularization * np.eye(self.n_factors)
                    # b = users_f^T * ratings
                    b = users_f.T @ ratings_i_centered
                    
                    self.item_factors[i] = np.linalg.solve(A, b)
            
            # Bias'ları güncelle (basit ortalama)
            if not implicit:
                for u in range(n_users):
                    rated_items = ~np.isnan(rating_matrix[u, :])
                    if np.sum(rated_items) > 0:
                        residuals = (rating_matrix[u, rated_items] - 
                                   self.global_mean - 
                                   self.item_biases[rated_items] -
                                   np.dot(self.user_factors[u], 
                                         self.item_factors[rated_items, :].T))
                        self.user_biases[u] = np.mean(residuals)
                
                for i in range(n_items):
                    rated_by = ~np.isnan(rating_matrix[:, i])
                    if np.sum(rated_by) > 0:
                        residuals = (rating_matrix[rated_by, i] - 
                                   self.global_mean - 
                                   self.user_biases[rated_by] -
                                   np.dot(self.user_factors[rated_by, :], 
                                         self.item_factors[i]))
                        self.item_biases[i] = np.mean(residuals)
        
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
            raise ValueError("Model henüz eğitilmemiş.")
        
        prediction = (self.global_mean + 
                     self.user_biases[user_idx] + 
                     self.item_biases[item_idx] +
                     np.dot(self.user_factors[user_idx], 
                           self.item_factors[item_idx]))
        
        return prediction
    
    def predict_all(self):
        """
        Tüm kullanıcı-ürün çiftleri için rating tahmin eder
        
        Returns:
            Tahmin matrisi (n_users x n_items)
        """
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model henüz eğitilmemiş.")
        
        predictions = (self.global_mean + 
                      self.user_biases[:, np.newaxis] + 
                      self.item_biases[np.newaxis, :] +
                      np.dot(self.user_factors, self.item_factors.T))
        
        return predictions
    
    def recommend(self, user_idx, n_recommendations=10, exclude_rated=True, rating_matrix=None):
        """
        Bir kullanıcı için öneriler üretir
        
        Args:
            user_idx: Kullanıcı indeksi
            n_recommendations: Önerilecek item sayısı
            exclude_rated: Zaten rating verilen item'ları hariç tut
            rating_matrix: Orijinal rating matrisi (exclude_rated için gerekli)
            
        Returns:
            (item_indices, predicted_ratings) tuple
        """
        predictions = self.predict_all()[user_idx]
        
        if exclude_rated and rating_matrix is not None:
            # Zaten rating verilen item'ları maskele
            rated_items = ~np.isnan(rating_matrix[user_idx, :])
            predictions[rated_items] = -np.inf
        
        # En yüksek skorlu item'ları seç
        top_indices = np.argsort(predictions)[::-1][:n_recommendations]
        top_ratings = predictions[top_indices]
        
        return top_indices, top_ratings
    
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
    
    def get_similar_items(self, item_idx, n_similar=10):
        """
        Benzer item'ları bulur (cosine similarity)
        
        Args:
            item_idx: Item indeksi
            n_similar: Bulunacak benzer item sayısı
            
        Returns:
            (similar_item_indices, similarity_scores) tuple
        """
        if self.item_factors is None:
            raise ValueError("Model henüz eğitilmemiş.")
        
        # Cosine similarity
        item_vector = self.item_factors[item_idx]
        similarities = np.dot(self.item_factors, item_vector) / (
            np.linalg.norm(self.item_factors, axis=1) * 
            np.linalg.norm(item_vector)
        )
        
        # Kendisini hariç tut
        similarities[item_idx] = -1
        
        # En benzer item'ları bul
        top_indices = np.argsort(similarities)[::-1][:n_similar]
        top_similarities = similarities[top_indices]
        
        return top_indices, top_similarities
    
    def get_similar_users(self, user_idx, n_similar=10):
        """
        Benzer kullanıcıları bulur (cosine similarity)
        
        Args:
            user_idx: Kullanıcı indeksi
            n_similar: Bulunacak benzer kullanıcı sayısı
            
        Returns:
            (similar_user_indices, similarity_scores) tuple
        """
        if self.user_factors is None:
            raise ValueError("Model henüz eğitilmemiş.")
        
        # Cosine similarity
        user_vector = self.user_factors[user_idx]
        similarities = np.dot(self.user_factors, user_vector) / (
            np.linalg.norm(self.user_factors, axis=1) * 
            np.linalg.norm(user_vector)
        )
        
        # Kendisini hariç tut
        similarities[user_idx] = -1
        
        # En benzer kullanıcıları bul
        top_indices = np.argsort(similarities)[::-1][:n_similar]
        top_similarities = similarities[top_indices]
        
        return top_indices, top_similarities

