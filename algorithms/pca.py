"""
Principal Component Analysis (PCA) Implementation
Kullanım: Veri görselleştirme, Özellik seçimi
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


class PCAAnalyzer:
    """
    PCA tabanlı veri analizi ve görselleştirme
    """
    
    def __init__(self, n_components=None, random_state=42):
        """
        Args:
            n_components: Kullanılacak bileşen sayısı (None ise tümü)
            random_state: Rastgelelik için seed
        """
        self.n_components = n_components
        self.random_state = random_state
        self.pca = None
        self.scaler = StandardScaler()
        self.explained_variance_ratio_ = None
        self.components_ = None
        
    def fit(self, X, standardize=True):
        """
        PCA modelini eğitir
        
        Args:
            X: Veri matrisi (n_samples x n_features)
            standardize: Veriyi standardize et (varsayılan: True)
        """
        # Veriyi standardize et
        if standardize:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
            self.scaler = None
        
        # PCA uygula
        self.pca = PCA(
            n_components=self.n_components,
            random_state=self.random_state
        )
        self.pca.fit(X_scaled)
        
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        self.components_ = self.pca.components_
        
        return self
    
    def transform(self, X):
        """
        Veriyi düşük boyutlu uzaya dönüştürür
        
        Args:
            X: Veri matrisi
            
        Returns:
            Dönüştürülmüş veri (n_samples x n_components)
        """
        if self.pca is None:
            raise ValueError("Model henüz eğitilmemiş. Önce fit() çağrılmalı.")
        
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return self.pca.transform(X_scaled)
    
    def fit_transform(self, X, standardize=True):
        """
        Fit ve transform işlemlerini birlikte yapar
        
        Args:
            X: Veri matrisi
            standardize: Veriyi standardize et
            
        Returns:
            Dönüştürülmüş veri
        """
        self.fit(X, standardize)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed):
        """
        Dönüştürülmüş veriyi orijinal boyuta geri getirir
        
        Args:
            X_transformed: Dönüştürülmüş veri
            
        Returns:
            Orijinal boyutta veri
        """
        if self.pca is None:
            raise ValueError("Model henüz eğitilmemiş.")
        
        X_reconstructed = self.pca.inverse_transform(X_transformed)
        
        if self.scaler is not None:
            X_reconstructed = self.scaler.inverse_transform(X_reconstructed)
        
        return X_reconstructed
    
    def get_optimal_components(self, X, variance_threshold=0.95):
        """
        Belirli bir varyans eşiğini korumak için gerekli bileşen sayısını bulur
        
        Args:
            X: Veri matrisi
            variance_threshold: Korunması istenen varyans oranı (0-1)
            
        Returns:
            Optimal bileşen sayısı
        """
        if self.pca is None:
            self.fit(X)
        
        cumulative_variance = np.cumsum(self.explained_variance_ratio_)
        optimal = np.argmax(cumulative_variance >= variance_threshold) + 1
        
        return optimal
    
    def get_feature_importance(self, feature_names=None):
        """
        Her özellik için PCA bileşenlerindeki önemini hesaplar
        
        Args:
            feature_names: Özellik isimleri listesi (opsiyonel)
            
        Returns:
            Özellik önem skorları
        """
        if self.components_ is None:
            raise ValueError("Model henüz eğitilmemiş.")
        
        # İlk birkaç bileşendeki mutlak yükleri topla
        importance = np.abs(self.components_[:min(3, len(self.components_))]).sum(axis=0)
        importance = importance / importance.sum()  # Normalize et
        
        if feature_names is not None:
            return dict(zip(feature_names, importance))
        
        return importance
    
    def plot_explained_variance(self, n_components=None, figsize=(10, 6)):
        """
        Açıklanan varyans grafiğini çizer
        
        Args:
            n_components: Gösterilecek maksimum bileşen sayısı
            figsize: Grafik boyutu
            
        Returns:
            matplotlib figure
        """
        if self.explained_variance_ratio_ is None:
            raise ValueError("Model henüz eğitilmemiş.")
        
        if n_components is None:
            n_components = len(self.explained_variance_ratio_)
        
        n_components = min(n_components, len(self.explained_variance_ratio_))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Bireysel varyans
        ax1.bar(range(1, n_components + 1), 
                self.explained_variance_ratio_[:n_components])
        ax1.set_xlabel('Bileşen')
        ax1.set_ylabel('Açıklanan Varyans Oranı')
        ax1.set_title('Bileşen Bazında Açıklanan Varyans')
        ax1.grid(True, alpha=0.3)
        
        # Kümülatif varyans
        cumulative = np.cumsum(self.explained_variance_ratio_[:n_components])
        ax2.plot(range(1, n_components + 1), cumulative, 'o-')
        ax2.axhline(y=0.95, color='r', linestyle='--', label='95% Eşiği')
        ax2.set_xlabel('Bileşen Sayısı')
        ax2.set_ylabel('Kümülatif Açıklanan Varyans')
        ax2.set_title('Kümülatif Açıklanan Varyans')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_2d_projection(self, X, y=None, labels=None, figsize=(10, 8)):
        """
        Veriyi 2D uzaya izdüşürür ve görselleştirir
        
        Args:
            X: Veri matrisi
            y: Sınıf etiketleri (opsiyonel, renklendirme için)
            labels: Veri noktaları için etiketler (opsiyonel)
            figsize: Grafik boyutu
            
        Returns:
            matplotlib figure
        """
        # İlk 2 bileşene izdüşür
        X_2d = self.transform(X)[:, :2]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if y is not None:
            scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, 
                               cmap='viridis', alpha=0.6, s=50)
            plt.colorbar(scatter, ax=ax)
        else:
            ax.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.6, s=50)
        
        ax.set_xlabel(f'PC1 ({self.explained_variance_ratio_[0]:.2%} varyans)')
        ax.set_ylabel(f'PC2 ({self.explained_variance_ratio_[1]:.2%} varyans)')
        ax.set_title('PCA 2D İzdüşümü')
        ax.grid(True, alpha=0.3)
        
        if labels is not None:
            for i, label in enumerate(labels):
                ax.annotate(label, (X_2d[i, 0], X_2d[i, 1]), 
                           fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    def plot_3d_projection(self, X, y=None, figsize=(12, 10)):
        """
        Veriyi 3D uzaya izdüşürür ve görselleştirir
        
        Args:
            X: Veri matrisi
            y: Sınıf etiketleri (opsiyonel)
            figsize: Grafik boyutu
            
        Returns:
            matplotlib figure
        """
        from mpl_toolkits.mplot3d import Axes3D
        
        # İlk 3 bileşene izdüşür
        X_3d = self.transform(X)[:, :3]
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        if y is not None:
            scatter = ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], 
                               c=y, cmap='viridis', alpha=0.6, s=50)
            plt.colorbar(scatter, ax=ax)
        else:
            ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], 
                      alpha=0.6, s=50)
        
        ax.set_xlabel(f'PC1 ({self.explained_variance_ratio_[0]:.2%})')
        ax.set_ylabel(f'PC2 ({self.explained_variance_ratio_[1]:.2%})')
        ax.set_zlabel(f'PC3 ({self.explained_variance_ratio_[2]:.2%})')
        ax.set_title('PCA 3D İzdüşümü')
        
        plt.tight_layout()
        return fig

