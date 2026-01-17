"""
Veri Görselleştirme Örneği
PCA kullanarak veri görselleştirme ve özellik seçimi
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.pca import PCAAnalyzer
from utils.data_loader import generate_sample_data


def example_pca_visualization():
    """PCA ile veri görselleştirme örneği"""
    print("=" * 60)
    print("PCA ile Veri Görselleştirme")
    print("=" * 60)
    
    # Örnek veri oluştur
    print("\n1. Örnek veri oluşturuluyor...")
    X, y = generate_sample_data(n_samples=500, n_features=50, n_clusters=5)
    print(f"   Veri boyutu: {X.shape}")
    print(f"   Sınıf sayısı: {len(np.unique(y))}")
    
    # PCA uygula
    print("\n2. PCA uygulanıyor...")
    pca = PCAAnalyzer(n_components=None)
    X_transformed = pca.fit_transform(X)
    
    # Açıklanan varyans
    print("\n3. Açıklanan varyans analizi...")
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_95 = pca.get_optimal_components(X, variance_threshold=0.95)
    print(f"   Toplam bileşen sayısı: {len(pca.explained_variance_ratio_)}")
    print(f"   İlk 5 bileşenin açıkladığı varyans: "
          f"{np.sum(pca.explained_variance_ratio_[:5]):.4f}")
    print(f"   %95 varyans için gerekli bileşen sayısı: {n_95}")
    
    # Görselleştirme
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Açıklanan varyans grafiği
    ax1 = fig.add_subplot(gs[0, :])
    pca.plot_explained_variance(n_components=20, figsize=(12, 4))
    fig.add_subplot(gs[0, :])
    
    # 2. 2D izdüşüm
    ax2 = fig.add_subplot(gs[1, 0])
    X_2d = X_transformed[:, :2]
    scatter = ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis', alpha=0.6)
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    ax2.set_title('2D İzdüşüm')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2)
    
    # 3. 3D izdüşüm
    from mpl_toolkits.mplot3d import Axes3D
    ax3 = fig.add_subplot(gs[1, 1:], projection='3d')
    X_3d = X_transformed[:, :3]
    scatter3d = ax3.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], 
                           c=y, cmap='viridis', alpha=0.6)
    ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    ax3.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%})')
    ax3.set_title('3D İzdüşüm')
    plt.colorbar(scatter3d, ax=ax3)
    
    # 4. Özellik önemi
    ax4 = fig.add_subplot(gs[2, :])
    feature_importance = pca.get_feature_importance()
    top_features = np.argsort(feature_importance)[-20:][::-1]
    ax4.barh(range(len(top_features)), feature_importance[top_features])
    ax4.set_yticks(range(len(top_features)))
    ax4.set_yticklabels([f'Özellik {i+1}' for i in top_features])
    ax4.set_xlabel('Önem Skoru')
    ax4.set_title('En Önemli 20 Özellik')
    ax4.grid(True, alpha=0.3, axis='x')
    
    plt.savefig('pca_visualization_results.png', dpi=150, bbox_inches='tight')
    print("\n   Grafik 'pca_visualization_results.png' olarak kaydedildi.")
    
    return pca, X_transformed


def example_feature_selection():
    """PCA ile özellik seçimi örneği"""
    print("\n" + "=" * 60)
    print("PCA ile Özellik Seçimi")
    print("=" * 60)
    
    # Yüksek boyutlu veri
    print("\n1. Yüksek boyutlu veri oluşturuluyor...")
    X, y = generate_sample_data(n_samples=300, n_features=100, n_clusters=3)
    print(f"   Orijinal boyut: {X.shape}")
    
    # PCA ile boyut azaltma
    print("\n2. PCA ile boyut azaltılıyor...")
    pca = PCAAnalyzer(n_components=10)
    X_reduced = pca.fit_transform(X)
    print(f"   Azaltılmış boyut: {X_reduced.shape}")
    print(f"   Boyut azaltma oranı: {(1 - X_reduced.shape[1]/X.shape[1])*100:.1f}%")
    
    # Varyans korunma
    variance_retained = np.sum(pca.explained_variance_ratio_)
    print(f"   Korunan varyans: {variance_retained:.4f} ({variance_retained*100:.2f}%)")
    
    # Geri dönüşüm hatası
    X_reconstructed = pca.inverse_transform(X_reduced)
    reconstruction_error = np.mean((X - X_reconstructed)**2)
    print(f"   Yeniden oluşturma hatası (MSE): {reconstruction_error:.6f}")
    
    return pca, X_reduced


if __name__ == "__main__":
    # Görselleştirme
    pca, X_transformed = example_pca_visualization()
    
    # Özellik seçimi
    pca_feature, X_reduced = example_feature_selection()
    
    plt.show()

